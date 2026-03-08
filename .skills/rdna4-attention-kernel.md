# RDNA4 Decode Attention Kernel Design

## Overview

Decode attention computes single-query attention against cached KV sequences.
Three proven approaches exist, each optimal for different BS×KV regimes:

| Approach | Kernel | Best For | File |
|---|---|---|---|
| WMMA Hybrid (single WG) | Q@K^T via WMMA, element-wise P@V | BS≤8, KV≤512 | `kernels/wmma_decode_attention.py` |
| Split-KV (two-stage) | Stage1: parallel KV splits, Stage2: merge | BS≥8 or KV≥512 | `kernels/wmma_decode_attention_splitkv.py` |
| Element-wise (single WG) | Wave-reduction dot products, no WMMA | Analysis/reference only | `kernels/wmma_decode_attention_elemwise.py` |

## Performance Results (vs Triton, GPU-only us, Qwen3-8B config)

| BS | KV | Best FlyDSL | Kernel | Triton | Speedup |
|---|---|---|---|---|---|
| 1 | 256 | 25.0 | WMMA single | 48.5 | 1.94x |
| 8 | 256 | 24.3 | WMMA single | 47.2 | 1.94x |
| 16 | 256 | 37.7 | Split-KV | 47.3 | 1.25x |
| 1 | 512 | 35.0 | WMMA single | 47.0 | 1.34x |
| 8 | 512 | 38.0 | Split-KV | 47.2 | 1.24x |
| 1 | 1024 | 38.0 | Split-KV | 47.2 | 1.24x |
| 32 | 1024 | 299.7 | Split-KV | 310.8 | 1.04x |

FlyDSL wins in 13/15 configurations tested.

## Critical Lesson: Dispatch Overhead Dominates at Small Scale

The single most important discovery: **FlyDSL's Python dispatch overhead was 71us per call**,
masking actual GPU compute time (~9us for BS=1 KV=256). After fixing the executor
(caching wrappers, memoizing ctypes structs), dispatch dropped to 18us. Without this
fix, all kernel approaches showed an identical ~80us floor regardless of actual work.

**Rule**: Always measure dispatch overhead separately (no-sync timing) before
concluding a kernel is slow. See `rdna4-dispatch-and-benchmarking.md` for methodology.

## Approach 1: WMMA Hybrid (Single Workgroup)

### Architecture

```
Grid:  (batch, num_q_heads) -- one workgroup per (batch, q_head)
Block: num_waves x 32 threads (e.g., 128 threads = 4 waves)
Each wave processes interleaved blocks of BLOCK_N=16 KV tokens
```

### Why WMMA for Q@K^T but Element-wise for P@V?

- **Q@K^T** is `[1, head_dim] x [head_dim, 16]` -- fits WMMA perfectly
  - Q loaded once, reused across all KV blocks (8 vec8 loads for head_dim=128)
  - K block of 16 tokens becomes WMMA B operand
  - 8 WMMA ops for head_dim=128 (128/16 = 8 K-tiles)

- **P@V** is `[1, 16] x [16, head_dim]` -- WMMA is counterproductive
  - Staging V into LDS for WMMA B format requires 512 scalar LDS reads per block
  - Element-wise: each thread handles `head_dim/32 = 4` V elements directly
  - V loaded directly from GMEM, no LDS staging needed
  - The WMMA overhead for P@V negates any compute benefit (verified experimentally)

### Q Loading (Once, Reused)

```python
# Load Q into WMMA A operand format
# Only lane16=0 (row 0) gets actual Q data, other rows are zero
q_wmma_vecs = []
for kt in range_constexpr(num_k_tiles):  # 8 tiles for head_dim=128
    k_offset = arith.index(kt * WMMA_K) + base8  # klane * 8
    q_raw = buffer_ops.buffer_load(q_rsrc, q_base + k_offset, vec_width=8, dtype=bf16)
    zero = arith.constant_vector(0.0, v8bf16_ty)
    q_selected = arith.select(is_row0, q_raw, zero)  # only lane16==0 gets Q
    q_wmma_vecs.append(vector.bitcast(v8i16_ty, q_selected))
```

### K Loading and Score Computation

```python
# Each lane16 loads one KV token's K vector
kv_local_idx = kv_start_token + lane16  # 16 tokens per block
kv_valid = kv_local_idx < seq_len_idx

# Indirect token lookup via kv_indices
kv_global = kv_start_idx + kv_local_idx
safe_global = arith.select(kv_valid, kv_global, kv_start_idx)
token_i32 = buffer_ops.buffer_load(indices_rsrc, safe_global, vec_width=1, dtype=i32)
token_idx = arith.index_cast(idx_type, token_i32)
k_base_token = token_idx * kv_token_stride + kv_head_off

# Load K for WMMA B operand
k_wmma_vecs = []
for kt in range_constexpr(num_k_tiles):
    k_raw = buffer_ops.buffer_load(k_rsrc, k_base_token + kt*16 + base8, vec_width=8, dtype=bf16)
    k_selected = arith.select(kv_valid, k_raw, zero_vec)
    k_wmma_vecs.append(vector.bitcast(v8i16_ty, k_selected))

# WMMA Q@K^T: accumulate across all K-tiles
qk_acc = arith.constant_vector(0.0, v8f32_ty)
for kt in range_constexpr(num_k_tiles):
    qk_acc = rocdl.wmma_f32_16x16x16_bf16(
        v8f32_ty, [q_wmma_vecs[kt], k_wmma_vecs[kt], arith.unwrap(qk_acc)])

# Extract score: row 0 result = klane=0, si=0 -> element 0
score_raw = vector.extract(qk_acc, static_position=[0], dynamic_position=[])
score_scaled = score_raw * c_sm_scale
score_masked = arith.select(kv_valid, score_scaled, neg_inf_val)
score_final = arith.select(is_klane0, score_masked, neg_inf_val)
```

### Score Broadcasting via LDS

Scores from WMMA reside only in lanes 0-15 (klane=0). All 32 lanes need all 16 scores:

```python
# Write: each lane writes its score to LDS (within same wave, no barrier needed)
score_lds_idx = wave_id * 32 + lane
memref.store(_unwrap(score_final), smem_scores, [_unwrap(score_lds_idx)])

# Read: all lanes read all 16 scores
for ni in range_constexpr(block_n):  # 16
    s_idx = wave_id * 32 + arith.index(ni)
    scores_f32[ni] = memref.load(smem_scores, [arith.as_value(s_idx)])
```

### Online Softmax with exp2

```python
LOG2E = 1.4426950408889634  # log2(e)

# Rescale existing accumulator when max changes
n_emax = arith.maximum(e_max, blk_max)
rescale = flydsl_math.exp2(arith.as_value((e_max - n_emax) * c_log2e))
e_sum = e_sum * rescale
e_max = n_emax

# Rescale existing V accumulator
for ei in range_constexpr(elems_per_thread):
    acc[ei] = acc[ei] * rescale

# Accumulate attention-weighted V
for ni in range_constexpr(BLOCK_N):  # 16 tokens
    p = flydsl_math.exp2(arith.as_value((scores_f32[ni] - e_max) * c_log2e))
    e_sum = e_sum + p
    # Load V for token ni, accumulate p * V
    v_vec_bf16 = buffer_ops.buffer_load(v_rsrc, v_off, vec_width=4, dtype=bf16)
    v_vec_f32 = flir.arith.extf(v4f32_ty, arith.as_value(v_vec_bf16))
    for ei in range_constexpr(elems_per_thread):
        v_f32 = vector.extract(v_vec_f32, static_position=[ei], dynamic_position=[])
        acc[ei] = acc[ei] + p * v_f32
```

### Cross-Wave Merge

After all KV blocks, merge partial results across waves via LDS:

```python
# Each wave writes: partial output, e_max, e_sum
for ei in range_constexpr(elems_per_thread):
    out_lds_idx = wave_id * head_dim + lane * elems_per_thread + arith.index(ei)
    memref.store(_unwrap(acc[ei]), smem_out, [_unwrap(out_lds_idx)])
memref.store(_unwrap(e_max), smem_max, [_unwrap(wave_id)])
memref.store(_unwrap(e_sum), smem_sum, [_unwrap(wave_id)])

gpu.barrier()

# Merge: rescale and combine across all waves
for w in range_constexpr(num_waves):
    w_max = memref.load(smem_max, [arith.as_value(arith.index(w))])
    w_sum = memref.load(smem_sum, [arith.as_value(arith.index(w))])
    n_merge_max = arith.maximum(merge_emax, w_max)
    old_scale = flydsl_math.exp2(arith.as_value((merge_emax - n_merge_max) * c_log2e))
    new_scale = flydsl_math.exp2(arith.as_value((w_max - n_merge_max) * c_log2e))
    merge_esum = merge_esum * old_scale + w_sum * new_scale
    for ei in range_constexpr(elems_per_thread):
        lds_idx = arith.index(w) * head_dim + lane * ept + arith.index(ei)
        w_val = memref.load(smem_out, [arith.as_value(lds_idx)])
        merge_acc[ei] = merge_acc[ei] * old_scale + w_val * new_scale
    merge_emax = n_merge_max

# Normalize and store
for ei in range_constexpr(elems_per_thread):
    out_val = merge_acc[ei] / merge_esum
    buffer_ops.buffer_store(arith.trunc_f(bf16, out_val), o_rsrc, o_offset + ei)
```

## Approach 2: Split-KV (Two-Stage)

### Architecture

```
Stage 1 Grid: (batch, num_q_heads, max_kv_splits)
  - Each workgroup processes a subset of KV tokens
  - Same WMMA Q@K^T + element-wise P@V as Approach 1
  - Outputs: att_out[batch, head, split, head_dim] (f32, normalized by local sum)
             att_lse[batch, head, split] (f32, log-sum-exp = max + log(sum))

Stage 2 Grid: (batch, num_q_heads)
  - Single wave per workgroup merges partial results
  - Log-sum-exp rescaling across splits
  - Outputs: O[batch, head, head_dim] (bf16)
```

### Split Range Computation (Must Match Triton Exactly)

```python
_MIN_BLOCK_KV = 32  # Alignment constant (same as Triton)

# Per-batch splits (Python-side)
max_kv_splits = max(1, math.ceil(kv_len / (_MIN_BLOCK_KV * 2)))
max_kv_splits = min(max_kv_splits, 32)

# In kernel: compute this split's range
inner_cdiv = (seq_len_idx + kv_splits_idx - c1) // kv_splits_idx
kv_len_per_split = ((inner_cdiv + c_min_block_kv - c1) // c_min_block_kv) * c_min_block_kv
split_kv_start = kv_len_per_split * split_kv_id
split_kv_end = min(split_kv_start + kv_len_per_split, seq_len)
```

### Stage 1 Output Format

```python
# att_out stores normalized partial results (divided by local sum)
inv_sum = 1.0 / final_esum
for ei in range_constexpr(elems_per_thread):
    out_val = final_acc[ei] * inv_sum
    buffer_ops.buffer_store(out_val, att_out_rsrc, att_out_offset + ei)

# att_lse stores log-sum-exp: max + log(sum)
lse_val = final_emax + flydsl_math.log(arith.as_value(final_esum))
buffer_ops.buffer_store(lse_val, att_lse_rsrc, att_lse_offset)
```

### Stage 2 Merge

```python
# For each split with work:
lse_val = buffer_ops.buffer_load(att_lse_rsrc, lse_off, ...)
n_e_max = arith.maximum(e_max, lse_val)
old_scale = flydsl_math.exp(arith.as_value(e_max - n_e_max))
exp_logic = flydsl_math.exp(arith.as_value(lse_val - n_e_max))
e_sum = e_sum * old_scale + exp_logic
for ei in range_constexpr(elems_per_thread):
    v_val = buffer_ops.buffer_load(att_out_rsrc, v_off, ...)
    acc[ei] = acc[ei] * old_scale + exp_logic * v_val
e_max = n_e_max
```

### When to Use Split-KV vs Single Workgroup

- **Single WG wins** when per-workgroup latency < dispatch overhead of second kernel
- **Split-KV wins** when KV is large enough that parallelizing across CUs reduces total time
- Crossover point: approximately BS×num_heads×KV > 32×32×512 (depends on CU count)
- In practice: use single WG for BS≤8 KV≤512, split-KV otherwise

## Approach 3: Element-wise (Reference)

No WMMA at all. Pure dot-product with wave-level reduction for Q@K^T:

```python
# Each lane handles 4 elements of head_dim=128
dot_local = zero_f32
for ei in range_constexpr(elems_per_thread):
    k_elem = vector.extract(k_vec_f32, static_position=[ei], dynamic_position=[])
    dot_local = dot_local + q_elems[ei] * k_elem

# Wave-level reduction (5 rounds for 32 lanes)
dot_val = dot_local
for shift in [16, 8, 4, 2, 1]:
    shift_val = arith.constant(shift, type=i32_type)
    width_val = arith.constant(32, type=i32_type)
    shuf = gpu.ShuffleOp(_unwrap(dot_val), _unwrap(shift_val), _unwrap(width_val), mode="xor")
    shuf_val = arith.ArithValue(shuf.shuffleResult)
    dot_val = dot_val + shuf_val
```

**Result**: Same ~80us floor as WMMA before dispatch fix. After fix, slightly slower than
WMMA for small KV (25us WMMA vs 27us element-wise at KV=256) due to 5 shuffle rounds
per token. WMMA amortizes across 16 tokens per WMMA op.

## LDS Budget

```python
# Cross-wave merge buffers:
smem_wave_max  = num_waves * 4          # 16 bytes (4 waves)
smem_wave_sum  = num_waves * 4          # 16 bytes
smem_wave_out  = num_waves * 128 * 4    # 2048 bytes (4 waves * 128 * sizeof(f32))
smem_scores    = num_waves * 32 * 4     # 512 bytes (4 * 32 * sizeof(f32))
# Total: ~2.6 KB -- very small, allows high occupancy
```

## GQA (Grouped Query Attention) Support

```python
kv_group_num = num_q_heads // num_kv_heads  # e.g., 32 // 8 = 4
cur_kv_head = cur_q_head // arith.index(kv_group_num)
```

## Paged KV Cache Access

```python
# kv_indptr: [batch+1] CSR-style pointers into kv_indices
# kv_indices: [total_kv_tokens] physical token indices in K/V buffers
kv_start = buffer_ops.buffer_load(indptr_rsrc, cur_batch, vec_width=1, dtype=i32)
kv_end = buffer_ops.buffer_load(indptr_rsrc, cur_batch + c1, vec_width=1, dtype=i32)
seq_len = kv_end - kv_start

# Map logical KV position to physical token
kv_global = kv_start_idx + kv_local_idx
token_i32 = buffer_ops.buffer_load(indices_rsrc, kv_global, vec_width=1, dtype=i32)
token_idx = arith.index_cast(idx_type, token_i32)
# Access K/V: buffer[token_idx, kv_head, :]
k_base = token_idx * num_kv_heads * head_dim + cur_kv_head * head_dim
```

## Common Bugs and Fixes

### 1. V LDS Shared Across Waves (Data Race)
If using WMMA for P@V with LDS staging, each wave needs its own V LDS region:
```python
# Wrong: shared V LDS
v_lds = allocator.allocate_array(f32, V_LDS_SIZE)
# Right: per-wave V LDS
v_lds = allocator.allocate_array(f32, num_waves * V_LDS_SIZE)
v_lds_offset = wave_id * V_LDS_SIZE
```

### 2. WMMA C Output Lane Mapping Mismatch
WMMA C result has layout `C[klane*8+si, lane16]`. When writing to LDS for broadcast,
must use `nt*16 + lane16` (WMMA N-tile layout), not `lane*4 + ei` (sequential).

### 3. LDS Write Race for klane=0/klane=1
Both klane groups have same lane16, so they write to the same LDS address.
Fix: use XOR shuffle to broadcast klane=0's value to klane=1 before write:
```python
shuf = gpu.ShuffleOp(_unwrap(val), _unwrap(arith.constant(16, i32)),
                      _unwrap(arith.constant(32, i32)), mode="xor")
val = arith.ArithValue(shuf.shuffleResult)  # klane=1 now has klane=0's value
```

### 4. Variables Defined Inside `if` in `range()` Loops
FlyDSL's AST rewriter has issues when variables defined inside `if` blocks are also
loop-carried. Workaround: use `arith.select()` instead of `if/else`:
```python
# Wrong: may cause AST rewriter issues
for blk in range(num_blocks):
    if blk < threshold:
        x = compute_a()
    else:
        x = compute_b()

# Right: use select
for blk in range(num_blocks):
    val_a = compute_a()
    val_b = compute_b()
    x = arith.select(blk < threshold, val_a, val_b)
```

### 5. Loop-Carried Vector Accumulators
When `acc[i]` becomes a `BlockArgument` inside SCF for-loop:
```python
# May need explicit wrapping for arithmetic
val = arith.ArithValue(pv_accs[nt])  # for arithmetic ops
raw = arith.unwrap(pv_accs[nt])      # for WMMA operand
```

## Test Configuration (Qwen3-8B)

```python
num_attention_heads = 32
num_key_value_heads = 8   # GQA, kv_group_num = 4
head_dim = 128
num_hidden_layers = 36
```
