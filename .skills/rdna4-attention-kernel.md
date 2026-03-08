# RDNA4 Decode Attention Kernel Design

## Overview

Decode attention computes single-query attention against cached KV sequences.
The RDNA4 implementation uses a **hybrid approach**: WMMA for Q@K^T score computation,
element-wise accumulation for P@V.

## Architecture

```
Grid:  (batch, num_q_heads) -- one workgroup per (batch, q_head)
Block: num_waves waves x 32 threads (e.g., 128 threads = 4 waves)
Each wave processes interleaved blocks of BLOCK_N=16 KV tokens
```

## Design Decisions

### Why WMMA for Q@K^T but Element-wise for P@V?

- **Q@K^T** is `[1, head_dim] x [head_dim, 16]` -- fits WMMA perfectly
  - Q is loaded once and reused across all KV blocks
  - K block of 16 tokens becomes WMMA B operand
  - 8 WMMA ops for head_dim=128 (128/16 = 8 K-tiles)

- **P@V** is `[1, 16] x [16, head_dim]` -- scattered access pattern
  - Each attention score multiplies one V row
  - Element-wise: each thread handles `head_dim/32 = 4` V elements
  - Avoids complex LDS staging for V

### Q Loading (Once, Reused)

```python
# Load Q into WMMA A operand format
# Q is [head_dim] for one query -- broadcast to 16x16 WMMA tile
# Only lane16=0 (row 0) gets actual Q data, other rows are zero
q_wmma_vecs = []
for kt in range(num_k_tiles):  # 8 tiles for head_dim=128
    k_offset = kt * WMMA_K + base8
    q_raw = buffer_load(q_rsrc, q_base + k_offset, vec_width=8, dtype=bf16)
    zero = constant_vector(0.0, v8bf16_ty)
    q_selected = arith.select(is_row0, q_raw, zero)  # only lane16==0 gets Q
    q_wmma_vecs.append(vector.bitcast(v8i16_ty, q_selected))
```

### K Loading and Score Computation

```python
# Each lane16 loads one KV token's K vector
kv_local_idx = kv_start_token + lane16  # 16 tokens per block
kv_valid = kv_local_idx < seq_len

# Load K for WMMA B operand (head_dim values across 8 WMMA tiles)
k_wmma_vecs = []
for kt in range(num_k_tiles):
    k_raw = buffer_load(k_rsrc, k_base + kt*16 + base8, vec_width=8, dtype=bf16)
    k_selected = arith.select(kv_valid, k_raw, zero_vec)
    k_wmma_vecs.append(vector.bitcast(v8i16_ty, k_selected))

# WMMA Q@K^T: accumulate across all K-tiles
qk_acc = constant_vector(0.0, v8f32_ty)
for kt in range(num_k_tiles):
    qk_acc = wmma_f32_16x16x16_bf16(v8f32_ty, [q_vecs[kt], k_vecs[kt], qk_acc])

# Extract score from WMMA C output row 0 (klane=0, si=0)
score = vector.extract(qk_acc, static_position=[0])
```

### Online Softmax

```python
# Rescale existing accumulator when max changes
n_emax = arith.maximum(e_max, blk_max)
rescale = exp2((e_max - n_emax) * LOG2E)
e_sum = e_sum * rescale
e_max = n_emax

# Rescale all V accumulator elements
for ei in range(elems_per_thread):
    acc[ei] = acc[ei] * rescale

# Accumulate attention-weighted V
for ni in range(BLOCK_N):  # 16 tokens
    p = exp2((scores[ni] - e_max) * LOG2E)
    e_sum += p
    v_vec = buffer_load(v_rsrc, v_offset, vec_width=elems_per_thread, dtype=bf16)
    v_f32 = arith.extf(v4f32_ty, v_vec)
    for ei in range(elems_per_thread):
        acc[ei] += p * v_f32[ei]
```

### Score Broadcasting via LDS

Scores computed by WMMA reside in specific lanes (lane16=0..15, klane=0).
To make all 32 lanes see all 16 scores, broadcast through LDS:

```python
# Write: each lane writes its score to LDS
score_lds_idx = wave_id * 32 + lane
memref.store(score_final, smem_scores, [score_lds_idx])

# Read: all lanes read all 16 scores
for ni in range(16):
    s_idx = wave_id * 32 + ni
    scores[ni] = memref.load(smem_scores, [s_idx])
```

### Cross-Wave Merge

After processing all KV blocks, merge partial results across waves via LDS:

```python
# Each wave writes: partial output, e_max, e_sum
memref.store(e_max, smem_max, [wave_id])
memref.store(e_sum, smem_sum, [wave_id])
for ei in range(elems_per_thread):
    out_idx = wave_id * head_dim + lane * elems_per_thread + ei
    memref.store(acc[ei], smem_out, [out_idx])

gpu.barrier()

# Merge: rescale and combine across all waves
for w in range(num_waves):
    w_max = memref.load(smem_max, [w])
    w_sum = memref.load(smem_sum, [w])
    n_merge_max = arith.maximum(merge_emax, w_max)
    old_scale = exp2((merge_emax - n_merge_max) * LOG2E)
    new_scale = exp2((w_max - n_merge_max) * LOG2E)
    merge_esum = merge_esum * old_scale + w_sum * new_scale
    for ei in range(elems_per_thread):
        w_val = memref.load(smem_out, [w * head_dim + lane * ept + ei])
        merge_acc[ei] = merge_acc[ei] * old_scale + w_val * new_scale
    merge_emax = n_merge_max

# Final normalize and store
for ei in range(elems_per_thread):
    out_val = merge_acc[ei] / merge_esum
    buffer_store(trunc_f(bf16, out_val), o_rsrc, o_offset + ei)
```

## LDS Budget for Attention

```python
# Cross-wave merge buffers:
smem_wave_max  = num_waves * sizeof(f32)        # 16 bytes
smem_wave_sum  = num_waves * sizeof(f32)        # 16 bytes
smem_wave_out  = num_waves * head_dim * sizeof(f32)  # 2048 bytes (4 waves * 128 * 4)
smem_scores    = num_waves * WAVE_SIZE * sizeof(f32)  # 512 bytes (4 * 32 * 4)
# Total: ~2.6 KB -- very small, allows high occupancy
```

## GQA (Grouped Query Attention) Support

```python
# Map Q head to KV head
kv_group_num = num_q_heads // num_kv_heads  # e.g., 32 // 8 = 4
cur_kv_head = cur_q_head // kv_group_num
```

## Paged KV Cache Access

```python
# kv_indptr: [batch+1] CSR-style pointers into kv_indices
# kv_indices: [total_kv_tokens] physical token indices
kv_start = buffer_load(indptr_rsrc, cur_batch, vec_width=1, dtype=i32)
kv_end = buffer_load(indptr_rsrc, cur_batch + 1, vec_width=1, dtype=i32)
seq_len = kv_end - kv_start

# Map logical KV position to physical token
kv_global = kv_start + kv_local_idx
token_i32 = buffer_load(indices_rsrc, kv_global, vec_width=1, dtype=i32)
# Access K/V: K_buf[token_idx, kv_head, :]
k_base = token_idx * num_kv_heads * head_dim + cur_kv_head * head_dim
```
