# RDNA4 MoE (Mixture of Experts) GEMM Kernel Design

## Overview

MoE kernels route tokens to different experts and compute per-expert GEMMs.
This skill covers the two-stage architecture, token routing, expert dispatch,
block validity checks, and the SiLU fused activation pattern.

### Kernel File

| File | Description |
|---|---|
| `kernels/wmma_moe_gemm.py` | Two-stage preshuffle MoE GEMM (Stage1: gate+up+SiLU, Stage2: down projection) |

## Two-Stage Architecture

### Stage 1: Gate+Up Projection with SiLU

Computes fused gate and up projections with SiLU activation:

```
out[t, slot, :] = SiLU(X[t] @ W_gate[e]) * (X[t] @ W_up[e])
```

- Each expert `e` has two weight matrices: `W_gate[e]` and `W_up[e]`
- Gate and up projections share the same input tokens but produce separate outputs
- SiLU activation is applied to the gate result, then multiplied element-wise with up result
- Output shape: `[tokens, topk, inter_dim]`

### Stage 2: Down Projection with Weighted Reduction

Computes down projection and accumulates with per-token routing weights:

```
Y[t] += weight[t, slot] * A2[t, slot] @ W_down[e]
```

- Input `A2` is the Stage 1 output
- Each expert's contribution is weighted by the routing weight
- Results are accumulated across top-k expert slots into the final output

## Token Routing via `sorted_token_ids`

MoE kernels use a **packed integer** encoding for efficient token-to-expert mapping:

```python
# sorted_token_ids is a 1D i32 array, one entry per (token, slot) pair
# Bit packing:
#   bits[0:24]  = token_id (masked with 0xFFFFFF)
#   bits[24:32] = slot_id (top-k index, 0..topk-1)

mask24 = arith.i32(0xFFFFFF)
c24 = arith.i32(24)

# Load fused id
fused_i = buffer_ops.buffer_load(sorted_rsrc, sorted_row, vec_width=1, dtype=i32)

# Unpack
token_id = arith.andi(fused_i, mask24)         # t_raw = fused_i & 0xFFFFFF
slot_id  = arith.shrui(fused_i, c24)           # s_raw = fused_i >> 24
```

### Per-Token Validity

After unpacking, each token must be checked against the total token count:

```python
t_valid = arith.cmpu(token_id, total_tokens_i32, "ult")
# Use t_valid to mask stores:
buffer_ops.buffer_store(val, out_rsrc, elem_off, mask=t_valid)
```

## Expert Dispatch

### Expert ID Mapping

An `expert_ids` array maps blockIdx.y to expert indices:

```python
# Each workgroup row (blockIdx.y = bx) corresponds to a block of sorted tokens
# expert_ids[bx] tells which expert this block belongs to

expert_idx = buffer_ops.buffer_load(expert_ids_rsrc, bx_i32, vec_width=1, dtype=i32)

# Compute weight base offset for this expert
expert_offset = expert_idx * arith.i32(N_per_expert * K * weight_stride)
```

### Weight Layout

Weights are stored per-expert with preshuffle layout:

```python
# For preshuffle:
#   W_gate[expert, N0, K0, KLane=2, NLane=16, KPack=8]  (bf16 as i16 bytes)
#   W_up  [expert, N0, K0, KLane=2, NLane=16, KPack=8]
#   W_down[expert, N0, K0, KLane=2, NLane=16, KPack=8]
#
# expert_offset selects which expert's weight block to read from
```

## Block Validity Check

MoE kernels skip empty blocks using `max_token_ids`:

```python
# max_token_ids[0] contains the maximum number of valid sorted token rows
bx_m_i32 = arith.index_cast(i32, bx * arith.index(tile_m))
max_token_id = buffer_ops.buffer_load(maxids_rsrc, arith.index(0), vec_width=1, dtype=i32)
blk_valid = arith.cmpu(bx_m_i32, max_token_id, "ult")

# Entire kernel body is wrapped in this condition
_if = scf.IfOp(blk_valid)
with _if.then():
    # ... full GEMM computation ...
```

This is critical for MoE because the routing table may have fewer valid entries
than the maximum grid size. Without this check, blocks would read garbage data.

## Grid Layout for MoE

```python
# blockIdx.x -> N tile (output dimension columns)
# blockIdx.y -> M tile (sorted token block index)
grid_x = inter_dim // tile_n      # number of N-tiles
grid_y = num_expert_blocks         # from routing table (variable per batch)
grid_z = 1

# For Stage 1 (gate+up): inter_dim = hidden_dim * 2 (gate and up concatenated)
# OR separate grid launches for gate and up
# For Stage 2 (down):    inter_dim = model_dim
```

## SiLU Activation (Stage 1)

SiLU is computed inline using AMD-specific fast math intrinsics:

```python
# SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
#
# Implementation using exp2 and rcp intrinsics:
LOG2E_NEG = -1.4426950408889634   # -log2(e)

neg_gate = gate_f32 * arith.f32(LOG2E_NEG)
emu = llvm.call_intrinsic(f32, "llvm.amdgcn.exp2.f32", [neg_gate])
den = arith.f32(1.0) + emu
sig = llvm.call_intrinsic(f32, "llvm.amdgcn.rcp.f32", [den])
silu_val = gate_f32 * sig

# Final fused result: SiLU(gate) * up
result = silu_val * up_f32
```

This avoids the standard `exp` intrinsic and uses `exp2` + `rcp` which map
directly to fast RDNA4 instructions.

## Per-Token Weight Application

Routing weights are applied during the store phase:

```python
# Stage 1 (optional, controlled by doweight_stage1):
if doweight_stage1:
    weight = buffer_ops.buffer_load(weights_rsrc, sorted_row, vec_width=1, dtype=f32)
    result = result * weight

# Stage 2 (always applied):
weight = buffer_ops.buffer_load(weights_rsrc, sorted_row, vec_width=1, dtype=f32)
for si in range_constexpr(8):
    val = vector.extract(acc, static_position=[si])
    val = val * weight
    val_bf16 = arith.trunc_f(bf16, val)
    buffer_ops.buffer_store(val_bf16, out_rsrc, elem_off, mask=t_valid)
```

## Output Layout

Both stages write to a 3D output `[tokens, topk, dim]`:

```python
# Element offset for output store:
# out_offset = (token_id * topk + slot_id) * dim + col
out_offset = (token_id * arith.i32(topk) + slot_id) * arith.i32(dim) + col_i32
```

## Weight Preshuffle Functions

Input activations (X) are NOT preshuffled because tokens are dynamically routed.
Only the expert weights are preshuffled at initialization time:

```python
def preshuffle_w_wmma(W_nk, WMMA_K=16):
    """Preshuffle weight W[N,K] for WMMA B operand layout.
    
    Output: [N0, K0, KLane=2, NLane=16, KPack=8]
    where N0 = N//16, K0 = K//16
    """
    N, K = W_nk.shape
    W_view = W_nk.view(torch.int16)
    W_reshaped = W_view.reshape(N // 16, 16, K // 16, 2, 8)
    return W_reshaped.permute(0, 2, 3, 1, 4).contiguous()

def preshuffle_a_wmma(A_mk, WMMA_K=16):
    """Preshuffle activation A[M,K] for WMMA A operand layout.
    
    Only used for static activations (not MoE input X).
    Output: [M0, K0, KLane=2, MLane=16, KPack=8]
    """
    M, K = A_mk.shape
    A_view = A_mk.view(torch.int16)
    A_reshaped = A_view.reshape(M // 16, 16, K // 16, 2, 8)
    return A_reshaped.permute(0, 2, 3, 1, 4).contiguous()
```

## Key Design Decisions

1. **No LDS for weights** - Preshuffle layout enables direct WMMA B operand loads from GMEM
2. **X loaded cooperatively** - Input activations are loaded from GMEM row-major (not preshuffled)
   because tokens are dynamically routed and cannot be preshuffled at init time
3. **Fused gate+up in Stage 1** - Both projections share the same input X tiles,
   reducing GMEM bandwidth for X by 2x
4. **Packed sorted_token_ids** - Token ID and slot packed in one i32 reduces memory
   traffic for the routing table
5. **Block validity skip** - Early exit for invalid blocks is essential because
   MoE routing is inherently sparse (most experts process only a fraction of tokens)
