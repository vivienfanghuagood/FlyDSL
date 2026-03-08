# RDNA4 Mixed-Precision and Quantization Kernels

## Overview

RDNA4 supports several mixed-precision paths for inference optimization.
This skill covers FP8, INT4 (W4A16), and mixed bf16+fp8 quantization patterns.

### Kernel Files

| File | Precision Paths | Description |
|---|---|---|
| `kernels/wmma_mixed_preshuffle_gemm.py` | fp8+fp8, bf16+fp8, bf16+int4 | Mixed-precision preshuffle GEMM (249T fp8) |
| `kernels/wmma_w4a16_gemv.py` | bf16+int4 | W4A16 GEMV for decode (small-M) |
| `kernels/wmma_moe_gemm.py` | bf16 (with SiLU) | Two-stage MoE GEMM |

## Precision Paths Summary

| Path | A dtype | B dtype | WMMA Used | Use Case |
|---|---|---|---|---|
| bf16+bf16 | bf16 | bf16 | wmma_f32_16x16x16_bf16 | Training, high-accuracy inference |
| fp8+fp8 | fp8_e4m3 | fp8_e4m3 | wmma_f32_16x16x16_fp8_fp8 | 2x throughput inference |
| bf16+fp8 | bf16 | fp8_e4m3 | wmma_f32_16x16x16_fp8_fp8 | Mixed-precision (truncate A) |
| bf16+int4 | bf16 | int4 (packed) | wmma_f32_16x16x16_bf16 | Weight-only quantization |

## INT4 Weight-Only Quantization (W4A16)

### Quantization Scheme (Symmetric Unsigned)

```python
# Quantize B[K,N] f32 -> int4 with per-group scales
group_size = 128
num_groups = K // group_size

# Per-group scale computation
B_grouped = B.reshape(num_groups, group_size, N)
amax = B_grouped.abs().amax(dim=1).clamp(min=1e-10)
scales = amax / 7.0  # [-7, 7] symmetric range

# Quantize: q = round(x / scale * 7) + 8, clamp [0, 15]
B_q = torch.round(B_grouped / amax * 7.0).to(int8) + 8
B_q = B_q.clamp(0, 15).to(uint8)

# Pack two int4 per byte: low nibble first
B_packed = B_even | (B_odd << 4)   # [K//2, N] uint8
```

### Dequantization: `(uint4_val - 8) * scale`

Optimized as FMA: `uint4_val * scale + (-8 * scale)`

```python
bias = -8.0 * scale_val   # pre-compute bias

for ni in range(8):       # 8 int4 values per i32
    shift = ni * 4        # each nibble is 4 bits
    nibble = (packed_i32 >> shift) & 0xF
    nibble_f32 = arith.uitofp(f32, nibble)
    dequant_f32 = nibble_f32 * scale_val + bias  # FMA
    bf16_val = arith.trunc_f(bf16, dequant_f32)
```

### W4A16 GEMV Design (Small-M Inference)

Optimized for decode phase (M=16..64):

```python
# Design choices:
# - Parallelize across N (each workgroup handles tile_n=128 columns)
# - Each workgroup iterates full K dimension
# - Weight layout: B_packed_t[N, K//2] (K contiguous for coalesced loads)
# - Scale layout:  scales_t[N, K//group_size] (same N-first order)

# Wave layout: all waves along N
waves_n = num_waves  # e.g., 4
wave_reg_n = (tile_n // WMMA_N) // waves_n

# Each wave handles wave_reg_n WMMA N-tiles
# M is small, so reg_m = M // 16 (1-4 WMMA M-tiles)
```

### INT4 Preshuffle for WMMA

Pack int4 values matching WMMA B operand layout:

```python
# B_int4_packed: [K//2, N] uint8
# -> B_shuf: [N0, K0, KLane=2, NLane=16, KPack_bytes=4] uint8
# where 4 bytes = 8 int4 values per lane per K-half

B_reshaped = B_packed.reshape(K0, 2, 4, N0, 16)  # 4 bytes per KLane
B_shuffled = B_reshaped.permute(3, 0, 1, 4, 2).contiguous()

# Scales: [K//group_size, N] -> [N0, num_groups, NLane=16] f32
scales_shuf = scales.reshape(num_groups, N0, 16).permute(1, 0, 2).contiguous()
```

## FP8 Path

### FP8 Preshuffle

Same structure as bf16 but with 8-byte operands (vs 16-byte for bf16):

```python
# A fp8 preshuffle: [M0, K0, KLane=2, MLane=16, KPack=8] bytes
# Each lane loads 8 fp8 bytes = 2xi32 via buffer_load dwordx2

def preshuffle_a_fp8(A_mk):
    A_view = A_mk.view(torch.uint8)
    A_reshaped = A_view.reshape(M0, 16, K0, 2, 8)
    return A_reshaped.permute(0, 2, 3, 1, 4).contiguous()
```

### FP8 WMMA Invocation

```python
# FP8 operands are vector<2xi32> (8 bytes = 8 fp8 values)
result = rocdl.wmma_f32_16x16x16_fp8_fp8(
    v8f32_ty,
    [a_v2i32, b_v2i32, acc]  # acc is vector<8xf32>
)
```

### FP8 with Per-Tensor Scaling

```python
# Load scales (scalar per tensor)
scale_a = buffer_ops.buffer_load(scale_a_rsrc, index(0), vec_width=1, dtype=f32)
scale_b = buffer_ops.buffer_load(scale_b_rsrc, index(0), vec_width=1, dtype=f32)
combined_scale = scale_a * scale_b

# After accumulation, apply scale to each output element
for si in range(8):
    val = vector.extract(acc, static_position=[si])
    val = val * combined_scale
    val_bf16 = arith.trunc_f(bf16, val)
    buffer_ops.buffer_store(val_bf16, c_rsrc, elem_off)
```

## Fast Math Intrinsics for Activation Functions

### SiLU (Sigmoid Linear Unit)

Used in MoE gate+up projection: `SiLU(x) = x * sigmoid(x)`

```python
# sigmoid(x) = 1 / (1 + exp(-x))
# Using exp2 intrinsic for speed:
neg_gate = gate_val * (-1.4426950408889634)  # -log2(e)
emu = llvm.call_intrinsic(f32, "llvm.amdgcn.exp2.f32", [neg_gate])
den = 1.0 + emu
sig = llvm.call_intrinsic(f32, "llvm.amdgcn.rcp.f32", [den])
silu_gate = gate_val * sig
result = silu_gate * up_val
```

### exp2 for Softmax

```python
LOG2E = 1.4426950408889634
# exp(x) = exp2(x * log2(e))
exp_val = flydsl_math.exp2(arith.as_value(x * LOG2E))
```

## MoE (Mixture of Experts) Kernel Design

### Two-Stage Architecture

**Stage 1**: Gate+Up projection with SiLU activation
```
out[t, slot, :] = SiLU(X[t] @ W_gate[e]) * (X[t] @ W_up[e])
```

**Stage 2**: Down projection with topk-weighted reduction
```
Y[t] += weight[t,slot] * A2[t,slot] @ W_down[e]
```

### Token Routing

MoE kernels use sorted_token_ids to map workgroup rows to actual tokens:

```python
# sorted_token_ids packs token_id and slot in one i32:
#   bits[0:24]  = token_id (masked with 0xFFFFFF)
#   bits[24:32] = slot_id (top-k index)
fused_i = buffer_load(sorted_rsrc, sorted_row, vec_width=1, dtype=i32)
token_id = fused_i & 0xFFFFFF
slot_id = fused_i >> 24
```

### Block Validity Check

MoE kernels check `max_token_ids` to skip empty blocks:

```python
bx_m_i32 = arith.index_cast(i32, bx * tile_m)
max_token_id = buffer_load(maxids_rsrc, index(0), vec_width=1, dtype=i32)
blk_valid = arith.cmpu(bx_m_i32, max_token_id, "ult")
with scf.IfOp(blk_valid).then():
    # ... kernel body ...
```

### Grid Layout for MoE

```python
# blockIdx.x -> N tile (output dimension)
# blockIdx.y -> M tile (expert block from routing)
gx = inter_dim / tile_n
gy = num_expert_blocks  # from routing table
```
