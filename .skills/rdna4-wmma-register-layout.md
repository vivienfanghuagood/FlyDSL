# RDNA4 WMMA Register Layout (gfx1201, wave32)

## Overview

AMD RDNA4 (gfx12xx) GPUs use **Wave Matrix Multiply-Accumulate (WMMA)** instructions
for accelerated matrix operations. These operate on **wave32** wavefronts (32 threads
per wave), performing 16x16x16 matrix multiply-accumulate per instruction.

All kernels in this project use these layouts:
- BF16 WMMA: `kernels/wmma_preshuffle_gemm.py`, `kernels/wmma_decode_attention.py`
- FP8 WMMA: `kernels/wmma_mixed_preshuffle_gemm.py`
- Preshuffle layout: `kernels/wmma_preshuffle_gemm.py`, `kernels/wmma_mixed_preshuffle_gemm.py`

## Instruction Variants

| Instruction | Input A | Input B | Accumulator | Notes |
|---|---|---|---|---|
| `v_wmma_f32_16x16x16_f16` | f16 | f16 | f32 | Standard half-precision |
| `v_wmma_f32_16x16x16_bf16` | bf16 | bf16 | f32 | Brain float, most common for LLM |
| `v_wmma_f32_16x16x16_fp8_fp8` | fp8_e4m3 | fp8_e4m3 | f32 | RDNA4 only, 2x throughput |
| `v_wmma_f32_16x16x16_fp8_bf8` | fp8_e4m3 | bf8_e5m2 | f32 | Mixed fp8 types |
| `v_wmma_f16_16x16x16_f16` | f16 | f16 | f16 | Half-precision accumulate |
| `v_wmma_bf16_16x16x16_bf16` | bf16 | bf16 | bf16 | BF16 accumulate |
| `v_wmma_i32_16x16x16_iu8` | int8 | int8 | i32 | Integer GEMM |
| `v_wmma_i32_16x16x16_iu4` | int4 | int4 | i32 | Integer GEMM |
| `v_wmma_i32_16x16x32_iu4` | int4 | int4 | i32 | gfx12 only, 2x K |

## Lane Mapping (Critical for Correctness)

In wave32, each of the 32 lanes is indexed as:

```
lane     = thread_id % 32        (0..31)
lane16   = lane % 16             (0..15) -- selects M-row (A) or N-column (B/C)
klane    = lane // 16            (0 or 1) -- selects K-half
base8    = klane * 8             (0 or 8)
```

### A Operand ("row-of-cols")

Lane `t` loads `A[t % 16, (t // 16) * 8 + i]` for `i = 0..7`

- `lane16` selects the **M-row** within the 16x16 tile
- `klane` selects which half of K: `klane=0` -> K[0:8], `klane=1` -> K[8:16]
- Each lane holds 8 contiguous elements along K dimension
- Register type: `vector<8xbf16>` (or f16, fp8)

### B Operand ("col-of-rows")

Lane `t` loads `B[(t // 16) * 8 + i, t % 16]` for `i = 0..7`

- `lane16` selects the **N-column** within the 16x16 tile
- `klane` selects which half of K: same as A
- Each lane holds 8 elements from different K rows at the same N column
- Register type: `vector<8xbf16>` (or f16, fp8)

### C/D Result ("col-of-rows")

Lane `t` holds `D[(t // 16) * 8 + i, t % 16]` for `i = 0..7`

- `lane16` -> N-column
- `klane*8 + si` -> M-row (si iterates 0..7)
- Register type: `vector<8xf32>` (f32 accumulation)
- Each lane holds 8 results: `D[base8+0][lane16]` through `D[base8+7][lane16]`

## BF16 Bitcast Requirement

For bf16 WMMA, operands must be bitcast to `vector<8xi16>` before the intrinsic call:

```python
a_i16 = vector.bitcast(v8i16_ty, a_vec)   # v8bf16 -> v8i16
b_i16 = vector.bitcast(v8i16_ty, b_vec)   # v8bf16 -> v8i16
result = rocdl.wmma_f32_16x16x16_bf16(
    v8f32_ty,               # result type
    [a_i16, b_i16, acc]     # operands: [A, B, C_accumulator]
)
```

For f16 WMMA, operands are passed directly as `vector<8xf16>`:

```python
result = rocdl.wmma_f32_16x16x16_f16(
    v8f32_ty,
    [a_vec, b_vec, acc]
)
```

## FP8 Operand Format

For fp8 WMMA, operands are `vector<2xi32>` (8 bytes = 8 fp8 values):

```python
result = rocdl.wmma_f32_16x16x16_fp8_fp8(
    v8f32_ty,
    [a_v2i32, b_v2i32, acc]
)
```

## Preshuffle Layout for Direct GMEM Loading

To avoid LDS staging, operands can be pre-arranged in memory to match WMMA register layout.

### A Preshuffle: `[M0, K0, KLane, MLane, KPack]`

```
M0    = M // 16          (WMMA M tiles)
K0    = K // 16          (WMMA K tiles)
KLane = 2                (lane // 16)
MLane = 16               (lane % 16 -> M-row)
KPack = 8                (8 elements per lane)
```

Strides (element units):
```
STRIDE_MLANE = KPack = 8
STRIDE_KLANE = 16 * KPack = 128
STRIDE_K0    = 2 * 16 * KPack = 256
STRIDE_M0    = K0_total * STRIDE_K0
```

Python preshuffle:
```python
def preshuffle_a(A_mk):
    M, K = A_mk.shape
    M0, K0 = M // 16, K // 16
    A_reshaped = A_mk.reshape(M0, 16, K0, 2, 8)
    return A_reshaped.permute(0, 2, 3, 1, 4).contiguous()
```

### B Preshuffle: `[N0, K0, KLane, NLane, KPack]`

```
N0    = N // 16
K0    = K // 16
KLane = 2
NLane = 16               (lane % 16 -> N-column)
KPack = 8
```

Python preshuffle:
```python
def preshuffle_b(B_kn):
    K, N = B_kn.shape
    N0, K0 = N // 16, K // 16
    B_reshaped = B_kn.reshape(K0, 2, 8, N0, 16)
    return B_reshaped.permute(3, 0, 1, 4, 2).contiguous()
```

## Extracting Results from Accumulator

After WMMA, the accumulator is `vector<8xf32>`. To store results:

```python
for si in range(8):
    g_row = tile_m0 + wmma_m_off + base8 + si  # klane*8 + si = M-row
    g_col = tile_n0 + wmma_n_off + lane16       # N-column
    val = vector.extract(acc, static_position=[si])
    # Optional: truncate f32 -> bf16
    val_bf16 = arith.trunc_f(bf16_ty, val)
    buffer_ops.buffer_store(val_bf16, c_rsrc, g_row * N + g_col)
```

## Key Differences from CDNA (MFMA)

| Feature | RDNA4 WMMA | CDNA MFMA |
|---|---|---|
| Wavefront size | 32 | 64 |
| Tile size | 16x16x16 | 16x16x16 (or 32x32x8) |
| Registers per lane | 8 elements | 4 elements |
| LDS bank count | 32 banks | 32 banks |
| Matrix unit | WMMA (smaller) | MFMA (larger) |
| Best pipeline | Preshuffle B + compute overlap | Async LDS pipeline |
