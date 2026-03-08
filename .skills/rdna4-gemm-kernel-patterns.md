# RDNA4 GEMM Kernel Design Patterns

## Overview

This skill documents proven patterns for writing high-performance GEMM kernels on
AMD RDNA4 (gfx1201) using WMMA instructions. These patterns have been validated
to achieve 93-136 TFLOPS on Radeon hardware.

### Kernel Files

| File | Pattern | Performance |
|---|---|---|
| `kernels/wmma_gemm.py` .. `kernels/wmma_gemm_v26.py` | All patterns below (evolution) | 40-134 TFLOPS |
| `kernels/wmma_preshuffle_gemm.py` | Pattern 2 (Preshuffle, production) | 136 TFLOPS (112% rocBLAS) |
| `kernels/wmma_moe_gemm.py` | Two-stage MoE GEMM | See `rdna4-mixed-precision-quantization.md` |
| `kernels/wmma_mixed_preshuffle_gemm.py` | Mixed-precision preshuffle | 249 TFLOPS fp8 (103% rocBLAS) |

## Architecture Constants

```
WMMA_M = 16            # WMMA tile M dimension
WMMA_N = 16            # WMMA tile N dimension
WMMA_K = 16            # WMMA tile K dimension
WAVE_SIZE = 32         # RDNA wave size (NOT 64 like CDNA)
```

## Kernel Configuration Template

A typical high-performance GEMM uses these parameters:

```python
# Register tiling: each wave computes reg_m x reg_n WMMA tiles
reg_m = 4              # 4 WMMA M-tiles per wave (64 M-rows per wave)
reg_n = 4              # 4 WMMA N-tiles per wave (64 N-cols per wave)
reg_k = 2              # 2 WMMA K-steps per tile load (K=32 per tile)

# Wave layout within workgroup
waves_m = 2            # 2 waves along M
waves_n = 2            # 2 waves along N
NUM_WAVES = 4          # Total waves = 2x2

# Derived tile sizes
BLOCK_M = WMMA_M * reg_m * waves_m  # 16*4*2 = 128
BLOCK_N = WMMA_N * reg_n * waves_n  # 16*4*2 = 128
BLOCK_K = WMMA_K * reg_k            # 16*2 = 32
THREADS_PER_BLOCK = NUM_WAVES * WAVE_SIZE  # 128
```

## Pattern 1: LDS-Based GEMM (Standard Approach)

Best for: Arbitrary B layout, when B cannot be pre-shuffled.

### Memory Flow
```
GMEM A [M,K] --buffer_load--> Registers --vector.store--> LDS
GMEM B [K,N] --buffer_load--> Registers --vector.store--> LDS
LDS A, LDS B --vector.load/ds_load--> Registers --WMMA--> Accumulators
```

### LDS Layout

**A in LDS**: Row-major `[BLOCK_M, BLOCK_K]` with optional K-padding for bank conflicts.
```python
BLOCK_K_PAD = BLOCK_K + 8   # 40 elements per row (8 elements padding)
LDS_A_SIZE = BLOCK_M * BLOCK_K_PAD
```

**B in LDS**: Row-major `[BLOCK_K, BLOCK_N + B_PAD]` for reduced bank conflicts.
```python
B_PAD = 8
B_LDS_STRIDE = BLOCK_N + B_PAD
LDS_B_SIZE = BLOCK_K * B_LDS_STRIDE
```

### Double-Buffered Pipeline

```
Prologue:
  Load tile_0 from GMEM -> LDS
  Barrier

Main loop (kt = 0 .. num_k_tiles-2):
  1. Issue GMEM loads for tile_{kt+1}  (non-blocking)
  2. Compute WMMA from current LDS buffer
  3. Store loaded data to alternate LDS buffer
  4. Barrier
  5. Swap read/write buffer offsets

Epilogue:
  Compute WMMA from last tile in LDS
```

### Cooperative GMEM Loading

All threads in the workgroup cooperatively load A and B tiles:

```python
LOAD_VEC = 8  # 8 bf16 = 16 bytes per load (buffer_load_b128)
A_TILE_ELEMS = BLOCK_M * BLOCK_K           # 128 * 32 = 4096
NUM_A_LOADS = A_TILE_ELEMS // (THREADS_PER_BLOCK * LOAD_VEC)  # 4

# Each thread's load position (invariant computation):
a_lin = tid * LOAD_VEC + load_idx * THREADS_PER_BLOCK * LOAD_VEC
a_load_row = a_lin // BLOCK_K
a_load_col = a_lin % BLOCK_K
```

## Pattern 2: Preshuffle GEMM (No LDS, Highest Performance)

Best for: When operands can be pre-arranged on the host. Achieves 134+ TFLOPS.

### Key Insight
Pre-arrange B (and optionally A) in GMEM to match WMMA register layout.
Each thread's buffer_load directly yields the correct WMMA operand.
**No LDS needed** -- register-only pipeline.

### Memory Flow
```
GMEM A_shuf [M0,K0,KLane,MLane,KPack] --buffer_load_b128--> v8bf16 (WMMA A operand)
GMEM B_shuf [N0,K0,KLane,NLane,KPack] --buffer_load_b128--> v8bf16 (WMMA B operand)
```

### Software-Pipelined K-Loop

```python
# Prologue
a_cur = load_a_tile(k=0)
b_cur = load_b_tile(k=0)

# Main loop with inner unrolling
for kt_outer in range(full_outer_iters):    # dynamic scf.for
    for j in range_constexpr(k_unroll):      # compile-time unroll
        a_next = load_a_tile(next_k)
        b_next = load_b_tile(next_k)
        accs = do_compute(accs, a_cur, b_cur)  # overlap compute with loads
        a_cur, b_cur = a_next, b_next

# Epilogue
accs = do_compute(accs, a_cur, b_cur)
```

### Preshuffle Address Computation

```python
def load_a_tile(k_tile_idx):
    m0 = tile_m0 // 16 + wave_m * wave_reg_m + rm
    k0 = k_tile_idx * reg_k + rk
    elem_off = (m0 * A_STRIDE_M0
                + k0 * A_STRIDE_K0
                + klane * A_STRIDE_KLANE
                + lane16 * A_STRIDE_MLANE)
    f32_off = elem_off // 2  # bf16 -> f32 offset for buffer_load
    return buffer_ops.buffer_load(a_rsrc, f32_off, vec_width=4, dtype=f32)
```

## Pattern 3: Inline ASM LDS Reads (Advanced)

For finer control over LDS read scheduling:

```python
# Combined B + A LDS reads with embedded waits
asm_lines = []
# 16x ds_load_u16 for B (scalar reads, one per WMMA B element)
for i in range(16):
    asm_lines.append(f"ds_load_u16 ${i}, $base offset:{even_offsets[i]}")
# 4x ds_load_b128 for A (vectorized reads)
for i in range(4):
    asm_lines.append(f"ds_load_b128 ${16+i}, $a_base offset:{a_offsets[i]}")
# Wait for A loads to complete (B still in flight)
asm_lines.append("s_wait_dscnt 0x4")
# Fill in B high halves
for i in range(16):
    asm_lines.append(f"ds_load_u16_d16_hi ${i}, $base offset:{odd_offsets[i]}")
asm_lines.append("s_wait_dscnt 0x0")
```

## Pattern 4: Inline ASM GMEM Batched Loads

Prevents LLVM from serializing loads due to VGPR reuse:

```python
# s_clause + multiple global_load_b128 in one asm block
asm_lines = [f"s_clause {TOTAL_LOADS - 1}"]
for i in range(TOTAL_LOADS):
    asm_lines.append(f"global_load_b128 ${i}, ${i + TOTAL_LOADS}, off")

result = llvm.inline_asm(
    struct_ty,          # output: struct of TOTAL_LOADS x v4i32
    addr_ptrs,          # input: TOTAL_LOADS pointer VGPRs
    asm_string,
    f"{out_constraints},{in_constraints}",
    has_side_effects=True,
)
```

## L2 Cache Swizzle (Grouped Block Scheduling)

Improves L2 cache reuse by scheduling nearby blocks together:

```python
GROUP_M = 8  # blocks grouped along M dimension

effective_group_m = min(GROUP_M, grid_m)
num_pid_in_group = effective_group_m * grid_n
group_id = pid // num_pid_in_group
first_pid_m = group_id * effective_group_m

pid_in_group = pid % num_pid_in_group
bid_m = first_pid_m + (pid_in_group % effective_group_m)
bid_n = pid_in_group // effective_group_m
```

This maps a linear block index to 2D (bid_m, bid_n) such that blocks in the
same group share M-tile rows, maximizing A reuse in L2.

## Compute-Load Overlap Best Practices

1. **Issue GMEM loads early**: Start loads before compute to hide latency
2. **Use `range_constexpr` for inner loops**: Compile-time unrolling eliminates loop overhead
3. **Use `range()` for outer loops**: Dynamic loops via `scf.for` keep IR size O(1)
4. **Load B first, then interleave A**: Keeps register pressure low
5. **Buffer loads (not raw pointers)**: Buffer descriptors enable hardware bounds checking

## Barrier Patterns for RDNA4

RDNA4 uses explicit barrier signal/wait (not `gpu.barrier()` which adds `buffer_gl_inv`):

```python
# Lightweight barrier (no global invalidate)
llvm.inline_asm(
    res=None, operands_=[],
    asm_string="s_barrier_signal -1\ns_barrier_wait -1",
    constraints="", has_side_effects=True,
)

# Full barrier with DS and store count waits
llvm.inline_asm(
    res=None, operands_=[],
    asm_string="s_wait_dscnt 0x0\ns_wait_storecnt 0x0\ns_barrier_signal -1\ns_barrier_wait -1",
    constraints="", has_side_effects=True,
)

# Wait for global loads
llvm.inline_asm(
    res=None, operands_=[],
    asm_string="s_wait_loadcnt 0x0",
    constraints="", has_side_effects=True,
)
```

## Register Pressure Guidelines

For 128x128x32 tile with 4 waves:

| Resource | Count | Description |
|---|---|---|
| Accumulators | 16 x v8f32 | reg_m * reg_n = 4*4 = 16 per wave |
| A operands | 4 x v8bf16 | reg_m per K-step |
| B operands | 4 x v8bf16 | reg_n per K-step |
| Total VGPRs | ~80-100 | Allows 2-3 waves per SIMD |

Strategy: Load all B for a K-step first, then load A one-at-a-time and compute:
```python
b_vecs = load_all_b(rk, buf_offset)  # 4 B operands in registers
for rm in range(reg_m):
    a_vec = load_single_a(rk, rm, buf_offset)  # 1 A operand
    for rn in range(reg_n):
        accs[rm*reg_n + rn] = wmma(a_vec, b_vecs[rn], accs[rm*reg_n + rn])
```
