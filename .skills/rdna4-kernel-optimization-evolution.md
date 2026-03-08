# RDNA4 Kernel Optimization Evolution: Lessons from 26 GEMM Iterations

## Overview

The FlyDSL repository contains 26+ iterations of WMMA GEMM kernels for gfx1201,
evolving from a basic implementation to 134+ TFLOPS. This skill captures the
key optimization lessons and decision points.

## Kernel File Index

| Kernel | File | Peak Performance | Description |
|---|---|---|---|
| GEMM v1-v26 | `kernels/wmma_gemm.py` .. `kernels/wmma_gemm_v26.py` | 134 TFLOPS (v15+) | Iterative WMMA GEMM evolution |
| Preshuffle GEMM | `kernels/wmma_preshuffle_gemm.py` | 136 TFLOPS (112% rocBLAS) | No-LDS preshuffle GEMM, production |
| MoE GEMM | `kernels/wmma_moe_gemm.py` | — | Two-stage MoE (gate+up SiLU, down) |
| Mixed-Prec GEMM | `kernels/wmma_mixed_preshuffle_gemm.py` | 249 TFLOPS (fp8, 103% rocBLAS) | fp8+fp8, bf16+fp8, bf16+int4 paths |
| W4A16 GEMV | `kernels/wmma_w4a16_gemv.py` | — | INT4 weight-only GEMV for decode |
| Decode Attn (WMMA) | `kernels/wmma_decode_attention.py` | 25us BS=1 KV=256 | WMMA Q@K^T + elem P@V, single WG |
| Decode Attn (Split-KV) | `kernels/wmma_decode_attention_splitkv.py` | 38us BS=1 KV=1024 | Two-stage split-KV for large KV |
| Decode Attn (Elem) | `kernels/wmma_decode_attention_elemwise.py` | 27us BS=1 KV=256 | No-WMMA reference implementation |

### Related Skills Files

| Skill | Coverage |
|---|---|
| `rdna4-gemm-kernel-patterns.md` | GEMM design patterns (LDS, preshuffle, inline ASM) |
| `rdna4-wmma-register-layout.md` | WMMA lane mapping, preshuffle layout |
| `rdna4-buffer-ops-and-memory.md` | Buffer load/store, LDS, memory coalescing |
| `rdna4-mixed-precision-quantization.md` | FP8, INT4, W4A16, MoE patterns |
| `rdna4-attention-kernel.md` | Decode attention (3 approaches, perf results) |
| `rdna4-flydsl-kernel-skeleton.md` | FlyDSL API reference, kernel template |
| `rdna4-dispatch-and-benchmarking.md` | Dispatch overhead fix, benchmarking methodology |

## Evolution Summary

### v1 (Baseline): LDS-based A and B with inline ASM LDS reads
- Both A and B loaded from GMEM -> LDS -> registers
- Used `ds_load_u16` + `ds_load_u16_d16_hi` for B operands
- **Problem**: WAW hazard on B reads serialized throughput
- **Performance**: ~40-50 TFLOPS

### v2 (Preshuffle B): Eliminated B LDS entirely
- Pre-shuffle B on host into WMMA-friendly layout
- Each thread's buffer_load directly yields WMMA B operand
- A still goes through LDS with single-buffer pipeline
- **Key insight**: Removing LDS for B eliminates ds_load WAW hazards
- **Performance**: ~65-70 TFLOPS

### v3-v14: Incremental optimizations
- Double-buffered LDS for A (ping-pong)
- Scheduling hints (`sched_group_barrier`)
- Better load/compute overlap
- Inline ASM batched global loads (`s_clause + global_load_b128`)
- XOR-swizzle on LDS stores for bank conflict avoidance

### v15 (Preshuffle A+B): No LDS at all
- Both A and B pre-shuffled on host
- Register-only pipeline: GMEM -> registers -> WMMA
- Software-pipelined K-loop with inner unrolling
- **Performance**: 134 TFLOPS (110% of rocBLAS at 4096^3)

### v25-v26: Alternative approaches
- v25: 8 warps (256 threads) -- explored higher occupancy
- v26: 4 warps with XOR-swizzle LDS (no K-padding)
  - 16KB per LDS buffer (vs 20KB with K-padding)
  - Load all B first, then A one-at-a-time per WMMA

## Key Optimization Principles

### 1. Eliminate LDS When Possible

LDS introduces latency, bank conflicts, and barriers. If data can be pre-arranged
in GMEM to match the WMMA register layout, direct GMEM loads are superior:

```
LDS path:  GMEM -> registers -> LDS (barrier) -> registers -> WMMA
Direct path: GMEM -> registers -> WMMA (no barrier needed between waves)
```

**Trade-off**: Preshuffle requires host-side layout transformation (one-time cost).
For weights in inference, this is negligible. For activations, it may add overhead.

### 2. Avoid LLVM VGPR Reuse Serialization

LLVM's register allocator may reuse VGPRs across loads, serializing them.
Use `s_clause` + batched loads in inline ASM to prevent this:

```
s_clause 7                    // hint: 8 consecutive loads follow
global_load_b128 v[0:3], ...  // load 1 (all 8 issued in parallel)
global_load_b128 v[4:7], ...  // load 2
...
global_load_b128 v[28:31], ...// load 8
```

### 3. Software Pipeline Design

The optimal pipeline overlaps:
- Current tile's WMMA compute
- Next tile's GMEM loads

```
Prologue: Load tile 0
Main loop:
  Load tile_{k+1}     <-- issue early (non-blocking)
  Compute tile_k      <-- overlaps with loads above
  (barrier if LDS)
  Swap buffers
Epilogue: Compute last tile
```

### 4. K-Unroll + Dynamic Outer Loop

Use `range_constexpr` for inner unroll and `range()` for outer loop:

```python
full_outer_iters = (num_k_tiles - 1) // k_unroll
for kt_outer in range(full_outer_iters):      # dynamic scf.for
    for j in range_constexpr(k_unroll):         # compile-time unroll
        # load next + compute current
```

This keeps IR size O(1) while still unrolling the critical inner loop.

### 5. LDS Bank Conflict Strategies

When LDS is required, two strategies avoid bank conflicts:

**K-padding**: Add 8 elements per row (wastes 20% LDS space)
```python
BLOCK_K_PAD = BLOCK_K + 8  # 32 + 8 = 40
lds_idx = row * BLOCK_K_PAD + col
```

**XOR-swizzle**: XOR K-group index with row (no wasted space)
```python
k_group = col // LOAD_VEC
k_group_swizzled = k_group ^ (row % 4)
lds_idx = row * BLOCK_K + k_group_swizzled * LOAD_VEC + (col % LOAD_VEC)
```

### 6. Register Pressure Management

For 128x128 tiles with 4x4 register tiling:
- 16 accumulators x 8 elements = 128 f32 values = 128 VGPRs
- Strategy: Load B first (4 vectors), then load A one-at-a-time

```python
b_vecs = load_all_b(rk)           # 4 B vectors in registers
for rm in range(reg_m):
    a_vec = load_single_a(rk, rm)  # 1 A vector at a time
    for rn in range(reg_n):
        accs[rm*reg_n + rn] = wmma(a_vec, b_vecs[rn], accs[...])
```

### 7. Scheduling Hints (Use Carefully)

`sched_group_barrier` can help or hurt:

```python
# Interleave VMEM reads, DS reads, and WMMA
rocdl.sched_vmem(2)   # Start 2 buffer loads
rocdl.sched_dsrd(2)   # Start 2 LDS reads
for group in range(8):
    rocdl.sched_mfma(4)  # 4 WMMAs
    rocdl.sched_vmem(1)  # 1 buffer load
rocdl.sched_barrier(0)   # Reset scheduling state
```

**Lesson from v2**: Removing scheduling hints improved performance from 64 to 68
TFLOPS because LLVM's native scheduler produced better interleaving. Test both.

### 8. Output Store Optimization

Use `buffer_ops.buffer_store` instead of `memref.store` for GMEM writes:

```python
# Bad: memref.store (generates flat_store, no coalescing hints)
memref.store(val, C, [row, col])

# Good: buffer_store (generates buffer_store_*, hardware coalescing)
buffer_ops.buffer_store(val, c_rsrc, row * N + col)
```

## Phase 6: FlashAttention Decode (3 Approaches, 1 Runtime Fix)

### Approach A: Full WMMA (Q@K^T + P@V via WMMA) -- CORRECT BUT SLOW
- Used WMMA for both Q@K^T and P@V
- **Problem**: P@V required staging V through LDS (512 scalar LDS reads per block)
- LDS staging overhead negated WMMA benefit
- Bugs found: V LDS shared across waves (data race), output LDS mapping mismatch,
  klane=0/klane=1 write race (fixed with XOR shuffle)
- Performance: same as scalar (~80us, dispatch-bound)

### Approach B: Hybrid WMMA Q@K^T + Element-wise P@V -- BEST SINGLE-KERNEL
- WMMA for Q@K^T (efficient 16-token score computation)
- Element-wise P@V (each thread handles 4 V elements, no LDS needed)
- **25us GPU time at BS=1 KV=256** (after dispatch fix)
- File: `kernels/wmma_decode_attention.py`

### Approach C: Split-KV (Two-Stage) -- BEST FOR LARGE KV
- Stage 1: parallel KV splits, each split computes partial softmax
- Stage 2: merge across splits using log-sum-exp rescaling
- **38us at BS=1 KV=1024** (constant regardless of KV within split capacity)
- File: `kernels/wmma_decode_attention_splitkv.py`

### Approach D: Pure Element-wise (No WMMA) -- REFERENCE ONLY
- Wave-level dot products with 5-round shuffle reduction for Q@K^T
- Slightly slower than WMMA hybrid (27us vs 25us at KV=256)
- Confirmed WMMA provides ~10% benefit for Q@K^T at this scale
- File: `kernels/wmma_decode_attention_elemwise.py`

### Critical Discovery: Dispatch Overhead
- All approaches showed identical ~80us floor before fix
- Root cause: FlyDSL executor rebuilt wrapper closure on every `exe()` call (71us)
- Fix: cache wrappers, memoize ctypes structures, pre-compute arg metadata
- After fix: 18us dispatch (2.6x faster than Triton's 48us)
- **Lesson**: Always measure dispatch separately; GPU compute can be hidden by CPU overhead

### Key Optimization Lesson
For decode attention (single query), the choice between WMMA and element-wise for
Q@K^T is minor (~10% difference). The dominant factors are:
1. **Dispatch overhead** (18-48us, dominates at small BS/KV)
2. **Parallelism strategy** (single WG vs split-KV, determines scaling)
3. **P@V approach** (element-wise always beats WMMA for P@V due to LDS staging cost)

## Performance Profiling Tips

### Environment for IR dumps
```bash
FLIR_DUMP_IR=1 FLIR_DUMP_DIR=./dumps python my_kernel.py
# Produces: 00_target_overridden.mlir through 15_final_isa.s
```

### Key ISA patterns to look for
- `buffer_load_b128`: Optimal 128-bit global loads
- `v_wmma_f32_16x16x16_bf16`: WMMA instructions
- `s_clause N`: Batched load hint (N+1 consecutive loads)
- `s_wait_loadcnt`: Global load wait (lower count = less stalling)
- `s_barrier_signal`/`s_barrier_wait`: Workgroup synchronization
- `ds_load_b128`: LDS 128-bit load (for LDS-based kernels)

### Register usage check
Look at the ISA dump header for VGPR/SGPR counts:
```
; NumVGPRs: 88
; NumSGPRs: 42
; Occupancy: 8 waves per SIMD (max)
```

Target: < 96 VGPRs for 8 waves/SIMD, < 128 for 4 waves/SIMD.
