# RDNA4 Kernel Development Guide

**Purpose**: Drive an agent through the full lifecycle of writing, verifying, profiling,
and optimizing a GPU kernel using FlyDSL. The agent must produce a **Correctness Report**
and an **Optimization Report** documenting every effort and result.

---

## Skill Index

| Skill File | What It Covers | When to Use |
|---|---|---|
| `flydsl-api-dictionary.md` | All FlyDSL functions/classes with signatures and examples | Every kernel — the primary API reference |
| `rdna4-flydsl-kernel-skeleton.md` | Complete boilerplate (MlirModule, @kernel, @jit, LaunchFuncOp), common pitfalls | Starting any new kernel |
| `rdna4-wmma-register-layout.md` | WMMA lane mapping, operand register layout, bf16/fp8 bitcast rules | Any kernel using WMMA/matrix ops |
| `rdna4-gemm-kernel-patterns.md` | 4 proven GEMM patterns (LDS-based, preshuffle, inline ASM) | GEMM/matmul kernels |
| `rdna4-buffer-ops-and-memory.md` | Buffer descriptors, load/store, LDS, bank conflicts, coalescing | Any kernel doing memory access |
| `rdna4-mixed-precision-quantization.md` | INT4 W4A16, FP8, dequantization, per-group scaling | Quantized inference kernels |
| `rdna4-moe-gemm-kernel.md` | Token routing, expert dispatch, SiLU, two-stage MoE | MoE/expert kernels |
| `rdna4-attention-kernel.md` | Decode attention (WMMA hybrid, Split-KV, element-wise), online softmax | Attention kernels |
| `rdna4-kernel-optimization-evolution.md` | 8 optimization principles from 26 GEMM iterations | Performance tuning any kernel |
| `rdna4-dispatch-and-benchmarking.md` | Dispatch overhead, benchmarking methodology, timing patterns | Benchmarking and profiling |

---

## Workflow Overview

```
Phase 1: Design & Implement ─── Write the kernel, get it compiling
Phase 2: Correctness         ─── Verify against reference, produce Correctness Report
Phase 3: Profile & Analyze   ─── Measure perf, roofline analysis, identify bottleneck
Phase 4: Optimize (loop)     ─── Apply targeted fix → re-profile → record result
Phase 5: Final Report        ─── Deliver Correctness Report + Optimization Report
```

The agent MUST track every optimization attempt and its measured impact.

---

## Phase 1: Design & Implement

### 1.1 Understand the problem

Determine before writing code:

| Question | Why It Matters |
|---|---|
| What is the computation? (matmul, attention, reduction, ...) | Selects kernel pattern |
| What are the shapes? (M, N, K, batch, seq_len) | Determines tile sizes, grid dimensions |
| What precision? (bf16, fp8, int4, mixed) | Selects WMMA variant, data layout |
| Is data preshuffled or row-major? | Determines memory access strategy |
| Latency-sensitive or throughput? (decode vs prefill) | Drives grid size and dispatch concerns |

### 1.2 Select kernel pattern and skills

```
What kind of kernel?
│
├─ Matrix multiply (GEMM)
│  ├─ Large M,N,K (prefill)
│  │   Skills: rdna4-gemm-kernel-patterns.md, rdna4-wmma-register-layout.md
│  │   Ref:    kernels/wmma_preshuffle_gemm.py, kernels/wmma_gemm_v26.py
│  ├─ Small M (decode GEMV)
│  │   Skills: rdna4-mixed-precision-quantization.md
│  │   Ref:    kernels/wmma_w4a16_gemv.py
│  └─ MoE routing
│      Skills: rdna4-moe-gemm-kernel.md, rdna4-gemm-kernel-patterns.md
│      Ref:    kernels/wmma_moe_gemm.py
│
├─ Attention
│  ├─ Decode (small Q)
│  │   Skills: rdna4-attention-kernel.md, rdna4-wmma-register-layout.md
│  │   Ref:    kernels/wmma_decode_attention.py
│  └─ Prefill (large Q)
│      Skills: rdna4-attention-kernel.md
│
├─ Quantized inference
│  │   Skills: rdna4-mixed-precision-quantization.md, rdna4-wmma-register-layout.md
│  │   Ref:    kernels/wmma_mixed_preshuffle_gemm.py
│
├─ Elementwise / Reduction / Normalization
│  │   Skills: flydsl-api-dictionary.md (arith, vector, scf, gpu sections)
│  │   Ref:    tests/kernels/test_softmax.py, test_layernorm.py, test_rmsnorm.py
│
└─ Something new
    Skills: rdna4-flydsl-kernel-skeleton.md + flydsl-api-dictionary.md +
            rdna4-buffer-ops-and-memory.md + rdna4-wmma-register-layout.md
```

### 1.3 Write the kernel

1. Copy skeleton from `rdna4-flydsl-kernel-skeleton.md`
2. Look up every API call in `flydsl-api-dictionary.md` — match by section:

   | Task | Dictionary Section |
   |---|---|
   | Constants, arithmetic | Section 2 (arith) |
   | Global memory load/store | Section 3 (buffer_ops) |
   | WMMA instructions | Section 4 (rocdl) |
   | Vector extract/bitcast/shuffle | Section 5 (vector) |
   | Loops (for/while) and if/else | Section 6 (scf) |
   | Thread IDs, barriers, shuffles | Section 7 (gpu) |
   | LDS load/store | Section 8 (memref) + Section 12 (SmemAllocator) |
   | Fast math intrinsics | Section 9 (llvm) |
   | Type constructors | Section 13 (Types / T) |

3. Check `rdna4-wmma-register-layout.md` for WMMA operand rules:
   - bf16 must be bitcast to `v8i16` before WMMA
   - fp8 operands are `v2i32`
   - Result is `v8f32`

4. Check `rdna4-buffer-ops-and-memory.md` for memory access rules:
   - `buffer_ops.create_buffer_resource()` for every tensor
   - Offsets are in elements (API converts to bytes)
   - Predicated loads use `mask=` parameter

5. Check `rdna4-flydsl-kernel-skeleton.md` for the common pitfalls checklist

### 1.4 Compile and fix errors

```bash
cd /root/e2e/FlyDSL
ROCR_VISIBLE_DEVICES=0 \
PYTHONPATH="$(pwd)/flydsl/src:$(pwd)/.flir/build/python_packages/flydsl:$(pwd):${PYTHONPATH}" \
python tests/kernels/test_my_kernel.py
```

Common compilation errors:

| Error | Cause | Fix |
|---|---|---|
| `ArithValue not accepted` | Wrapper passed to raw MLIR | `arith.unwrap(val)` |
| `type mismatch in operand` | Wrong vector type for WMMA | Check bitcast rules |
| `offset must be i32` | index passed to buffer_load | `arith.index_cast(i32, val)` |
| `scf.for body not terminated` | Missing yield | Add `scf.yield_([...])` |
| `does not dominate use` | Value defined inside if | Use `scf.IfOp` with results |
| Shared memory overflow | LDS > 64KB | Reduce `SmemAllocator` total |

---

## Phase 2: Correctness

### 2.1 Write reference implementation

**Rules**:
1. Compute in **float32**: `A.float() @ B.float()`, never `A @ B` in reduced precision
2. Use `torch.manual_seed(42)` for reproducibility
3. Scale inputs: `* 0.1` to avoid bf16 overflow
4. Mask sparse outputs: MoE and attention may have zero-padded slots

### 2.2 Correctness metrics: rtol and atol

Use **rtol** (relative tolerance) and **atol** (absolute tolerance) as the primary
correctness metrics. These check every element individually and catch outliers that
aggregate metrics like cosine similarity would hide.

**Why not cosine similarity?** Cosine similarity measures global directional agreement
across all elements. A result with a few wildly wrong values (e.g., 10x off) can still
show cos_sim > 0.999 if the remaining thousands of elements are close. This masks
real bugs. Use `rtol` and `atol` with per-element checking instead.

**Per-element correctness check**:
```
For each element i:
  |actual[i] - expected[i]| <= atol + rtol * |expected[i]|
```

This is the same formula as `torch.allclose(actual, expected, rtol=R, atol=A)`.

### 2.3 Correctness test structure

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import torch
from flydsl.runtime.device import get_rocm_arch

def reference_compute(A, B):
    return (A.float() @ B.float())  # always float32

def check_correctness(actual, expected, rtol, atol, label=""):
    """Per-element rtol/atol check with detailed error reporting."""
    diff = (actual.float() - expected.float()).abs()
    threshold = atol + rtol * expected.float().abs()
    violations = diff > threshold
    num_violations = violations.sum().item()
    total = actual.numel()
    pass_rate = 1.0 - num_violations / total

    # Worst-case metrics
    max_abs_err = diff.max().item()
    max_rel_err = (diff / (expected.float().abs() + 1e-8)).max().item()

    # Percentile errors (more informative than max alone)
    sorted_diff = diff.flatten().sort().values
    p99_abs_err = sorted_diff[int(0.99 * total)].item()
    p999_abs_err = sorted_diff[min(int(0.999 * total), total - 1)].item()

    result = {
        "pass": num_violations == 0,
        "pass_rate": pass_rate,
        "num_violations": num_violations,
        "total_elements": total,
        "max_abs_err": max_abs_err,
        "max_rel_err": max_rel_err,
        "p99_abs_err": p99_abs_err,
        "p999_abs_err": p999_abs_err,
        "rtol": rtol,
        "atol": atol,
    }
    status = "PASS" if result["pass"] else "FAIL"
    print(f"[{label}] {status}: {num_violations}/{total} violations "
          f"(max_abs={max_abs_err:.6f}, max_rel={max_rel_err:.6f}, "
          f"p99_abs={p99_abs_err:.6f}, pass_rate={pass_rate*100:.4f}%)")
    return result

def test_correctness(M, N, K):
    torch.manual_seed(42)
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.1
    B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda") * 0.1
    C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    expected = reference_compute(A, B)

    exe = compile_my_kernel(M=M, N=N, K=K)
    exe(C, A, B, torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    return check_correctness(C, expected, rtol=1e-2, atol=1e-3,
                             label=f"{M}x{N}x{K}")
```

### 2.4 Tolerance thresholds

| Kernel Type | rtol | atol | Rationale |
|---|---|---|---|
| GEMM (bf16) | 1e-2 | 1e-3 | bf16 has ~3 decimal digits of precision |
| GEMM (fp8) | 5e-2 | 1e-2 | FP8 has ~1.5 decimal digits |
| Attention | 1e-2 | 1e-3 | Softmax amplifies errors in tail |
| MoE | 5e-2 | 1e-2 | Routing + SiLU + multi-expert accumulation |
| W4A16 | 5e-2 | 1e-2 | INT4 quantization noise |
| Elementwise | 1e-5 | 1e-6 | Same-precision ops should be nearly exact |
| Softmax / Norm | 1e-3 | 1e-4 | Output distribution sensitive to precision |

**If strict rtol/atol fails**, report per-element pass rate and percentile errors:
- **pass_rate >= 99.9%** with reasonable max error: likely acceptable (tail outliers from precision)
- **pass_rate < 99%**: real bug — investigate the failing elements' positions for patterns
  (e.g., tile boundaries, last row/column, masked regions)

### 2.5 Run across multiple shapes

Test at least 3 shape categories:
- **Small**: exercises edge cases (M=16, N=16, K=16)
- **Medium**: typical workload (M=256, N=256, K=256 or problem-specific)
- **Large**: production scale (M=4096, N=4096, K=4096 or problem-specific)

### 2.7 Produce Correctness Report

```
## Correctness Report: <kernel_name>

### Test Configuration
- GPU: gfx1201 (RDNA4)
- Precision: bf16 (or fp8/int4)
- Reference: float32 PyTorch
- Tolerances: rtol=1e-2, atol=1e-3

### Results

| Shape (M,N,K) | Max Abs Err | Max Rel Err | P99 Abs Err | Pass Rate  | Status |
|---|---|---|---|---|---|
| 256,256,256    | 0.00123     | 0.0031      | 0.00089     | 100.0000%  | PASS   |
| 2048,2048,2048 | 0.00456     | 0.0089      | 0.00234     | 100.0000%  | PASS   |
| 4096,4096,4096 | 0.00567     | 0.0092      | 0.00312     | 99.9998%   | PASS   |

### Verdict: PASS (all shapes within bf16 GEMM tolerance, rtol=1e-2, atol=1e-3)
```

---

## Phase 3: Profile & Analyze

### 3.1 Baseline measurement

Measure the kernel AND the baseline(s) using the same methodology.

**Compute-bound kernels** (GEMM, attention matmul):

```python
def benchmark(M, N, K, iters=100):
    # ... setup ...
    # Warmup
    for _ in range(10):
        exe(C, A, B, stream_ptr)
    torch.cuda.synchronize()

    # Timed loop
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        exe(C, A, B, stream_ptr)
    torch.cuda.synchronize()
    avg_ms = (time.perf_counter() - t0) / iters * 1000

    tflops = 2 * M * N * K / (avg_ms / 1000) / 1e12
    return avg_ms, tflops

# Baseline: PyTorch/rocBLAS
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(iters):
    torch.mm(A, B, out=C_ref)
torch.cuda.synchronize()
rocblas_ms = (time.perf_counter() - t0) / iters * 1000
rocblas_tflops = 2 * M * N * K / (rocblas_ms / 1000) / 1e12
```

**Memory-bound kernels** (softmax, layernorm, elementwise):

```python
total_bytes = read_bytes + write_bytes
# Example: softmax reads MxN, writes MxN -> 2 * M * N * elem_size
bandwidth_gbs = total_bytes / (avg_us / 1e6) / 1e9
```

**Latency-sensitive kernels** (decode attention, GEMV):

Also measure dispatch overhead separately — see `rdna4-dispatch-and-benchmarking.md`:
```python
# Dispatch-only (no sync)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(5000):
    exe(args...)
t1 = time.perf_counter()
dispatch_us = (t1 - t0) / 5000 * 1e6
torch.cuda.synchronize()
```

### 3.2 Roofline analysis

Determine whether the kernel is compute-bound or memory-bound.

#### RDNA4 gfx1201 hardware specs

| Parameter | Value |
|---|---|
| CUs | 64 |
| SIMDs per CU | 2 |
| Wave size | 32 |
| VGPRs per SIMD | 512 (max 256 per wave) |
| LDS per workgroup | 64 KB |
| Peak bf16 TFLOPS | ~122 TFLOPS |
| Peak fp8 TFLOPS | ~244 TFLOPS |
| Peak memory bandwidth | ~492 GB/s (measured) |
| L2 cache | 4 MB |
| Kernel launch overhead | ~5.6 us (Triton noop) |
| FlyDSL dispatch overhead | ~18 us (optimized) |

#### Arithmetic intensity calculation

```
Arithmetic Intensity (AI) = FLOPs / Bytes_accessed

For GEMM C[M,N] = A[M,K] * B[K,N]:
  FLOPs = 2 * M * N * K
  Bytes  = (M*K + K*N + M*N) * elem_bytes
  AI     = 2*M*N*K / ((M*K + K*N + M*N) * elem_bytes)

For bf16 (2 bytes): AI = 2*M*N*K / ((M*K + K*N + M*N) * 2)
For 4096x4096x4096 bf16: AI = 2*4096^3 / (3*4096^2*2) = 1365 FLOPs/byte
```

#### Roofline ridge point

```
Ridge Point = Peak TFLOPS / Peak Bandwidth
            = 122 TFLOPS / 0.492 TB/s
            = 248 FLOPs/byte

If AI > 248: kernel is compute-bound → optimize for WMMA utilization
If AI < 248: kernel is memory-bound → optimize for bandwidth utilization
```

#### Roofline achievable performance

```
Achievable TFLOPS = min(Peak_TFLOPS, AI * Peak_BW)

For GEMM 4096^3 bf16: AI=1365 >> 248, so compute-bound
  Achievable = 122 TFLOPS (peak)
  Measured   = 134 TFLOPS (preshuffle)
  Efficiency = 134/122 = 110% (exceeds nominal peak via instruction overlap)

For softmax 1x8192 bf16: AI = 2 FLOPs/byte, memory-bound
  Achievable = 2 * 492 GB/s = 0.984 TFLOPS
  Optimize for bandwidth, not compute
```

### 3.3 ISA analysis

Dump and analyze the generated assembly:

```bash
FLIR_DUMP_IR=1 FLIR_DUMP_DIR=my_ir_dumps python test_my_kernel.py
# Check: my_ir_dumps/<kernel_name>/15_final_isa.s
```

**Extract from ISA header** (look for these in the `.s` file):
```
; NumVGPRs: 88         → register pressure
; NumSGPRs: 42         → scalar register usage
; ScratchSize: 0       → spilling (must be 0 for good perf)
; LDS Size: 16384      → shared memory usage
```

**Occupancy from VGPR count**:

| VGPRs | Waves/SIMD | Occupancy Level |
|---|---|---|
| <= 96 | 5 (max on RDNA4) | Maximum |
| <= 128 | 4 | High |
| <= 168 | 3 | Medium |
| <= 256 | 2 | Low |
| > 256 | 1 | Minimum (avoid) |

**Key ISA patterns to check**:

| Pattern | What It Means | Good/Bad |
|---|---|---|
| `s_clause N` | Batched N+1 consecutive loads | Good — memory latency hiding |
| `v_wmma_f32_16x16x16_bf16` | WMMA instruction | Good — matrix compute |
| `buffer_load_b128` | 128-bit global load | Good — max bandwidth |
| `buffer_load_b32` | 32-bit global load | Bad — 4x less bandwidth |
| `scratch_load/store` | Register spilling | Bad — VGPR overflow |
| `s_waitcnt vmcnt(0)` | Waiting for all loads | Bad — no latency hiding |
| `s_wait_loadcnt 0` | Waiting for all loads (RDNA4) | Bad — same issue |
| `ds_load_b128` | 128-bit LDS load | Good — max LDS bandwidth |

### 3.4 Gap analysis

Compare measured performance against achievable ceiling:

```
Performance Gap = 1 - (Measured / Achievable)

For compute-bound GEMM:
  Measured = 100 TFLOPS, Peak = 122 TFLOPS
  Gap = 1 - 100/122 = 18%
  → Room for improvement: optimize WMMA utilization, reduce stalls

For memory-bound softmax:
  Measured BW = 350 GB/s, Peak BW = 492 GB/s
  Gap = 1 - 350/492 = 29%
  → Room for improvement: optimize coalescing, reduce redundant loads
```

**Identify the bottleneck**:

| Symptom | Bottleneck | Action |
|---|---|---|
| TFLOPS < 50% of peak, ISA shows many `s_waitcnt` | Memory latency | Add prefetching / software pipeline |
| TFLOPS < 50% of peak, ISA shows `scratch_load` | Register spilling | Reduce VGPRs, smaller tiles |
| TFLOPS plateaus at ~80% of peak | Instruction scheduling | Try scheduling hints (`sched_barrier`) |
| BW < 50% of peak | Poor coalescing | Check access patterns, add vectorization |
| BW < 50% of peak, high LDS usage | LDS bank conflicts | Add K-padding or XOR-swizzle |
| Dispatch time > GPU time | Dispatch overhead | Measure dispatch separately (see 3.1) |

---

## Phase 4: Optimize (Iterative Loop)

**Skill**: `rdna4-kernel-optimization-evolution.md`

### Optimization strategy priority

Apply in this order. **After each change, re-measure and record the result.**

| Priority | Strategy | Applies When | Expected Gain |
|---|---|---|---|
| 1 | Eliminate LDS (preshuffle layout) | GEMM with LDS bottleneck | 1.5-3x |
| 2 | Maximize memory coalescing | Any kernel with strided access | 1.2-2x |
| 3 | Fix VGPR reuse serialization | ISA shows back-to-back WAW | 1.1-1.3x |
| 4 | Software pipeline (double buffer) | GMEM loads stalling WMMA | 1.2-1.5x |
| 5 | K-unroll (2x or 4x) | Compute-bound, short inner loop | 1.1-1.3x |
| 6 | LDS bank conflict avoidance | LDS-based kernel, ds_load stalls | 1.1-1.2x |
| 7 | Register pressure reduction | VGPRs > 128, low occupancy | 1.1-1.3x |
| 8 | L2 cache swizzle | Large problem, poor L2 hit rate | 1.05-1.15x |
| 9 | Scheduling hints | Fine-tuning after other opts | varies |

### Iteration tracking template

For EACH optimization attempt, record:

```
### Attempt N: <strategy name>

**Change**: <what was modified>
**Hypothesis**: <why this should help>
**Measured**:
  - Before: <TFLOPS / BW / latency>
  - After:  <TFLOPS / BW / latency>
  - Delta:  <+X% or -Y%>
**ISA impact**: VGPRs <before→after>, scratch <before→after>
**Verdict**: KEEP / REVERT
**Notes**: <observations, surprises>
```

### When to stop

| Condition | Action |
|---|---|
| >= 90% of rocBLAS/peak | Stop — production ready |
| 80-90% of peak | Acceptable. Try 1-2 more strategies, then stop |
| 50-80% of peak | Review memory access patterns and ISA for obvious issues |
| < 50% of peak | Fundamental design issue — reconsider the kernel pattern |

---

## Phase 5: Final Reports

The agent MUST deliver two reports at the end.

### 5.1 Correctness Report

```
═══════════════════════════════════════════════════════
CORRECTNESS REPORT: <kernel_name>
═══════════════════════════════════════════════════════

Kernel:      <kernel_name>
File:        kernels/<kernel_file>.py
Test:        tests/kernels/test_<kernel_name>.py
GPU:         gfx1201 (RDNA4)
Precision:   <bf16 / fp8 / int4 / mixed>
Reference:   float32 PyTorch
Tolerances:  rtol=1e-2, atol=1e-3

RESULTS:
┌──────────────────┬─────────────┬─────────────┬─────────────┬───────────┬────────┐
│ Shape            │ Max Abs Err │ Max Rel Err │ P99 Abs Err │ Pass Rate │ Status │
├──────────────────┼─────────────┼─────────────┼─────────────┼───────────┼────────┤
│ 256x256x256      │ 0.00123     │ 0.0031      │ 0.00089     │ 100.000%  │ PASS   │
│ 2048x2048x2048   │ 0.00456     │ 0.0089      │ 0.00234     │ 100.000%  │ PASS   │
│ 4096x4096x4096   │ 0.00567     │ 0.0092      │ 0.00312     │ 99.999%   │ PASS   │
└──────────────────┴─────────────┴─────────────┴─────────────┴───────────┴────────┘

VERDICT: PASS — all shapes within tolerance (rtol=1e-2, atol=1e-3 for bf16 GEMM)

Note: Pass Rate = % of elements satisfying |actual-expected| <= atol + rtol*|expected|
      P99 Abs Err = 99th percentile of per-element absolute errors
```

### 5.2 Optimization Report

```
═══════════════════════════════════════════════════════
OPTIMIZATION REPORT: <kernel_name>
═══════════════════════════════════════════════════════

Kernel:      <kernel_name>
GPU:         gfx1201 (RDNA4), 64 CUs, wave32
Precision:   bf16
Peak:        122 TFLOPS (bf16)
Peak BW:     492 GB/s

─── ROOFLINE ANALYSIS ───

Arithmetic Intensity: 1365 FLOPs/byte (4096x4096x4096 bf16)
Ridge Point:          248 FLOPs/byte
Classification:       COMPUTE-BOUND
Achievable Ceiling:   122 TFLOPS

─── ISA ANALYSIS ───

VGPRs:      88
SGPRs:      42
Scratch:    0 bytes (no spilling)
LDS:        0 bytes (preshuffle, no LDS)
Occupancy:  5 waves/SIMD

─── BASELINE MEASUREMENTS ───

┌──────────────────┬────────────┬────────────────┬─────────────┐
│ Shape            │ rocBLAS    │ Kernel v1      │ Efficiency  │
├──────────────────┼────────────┼────────────────┼─────────────┤
│ 2048x2048x2048   │ 95 TFLOPS  │ 60 TFLOPS      │ 63%         │
│ 4096x4096x4096   │ 120 TFLOPS │ 72 TFLOPS      │ 60%         │
└──────────────────┴────────────┴────────────────┴─────────────┘

─── OPTIMIZATION ITERATIONS ───

Attempt 1: Preshuffle B operand
  Change:     Removed B LDS loads, use preshuffled GMEM layout
  Hypothesis: Eliminate ds_load WAW hazards on B
  Before:     72 TFLOPS (4096^3)
  After:      95 TFLOPS (4096^3)
  Delta:      +32%
  ISA:        VGPRs 112→96, removed all ds_load_b128 for B
  Verdict:    KEEP

Attempt 2: Preshuffle A operand
  Change:     Also preshuffle A, eliminate all LDS
  Hypothesis: Remove remaining LDS overhead
  Before:     95 TFLOPS
  After:      120 TFLOPS
  Delta:      +26%
  ISA:        VGPRs 96→88, LDS 16KB→0
  Verdict:    KEEP

Attempt 3: K-unroll x4
  Change:     Unroll inner K loop by 4
  Hypothesis: Better instruction scheduling, fill pipeline
  Before:     120 TFLOPS
  After:      134 TFLOPS
  Delta:      +12%
  ISA:        VGPRs 88→112, s_clause 7 (8 batched loads)
  Verdict:    KEEP

Attempt 4: Scheduling hints (sched_barrier)
  Change:     Added rocdl.sched_barrier between load and compute
  Hypothesis: Force better instruction interleaving
  Before:     134 TFLOPS
  After:      131 TFLOPS
  Delta:      -2%
  Verdict:    REVERT — RDNA4 scheduler handles this well without hints

─── FINAL RESULTS ───

┌──────────────────┬────────────┬────────────────┬─────────────┬───────────┐
│ Shape            │ rocBLAS    │ Final Kernel   │ Efficiency  │ vs Peak   │
├──────────────────┼────────────┼────────────────┼─────────────┼───────────┤
│ 2048x2048x2048   │ 95 TFLOPS  │ 105 TFLOPS     │ 111%        │ 86%       │
│ 4096x4096x4096   │ 120 TFLOPS │ 134 TFLOPS     │ 112%        │ 110%      │
└──────────────────┴────────────┴────────────────┴─────────────┴───────────┘

Gap: 0% (exceeds rocBLAS at large shapes)
Verdict: PRODUCTION READY

─── EFFORTS SUMMARY ───

Total attempts:  4
Kept:            3 (preshuffle B, preshuffle A, K-unroll x4)
Reverted:        1 (scheduling hints — hurt performance)
Total speedup:   72 → 134 TFLOPS (1.86x from v1)
Key insight:     Eliminating LDS entirely via preshuffle was the single
                 biggest win (60% → 110% of rocBLAS). Scheduling hints
                 should be avoided on RDNA4 — the hardware scheduler is
                 already effective.
```

---

## Run Environment

```bash
cd /root/e2e/FlyDSL
ROCR_VISIBLE_DEVICES=0 \
PYTHONPATH="$(pwd)/flydsl/src:$(pwd)/.flir/build/python_packages/flydsl:$(pwd):${PYTHONPATH}" \
python <script>
```

### Useful commands

```bash
# Correctness test
python tests/kernels/test_my_kernel.py

# Dump MLIR IR at each compilation stage
FLIR_DUMP_IR=1 FLIR_DUMP_DIR=my_ir_dumps python tests/kernels/test_my_kernel.py

# Compile without GPU (cross-compilation check)
FLYDSL_COMPILE_ONLY=1 FLYDSL_TARGET_ARCH=gfx1201 python tests/kernels/test_my_kernel.py

# Profile with rocprofv3
rocprofv3 --hip-trace python tests/kernels/profile_gemm.py flydsl 4096

# Occupancy control
# In kernel code: _apply_waves_per_eu_hint(m.module, waves_per_eu=2)
```

---

## File Organization

```
FlyDSL/
├── kernels/
│   ├── my_new_kernel.py           # Kernel implementation
│   └── kernels_common.py          # Shared utilities
├── tests/kernels/
│   ├── test_my_new_kernel.py      # Correctness test + benchmark
│   └── bench_my_new_kernel.py     # Dedicated benchmark (optional)
├── flydsl/src/flydsl/             # FlyDSL framework (don't modify for kernel dev)
└── .skills/                       # This documentation
```
