# RDNA4 Kernel Development Guide

Step-by-step workflow for developing new GPU kernels using FlyDSL on RDNA4 (gfx1201).
This guide tells you **which skills to consult** and **what to do** at each stage.

---

## Skill Index

| Skill File | What It Covers | When to Use |
|---|---|---|
| `flydsl-api-dictionary.md` | All FlyDSL functions/classes with signatures and examples | Every kernel — the primary API reference |
| `rdna4-flydsl-kernel-skeleton.md` | Complete boilerplate template (MlirModule, @kernel, @jit, LaunchFuncOp) | Starting any new kernel |
| `rdna4-wmma-register-layout.md` | WMMA lane mapping, operand register layout, bf16/fp8 bitcast rules | Any kernel using WMMA/matrix ops |
| `rdna4-gemm-kernel-patterns.md` | 4 proven GEMM patterns (LDS-based, preshuffle, inline ASM) | GEMM/matmul kernels |
| `rdna4-buffer-ops-and-memory.md` | Buffer descriptors, load/store patterns, LDS, bank conflicts | Any kernel doing global/shared memory access |
| `rdna4-mixed-precision-quantization.md` | INT4 W4A16, FP8, dequantization, per-group scaling | Quantized inference kernels |
| `rdna4-moe-gemm-kernel.md` | Token routing, expert dispatch, SiLU, two-stage MoE | MoE/expert kernels |
| `rdna4-attention-kernel.md` | Decode attention (WMMA hybrid, Split-KV, element-wise), online softmax | Attention kernels |
| `rdna4-kernel-optimization-evolution.md` | 8 optimization principles from 26 GEMM iterations | Performance tuning any kernel |
| `rdna4-dispatch-and-benchmarking.md` | Dispatch overhead, benchmarking methodology, timing patterns | Benchmarking and profiling |

---

## Workflow Overview

```
1. Understand the Problem   →  What computation? What shapes? What precision?
2. Choose Kernel Pattern    →  Match to an existing pattern or design a new one
3. Write the Kernel         →  Use skeleton + API dictionary
4. Write Correctness Test   →  Reference in float32, tolerance-based comparison
5. Run & Debug              →  Fix compilation errors, verify correctness
6. Benchmark                →  Measure TFLOPS, compare to baselines
7. Optimize                 →  Apply optimization principles, iterate
```

---

## Step 1: Understand the Problem

Before writing any code, determine:

- **Computation**: matmul, attention, elementwise, reduction, etc.
- **Shapes**: M, N, K dimensions; batch size; sequence length
- **Precision**: bf16, fp8, int4 (W4A16), mixed
- **Memory access pattern**: is data contiguous? preshuffled? paged?
- **Latency vs throughput**: small problem (decode) or large (prefill)?

---

## Step 2: Choose Kernel Pattern

Match your problem to a known pattern:

### GEMM / Matrix Multiply
- **Skills**: `rdna4-gemm-kernel-patterns.md`, `rdna4-wmma-register-layout.md`
- **Reference kernels**: `kernels/wmma_preshuffle_gemm.py`, `kernels/wmma_gemm_v26.py`
- **Key decisions**: LDS-based vs preshuffle, tile sizes, double buffering

### Attention (Decode)
- **Skills**: `rdna4-attention-kernel.md`, `rdna4-wmma-register-layout.md`
- **Reference kernels**: `kernels/wmma_decode_attention.py`
- **Key decisions**: WMMA hybrid vs element-wise for P@V, split-KV for long sequences

### Quantized Inference (W4A16, FP8)
- **Skills**: `rdna4-mixed-precision-quantization.md`, `rdna4-wmma-register-layout.md`
- **Reference kernels**: `kernels/wmma_w4a16_gemv.py`, `kernels/wmma_mixed_preshuffle_gemm.py`
- **Key decisions**: dequant inline vs precompute, group size, GEMM vs GEMV

### MoE (Mixture of Experts)
- **Skills**: `rdna4-moe-gemm-kernel.md`, `rdna4-gemm-kernel-patterns.md`
- **Reference kernels**: `kernels/wmma_moe_gemm.py`
- **Key decisions**: two-stage vs single-stage, token routing, SiLU fusion

### Elementwise / Reduction / Other
- **Skills**: `flydsl-api-dictionary.md` (arith, vector, scf, gpu sections)
- **Reference kernels**: `tests/kernels/test_eltwise_add.py`, `tests/kernels/test_softmax.py`, `tests/kernels/test_layernorm.py`
- **Key decisions**: threads per element, vectorization width, shared memory for reductions

---

## Step 3: Write the Kernel

### 3a. Start from the skeleton

**Skill**: `rdna4-flydsl-kernel-skeleton.md`

Copy the template and fill in:
1. Module name, GPU targets
2. Kernel arguments (memrefs, scalars)
3. Thread/block index computation
4. Buffer resource creation
5. Main computation loop
6. Result stores
7. `__call__` launcher with grid/block sizes

### 3b. Look up API functions

**Skill**: `flydsl-api-dictionary.md`

This is the primary reference. Key sections by task:

| Task | Dictionary Section |
|---|---|
| Create constants, do arithmetic | Section 2 (arith) |
| Load/store global memory | Section 3 (buffer_ops) |
| Call WMMA instructions | Section 4 (rocdl) — WMMA subsection |
| Vector manipulation (extract, bitcast) | Section 5 (vector) |
| Loops and conditionals | Section 6 (scf) |
| Thread IDs, barriers, shuffles | Section 7 (gpu) |
| LDS (shared memory) access | Section 8 (memref) + Section 12 (SmemAllocator) |
| Fast math (exp2, rcp) | Section 9 (llvm) — intrinsic calls |
| Type constructors | Section 13 (Types / T) |
| Compile and run | Section 14 (Compiler) |

### 3c. Handle WMMA specifics

**Skill**: `rdna4-wmma-register-layout.md`

Critical rules:
- bf16 operands must be bitcast to `v8i16` before WMMA
- fp8 operands are `v2i32` (8 bytes packed as 2x i32)
- Result is `v8f32` — extract with `vector.extract(acc, static_position=[i])`
- Lane mapping: `lane16 = lane % 16`, `klane = lane // 16`

### 3d. Handle memory access

**Skill**: `rdna4-buffer-ops-and-memory.md`

Key patterns:
- `buffer_ops.create_buffer_resource()` for every tensor
- Element offsets (API converts to bytes internally)
- Predicated loads with `mask=` parameter
- LDS via `SmemAllocator` + `memref.view` + `memref.load/store`

---

## Step 4: Write Correctness Test

### Test file structure

Place tests in `tests/kernels/test_<kernel_name>.py`. Follow this pattern:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
from flydsl.runtime.device import get_rocm_arch
from kernels.my_kernel import compile_my_kernel

def reference_compute(A, B, ...):
    """Reference implementation in float32."""
    return (A.float() @ B.float()).to(torch.bfloat16)

def test_correctness():
    arch = get_rocm_arch()
    if not arch.startswith("gfx12"):
        pytest.skip("RDNA4 only")

    torch.manual_seed(42)
    # Small-magnitude inputs for numerical stability
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.1
    B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda") * 0.1
    C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

    expected = reference_compute(A, B)

    exe = compile_my_kernel(M=M, N=N, K=K)
    exe(C, A, B, torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    # Check correctness
    max_abs_err = (C.float() - expected.float()).abs().max().item()
    rel_err = max_abs_err / (expected.float().abs().max().item() + 1e-8)
    print(f"Relative error: {rel_err:.6f}")
    assert rel_err < TOLERANCE, f"Relative error {rel_err} exceeds {TOLERANCE}"
```

### Tolerance guidelines

| Kernel Type | Metric | Threshold | Rationale |
|---|---|---|---|
| GEMM (bf16) | Relative error | < 0.01 (1%) | Straightforward matmul |
| GEMM (fp8) | Relative error | < 0.05 (5%) | FP8 has limited precision |
| Attention | Cosine similarity | > 0.99 | Softmax amplifies errors |
| MoE | Relative error (nonzero mask) | < 0.15 (15%) | Routing + SiLU + multi-expert accumulation |
| W4A16 | Relative error | < 0.05 (5%) | INT4 quantization noise |
| Elementwise | Max absolute error | < 1e-5 | Should be nearly exact |

### Reference computation rules

1. **Always compute in float32**: `A.float() @ B.float()`, not `A @ B`
2. **Use nonzero masking for sparse outputs**: MoE and attention may have zero-padded slots
3. **Match the kernel's output dtype for comparison**: Cast reference back to output dtype only if needed
4. **Seed random inputs**: `torch.manual_seed(42)` for reproducibility
5. **Scale inputs**: Use `* 0.1` to keep values small (avoids fp16/bf16 overflow)

---

## Step 5: Run & Debug

### Run the test

```bash
cd /root/e2e/FlyDSL
ROCR_VISIBLE_DEVICES=0 \
PYTHONPATH="$(pwd)/flydsl/src:$(pwd)/.flir/build/python_packages/flydsl:$(pwd):${PYTHONPATH}" \
python tests/kernels/test_my_kernel.py
```

### Debug with IR dumps

```bash
FLIR_DUMP_IR=1 FLIR_DUMP_DIR=my_ir_dumps python tests/kernels/test_my_kernel.py
# Check my_ir_dumps/<kernel_name>/ for intermediate MLIR stages
```

### Compile-only mode (no GPU needed)

```bash
FLYDSL_COMPILE_ONLY=1 FLYDSL_TARGET_ARCH=gfx1201 python tests/kernels/test_my_kernel.py
```

### Common errors and fixes

| Error | Likely Cause | Fix |
|---|---|---|
| `ArithValue not accepted` | Passed wrapper to raw MLIR API | Use `arith.unwrap(val)` or `_unwrap(val)` |
| `type mismatch in operand` | Wrong vector type for WMMA | Check bitcast: bf16->i16, fp8->i32 |
| `offset must be i32` | Passed index to buffer_load | Use `arith.index_cast(i32, val)` |
| `scf.for body not terminated` | Missing yield in loop | Add `scf.yield_([...])` |
| `operation does not dominate use` | Value defined inside if used outside | Use `scf.IfOp` with result types |
| Shared memory overflow | Too much LDS allocated | Check `SmemAllocator` total vs 64KB limit |

---

## Step 6: Benchmark

**Skill**: `rdna4-dispatch-and-benchmarking.md`

### Benchmark function template

```python
def benchmark(M, N, K, iters=100):
    exe = compile_my_kernel(M=M, N=N, K=K)
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.1
    B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda") * 0.1
    C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

    # Warmup
    for _ in range(5):
        exe(C, A, B, torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    # Timed loop
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        exe(C, A, B, torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    avg_ms = (time.time() - t0) / iters * 1000

    tflops = 2 * M * N * K / (avg_ms / 1000) / 1e12
    print(f"M={M} N={N} K={K}: {avg_ms:.3f} ms, {tflops:.1f} TFLOPS")
```

### Baselines to compare against

| Baseline | How to Measure |
|---|---|
| rocBLAS | `torch.mm(A, B)` with same dtype, timed the same way |
| Triton | Write equivalent Triton kernel, time with same methodology |
| Theoretical peak | RDNA4 gfx1201: ~122 TFLOPS bf16, ~244 TFLOPS fp8 |

### Measuring dispatch overhead

For latency-sensitive kernels (decode attention, GEMV), measure dispatch separately:

```python
# No-sync timing (dispatch only)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(5000):
    exe(args...)
t1 = time.perf_counter()
dispatch_us = (t1 - t0) / 5000 * 1e6
torch.cuda.synchronize()
```

---

## Step 7: Optimize

**Skill**: `rdna4-kernel-optimization-evolution.md`

### Optimization checklist (in priority order)

1. **Eliminate LDS if possible** — preshuffle layout enables direct GMEM->register WMMA
2. **Maximize memory coalescing** — contiguous lane access to contiguous addresses
3. **Avoid VGPR reuse serialization** — don't write-then-read the same register bank
4. **Software pipeline** — overlap GMEM loads with WMMA compute (double buffering)
5. **K-unroll** — 2x or 4x K-unroll to fill the pipeline
6. **Bank conflict avoidance** — K-padding or XOR-swizzle for LDS
7. **Register pressure** — keep VGPRs under 128 for 2 waves/EU occupancy
8. **L2 cache swizzle** — reorder block IDs for spatial locality

### Performance profiling

```bash
# Dump ISA for analysis
FLIR_DUMP_IR=1 python test_my_kernel.py
# Check my_ir_dumps/<kernel>/15_final_isa.s

# Count VGPRs, SGPRs, occupancy from ISA header
# Look for: .vgpr_count, .sgpr_count, .lds_size
```

### When to stop optimizing

- Within 90% of rocBLAS → good for production
- Within 80% → review memory access patterns
- Below 50% → likely a fundamental design issue, reconsider the pattern

---

## Quick Decision Tree

```
What kind of kernel?
│
├─ Matrix multiply (GEMM)
│  ├─ Large M,N,K (prefill) → rdna4-gemm-kernel-patterns.md (preshuffle pattern)
│  ├─ Small M (decode GEMV) → rdna4-mixed-precision-quantization.md (W4A16 GEMV)
│  └─ MoE routing          → rdna4-moe-gemm-kernel.md
│
├─ Attention
│  ├─ Decode (small Q)      → rdna4-attention-kernel.md (WMMA hybrid)
│  └─ Prefill (large Q)     → rdna4-attention-kernel.md (Split-KV)
│
├─ Quantized inference
│  ├─ INT4 weight-only      → rdna4-mixed-precision-quantization.md
│  └─ FP8 compute           → rdna4-mixed-precision-quantization.md
│
├─ Elementwise / Reduction
│  └─ Use API dictionary directly → flydsl-api-dictionary.md
│
└─ Something new
   ├─ Start with skeleton   → rdna4-flydsl-kernel-skeleton.md
   ├─ API reference          → flydsl-api-dictionary.md
   ├─ Memory patterns        → rdna4-buffer-ops-and-memory.md
   └─ WMMA details           → rdna4-wmma-register-layout.md
```

---

## File Organization Convention

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
