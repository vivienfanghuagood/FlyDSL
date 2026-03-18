# FlyDSL Dispatch Optimization and Benchmarking Methodology

## Overview

For latency-sensitive GPU kernels (decode attention, GEMV), the CPU-side dispatch
overhead of launching a kernel can dominate total execution time. This skill documents
the dispatch optimization applied to FlyDSL and the correct benchmarking methodology.

## The Dispatch Overhead Problem

### Symptoms
- Kernel timing shows a constant floor (~80us) regardless of problem size
- GPU event timing and wall-clock timing are nearly identical
- Varying BS, KV length, or algorithm has no effect on measured time

### Root Cause
FlyDSL's `ExecutionEngineExecutor.__call__()` had three sources of per-call overhead:

1. **Wrapper re-creation**: `__getattr__("__call__")` created a new closure on every call
   including `raw_lookup()`, `CFUNCTYPE` construction, and closure allocation
2. **Dynamic ctypes Structure**: `_make_memref_desc_type(rank)` created a new
   `ctypes.Structure` subclass via `type()` on every call (Python class creation is expensive)
3. **Runtime type checking**: Per-argument `hasattr`/`isinstance`/`callable` duck-typing
   checks, ciface descriptor inference, and signature scanning on every call

### Fix Applied (executor.py)

```python
# 1. Cache wrapper closure in __dict__ (called once, reused forever)
def __getattr__(self, name):
    cached = self._wrapper_cache.get(name)
    if cached is not None:
        return cached
    # ... build wrapper ...
    self._wrapper_cache[name] = wrapper
    return wrapper

# 2. Memoize ctypes Structure by rank (module-level cache)
_memref_desc_cache: Dict[int, type] = {}
def _make_memref_desc_type(rank: int):
    cached = _memref_desc_cache.get(rank)
    if cached is not None:
        return cached
    class _MemRefDesc(ctypes.Structure): ...
    _memref_desc_cache[rank] = _MemRefDesc
    return _MemRefDesc

# 3. Pre-compute per-argument metadata at wrapper creation time
# arg_kinds, arg_ctypes, arg_memref_ranks, desc_types all computed once
# Fast path uses indexed arrays instead of hasattr/isinstance checks
```

### Results

| Metric | Before | After | Improvement |
|---|---|---|---|
| FlyDSL dispatch (no sync) | 71 us | 18 us | 3.9x faster |
| FlyDSL total (with sync) | 80 us | 23 us | 3.5x faster |
| vs Triton dispatch | 1.5x slower | 2.6x faster | - |

## Benchmarking Methodology

### Step 1: Measure Dispatch Overhead (No Sync)

This measures pure CPU-side time to enqueue the kernel, without waiting for GPU:

```python
import time, torch

# Warmup
for _ in range(200):
    exe(args...)
torch.cuda.synchronize()

# Dispatch-only timing
N = 5000
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N):
    exe(args...)
t1 = time.perf_counter()  # NO sync here
dispatch_us = (t1 - t0) / N * 1e6
torch.cuda.synchronize()  # sync after measurement
print(f"Dispatch (no sync): {dispatch_us:.1f} us")
```

### Step 2: Measure Total Time (With Sync)

```python
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N):
    exe(args...)
torch.cuda.synchronize()
total_us = (time.perf_counter() - t0) / N * 1e6
print(f"Total (with sync): {total_us:.1f} us")
print(f"GPU compute: {total_us - dispatch_us:.1f} us")
```

### Step 3: GPU-Only Timing (CUDA Events)

CUDA events measure GPU-side wall time including any gaps between kernels:

```python
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
for _ in range(iters):
    exe(args...)
end_event.record()
torch.cuda.synchronize()

gpu_us = start_event.elapsed_time(end_event) / iters * 1000
```

**Warning**: GPU event timing includes dispatch gaps between iterations. If dispatch
takes longer than GPU execution, events will report dispatch time, not GPU time.

### Step 4: Kernel Launch Overhead Baseline

Measure the minimum cost of launching any kernel (use a no-op Triton kernel):

```python
# tests/kernels/bench_launch_overhead.py
@triton.jit
def _noop_kernel(x_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    tl.store(x_ptr + offs, x)

# Result on RDNA4: ~5.6us per kernel launch
```

### Interpretation Guide

| Scenario | Dispatch | Total | GPU | Conclusion |
|---|---|---|---|---|
| 70 us | 80 us | 10 us | Dispatch-bound, optimize executor |
| 10 us | 100 us | 90 us | Compute-bound, optimize kernel |
| 10 us | 10 us | ~0 us | Both fast, kernel finishes before CPU returns |
| 50 us | 50 us | ~0 us | Dispatch-bound (Triton's typical pattern for small kernels) |

## Running Benchmarks

### Environment Setup

```bash
cd /root/e2e/FlyDSL
ROCR_VISIBLE_DEVICES=0 \
PYTHONPATH="$(pwd)/flydsl/src:$(pwd)/.flir/build/python_packages/flydsl:$(pwd):${PYTHONPATH}" \
python <script>
```

### Benchmark Files

| File | Purpose |
|---|---|
| `tests/kernels/bench_gpu_only.py` | FlyDSL WMMA hybrid decode attention GPU timing |
| `tests/kernels/bench_triton_gpu_only.py` | Triton split-KV decode attention GPU timing |
| `tests/kernels/bench_dispatch_overhead.py` | Side-by-side dispatch overhead comparison |
| `tests/kernels/bench_launch_overhead.py` | Minimum kernel launch cost measurement |

### Comparison Table Format

Report FlyDSL/Triton ratio where >1.0 means FlyDSL wins:

```
| BS | KV | FlyDSL(us) | Triton(us) | Ratio |
|---|---|---|---|---|
| 1 | 256 | 25.0 | 48.5 | 1.94x |
```

## Hardware Constants (gfx1201 RDNA4)

- **GPU**: RX 9070 XT class, 64 CUs, 2 SIMDs per CU, wave32
- **VGPRs per SIMD**: 512 (max 256 per wave)
- **LDS**: up to 64KB per workgroup
- **Memory bandwidth**: ~492 GB/s measured
- **ROCm**: 7.1.0
- **PyTorch**: 2.9.1+rocm7.1.0
- **Triton**: 3.5.1+rocm7.1.0
- **Kernel launch overhead**: ~5.6us (Triton noop)
- **FlyDSL dispatch overhead**: ~18us (after optimization)
- **Triton dispatch overhead**: ~48us (includes Python wrapper + intermediate allocation)
