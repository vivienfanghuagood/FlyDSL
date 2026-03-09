# FlyDSL Kernel Skeleton for RDNA4

## Overview

FlyDSL is a Python-embedded DSL that generates MLIR for AMD GPU kernels.
This skill provides the complete boilerplate for writing new RDNA4 kernels.

## Complete Kernel Template

```python
#!/usr/bin/env python3
"""[Kernel Name] for RDNA4 (gfx12xx, wave32).

[Description of what the kernel computes]
"""

import os
import functools

import flydsl
from flydsl.dialects.ext import (
    flir,       # Module/kernel decorators, thread/block IDs
    arith,      # Arithmetic operations, constants, type conversions
    gpu,        # GPU barriers, launch
    buffer_ops, # AMD buffer load/store
    vector,     # Vector operations, bitcast, extract, load/store
    rocdl,      # WMMA/MFMA intrinsics, synchronization
    scf,        # Structured control flow (if/else)
    memref,     # Memory operations (LDS load/store)
    llvm,       # Inline assembly, intrinsics
)
from flydsl.dialects.ext.python_control_flow import range_constexpr
from flydsl.runtime.device import get_rocm_arch
from flydsl.lang.ir.types import T as I        # Type shortcuts (I.f32, I.i64, etc.)
from flydsl.kernels.kernels_common import stream_ptr_to_async_token
from flydsl.utils import SmemAllocator

from _mlir import ir
import _mlir.extras.types as T                  # MLIR type constructors


WMMA_M = 16
WMMA_N = 16
WMMA_K = 16


def _unwrap(v):
    """Unwrap ArithValue to raw MLIR Value."""
    while hasattr(v, "_value"):
        v = v._value
    return v


@functools.lru_cache(maxsize=64)
def compile_my_kernel(*, M: int, N: int, K: int, tile_m: int = 128, ...):
    """Compile the kernel.

    Args:
        M, N, K: Problem dimensions
        tile_m: Tile size along M

    Returns:
        Compiled executable: exe(output, input_a, input_b, ..., stream_ptr)
    """
    gpu_arch = get_rocm_arch()
    WAVE_SIZE = 32
    DYN = ir.ShapedType.get_dynamic_size()

    # --- Derived constants ---
    reg_m = tile_m // WMMA_M
    # ... more config ...
    THREADS_PER_BLOCK = num_waves * WAVE_SIZE

    # --- Optional: LDS allocator ---
    allocator = SmemAllocator(None, arch=gpu_arch)

    module_name = "my_kernel"

    class _Kernel(flir.MlirModule):
        GPU_MODULE_NAME = module_name
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}">']

        # Optional: LDS initialization
        def init_gpu_module(self):
            self._smem = allocator.allocate_array(T.f32(), num_elements)
            allocator.finalize()

        @flir.kernel
        def my_kernel_func(
            self: flir.T.i64,
            arg_out: lambda: T.memref(DYN, T.bf16()),
            arg_a: lambda: T.memref(DYN, T.bf16()),
            arg_b: lambda: T.memref(DYN, T.bf16()),
            c_m: lambda: I.index,
            c_n: lambda: I.index,
        ):
            # === Types ===
            f32 = ir.F32Type.get()
            bf16 = ir.BF16Type.get()
            i32 = ir.IntegerType.get_signless(32)
            i16_ty = ir.IntegerType.get_signless(16)
            v8bf16_ty = ir.VectorType.get([8], bf16)
            v8i16_ty = ir.VectorType.get([8], i16_ty)
            v8f32_ty = I.vec(8, I.f32)

            # === Thread/block IDs ===
            tid = flir.thread_idx("x")
            pid = flir.block_idx("x")

            c32 = arith.index(32)
            c16 = arith.index(16)
            c8 = arith.index(8)
            wave_id = tid // c32
            lane = tid % c32
            lane16 = lane % c16
            klane = lane // c16
            base8 = klane * c8

            # === Buffer resources ===
            a_rsrc = buffer_ops.create_buffer_resource(_unwrap(arg_a), max_size=True)
            out_rsrc = buffer_ops.create_buffer_resource(_unwrap(arg_out), max_size=True)

            # === LDS setup (if needed) ===
            # base_ptr = allocator.get_base()
            # smem_view = self._smem(base_ptr).get()

            # === Main computation ===
            # ... load, compute, store ...

            # === WMMA example ===
            zero_acc = arith.constant_vector(0.0, v8f32_ty)
            a_i16 = vector.bitcast(v8i16_ty, a_vec)
            b_i16 = vector.bitcast(v8i16_ty, b_vec)
            acc = rocdl.wmma_f32_16x16x16_bf16(
                v8f32_ty, [a_i16, b_i16, arith.unwrap(zero_acc)]
            )

            # === Store results ===
            for si in range_constexpr(8):
                val = vector.extract(acc, static_position=[si], dynamic_position=[])
                val_bf16 = arith.trunc_f(bf16, val)
                elem_off = g_row * c_n + g_col
                buffer_ops.buffer_store(val_bf16, out_rsrc, elem_off)

        @flir.jit
        def __call__(
            self: flir.T.i64,
            arg_out: lambda: T.memref(DYN, T.bf16()),
            arg_a: lambda: T.memref(DYN, T.bf16()),
            arg_b: lambda: T.memref(DYN, T.bf16()),
            c_m: lambda: I.index,
            c_n: lambda: I.index,
            stream_ptr: lambda: I.i64,
        ):
            c1 = arith.constant(1, index=True)
            bdx = arith.constant(THREADS_PER_BLOCK, index=True)
            total_blocks = c_m / arith.index(tile_m)
            stream_token = stream_ptr_to_async_token(stream_ptr)
            flir.gpu_ext.LaunchFuncOp(
                [module_name, "my_kernel_func"],
                grid_size=(total_blocks, c1, c1),
                block_size=(bdx, c1, c1),
                kernel_operands=[arg_out, arg_a, arg_b, c_m, c_n],
                async_dependencies=[stream_token],
            )

    m = _Kernel()
    return flydsl.compile(m)
```

## Notes

- For the complete API reference of all FlyDSL functions, see `flydsl-api-dictionary.md`
- For WMMA register layouts and lane mapping, see `rdna4-wmma-register-layout.md`
- For GEMM kernel design patterns, see `rdna4-gemm-kernel-patterns.md`

## Common Pitfalls

1. **bf16 WMMA requires i16 bitcast**: Always bitcast v8bf16 -> v8i16 before WMMA
2. **buffer_load offset is in elements**: The API handles byte conversion internally
3. **For bf16 via f32 loads**: Divide element offset by 2 (elem_off // 2)
4. **gpu.barrier() adds buffer_gl_inv**: Use inline asm barrier for RDNA4 to avoid it
5. **range_constexpr vs range()**: Use constexpr for small fixed counts, range() for dynamic
6. **_unwrap() before raw MLIR ops**: ArithValue wrappers must be unwrapped for direct MLIR API
7. **Wave size is 32**: Not 64 -- all thread indexing uses wave32 math
8. **ShuffleOp args must be raw Values**: All 3 args to gpu.ShuffleOp must be _unwrap()'d
9. **arith.cmpi does NOT exist**: Use Python operators (==, <, >) or arith.cmpu() for unsigned
10. **Variables inside if in range() loops**: FlyDSL AST rewriter breaks when if-defined vars
    are loop-carried. Use arith.select() instead of if/else inside range() loops
11. **Dispatch overhead**: FlyDSL dispatch is ~18us after optimization. For latency-sensitive
    kernels (<50us GPU time), dispatch overhead matters. Measure with no-sync timing.
12. **f32 accumulation required**: Use wmma_f32_16x16x16_bf16, NOT bf16 accumulation variants
