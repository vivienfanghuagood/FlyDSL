"""WMMA Preshuffle GEMM kernel for RDNA4 (gfx12xx, wave32).

This kernel performs C[M,N] = A[M,K] @ B[K,N] using WMMA instructions
(v_wmma_f32_16x16x16_bf16 / v_wmma_f32_16x16x16_f16) with f32 accumulation.

Architecture: "preshuffle A + preshuffle B" — both operands are pre-shuffled
on the host into WMMA-friendly layouts for direct buffer_load from GMEM.
No LDS usage (register-only pipeline).

Based on the v15 kernel which achieves 134 TFLOPS (110% of rocBLAS) at 4096³.

Interface matches compile_preshuffle_gemm_a8() for drop-in usage:
  exe(c, a, b, scale_a, scale_b, M, N, K, stream_ptr)

Supported dtypes:
  - in_dtype: "bf16", "fp16"
  - out_dtype: "bf16", "fp16", "f32"
"""

import os

import flydsl
from flydsl.dialects.ext import flir, arith, gpu, buffer_ops, vector, rocdl
from flydsl.dialects.ext.python_control_flow import range_constexpr
from flydsl.runtime.device import get_rocm_arch
from flydsl.lang.ir.types import T, memref
from flydsl.kernels.kernels_common import stream_ptr_to_async_token
from flydsl.compiler.compiler import _apply_waves_per_eu_hint

from _mlir import ir
import _mlir.extras.types as Textra

# =============================================================================
# Constants
# =============================================================================

WMMA_M = 16
WMMA_N = 16
WMMA_K = 16


def _unwrap(v):
    """Unwrap ArithValue to raw MLIR Value."""
    while hasattr(v, "_value"):
        v = v._value
    return v


# =============================================================================
# Host-side pre-shuffle functions
# =============================================================================


def preshuffle_a_wmma(A_mk):
    """Pre-shuffle A[M,K] for WMMA register layout.

    Output shape: [M0, K0, KLane, MLane, KPack]
    where M0=M//16, K0=K//16, KLane=2, MLane=16, KPack=8
    """
    import torch

    M, K = A_mk.shape
    assert M % 16 == 0 and K % 16 == 0
    M0 = M // 16
    K0 = K // 16
    A_reshaped = A_mk.reshape(M0, 16, K0, 2, 8)
    A_shuffled = A_reshaped.permute(0, 2, 3, 1, 4).contiguous()
    return A_shuffled


def preshuffle_b_wmma(B_kn):
    """Pre-shuffle B[K,N] for WMMA register layout.

    Output shape: [N0, K0, KLane, NLane, KPack]
    where N0=N//16, K0=K//16, KLane=2, NLane=16, KPack=8

    NOTE: Input must be B[K,N], NOT B_T[N,K]!
    """
    import torch

    K, N = B_kn.shape
    assert K % 16 == 0 and N % 16 == 0
    N0 = N // 16
    K0 = K // 16
    B_reshaped = B_kn.reshape(K0, 2, 8, N0, 16)
    B_shuffled = B_reshaped.permute(3, 0, 1, 4, 2).contiguous()
    return B_shuffled


# =============================================================================
# Kernel compiler
# =============================================================================


def compile_wmma_preshuffle_gemm(
    *,
    M: int,
    N: int,
    K: int,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 32,
    in_dtype: str = "bf16",
    out_dtype: str = "bf16",
    # Tuning parameters
    k_unroll: int = 4,
    group_m: int = 8,
    # Occupancy control
    waves_per_eu: int = None,
):
    """Compile the WMMA preshuffle GEMM kernel for RDNA4 (gfx12xx).

    Args:
        M, N, K: Matrix dimensions. A[M,K] @ B[K,N] = C[M,N].
        tile_m, tile_n, tile_k: Block tile sizes. Must be multiples of 16.
        in_dtype: Input element type ("bf16" or "fp16").
        out_dtype: Output element type ("bf16", "fp16", or "f32").
        k_unroll: Inner K-loop unroll factor for software pipelining.
        group_m: L2 cache swizzle group size along M.
        waves_per_eu: Occupancy hint (None = default).

    Returns:
        Compiled executable with signature:
          exe(c, a_shuf, b_shuf, scale_a, scale_b, M, N, K, stream_ptr)
        where scale_a and scale_b are unused (empty tensors for API compat).
    """
    if in_dtype not in ("bf16", "fp16"):
        raise ValueError(f"in_dtype must be 'bf16' or 'fp16', got {in_dtype!r}")
    if out_dtype not in ("bf16", "fp16", "f32"):
        raise ValueError(
            f"out_dtype must be 'bf16', 'fp16', or 'f32', got {out_dtype!r}"
        )

    is_bf16 = in_dtype == "bf16"

    # WMMA tile parameters derived from block tiles
    assert tile_m % WMMA_M == 0, f"tile_m ({tile_m}) must be multiple of {WMMA_M}"
    assert tile_n % WMMA_N == 0, f"tile_n ({tile_n}) must be multiple of {WMMA_N}"
    assert tile_k % WMMA_K == 0, f"tile_k ({tile_k}) must be multiple of {WMMA_K}"

    reg_m = tile_m // WMMA_M  # WMMA tiles per wave along M
    reg_n = tile_n // WMMA_N  # WMMA tiles per wave along N
    reg_k = tile_k // WMMA_K  # WMMA tiles along K per block tile

    # Wave layout: determine from tile sizes
    # For 128x128: 4x4 wmma tiles per block → 2x2 waves, each wave does 2x2 wmma tiles
    # For 64x128: 4x8 wmma tiles → 2x2 waves, wave does 2x4
    # We use a fixed wave layout heuristic:
    WAVE_SIZE = 32

    # Choose wave layout to balance M and N work
    if tile_m >= 128 and tile_n >= 128:
        waves_m, waves_n = 2, 2
    elif tile_m >= 128:
        waves_m, waves_n = 2, 1
    elif tile_n >= 128:
        waves_m, waves_n = 1, 2
    else:
        waves_m, waves_n = 1, 1

    NUM_WAVES = waves_m * waves_n
    THREADS_PER_BLOCK = NUM_WAVES * WAVE_SIZE

    # Per-wave tile sizes
    wave_tile_m = tile_m // waves_m
    wave_tile_n = tile_n // waves_n
    wave_reg_m = wave_tile_m // WMMA_M
    wave_reg_n = wave_tile_n // WMMA_N

    gpu_arch = get_rocm_arch()
    DYN = ir.ShapedType.get_dynamic_size()

    assert M % tile_m == 0, f"M ({M}) must be multiple of tile_m ({tile_m})"
    assert N % tile_n == 0, f"N ({N}) must be multiple of tile_n ({tile_n})"
    assert K % tile_k == 0, f"K ({K}) must be multiple of tile_k ({tile_k})"

    num_k_tiles = K // tile_k
    assert num_k_tiles % k_unroll == 0, (
        f"num_k_tiles ({num_k_tiles}) must be divisible by k_unroll ({k_unroll})"
    )
    assert num_k_tiles > k_unroll, (
        f"Need num_k_tiles ({num_k_tiles}) > k_unroll ({k_unroll})"
    )

    grid_m = M // tile_m
    grid_n = N // tile_n

    # Preshuffle stride constants
    M0_total = M // 16
    K0_total = K // 16
    A_KPACK = 8
    A_STRIDE_MLANE = A_KPACK
    A_STRIDE_KLANE = 16 * A_KPACK
    A_STRIDE_K0 = 2 * 16 * A_KPACK
    A_STRIDE_M0 = K0_total * A_STRIDE_K0

    N0_total = N // 16
    B_KPACK = 8
    B_STRIDE_NLANE = B_KPACK
    B_STRIDE_KLANE = 16 * B_KPACK
    B_STRIDE_K0 = 2 * 16 * B_KPACK
    B_STRIDE_N0 = K0_total * B_STRIDE_K0

    module_name = "wmma_preshuffle_gemm"

    def _in_elem_ty():
        return Textra.bf16() if is_bf16 else Textra.f16()

    def _out_elem_ty():
        if out_dtype == "f32":
            return Textra.f32()
        elif out_dtype == "bf16":
            return Textra.bf16()
        else:
            return Textra.f16()

    def _wmma_op(result_type, a_vec, b_vec, acc, v8i16_ty):
        if is_bf16:
            a_i16 = vector.bitcast(v8i16_ty, a_vec)
            b_i16 = vector.bitcast(v8i16_ty, b_vec)
            return rocdl.wmma_f32_16x16x16_bf16(
                result_type, [a_i16, b_i16, arith.unwrap(acc)]
            )
        else:
            return rocdl.wmma_f32_16x16x16_f16(
                result_type,
                [arith.unwrap(a_vec), arith.unwrap(b_vec), arith.unwrap(acc)],
            )

    class _GEMM(flir.MlirModule):
        GPU_MODULE_NAME = module_name
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}">']

        @flir.kernel
        def kernel_gemm(
            self: flir.T.i64,
            arg_c: lambda: Textra.memref(DYN, _out_elem_ty()),
            arg_a: lambda: Textra.memref(DYN, _in_elem_ty()),
            arg_b: lambda: Textra.memref(DYN, _in_elem_ty()),
            arg_scale_a: lambda: Textra.memref(DYN, Textra.f32()),
            arg_scale_b: lambda: Textra.memref(DYN, Textra.f32()),
            c_m: lambda: T.index,
            c_n: lambda: T.index,
            c_k: lambda: T.index,
        ):
            in_ir_ty = ir.BF16Type.get() if is_bf16 else ir.F16Type.get()
            v8_in_ty = ir.VectorType.get([8], in_ir_ty)
            v8f32_ty = T.vec(8, T.f32)
            i16_ty = ir.IntegerType.get_signless(16)
            v8i16_ty = ir.VectorType.get([8], i16_ty)

            tid = flir.thread_idx("x")
            pid = flir.block_idx("x")

            c32 = arith.index(32)
            c16 = arith.index(16)
            c8 = arith.index(8)
            wave_id = tid // c32
            lane = tid % c32
            lane16 = lane % c16
            klane = lane // c16

            # L2 cache swizzle
            effective_group_m = min(group_m, grid_m)
            c_grid_n = arith.index(grid_n)
            c_group_m = arith.index(effective_group_m)
            num_pid_in_group = c_group_m * c_grid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * c_group_m
            group_size_m = c_group_m

            pid_in_group = pid % num_pid_in_group
            bid_m = first_pid_m + (pid_in_group % group_size_m)
            bid_n = pid_in_group // group_size_m

            c_wn = arith.index(waves_n)
            wave_m = wave_id // c_wn
            wave_n = wave_id % c_wn

            tile_m0 = bid_m * arith.index(tile_m)
            tile_n0 = bid_n * arith.index(tile_n)

            a_rsrc = buffer_ops.create_buffer_resource(_unwrap(arg_a), max_size=True)
            b_rsrc = buffer_ops.create_buffer_resource(_unwrap(arg_b), max_size=True)
            c_rsrc = buffer_ops.create_buffer_resource(_unwrap(arg_c), max_size=True)

            def _load_a_tile(k_tile_idx):
                """Load A operands from pre-shuffled GMEM."""
                a_vecs = []
                m0_base = tile_m0 // c16 + wave_m * arith.index(wave_reg_m)
                for rk in range_constexpr(reg_k):
                    rk_vecs = []
                    k0 = k_tile_idx * arith.index(reg_k) + arith.index(rk)
                    for rm in range_constexpr(wave_reg_m):
                        m0 = m0_base + arith.index(rm)
                        elem_off = (
                            m0 * arith.index(A_STRIDE_M0)
                            + k0 * arith.index(A_STRIDE_K0)
                            + klane * arith.index(A_STRIDE_KLANE)
                            + lane16 * arith.index(A_STRIDE_MLANE)
                        )
                        f32_off = elem_off // arith.index(2)
                        a_raw = buffer_ops.buffer_load(
                            a_rsrc,
                            f32_off,
                            vec_width=4,
                            dtype=ir.F32Type.get(),
                        )
                        a_vec = vector.bitcast(v8_in_ty, a_raw)
                        rk_vecs.append(a_vec)
                    a_vecs.append(rk_vecs)
                return a_vecs

            def _load_b_tile(k_tile_idx):
                """Load B operands from pre-shuffled GMEM."""
                b_vecs = []
                n0_base = tile_n0 // c16 + wave_n * arith.index(wave_reg_n)
                for rk in range_constexpr(reg_k):
                    rk_vecs = []
                    k0 = k_tile_idx * arith.index(reg_k) + arith.index(rk)
                    for rn in range_constexpr(wave_reg_n):
                        n0 = n0_base + arith.index(rn)
                        elem_off = (
                            n0 * arith.index(B_STRIDE_N0)
                            + k0 * arith.index(B_STRIDE_K0)
                            + klane * arith.index(B_STRIDE_KLANE)
                            + lane16 * arith.index(B_STRIDE_NLANE)
                        )
                        f32_off = elem_off // arith.index(2)
                        b_raw = buffer_ops.buffer_load(
                            b_rsrc,
                            f32_off,
                            vec_width=4,
                            dtype=ir.F32Type.get(),
                        )
                        b_vec = vector.bitcast(v8_in_ty, b_raw)
                        rk_vecs.append(b_vec)
                    b_vecs.append(rk_vecs)
                return b_vecs

            def _do_compute(accs_in, a_vecs, b_vecs):
                new_accs = list(accs_in)
                for rk in range_constexpr(reg_k):
                    for rm in range_constexpr(wave_reg_m):
                        for rn in range_constexpr(wave_reg_n):
                            idx = rm * wave_reg_n + rn
                            new_accs[idx] = _wmma_op(
                                v8f32_ty,
                                a_vecs[rk][rm],
                                b_vecs[rk][rn],
                                new_accs[idx],
                                v8i16_ty,
                            )
                return new_accs

            # Initialize accumulators
            zero_acc = arith.constant_vector(0.0, v8f32_ty)
            accs = [zero_acc for _ in range_constexpr(wave_reg_m * wave_reg_n)]

            # ============================================================
            # PIPELINED K-LOOP WITH range_constexpr INNER UNROLLING
            # ============================================================
            # Prologue: load first tile
            a_cur = _load_a_tile(arith.index(0))
            b_cur = _load_b_tile(arith.index(0))

            # Main loop
            full_outer_iters = (num_k_tiles - 1) // k_unroll
            remainder = (num_k_tiles - 1) % k_unroll

            for kt_outer in range(full_outer_iters):
                for j in range_constexpr(k_unroll):
                    next_kt = kt_outer * arith.index(k_unroll) + arith.index(j + 1)
                    a_next = _load_a_tile(next_kt)
                    b_next = _load_b_tile(next_kt)
                    accs = _do_compute(accs, a_cur, b_cur)
                    a_cur = a_next
                    b_cur = b_next

            # Handle remainder tiles
            if remainder > 0:
                for j in range_constexpr(remainder):
                    next_kt = arith.index(full_outer_iters * k_unroll + j + 1)
                    a_next = _load_a_tile(next_kt)
                    b_next = _load_b_tile(next_kt)
                    accs = _do_compute(accs, a_cur, b_cur)
                    a_cur = a_next
                    b_cur = b_next

            # Epilogue: compute with last loaded tile
            accs = _do_compute(accs, a_cur, b_cur)

            # ========== Store results ==========
            c_layout_n = c_n
            base8 = klane * c8

            for rm in range_constexpr(wave_reg_m):
                for rn in range_constexpr(wave_reg_n):
                    idx = rm * wave_reg_n + rn
                    wmma_m_off = wave_m * arith.index(
                        wave_reg_m * WMMA_M
                    ) + arith.index(rm * WMMA_M)
                    wmma_n_off = wave_n * arith.index(
                        wave_reg_n * WMMA_N
                    ) + arith.index(rn * WMMA_N)
                    for si in range_constexpr(8):
                        g_row = tile_m0 + wmma_m_off + base8 + arith.index(si)
                        g_col = tile_n0 + wmma_n_off + lane16
                        val = vector.extract(
                            accs[idx],
                            static_position=[si],
                            dynamic_position=[],
                        )
                        if out_dtype == "bf16":
                            val = arith.trunc_f(ir.BF16Type.get(), val)
                        elif out_dtype == "fp16":
                            val = arith.trunc_f(ir.F16Type.get(), val)
                        # f32: no truncation needed
                        elem_off = g_row * c_layout_n + g_col
                        buffer_ops.buffer_store(val, c_rsrc, elem_off)

        @flir.jit
        def __call__(
            self: flir.T.i64,
            arg_c: lambda: memref(DYN, _out_elem_ty()),
            arg_a: lambda: memref(DYN, _in_elem_ty()),
            arg_b: lambda: memref(DYN, _in_elem_ty()),
            arg_scale_a: lambda: memref(DYN, T.f32),
            arg_scale_b: lambda: memref(DYN, T.f32),
            c_m: lambda: T.index,
            c_n: lambda: T.index,
            c_k: lambda: T.index,
            stream_ptr: lambda: T.i64,
        ):
            c1 = arith.constant(1, index=True)
            bdx = arith.constant(THREADS_PER_BLOCK, index=True)
            tm = arith.constant(tile_m, index=True)
            tn = arith.constant(tile_n, index=True)
            one = arith.constant(1, index=True)
            gx = (c_m + tm - one) / tm
            gy = c_n / tn
            total_blocks = gx * gy
            stream_token = stream_ptr_to_async_token(stream_ptr)
            flir.gpu_ext.LaunchFuncOp(
                [module_name, "kernel_gemm"],
                grid_size=(total_blocks, c1, c1),
                block_size=(bdx, c1, c1),
                kernel_operands=[
                    arg_c,
                    arg_a,
                    arg_b,
                    arg_scale_a,
                    arg_scale_b,
                    c_m,
                    c_n,
                    c_k,
                ],
                async_dependencies=[stream_token],
            )

    m = _GEMM()

    if waves_per_eu is not None:
        _apply_waves_per_eu_hint(m.module, waves_per_eu)

    return flydsl.compile(
        m,
        use_bare_ptr_memref_call_conv=False,
        use_bare_pointers_for_host=False,
        use_bare_pointers_for_kernels=False,
    )


__all__ = [
    "compile_wmma_preshuffle_gemm",
    "preshuffle_a_wmma",
    "preshuffle_b_wmma",
]
