"""Fast Float8 Preshuffle GEMM for RDNA4 (gfx12xx, wave32).

Optimized for M=32, N=8192, K=6144 (decode-phase inference shape).

  C[M,N] = A[M,K] @ B[K,N]

Both A and B are fp8_e4m3fn with per-tensor scales.
Output is bf16.  Accumulation in f32.

Uses the preshuffle pattern (Pattern 2) for maximum throughput:
  - A preshuffled to [M0, K0, KLane=2, MLane=16, KPack=8] bytes
  - B preshuffled to [N0, K0, KLane=2, NLane=16, KPack=8] bytes
  - No LDS needed — direct GMEM -> register -> WMMA pipeline
  - Software-pipelined K-loop with compile-time inner unrolling

Tile config (tuned for M=32):
  tile_m=32  (2 WMMA M-tiles)
  tile_n=128 (8 WMMA N-tiles)
  tile_k=32  (2 WMMA K-tiles)
  waves_m=1, waves_n=2 → 2 waves = 64 threads per block
  wave_reg_m=2, wave_reg_n=4 → 8 accumulators per wave
"""

import os
import functools
import time

import flydsl
from flydsl.dialects.ext import (
    flir,
    arith,
    gpu,
    buffer_ops,
    vector,
    rocdl,
    scf,
    memref,
    llvm,
)
from flydsl.dialects.ext.python_control_flow import range_constexpr
from flydsl.runtime.device import get_rocm_arch
from flydsl.lang.ir.types import T as I
from flydsl.kernels.kernels_common import stream_ptr_to_async_token

from _mlir import ir
import _mlir.extras.types as T


WMMA_M = 16
WMMA_N = 16
WMMA_K = 16


def _unwrap(v):
    """Unwrap ArithValue to raw MLIR Value."""
    while hasattr(v, "_value"):
        v = v._value
    return v


# =============================================================================
# Host-side preshuffle functions
# =============================================================================


def preshuffle_a_fp8(A_mk):
    """Preshuffle A[M,K] fp8 for WMMA A operand layout.

    Layout: [M0, K0, KLane=2, MLane=16, KPack=8] bytes.
    lane16 selects M row, klane selects K half (0=K[0:8], 1=K[8:16]).
    Each lane loads 8 contiguous fp8 bytes as 2xi32.
    """
    import torch

    M, K = A_mk.shape
    assert M % 16 == 0 and K % 16 == 0
    M0 = M // 16
    K0 = K // 16
    A_view = A_mk.view(torch.uint8)
    A_reshaped = A_view.reshape(M0, 16, K0, 2, 8)
    return A_reshaped.permute(0, 2, 3, 1, 4).contiguous()  # [M0, K0, 2, 16, 8]


def preshuffle_b_fp8(B_kn):
    """Preshuffle B[K,N] fp8 for WMMA B operand layout.

    Layout: [N0, K0, KLane=2, NLane=16, KPack=8] bytes.
    lane16 selects N column, klane selects K half.
    """
    import torch

    K, N = B_kn.shape
    assert K % 16 == 0 and N % 16 == 0
    N0 = N // 16
    K0 = K // 16
    B_view = B_kn.view(torch.uint8)
    B_reshaped = B_view.reshape(K0, 2, 8, N0, 16)
    return B_reshaped.permute(3, 0, 1, 4, 2).contiguous()  # [N0, K0, 2, 16, 8]


def fp8_quantize_per_tensor(x_f32):
    """Quantize f32 tensor to fp8_e4m3fn with per-tensor scale.

    Returns (x_fp8, scale) where x_f32 ~ x_fp8.float() * scale.
    """
    import torch

    amax = x_f32.abs().amax().clamp(min=1e-12)
    scale = amax / 448.0  # fp8_e4m3fn max = 448.0
    x_scaled = (x_f32 / scale).clamp(-448.0, 448.0)
    x_fp8 = x_scaled.to(torch.float8_e4m3fn)
    return x_fp8, scale.item()


# =============================================================================
# Kernel compiler
# =============================================================================


@functools.lru_cache(maxsize=64)
def compile_fp8_gemm(
    *,
    M: int,
    N: int,
    K: int,
    tile_m: int = 32,
    tile_n: int = 128,
    tile_k: int = 32,
    k_unroll: int = 4,
    group_m: int = 8,
):
    """Compile fp8 preshuffle GEMM for RDNA4.

    Optimized for small-M shapes (e.g., M=32, decode phase).

    Args:
        M, N, K: Matrix dimensions. Must be divisible by tile sizes.
        tile_m: Block tile M (default 32 for small-M).
        tile_n: Block tile N (default 128).
        tile_k: Block tile K (default 32 = 2 WMMA K-tiles).
        k_unroll: Inner K-loop unroll factor.
        group_m: L2 cache swizzle group size.

    Returns:
        exe(c, a_shuf, b_shuf, scale_a, scale_b, M, N, K, stream_ptr)
    """
    gpu_arch = get_rocm_arch()

    WAVE_SIZE = 32
    assert tile_m % WMMA_M == 0, f"tile_m={tile_m} must be multiple of {WMMA_M}"
    assert tile_n % WMMA_N == 0, f"tile_n={tile_n} must be multiple of {WMMA_N}"
    assert tile_k % WMMA_K == 0, f"tile_k={tile_k} must be multiple of {WMMA_K}"
    assert M % tile_m == 0, f"M={M} must be multiple of tile_m={tile_m}"
    assert N % tile_n == 0, f"N={N} must be multiple of tile_n={tile_n}"
    assert K % tile_k == 0, f"K={K} must be multiple of tile_k={tile_k}"

    reg_m = tile_m // WMMA_M  # 32/16 = 2
    reg_n = tile_n // WMMA_N  # 128/16 = 8
    reg_k = tile_k // WMMA_K  # 32/16 = 2

    # Wave layout: for small M, put all waves along N
    if tile_m >= 128 and tile_n >= 128:
        waves_m, waves_n = 2, 2
    elif tile_m >= 64 and tile_n >= 128:
        waves_m, waves_n = 2, 2
    elif tile_m >= 64:
        waves_m, waves_n = 2, 1
    elif tile_n >= 128:
        waves_m, waves_n = 1, 2
    else:
        waves_m, waves_n = 1, 1

    NUM_WAVES = waves_m * waves_n
    THREADS_PER_BLOCK = NUM_WAVES * WAVE_SIZE
    wave_reg_m = reg_m // waves_m
    wave_reg_n = reg_n // waves_n

    num_k_tiles = K // tile_k
    grid_m = M // tile_m
    grid_n = N // tile_n

    K0_total = K // 16  # total WMMA K-tiles across full K dimension

    # Preshuffle strides (byte-based for fp8)
    # Layout: [M0/N0, K0, KLane=2, MLane/NLane=16, KPack=8] bytes
    A_KPACK = 8  # 8 fp8 bytes per lane
    A_STRIDE_MLANE = A_KPACK  # 8
    A_STRIDE_KLANE = 16 * A_KPACK  # 128
    A_STRIDE_K0 = 2 * 16 * A_KPACK  # 256
    A_STRIDE_M0 = K0_total * A_STRIDE_K0

    B_KPACK = 8
    B_STRIDE_NLANE = B_KPACK  # 8
    B_STRIDE_KLANE = 16 * B_KPACK  # 128
    B_STRIDE_K0 = 2 * 16 * B_KPACK  # 256
    B_STRIDE_N0 = K0_total * B_STRIDE_K0

    DYN = ir.ShapedType.get_dynamic_size()

    module_name = "wmma_fp8_gemm"

    class _GEMM(flir.MlirModule):
        GPU_MODULE_NAME = module_name
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}">']

        @flir.kernel
        def kernel_gemm(
            self: flir.T.i64,
            arg_c: lambda: T.memref(DYN, T.bf16()),
            arg_a: lambda: T.memref(DYN, T.f32()),  # fp8 bytes viewed as f32
            arg_b: lambda: T.memref(DYN, T.f32()),  # fp8 bytes viewed as f32
            arg_scale_a: lambda: T.memref(DYN, T.f32()),
            arg_scale_b: lambda: T.memref(DYN, T.f32()),
            c_m: lambda: I.index,
            c_n: lambda: I.index,
            c_k: lambda: I.index,
        ):
            # === Types ===
            f32 = ir.F32Type.get()
            bf16 = ir.BF16Type.get()
            i32 = ir.IntegerType.get_signless(32)
            v8f32_ty = I.vec(8, I.f32)
            v2i32_ty = ir.VectorType.get([2], i32)

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

            # === L2 cache swizzle ===
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

            # === Wave position within workgroup ===
            c_wn = arith.index(waves_n)
            wave_m = wave_id // c_wn
            wave_n = wave_id % c_wn

            tile_m0 = bid_m * arith.index(tile_m)
            tile_n0 = bid_n * arith.index(tile_n)

            # === Buffer resources ===
            a_rsrc = buffer_ops.create_buffer_resource(_unwrap(arg_a), max_size=True)
            b_rsrc = buffer_ops.create_buffer_resource(_unwrap(arg_b), max_size=True)
            c_rsrc = buffer_ops.create_buffer_resource(_unwrap(arg_c), max_size=True)
            scale_a_rsrc = buffer_ops.create_buffer_resource(
                _unwrap(arg_scale_a), max_size=True
            )
            scale_b_rsrc = buffer_ops.create_buffer_resource(
                _unwrap(arg_scale_b), max_size=True
            )

            # === Load per-tensor scales ===
            scale_a_val = buffer_ops.buffer_load(
                scale_a_rsrc, arith.index(0), vec_width=1, dtype=f32
            )
            scale_b_val = buffer_ops.buffer_load(
                scale_b_rsrc, arith.index(0), vec_width=1, dtype=f32
            )
            combined_scale = scale_a_val * scale_b_val

            # === Tile load functions ===

            def _load_a_tile(k_tile_idx):
                """Load A fp8 tile. Returns [reg_k][wave_reg_m] of v2i32."""
                a_vecs = []
                m0_base = tile_m0 // c16 + wave_m * arith.index(wave_reg_m)
                for rk in range_constexpr(reg_k):
                    rk_vecs = []
                    k0 = k_tile_idx * arith.index(reg_k) + arith.index(rk)
                    for rm in range_constexpr(wave_reg_m):
                        m0 = m0_base + arith.index(rm)
                        # Byte offset into preshuffled A
                        byte_off = (
                            m0 * arith.index(A_STRIDE_M0)
                            + k0 * arith.index(A_STRIDE_K0)
                            + klane * arith.index(A_STRIDE_KLANE)
                            + lane16 * arith.index(A_STRIDE_MLANE)
                        )
                        # Load 8 bytes as dwordx2 (2 x i32)
                        dword_off = byte_off // arith.index(4)
                        a_raw = buffer_ops.buffer_load(
                            a_rsrc, dword_off, vec_width=2, dtype=i32
                        )
                        rk_vecs.append(a_raw)
                    a_vecs.append(rk_vecs)
                return a_vecs

            def _load_b_tile(k_tile_idx):
                """Load B fp8 tile. Returns [reg_k][wave_reg_n] of v2i32."""
                b_vecs = []
                n0_base = tile_n0 // c16 + wave_n * arith.index(wave_reg_n)
                for rk in range_constexpr(reg_k):
                    rk_vecs = []
                    k0 = k_tile_idx * arith.index(reg_k) + arith.index(rk)
                    for rn in range_constexpr(wave_reg_n):
                        n0 = n0_base + arith.index(rn)
                        byte_off = (
                            n0 * arith.index(B_STRIDE_N0)
                            + k0 * arith.index(B_STRIDE_K0)
                            + klane * arith.index(B_STRIDE_KLANE)
                            + lane16 * arith.index(B_STRIDE_NLANE)
                        )
                        dword_off = byte_off // arith.index(4)
                        b_raw = buffer_ops.buffer_load(
                            b_rsrc, dword_off, vec_width=2, dtype=i32
                        )
                        rk_vecs.append(b_raw)
                    b_vecs.append(rk_vecs)
                return b_vecs

            # === Compute function ===

            def _do_compute(accs_in, a_vecs, b_vecs):
                """Run WMMA fp8 multiply-accumulate for one tile."""
                new_accs = list(accs_in)
                for rk in range_constexpr(reg_k):
                    # Load all B for this rk, then iterate A (minimize reg pressure)
                    for rm in range_constexpr(wave_reg_m):
                        for rn in range_constexpr(wave_reg_n):
                            idx = rm * wave_reg_n + rn
                            new_accs[idx] = rocdl.wmma_f32_16x16x16_fp8_fp8(
                                v8f32_ty,
                                [
                                    a_vecs[rk][rm],
                                    b_vecs[rk][rn],
                                    arith.unwrap(new_accs[idx]),
                                ],
                            )
                return new_accs

            # === Initialize accumulators ===
            zero_acc = arith.constant_vector(0.0, v8f32_ty)
            accs = [zero_acc for _ in range_constexpr(wave_reg_m * wave_reg_n)]

            # === Software-pipelined K-loop ===
            # Prologue: load first tile
            a_cur = _load_a_tile(arith.index(0))
            b_cur = _load_b_tile(arith.index(0))

            full_outer_iters = (num_k_tiles - 1) // k_unroll
            remainder = (num_k_tiles - 1) % k_unroll

            # Main loop: load next tile, compute current tile
            for kt_outer in range(full_outer_iters):
                for j in range_constexpr(k_unroll):
                    next_kt = kt_outer * arith.index(k_unroll) + arith.index(j + 1)
                    a_next = _load_a_tile(next_kt)
                    b_next = _load_b_tile(next_kt)
                    accs = _do_compute(accs, a_cur, b_cur)
                    a_cur = a_next
                    b_cur = b_next

            # Handle remainder iterations
            if remainder > 0:
                for j in range_constexpr(remainder):
                    next_kt = arith.index(full_outer_iters * k_unroll + j + 1)
                    a_next = _load_a_tile(next_kt)
                    b_next = _load_b_tile(next_kt)
                    accs = _do_compute(accs, a_cur, b_cur)
                    a_cur = a_next
                    b_cur = b_next

            # Epilogue: compute last loaded tile
            accs = _do_compute(accs, a_cur, b_cur)

            # === Store results with scaling ===
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
                        # Apply combined scale and truncate to bf16
                        val = val * combined_scale
                        val_bf16 = arith.trunc_f(bf16, val)
                        elem_off = g_row * c_n + g_col
                        buffer_ops.buffer_store(val_bf16, c_rsrc, elem_off)

        @flir.jit
        def __call__(
            self: flir.T.i64,
            arg_c: lambda: T.memref(DYN, T.bf16()),
            arg_a: lambda: T.memref(DYN, T.f32()),
            arg_b: lambda: T.memref(DYN, T.f32()),
            arg_scale_a: lambda: T.memref(DYN, T.f32()),
            arg_scale_b: lambda: T.memref(DYN, T.f32()),
            c_m: lambda: I.index,
            c_n: lambda: I.index,
            c_k: lambda: I.index,
            stream_ptr: lambda: I.i64,
        ):
            c1 = arith.constant(1, index=True)
            bdx = arith.constant(THREADS_PER_BLOCK, index=True)
            tm = arith.constant(tile_m, index=True)
            tn = arith.constant(tile_n, index=True)
            gx = c_m / tm
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
    return flydsl.compile(m)


__all__ = [
    "compile_fp8_gemm",
    "preshuffle_a_fp8",
    "preshuffle_b_fp8",
    "fp8_quantize_per_tensor",
]
