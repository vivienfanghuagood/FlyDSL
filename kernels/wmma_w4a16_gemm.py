"""WMMA W4A16 GEMM kernel for RDNA4 (gfx12xx, wave32).

Optimized for memory-bound small-M inference: C[M,N] = A[M,K] @ dequant(B_int4[K,N]).
Target shape: M=16, N=8192, K=6144 (decode phase, single token batch).

Key design:
  - Parallelizes across N (each workgroup handles tile_n=128 columns)
  - Each workgroup iterates over full K dimension
  - A is bf16 preshuffled into WMMA A operand layout (direct buffer_load)
  - B is int4 packed in transposed layout: B_packed_t[N, K//2] uint8
    (K contiguous for coalesced loads)
  - Scales: scales_t[N, K//group_size] f32 (same N-first order)
  - Uses WMMA bf16 for 16x16x16 tiles, f32 accumulation
  - Software-pipelined K-loop with configurable unroll
  - No LDS needed (A is preshuffled, B dequant in registers)

Quantization: symmetric unsigned, zero_point=8
  float_val = (uint4_val - 8) * scale
  Optimized as FMA: uint4_val * scale + (-8 * scale)

Memory analysis (M=16, N=8192, K=6144):
  B weights: 8192 * 6144 / 2 = 25.2 MB (dominant)
  Scales:    8192 * (6144/128) * 4 = 1.57 MB
  A:         16 * 6144 * 2 = 192 KB (cached in L2)
  C:         16 * 8192 * 2 = 256 KB
  AI ≈ 64 FLOPs/byte => MEMORY-BOUND (ridge point = 248)
"""

import os
import functools

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
    while hasattr(v, "_value"):
        v = v._value
    return v


# =============================================================================
# Host-side helpers
# =============================================================================


def quantize_int4_symmetric(B_kn_f32, group_size=128):
    """Quantize B[K,N] f32 to symmetric int4 with per-group scales.

    Returns:
      B_packed_t: [N, K//2] uint8 (transposed, K contiguous for coalesced loads)
      scales_t: [N, K//group_size] f32 (transposed)
    """
    import torch

    K, N = B_kn_f32.shape
    assert K % group_size == 0

    num_groups = K // group_size
    B_grouped = B_kn_f32.reshape(num_groups, group_size, N)

    amax = B_grouped.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)
    scales = (amax / 7.0).squeeze(1)  # [groups, N]

    B_q = torch.round(B_grouped / amax * 7.0).to(torch.int8) + 8
    B_q = B_q.clamp(0, 15).to(torch.uint8)
    B_q = B_q.reshape(K, N)

    B_even = B_q[0::2, :]  # [K//2, N]
    B_odd = B_q[1::2, :]
    B_packed = B_even | (B_odd << 4)  # [K//2, N]

    # Transpose for K-contiguous access
    B_packed_t = B_packed.t().contiguous()  # [N, K//2]
    scales_t = scales.t().contiguous()  # [N, num_groups]

    return B_packed_t, scales_t


def preshuffle_a_bf16(A_mk):
    """Preshuffle A[M,K] in bf16 for WMMA A operand layout.

    Layout: [M0, K0, KLane=2, MLane=16, KPack=8] bf16 (16 bytes per lane).
    Each lane's buffer_load directly yields the WMMA A operand.
    """
    import torch

    M, K = A_mk.shape
    assert M % 16 == 0 and K % 16 == 0
    M0 = M // 16
    K0 = K // 16
    A_reshaped = A_mk.reshape(M0, 16, K0, 2, 8)
    A_shuffled = A_reshaped.permute(0, 2, 3, 1, 4).contiguous()
    return A_shuffled


# =============================================================================
# Kernel compiler
# =============================================================================


@functools.lru_cache(maxsize=64)
def compile_w4a16_gemm(
    *,
    M: int,
    N: int,
    K: int,
    tile_n: int = 128,
    group_size: int = 128,
    num_waves: int = 4,
):
    """Compile W4A16 GEMM kernel optimized for small M, large N/K.

    Design:
      - Grid: one workgroup per tile_n columns
      - Wave layout: waves_m=1, waves_n=num_waves (all waves along N)
      - Each wave: wave_reg_n WMMA N-tiles, iterates full K
      - A is bf16 preshuffled [M0, K0, KLane=2, MLane=16, KPack=8]
      - B_packed_t[N, K//2] with K contiguous (coalesced loads)
      - Dequants int4->bf16 in registers, accumulates via WMMA f32

    Args:
        M: Rows (tokens), must be multiple of 16
        N: Columns (output features)
        K: Inner dimension (input features)
        tile_n: N-tile per workgroup (default 128)
        group_size: INT4 quantization group size
        num_waves: Waves per workgroup

    Returns:
        exe(c, a_shuf, b_packed_t, scales_t, M, N, K, stream_ptr)
        where a_shuf: preshuffled [M0, K0, 2, 16, 8] bf16
              b_packed_t: [N, K//2] uint8 (via f32 memref)
              scales_t: [N, K//group_size] f32
              c: [M, N] bf16 (standard row-major)
    """
    gpu_arch = get_rocm_arch()

    WAVE_SIZE = 32
    assert M % WMMA_M == 0, f"M={M} must be multiple of {WMMA_M}"
    assert N % tile_n == 0, f"N={N} must be multiple of tile_n={tile_n}"
    assert K % WMMA_K == 0, f"K={K} must be multiple of {WMMA_K}"
    assert K % group_size == 0
    assert tile_n % WMMA_N == 0

    reg_m = M // WMMA_M  # WMMA tiles in M (1 for M=16)

    # Wave layout: all waves along N
    waves_n = num_waves
    reg_n_total = tile_n // WMMA_N
    wave_reg_n = reg_n_total // waves_n  # WMMA N-tiles per wave

    THREADS_PER_BLOCK = num_waves * WAVE_SIZE
    grid_n = N // tile_n
    num_k_iters = K // WMMA_K
    K_packed = K // 2  # number of packed bytes per column

    # A preshuffle strides (in bf16 elements)
    K0_total = K // 16
    A_KPACK = 8
    A_STRIDE_MLANE = A_KPACK  # 8
    A_STRIDE_KLANE = 16 * A_KPACK  # 128
    A_STRIDE_K0 = 2 * 16 * A_KPACK  # 256
    A_STRIDE_M0 = K0_total * A_STRIDE_K0

    DYN = ir.ShapedType.get_dynamic_size()

    # K-loop unrolling
    k_unroll = min(4, num_k_iters - 1) if num_k_iters > 1 else 1
    while k_unroll > 1 and (num_k_iters - 1) % k_unroll != 0:
        k_unroll -= 1

    module_name = f"w4a16_gemm_M{M}_N{N}"

    class _GEMM(flir.MlirModule):
        GPU_MODULE_NAME = module_name
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}">']

        @flir.kernel
        def kernel_gemm(
            self: flir.T.i64,
            arg_c: lambda: T.memref(DYN, T.bf16()),
            arg_a: lambda: T.memref(DYN, T.bf16()),
            arg_b: lambda: T.memref(DYN, T.f32()),  # B packed bytes via f32
            arg_scales: lambda: T.memref(DYN, T.f32()),
            c_m: lambda: I.index,
            c_n: lambda: I.index,
            c_k: lambda: I.index,
        ):
            f32 = ir.F32Type.get()
            bf16 = ir.BF16Type.get()
            i32 = ir.IntegerType.get_signless(32)
            i16_ty = ir.IntegerType.get_signless(16)
            v8f32_ty = I.vec(8, I.f32)
            v8bf16_ty = ir.VectorType.get([8], bf16)
            v8i16_ty = ir.VectorType.get([8], i16_ty)

            tid = flir.thread_idx("x")
            bid = flir.block_idx("x")

            c32 = arith.index(32)
            c16 = arith.index(16)
            c8 = arith.index(8)
            wave_id = tid // c32
            lane = tid % c32
            lane16 = lane % c16
            klane = lane // c16

            wave_n = wave_id  # wave_id indexes along N

            tile_n0 = bid * arith.index(tile_n)

            a_rsrc = buffer_ops.create_buffer_resource(_unwrap(arg_a), max_size=True)
            b_rsrc = buffer_ops.create_buffer_resource(_unwrap(arg_b), max_size=True)
            c_rsrc = buffer_ops.create_buffer_resource(_unwrap(arg_c), max_size=True)
            scale_rsrc = buffer_ops.create_buffer_resource(
                _unwrap(arg_scales), max_size=True
            )

            # B_packed_t layout: [N, K//2] uint8
            c_k_packed = arith.index(K_packed)

            # scales_t layout: [N, K//group_size] f32
            num_k_groups = K // group_size
            c_num_k_groups = arith.index(num_k_groups)

            # --------------------------------------------------------
            # Load helpers
            # --------------------------------------------------------

            def _load_a_tile(k_base_idx):
                """Load A bf16 from preshuffled GMEM.
                Returns list of v8i16 (bitcast from v8bf16) per reg_m.
                """
                vecs = []
                k0 = k_base_idx  # k0 index into preshuffle
                for rm in range_constexpr(reg_m):
                    m0 = arith.index(rm)
                    elem_off = (
                        m0 * arith.index(A_STRIDE_M0)
                        + k0 * arith.index(A_STRIDE_K0)
                        + klane * arith.index(A_STRIDE_KLANE)
                        + lane16 * arith.index(A_STRIDE_MLANE)
                    )
                    # bf16 elements -> f32 dword offset for buffer_load
                    f32_off = elem_off // arith.index(2)
                    a_raw = buffer_ops.buffer_load(
                        a_rsrc, f32_off, vec_width=4, dtype=f32
                    )
                    a_vec = vector.bitcast(v8bf16_ty, a_raw)
                    vecs.append(vector.bitcast(v8i16_ty, a_vec))
                return vecs

            def _load_b_dequant_tile(k_base_idx):
                """Load B int4, dequant to bf16 for WMMA.
                Returns list of v8i16 (bitcast from v8bf16) per wave_reg_n.

                k_base_idx is the WMMA K-tile index (0..num_k_iters-1).
                Each WMMA K-tile = 16 K values = 8 packed bytes per lane.
                """
                vecs = []
                k_base = k_base_idx * arith.index(WMMA_K)
                for rn in range_constexpr(wave_reg_n):
                    n_col_base = (
                        tile_n0
                        + wave_n * arith.index(wave_reg_n * WMMA_N)
                        + arith.index(rn * WMMA_N)
                    )
                    n_col = n_col_base + lane16

                    # Each lane loads 4 packed bytes (8 int4 values) for its K-half
                    # klane=0: K[0:8], klane=1: K[8:16]
                    k_half_start = k_base // arith.index(2) + klane * arith.index(4)
                    b_byte_off = n_col * c_k_packed + k_half_start
                    b_dword_off = b_byte_off // arith.index(4)
                    packed_i32 = buffer_ops.buffer_load(
                        b_rsrc, b_dword_off, vec_width=1, dtype=i32
                    )

                    # Load scale for this (n_col, K-group)
                    k_abs = k_base + klane * c8
                    group_idx = k_abs // arith.index(group_size)
                    scale_off = n_col * c_num_k_groups + group_idx
                    scale_val = buffer_ops.buffer_load(
                        scale_rsrc, scale_off, vec_width=1, dtype=f32
                    )

                    # Pre-compute bias = -8.0 * scale for FMA dequant
                    bias = arith.constant(-8.0, type=f32) * scale_val

                    # Unpack 8 int4 nibbles -> 8 bf16
                    bf16_vals = []
                    for ni in range_constexpr(8):
                        shift = arith.constant(ni * 4, type=i32)
                        nibble = arith.andi(
                            arith.shrui(packed_i32, shift),
                            arith.constant(0xF, type=i32),
                        )
                        nibble_f32 = arith.uitofp(f32, nibble)
                        dequant_f32 = nibble_f32 * scale_val + bias
                        bf16_vals.append(arith.trunc_f(bf16, dequant_f32))

                    b_vec = vector.from_elements(v8bf16_ty, bf16_vals)
                    vecs.append(vector.bitcast(v8i16_ty, b_vec))
                return vecs

            def _do_compute(accs_in, a_vecs, b_vecs):
                """WMMA compute step. Returns new accs list."""
                new_accs = list(accs_in)
                for rm in range_constexpr(reg_m):
                    for rn in range_constexpr(wave_reg_n):
                        idx = rm * wave_reg_n + rn
                        new_accs[idx] = rocdl.wmma_f32_16x16x16_bf16(
                            v8f32_ty,
                            [a_vecs[rm], b_vecs[rn], arith.unwrap(new_accs[idx])],
                        )
                return new_accs

            # --------------------------------------------------------
            # Initialize accumulators
            # --------------------------------------------------------
            zero_acc = arith.constant_vector(0.0, v8f32_ty)
            accs = [zero_acc for _ in range_constexpr(reg_m * wave_reg_n)]

            # --------------------------------------------------------
            # Pipelined K-loop
            # --------------------------------------------------------
            a_cur = _load_a_tile(arith.index(0))
            b_cur = _load_b_dequant_tile(arith.index(0))

            full_outer_iters = (num_k_iters - 1) // k_unroll
            remainder = (num_k_iters - 1) % k_unroll

            for kt_outer in range(full_outer_iters):
                for j in range_constexpr(k_unroll):
                    next_kt = kt_outer * arith.index(k_unroll) + arith.index(j + 1)
                    a_next = _load_a_tile(next_kt)
                    b_next = _load_b_dequant_tile(next_kt)
                    accs = _do_compute(accs, a_cur, b_cur)
                    a_cur = a_next
                    b_cur = b_next

            if remainder > 0:
                for j in range_constexpr(remainder):
                    next_kt = arith.index(full_outer_iters * k_unroll + j + 1)
                    a_next = _load_a_tile(next_kt)
                    b_next = _load_b_dequant_tile(next_kt)
                    accs = _do_compute(accs, a_cur, b_cur)
                    a_cur = a_next
                    b_cur = b_next

            # Epilogue: compute with last loaded tile
            accs = _do_compute(accs, a_cur, b_cur)

            # --------------------------------------------------------
            # Store output: C[M, N] bf16 row-major
            # --------------------------------------------------------
            base8 = klane * c8
            for rm in range_constexpr(reg_m):
                for rn in range_constexpr(wave_reg_n):
                    idx = rm * wave_reg_n + rn
                    n_col_base = (
                        tile_n0
                        + wave_n * arith.index(wave_reg_n * WMMA_N)
                        + arith.index(rn * WMMA_N)
                    )
                    for si in range_constexpr(8):
                        g_row = arith.index(rm * WMMA_M) + base8 + arith.index(si)
                        g_col = n_col_base + lane16
                        val = vector.extract(
                            accs[idx],
                            static_position=[si],
                            dynamic_position=[],
                        )
                        val = arith.trunc_f(bf16, val)
                        elem_off = g_row * c_n + g_col
                        buffer_ops.buffer_store(val, c_rsrc, elem_off)

        @flir.jit
        def __call__(
            self: flir.T.i64,
            arg_c: lambda: T.memref(DYN, T.bf16()),
            arg_a: lambda: T.memref(DYN, T.bf16()),
            arg_b: lambda: T.memref(DYN, T.f32()),
            arg_scales: lambda: T.memref(DYN, T.f32()),
            c_m: lambda: I.index,
            c_n: lambda: I.index,
            c_k: lambda: I.index,
            stream_ptr: lambda: I.i64,
        ):
            c1 = arith.constant(1, index=True)
            bdx = arith.constant(THREADS_PER_BLOCK, index=True)
            tn = arith.constant(tile_n, index=True)
            total_blocks = c_n / tn
            stream_token = stream_ptr_to_async_token(stream_ptr)
            flir.gpu_ext.LaunchFuncOp(
                [module_name, "kernel_gemm"],
                grid_size=(total_blocks, c1, c1),
                block_size=(bdx, c1, c1),
                kernel_operands=[
                    arg_c,
                    arg_a,
                    arg_b,
                    arg_scales,
                    c_m,
                    c_n,
                    c_k,
                ],
                async_dependencies=[stream_token],
            )

    m = _GEMM()
    return flydsl.compile(m)


__all__ = [
    "compile_w4a16_gemm",
    "quantize_int4_symmetric",
    "preshuffle_a_bf16",
]
