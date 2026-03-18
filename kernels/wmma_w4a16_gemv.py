"""WMMA W4A16 GEMV kernel for RDNA4 (gfx12xx, wave32).

Optimized for memory-bound small-M inference (M=16..64, decode phase).
Weight-only INT4 quantization with per-group scales, bf16 activations.

Key design:
  - Parallelizes across N dimension (each workgroup handles tile_n columns)
  - Each workgroup iterates over full K dimension
  - Weight layout: B_packed_t[N, K//2] uint8 — K is contiguous for coalesced loads
  - Scale layout:  scales_t[N, K//group_size] f32 — same N-first order
  - Uses WMMA bf16 for M=16/32/64 tiles (16x16x16 matrix multiply)

Quantization: symmetric unsigned, zero_point=8
  float_val = (uint4_val - 8) * scale
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


# =============================================================================
# Kernel compiler
# =============================================================================


@functools.lru_cache(maxsize=64)
def compile_wmma_w4a16_gemv(
    *,
    M: int,
    N: int,
    K: int,
    tile_n: int = 128,
    group_size: int = 128,
    num_waves: int = 4,
):
    """Compile W4A16 GEMV kernel optimized for small M.

    Design:
      - Grid: one workgroup per tile_n columns
      - Wave layout: waves_m=1, waves_n=num_waves (all waves along N)
      - Each wave: (tile_n/num_waves/16) WMMA N-tiles, iterates full K
      - Loads A[M, K] as bf16 (cached, small)
      - Loads B_packed_t[N, K//2] with K contiguous (coalesced loads)
      - Dequants int4->bf16, accumulates via WMMA f32

    Args:
        M: Rows (tokens), must be multiple of 16, typically 16-64
        N: Columns (output features)
        K: Inner dimension (input features)
        tile_n: N-tile per workgroup (default 128)
        group_size: INT4 quantization group size
        num_waves: Waves per workgroup

    Returns:
        exe(c, a, b_packed_t, scales_t, M, N, K, stream_ptr)
        where a: [M, K] bf16 (standard row-major)
              b_packed_t: [N, K//2] uint8 (transposed packed, K contiguous)
              scales_t: [N, K//group_size] f32 (transposed)
              c: [M, N] bf16 (standard row-major)
    """
    gpu_arch = get_rocm_arch()

    WAVE_SIZE = 32
    assert M % WMMA_M == 0, f"M={M} must be multiple of {WMMA_M}"
    assert N % tile_n == 0, f"N={N} must be multiple of tile_n={tile_n}"
    assert K % WMMA_K == 0, f"K={K} must be multiple of {WMMA_K}"
    assert K % group_size == 0
    assert tile_n % WMMA_N == 0

    reg_m = M // WMMA_M  # WMMA tiles in M (1 for M=16, 2 for M=32, etc.)

    # Wave layout: all waves along N
    waves_n = num_waves
    reg_n_total = tile_n // WMMA_N
    wave_reg_n = reg_n_total // waves_n  # WMMA N-tiles per wave

    THREADS_PER_BLOCK = num_waves * WAVE_SIZE
    grid_n = N // tile_n
    num_k_iters = K // WMMA_K
    K_packed = K // 2  # number of packed bytes per column

    DYN = ir.ShapedType.get_dynamic_size()

    module_name = f"wmma_w4a16_gemv_M{M}_N{N}"

    class _GEMV(flir.MlirModule):
        GPU_MODULE_NAME = module_name
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}">']

        @flir.kernel
        def kernel_gemv(
            self: flir.T.i64,
            arg_c: lambda: T.memref(DYN, T.bf16()),
            arg_a: lambda: T.memref(DYN, T.bf16()),
            arg_b: lambda: T.memref(DYN, T.f32()),  # B packed as raw bytes via f32
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
            # For column n, packed byte at K-position k: offset = n * K_packed + k//2
            c_k_packed = arith.index(K_packed)

            # scales_t layout: [N, K//group_size] f32
            num_k_groups = K // group_size
            c_num_k_groups = arith.index(num_k_groups)

            # Helper functions for pipelined K-loop
            def _load_a_tile(k_base_idx):
                """Load A bf16 tiles. Returns list of v8i16 per reg_m."""
                vecs = []
                k_base = k_base_idx * arith.index(WMMA_K)
                for rm in range_constexpr(reg_m):
                    m_row = arith.index(rm * WMMA_M) + lane16
                    k_col = k_base + klane * c8
                    a_elem_off = m_row * c_k + k_col
                    a_raw = buffer_ops.buffer_load(
                        a_rsrc, a_elem_off, vec_width=8, dtype=bf16
                    )
                    vecs.append(vector.bitcast(v8i16_ty, a_raw))
                return vecs

            def _load_b_dequant_tile(k_base_idx):
                """Load B int4, dequant to bf16. Returns list of v8i16 per wave_reg_n."""
                vecs = []
                k_base = k_base_idx * arith.index(WMMA_K)
                for rn in range_constexpr(wave_reg_n):
                    n_col_base = (
                        tile_n0
                        + wave_n * arith.index(wave_reg_n * WMMA_N)
                        + arith.index(rn * WMMA_N)
                    )
                    n_col = n_col_base + lane16

                    k_half_start = k_base // arith.index(2) + klane * arith.index(4)
                    b_byte_off = n_col * c_k_packed + k_half_start
                    b_dword_off = b_byte_off // arith.index(4)
                    packed_i32 = buffer_ops.buffer_load(
                        b_rsrc, b_dword_off, vec_width=1, dtype=i32
                    )

                    k_abs = k_base + klane * c8
                    group_idx = k_abs // arith.index(group_size)
                    scale_off = n_col * c_num_k_groups + group_idx
                    scale_val = buffer_ops.buffer_load(
                        scale_rsrc, scale_off, vec_width=1, dtype=f32
                    )

                    bias = arith.constant(-8.0, type=f32) * scale_val

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

            # Initialize accumulators
            zero_acc = arith.constant_vector(0.0, v8f32_ty)
            accs = [zero_acc for _ in range_constexpr(reg_m * wave_reg_n)]

            # Pipelined K-loop (same pattern as preshuffle GEMM)
            k_unroll = min(4, num_k_iters - 1) if num_k_iters > 1 else 1
            while k_unroll > 1 and (num_k_iters - 1) % k_unroll != 0:
                k_unroll -= 1

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

            accs = _do_compute(accs, a_cur, b_cur)

            # Store output: C[M, N] bf16 row-major
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
                [module_name, "kernel_gemv"],
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

    m = _GEMV()
    return flydsl.compile(m)
