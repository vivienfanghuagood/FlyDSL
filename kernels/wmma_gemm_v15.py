#!/usr/bin/env python3
"""WMMA GEMM kernel v15 for RDNA4 (gfx12xx, wave32).

Key improvement: Hybrid approach combining v7's pipelining with range_constexpr.

Strategy:
  - Outer loop: range(num_k_tiles // UNROLL) with loop-carried vars for
    "prefetched" data (loads from last unrolled iteration of previous outer loop)
  - Inner loop: range_constexpr(UNROLL) fully unrolled
  - Within each constexpr iteration: compute with "current" data, then load "next"
  - This means N+1's loads overlap with N's compute, similar to v7
  - But the inner unroll gives the compiler a larger scheduling window

The key difference from v7: instead of load→copy→compute per iteration with
23 register copies, we have UNROLL iterations of load→compute inline,
and only need register copies at the outer loop boundary.

Architecture: "preshuffle A + preshuffle B" (same as v6/v7)

Computes C[M,N] = A_shuffled @ B_shuffled
"""

import os

from flydsl.dialects.ext import flir, arith, memref, vector, rocdl, gpu
from flydsl.dialects.ext.python_control_flow import range_constexpr
from flydsl.runtime.device import get_rocm_arch
from flydsl.lang.ir.types import T
from flydsl.dialects.ext import buffer_ops
from _mlir import ir
from _mlir.dialects import llvm as _llvm
from _mlir.dialects import arith as _std_arith
from _mlir.dialects import memref as _std_memref
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
# Host-side pre-shuffle (same as v6/v7)
# =============================================================================


def preshuffle_a_wmma(A_mk):
    import torch

    M, K = A_mk.shape
    assert M % 16 == 0 and K % 16 == 0
    M0 = M // 16
    K0 = K // 16
    A_reshaped = A_mk.reshape(M0, 16, K0, 2, 8)
    A_shuffled = A_reshaped.permute(0, 2, 3, 1, 4).contiguous()
    return A_shuffled


def preshuffle_b_wmma(B_kn):
    import torch

    K, N = B_kn.shape
    assert K % 16 == 0 and N % 16 == 0
    N0 = N // 16
    K0 = K // 16
    B_reshaped = B_kn.reshape(K0, 2, 8, N0, 16)
    B_shuffled = B_reshaped.permute(3, 0, 1, 4, 2).contiguous()
    return B_shuffled


# =============================================================================
# Kernel module
# =============================================================================


def create_wmma_gemm_v15_module(
    M: int,
    N: int,
    K: int,
    in_dtype="bf16",
    out_dtype="bf16",
    *,
    reg_m=4,
    reg_n=4,
    reg_k=2,
    waves_m=2,
    waves_n=2,
    group_m=8,
    k_unroll=4,
):
    """Create WMMA GEMM v15 module: pipelined with range_constexpr inner unrolling."""
    BLOCK_M = WMMA_M * reg_m * waves_m
    BLOCK_N = WMMA_N * reg_n * waves_n
    BLOCK_K = WMMA_K * reg_k
    NUM_WAVES = waves_m * waves_n
    WAVE_SIZE = 32
    THREADS_PER_BLOCK = NUM_WAVES * WAVE_SIZE

    gpu_arch = get_rocm_arch()
    S = ir.ShapedType.get_dynamic_size()

    assert M % BLOCK_M == 0
    assert N % BLOCK_N == 0
    assert K % BLOCK_K == 0

    num_k_tiles = K // BLOCK_K
    assert num_k_tiles % k_unroll == 0
    # We need at least k_unroll+1 tiles for the pipelining to work
    # (prologue loads tile 0, then the loop processes k_unroll tiles at a time)
    assert num_k_tiles > k_unroll, (
        f"Need num_k_tiles ({num_k_tiles}) > k_unroll ({k_unroll})"
    )
    # outer loop runs (num_k_tiles - 1) // k_unroll times, but we want clean math
    # Actually: prologue loads tile[0], loop body processes k_unroll tiles starting from
    # the prefetched data, then loads the next k_unroll tiles.
    # Total: 1 prologue + outer_iters * k_unroll = num_k_tiles
    # So outer_iters = (num_k_tiles - 1) // k_unroll ... but that doesn't divide cleanly.
    #
    # Simpler: let's just use (num_k_tiles // k_unroll) outer iterations.
    # Prologue: load tile 0
    # Each outer iteration i: for j in range_constexpr(k_unroll):
    #   compute with current data
    #   if not last iteration overall: load next tile
    # Epilogue: compute with last tile
    #
    # But this is complex. Let's do it differently:
    # Just have the outer loop carry the "current" data, and within each
    # range_constexpr iteration, compute then load next.
    #
    # Actually the cleanest approach for FlyDSL:
    # Prologue: load tiles 0..k_unroll-1 using range_constexpr
    # Main loop: for kt_outer in range(num_k_tiles // k_unroll - 1):
    #   for j in range_constexpr(k_unroll):
    #     load next tile (kt_outer*k_unroll + k_unroll + j)
    #     compute with current tile (already loaded)
    #     current = loaded_next
    # Epilogue: compute last k_unroll tiles
    #
    # This way each outer iteration has k_unroll loads and k_unroll computes,
    # and the loads from iteration j overlap with compute from iteration j.
    # The loop-carried state is k_unroll sets of A/B tiles.
    #
    # But that's k_unroll * 16 * 4 = 256 VGPRs for loads alone (at k_unroll=4),
    # plus 128 for accumulators = 384 total. Would spill.
    #
    # Better: carry only 1 tile's worth of data as loop-carried,
    # and do a mini software pipeline within the constexpr block.

    outer_k_iters = num_k_tiles // k_unroll
    grid_m = M // BLOCK_M
    grid_n = N // BLOCK_N
    is_bf16 = in_dtype == "bf16"

    # A preshuffle constants
    M0_total = M // 16
    K0_total = K // 16
    A_KPACK = 8
    A_STRIDE_MLANE = A_KPACK
    A_STRIDE_KLANE = 16 * A_KPACK
    A_STRIDE_K0 = 2 * 16 * A_KPACK
    A_STRIDE_M0 = K0_total * A_STRIDE_K0

    # B preshuffle constants
    N0_total = N // 16
    B_KPACK = 8
    B_STRIDE_NLANE = B_KPACK
    B_STRIDE_KLANE = 16 * B_KPACK
    B_STRIDE_K0 = 2 * 16 * B_KPACK
    B_STRIDE_N0 = K0_total * B_STRIDE_K0

    def _in_elem_ty():
        return Textra.bf16() if is_bf16 else Textra.f16()

    def _out_elem_ty():
        return Textra.f32() if out_dtype == "f32" else Textra.bf16()

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

    class _WmmaGemmV15(flir.MlirModule):
        GPU_MODULE_NAME = "wmma_gemm_v15"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}">']

        @flir.kernel
        def wmma_gemm_v15_kernel(
            self: flir.T.i64,
            A_shuf: lambda: Textra.memref(S, _in_elem_ty()),
            B_shuf: lambda: Textra.memref(S, _in_elem_ty()),
            C: lambda: Textra.memref(S, S, _out_elem_ty()),
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

            tile_m0 = bid_m * arith.index(BLOCK_M)
            tile_n0 = bid_n * arith.index(BLOCK_N)

            a_rsrc = buffer_ops.create_buffer_resource(_unwrap(A_shuf), max_size=True)
            b_rsrc = buffer_ops.create_buffer_resource(_unwrap(B_shuf), max_size=True)
            c_rsrc = buffer_ops.create_buffer_resource(_unwrap(C), max_size=True)

            def _load_a_tile(k_tile_idx):
                """Load A operands from pre-shuffled GMEM."""
                a_vecs = []
                m0_base = tile_m0 // c16 + wave_m * arith.index(reg_m)
                for rk in range_constexpr(reg_k):
                    rk_vecs = []
                    k0 = k_tile_idx * arith.index(reg_k) + arith.index(rk)
                    for rm in range_constexpr(reg_m):
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
                n0_base = tile_n0 // c16 + wave_n * arith.index(reg_n)
                for rk in range_constexpr(reg_k):
                    rk_vecs = []
                    k0 = k_tile_idx * arith.index(reg_k) + arith.index(rk)
                    for rn in range_constexpr(reg_n):
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
                    for rm in range_constexpr(reg_m):
                        for rn in range_constexpr(reg_n):
                            idx = rm * reg_n + rn
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
            accs = [zero_acc for _ in range_constexpr(reg_m * reg_n)]

            # ============================================================
            # PIPELINED K-LOOP WITH range_constexpr INNER UNROLLING
            # ============================================================
            #
            # Structure (k_unroll=4, num_k_tiles=128):
            #   Prologue: load tile[0]
            #   Outer loop (32 iterations):
            #     for j in range_constexpr(k_unroll):  # j = 0,1,2,3
            #       load tile[outer*4 + j + 1]  (next tile)
            #       compute with current tile
            #       current = next
            #   Epilogue: compute with last loaded tile
            #
            # This gives 4 loads + 4 computes per constexpr block.
            # The loads for tile[j+1] overlap with compute of tile[j].
            # Only at the outer loop boundary do we need register copies.
            # With k_unroll=4, we get 4x fewer register copies than v7.

            # Prologue: load first tile
            a_cur = _load_a_tile(arith.index(0))
            b_cur = _load_b_tile(arith.index(0))

            # Main loop: process k_unroll tiles per outer iteration
            # Iteration i processes tiles [i*k_unroll .. (i+1)*k_unroll - 1]
            # But we've already loaded tile 0, so we process:
            #   tile 0 (already loaded), load tile 1
            #   tile 1, load tile 2
            #   ...
            #   tile k_unroll-1, load tile k_unroll (=next outer iteration's first tile)
            #
            # For the last outer iteration, we don't load beyond num_k_tiles-1.
            # So the outer loop runs (num_k_tiles - 1) // k_unroll times for the
            # "full" iterations, and we handle the remainder specially.
            #
            # Actually with FlyDSL's peeling, let's keep it simple:
            # outer loop: range(num_k_tiles - 1) with sw pipeline (like v7)
            # but we process k_unroll tiles at once within range_constexpr.
            #
            # Even simpler: Just do v7's approach but with k_unroll inner iterations.
            # outer loop: range((num_k_tiles - 1) // k_unroll)
            # inner: range_constexpr(k_unroll) doing load_next, compute_current
            # epilogue_inner: compute last group

            # Process (num_k_tiles - 1) tiles with pipelining
            # Each outer iteration processes k_unroll tiles
            full_outer_iters = (num_k_tiles - 1) // k_unroll
            remainder = (num_k_tiles - 1) % k_unroll

            for kt_outer in range(full_outer_iters):
                for j in range_constexpr(k_unroll):
                    # Load next tile
                    next_kt = kt_outer * arith.index(k_unroll) + arith.index(j + 1)
                    a_next = _load_a_tile(next_kt)
                    b_next = _load_b_tile(next_kt)
                    # Compute with current
                    accs = _do_compute(accs, a_cur, b_cur)
                    # Advance
                    a_cur = a_next
                    b_cur = b_next

            # Handle remainder tiles (if any)
            if remainder > 0:
                for j in range_constexpr(remainder):
                    next_kt = arith.index(full_outer_iters * k_unroll + j + 1)
                    a_next = _load_a_tile(next_kt)
                    b_next = _load_b_tile(next_kt)
                    accs = _do_compute(accs, a_cur, b_cur)
                    a_cur = a_next
                    b_cur = b_next

            # Epilogue: compute with the last loaded tile
            accs = _do_compute(accs, a_cur, b_cur)

            # ========== Store results ==========
            c_layout_n = arith.index(N)
            base8 = klane * c8

            for rm in range_constexpr(reg_m):
                for rn in range_constexpr(reg_n):
                    idx = rm * reg_n + rn
                    wmma_m_off = wave_m * arith.index(reg_m * WMMA_M) + arith.index(
                        rm * WMMA_M
                    )
                    wmma_n_off = wave_n * arith.index(reg_n * WMMA_N) + arith.index(
                        rn * WMMA_N
                    )
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
                        elem_off = g_row * c_layout_n + g_col
                        buffer_ops.buffer_store(val, c_rsrc, elem_off)

        @flir.jit
        def __call__(
            self: flir.T.i64,
            A_shuf: lambda: Textra.memref(S, _in_elem_ty()),
            B_shuf: lambda: Textra.memref(S, _in_elem_ty()),
            C: lambda: Textra.memref(S, S, _out_elem_ty()),
        ):
            c1 = arith.index(1)
            total_blocks = arith.index(grid_m * grid_n)
            bk = arith.index(THREADS_PER_BLOCK)
            flir.gpu_ext.LaunchFuncOp(
                ["wmma_gemm_v15", "wmma_gemm_v15_kernel"],
                grid_size=(total_blocks, c1, c1),
                block_size=(bk, c1, c1),
                kernel_operands=[A_shuf, B_shuf, C],
            )

    return _WmmaGemmV15(), BLOCK_M, BLOCK_N, BLOCK_K
