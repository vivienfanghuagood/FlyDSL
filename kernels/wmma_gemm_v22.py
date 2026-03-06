#!/usr/bin/env python3
"""WMMA GEMM kernel v22 for RDNA4 (gfx12xx, wave32).

Key insight from v15 ISA analysis: WMMA utilization is only 28.6% because
buffer_loads don't have enough runway before use. In v15, each load is only
~16 WMMAs (496 SIMD cycles) ahead of its use. This is barely enough for
L2 cache hits (~200 cycles) but not for VRAM/L3 misses (~400+ cycles).

v22 strategy: Increase load-ahead distance to ~4 k-tiles (64 WMMAs = ~2000
SIMD cycles) by loading in a "batch-ahead" pattern within constexpr unroll.

Pipeline structure (k_unroll=8, PREFETCH_DIST=4):
  Prologue: load tiles[0..3] (4 tiles ahead)
  Main loop:
    for j in range_constexpr(4):  # first half of k_unroll
      load tile[j+4]              # load 4 tiles ahead
      compute tile[j]             # use tile loaded 4 tiles ago
    for j in range_constexpr(4):  # second half
      load tile[j+8]              # load next batch
      compute tile[j+4]           # use from first half loads

This gives 64 WMMAs between load issue and use (~2000 SIMD cycles at 31 cyc/WMMA).

Architecture: "preshuffle A + preshuffle B" (same as v15)
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
import _mlir.extras.types as Textra


WMMA_M = 16
WMMA_N = 16
WMMA_K = 16


def _unwrap(v):
    while hasattr(v, "_value"):
        v = v._value
    return v


def preshuffle_a_wmma(A_mk):
    M, K = A_mk.shape
    assert M % 16 == 0 and K % 16 == 0
    M0 = M // 16
    K0 = K // 16
    A_reshaped = A_mk.reshape(M0, 16, K0, 2, 8)
    A_shuffled = A_reshaped.permute(0, 2, 3, 1, 4).contiguous()
    return A_shuffled


def preshuffle_b_wmma(B_kn):
    K, N = B_kn.shape
    assert K % 16 == 0 and N % 16 == 0
    N0 = N // 16
    K0 = K // 16
    B_reshaped = B_kn.reshape(K0, 2, 8, N0, 16)
    B_shuffled = B_reshaped.permute(3, 0, 1, 4, 2).contiguous()
    return B_shuffled


def create_wmma_gemm_v22_module(
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
    prefetch_dist=4,  # Load this many k-tiles ahead
):
    BLOCK_M = WMMA_M * reg_m * waves_m  # 128
    BLOCK_N = WMMA_N * reg_n * waves_n  # 128
    BLOCK_K = WMMA_K * reg_k  # 32
    NUM_WAVES = waves_m * waves_n  # 4
    WAVE_SIZE = 32
    THREADS_PER_BLOCK = NUM_WAVES * WAVE_SIZE  # 128

    gpu_arch = get_rocm_arch()
    S = ir.ShapedType.get_dynamic_size()

    assert M % BLOCK_M == 0
    assert N % BLOCK_N == 0
    assert K % BLOCK_K == 0

    num_k_tiles = K // BLOCK_K
    PD = prefetch_dist
    assert num_k_tiles > PD, f"Need num_k_tiles ({num_k_tiles}) > prefetch_dist ({PD})"
    assert (num_k_tiles - PD) % PD == 0, (
        f"(num_k_tiles-PD)={num_k_tiles - PD} must be divisible by PD={PD}"
    )

    # Outer loop: process PD tiles per iteration
    # Structure:
    #   Prologue: load tiles[0..PD-1]
    #   Loop iter i (runs (num_k_tiles - PD) / PD times):
    #     for j in range_constexpr(PD):
    #       load tile[PD + i*PD + j]  (next batch)
    #       compute tile[i*PD + j]    (from previous batch)
    #   Epilogue: compute last PD tiles (no more loads needed)

    outer_iters = (num_k_tiles - PD) // PD

    grid_m = M // BLOCK_M
    grid_n = N // BLOCK_N
    is_bf16 = in_dtype == "bf16"

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

    class _WmmaGemmV22(flir.MlirModule):
        GPU_MODULE_NAME = "wmma_gemm_v22"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}">']

        @flir.kernel
        def wmma_gemm_v22_kernel(
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

            zero_acc = arith.constant_vector(0.0, v8f32_ty)
            accs = [zero_acc for _ in range_constexpr(reg_m * reg_n)]

            # ============================================================
            # DEEP PREFETCH with PD=2:
            # Load 2 tiles ahead, compute 2 tiles behind.
            # Each iteration: compute 2 tiles (64 WMMAs) while loading 2 more.
            # Load-to-use distance: 2 tiles × 32 WMMAs = 64 WMMAs ≈ 2000 SIMD cycles
            # ============================================================

            assert PD == 2, "Currently only PD=2 is supported"

            # Prologue: load tiles 0 and 1
            a_tile0 = _load_a_tile(arith.index(0))
            b_tile0 = _load_b_tile(arith.index(0))
            a_tile1 = _load_a_tile(arith.index(1))
            b_tile1 = _load_b_tile(arith.index(1))

            # Flatten tile data into flat lists for loop-carried variables
            # Each tile: a = reg_k * reg_m vecs, b = reg_k * reg_n vecs
            # Flatten: a_tile = [a[0][0], a[0][1], ..., a[1][0], ...]
            def _flatten_a(a_vecs):
                flat = []
                for rk in range_constexpr(reg_k):
                    for rm in range_constexpr(reg_m):
                        flat.append(a_vecs[rk][rm])
                return flat

            def _flatten_b(b_vecs):
                flat = []
                for rk in range_constexpr(reg_k):
                    for rn in range_constexpr(reg_n):
                        flat.append(b_vecs[rk][rn])
                return flat

            def _unflatten_a(flat):
                result = []
                idx = 0
                for rk in range_constexpr(reg_k):
                    rk_vecs = []
                    for rm in range_constexpr(reg_m):
                        rk_vecs.append(flat[idx])
                        idx += 1
                    result.append(rk_vecs)
                return result

            def _unflatten_b(flat):
                result = []
                idx = 0
                for rk in range_constexpr(reg_k):
                    rk_vecs = []
                    for rn in range_constexpr(reg_n):
                        rk_vecs.append(flat[idx])
                        idx += 1
                    result.append(rk_vecs)
                return result

            # Main loop: process 2 tiles per iteration, load 2 more ahead
            # outer_iters = (num_k_tiles - 2) / 2
            for kt_outer in range(outer_iters):
                # Compute tile 0 (loaded 2 ago), load tile 0 of next batch
                next_kt0 = kt_outer * arith.index(PD) + arith.index(PD)
                a_next0 = _load_a_tile(next_kt0)
                b_next0 = _load_b_tile(next_kt0)
                accs = _do_compute(accs, a_tile0, b_tile0)

                # Compute tile 1 (loaded 2 ago), load tile 1 of next batch
                next_kt1 = kt_outer * arith.index(PD) + arith.index(PD + 1)
                a_next1 = _load_a_tile(next_kt1)
                b_next1 = _load_b_tile(next_kt1)
                accs = _do_compute(accs, a_tile1, b_tile1)

                # Rotate: next becomes current
                a_tile0 = a_next0
                b_tile0 = b_next0
                a_tile1 = a_next1
                b_tile1 = b_next1

            # Epilogue: compute last 2 tiles
            accs = _do_compute(accs, a_tile0, b_tile0)
            accs = _do_compute(accs, a_tile1, b_tile1)

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
                            accs[idx], static_position=[si], dynamic_position=[]
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
                ["wmma_gemm_v22", "wmma_gemm_v22_kernel"],
                grid_size=(total_blocks, c1, c1),
                block_size=(bk, c1, c1),
                kernel_operands=[A_shuf, B_shuf, C],
            )

    return _WmmaGemmV22(), BLOCK_M, BLOCK_N, BLOCK_K
