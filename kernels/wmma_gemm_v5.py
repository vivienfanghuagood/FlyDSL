#!/usr/bin/env python3
"""Tunable WMMA GEMM kernel v5 for RDNA4 (gfx12xx, wave32).

Key improvements over v2/v3:
  1. Parameterized tiling: REG_M, REG_N, WAVES_M, WAVES_N, REG_K all configurable
  2. Support for different wave counts (4, 8) via WAVES_M * WAVES_N
  3. Same preshuffle-B + LDS-A architecture as v2

Architecture: "preshuffle B + LDS A"
  - B is pre-shuffled in global memory for direct GMEM loading
  - A goes through LDS with barrier-protected single buffer

Computes C[M,N] = A[M,K] @ B_shuffled[N0,K0,KLane,NLane,KPack]
"""

import os

from flydsl.dialects.ext import flir, arith, memref, vector, rocdl, gpu
from flydsl.dialects.ext.python_control_flow import range_constexpr
from flydsl.runtime.device import get_rocm_arch
from flydsl.lang.ir.types import T
from flydsl.utils import SmemAllocator
from flydsl.dialects.ext import buffer_ops
from _mlir import ir
from _mlir.dialects import llvm as _llvm
from _mlir.dialects import arith as _std_arith
from _mlir.dialects import memref as _std_memref
import _mlir.extras.types as Textra


# =============================================================================
# Kernel configuration (defaults, overridable via create function)
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
# Host-side pre-shuffle for B (same as v2)
# =============================================================================


def preshuffle_b_wmma(B_kn):
    """Pre-shuffle B[K,N] into WMMA-friendly layout for direct GMEM loading."""
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


def create_wmma_gemm_v5_module(
    M: int,
    N: int,
    K: int,
    in_dtype="bf16",
    out_dtype="bf16",
    *,
    reg_m=2,
    reg_n=4,
    reg_k=2,
    waves_m=2,
    waves_n=2,
    group_m=8,
):
    """Create tunable WMMA GEMM v5 module.

    Args:
        M, N, K: matrix dimensions
        in_dtype: "bf16" or "f16"
        out_dtype: "f32" or "bf16"
        reg_m: WMMA M-tiles per wave
        reg_n: WMMA N-tiles per wave
        reg_k: WMMA K-steps per tile load
        waves_m: waves along M
        waves_n: waves along N
        group_m: L2 cache swizzle group size
    """
    # Derived constants
    BLOCK_M = WMMA_M * reg_m * waves_m
    BLOCK_N = WMMA_N * reg_n * waves_n
    BLOCK_K = WMMA_K * reg_k
    NUM_WAVES = waves_m * waves_n
    WAVE_SIZE = 32
    THREADS_PER_BLOCK = NUM_WAVES * WAVE_SIZE

    A_TILE_ELEMS = BLOCK_M * BLOCK_K
    A_LOAD_VEC = 8
    NUM_A_LOADS = A_TILE_ELEMS // (THREADS_PER_BLOCK * A_LOAD_VEC)
    assert NUM_A_LOADS >= 1, f"Need at least 1 A load, got {NUM_A_LOADS}"

    LDS_A_ELEMS = BLOCK_M * BLOCK_K

    gpu_arch = get_rocm_arch()
    S = ir.ShapedType.get_dynamic_size()

    assert M % BLOCK_M == 0, f"M={M} must be multiple of BLOCK_M={BLOCK_M}"
    assert N % BLOCK_N == 0, f"N={N} must be multiple of BLOCK_N={BLOCK_N}"
    assert K % BLOCK_K == 0, f"K={K} must be multiple of BLOCK_K={BLOCK_K}"

    num_k_tiles = K // BLOCK_K
    grid_m = M // BLOCK_M
    grid_n = N // BLOCK_N
    is_bf16 = in_dtype == "bf16"

    # B preshuffle constants
    N0 = N // 16
    K0_total = K // 16
    B_STRIDE_NLANE = 8
    B_STRIDE_KLANE = 16 * 8  # 128
    B_STRIDE_K0 = 2 * 16 * 8  # 256
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

    allocator = SmemAllocator(None, arch=gpu_arch)
    _state = {}

    class _WmmaGemmV5(flir.MlirModule):
        GPU_MODULE_NAME = "wmma_gemm_v5"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}">']

        def init_gpu_module(self):
            _state["lds_a"] = allocator.allocate_array(_in_elem_ty(), LDS_A_ELEMS)
            allocator.finalize()

        @flir.kernel
        def wmma_gemm_v5_kernel(
            self: flir.T.i64,
            A: lambda: Textra.memref(S, S, _in_elem_ty()),
            B_shuf: lambda: Textra.memref(S, _in_elem_ty()),
            C: lambda: Textra.memref(S, S, _out_elem_ty()),
        ):
            in_ir_ty = ir.BF16Type.get() if is_bf16 else ir.F16Type.get()
            v8_in_ty = ir.VectorType.get([8], in_ir_ty)
            v8f32_ty = T.vec(8, T.f32)
            i16_ty = ir.IntegerType.get_signless(16)
            v8i16_ty = ir.VectorType.get([8], i16_ty)
            i32_ty = ir.IntegerType.get_signless(32)
            i64_ty = ir.IntegerType.get_signless(64)
            ptr_ty = ir.Type.parse("!llvm.ptr")
            v4i32_ty = ir.VectorType.get([4], ir.IntegerType.get_signless(32))

            tid = flir.thread_idx("x")
            pid = flir.block_idx("x")

            c32 = arith.index(32)
            c16 = arith.index(16)
            c8 = arith.index(8)
            wave_id = tid // c32
            lane = tid % c32
            lane16 = lane % c16
            base8 = (lane // c16) * c8

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

            lds_base = allocator.get_base()
            lds_a_view = _state["lds_a"](lds_base).get()

            a_rsrc = buffer_ops.create_buffer_resource(_unwrap(A), max_size=True)
            b_rsrc = buffer_ops.create_buffer_resource(_unwrap(B_shuf), max_size=True)
            c_rsrc = buffer_ops.create_buffer_resource(_unwrap(C), max_size=True)

            # Pre-compute A LDS store addresses
            a_lds_addrs = []
            for al in range_constexpr(NUM_A_LOADS):
                a_lin = tid * c8 + arith.index(al * THREADS_PER_BLOCK * A_LOAD_VEC)
                a_load_row = a_lin // arith.index(BLOCK_K)
                a_load_col = a_lin % arith.index(BLOCK_K)
                a_lds_addrs.append(a_load_row * arith.index(BLOCK_K) + a_load_col)

            # Inline asm for batched A global loads
            if NUM_A_LOADS == 1:
                # Single load: no struct needed, just use v4i32 directly
                a_struct_ty = v4i32_ty
                asm_a_load_str = "global_load_b128 $0, $1, off"
                asm_a_constraints = "=&v,v"
            else:
                a_struct_ty = _llvm.StructType.get_literal([v4i32_ty] * NUM_A_LOADS)
                asm_a_load_lines = [f"s_clause {NUM_A_LOADS - 1}"]
                for i in range_constexpr(NUM_A_LOADS):
                    asm_a_load_lines.append(
                        f"global_load_b128 ${i}, ${i + NUM_A_LOADS}, off"
                    )
                asm_a_load_str = "\n".join(asm_a_load_lines)
                a_out_constraints = ",".join(["=&v"] * NUM_A_LOADS)
                a_in_constraints = ",".join(["v"] * NUM_A_LOADS)
                asm_a_constraints = f"{a_out_constraints},{a_in_constraints}"

            elem_bytes = 2
            a_base_i64 = _unwrap(
                _std_arith.IndexCastOp(
                    i64_ty,
                    _unwrap(
                        _std_memref.ExtractAlignedPointerAsIndexOp(_unwrap(A)).result
                    ),
                ).result
            )

            def _compute_a_load_addrs(k_base):
                addrs = []
                for al in range_constexpr(NUM_A_LOADS):
                    a_lin = tid * c8 + arith.index(al * THREADS_PER_BLOCK * A_LOAD_VEC)
                    a_load_row = a_lin // arith.index(BLOCK_K)
                    a_load_col = a_lin % arith.index(BLOCK_K)
                    g_a_row = tile_m0 + a_load_row
                    g_a_col = k_base + a_load_col
                    byte_off = (g_a_row * arith.index(K) + g_a_col) * arith.index(
                        elem_bytes
                    )
                    byte_off_i64 = _unwrap(
                        _std_arith.IndexCastOp(
                            i64_ty, _unwrap(arith.unwrap(byte_off))
                        ).result
                    )
                    addr_i64 = _unwrap(
                        _std_arith.AddIOp(a_base_i64, byte_off_i64).result
                    )
                    addr_ptr = _unwrap(_llvm.IntToPtrOp(ptr_ty, addr_i64).result)
                    addrs.append(addr_ptr)
                return addrs

            def _issue_a_loads(addrs):
                return _llvm.inline_asm(
                    a_struct_ty,
                    addrs,
                    asm_a_load_str,
                    asm_a_constraints,
                    has_side_effects=True,
                )

            def _store_a_to_lds(asm_result, lds_view):
                if NUM_A_LOADS == 1:
                    # Single load returns v4i32 directly
                    bf16_vec = vector.bitcast(v8_in_ty, asm_result)
                    vector.store(bf16_vec, lds_view, [a_lds_addrs[0]])
                else:
                    for al in range_constexpr(NUM_A_LOADS):
                        pos_attr = ir.DenseI64ArrayAttr.get([al])
                        v4i32_val = _llvm.ExtractValueOp(
                            v4i32_ty, asm_result, pos_attr
                        ).result
                        bf16_vec = vector.bitcast(v8_in_ty, v4i32_val)
                        vector.store(bf16_vec, lds_view, [a_lds_addrs[al]])

            def _load_b_tile(k_tile_idx):
                b_vecs = []
                n0_base = tile_n0 // c16 + wave_n * arith.index(reg_n)
                klane = lane // c16

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

            def _load_a_from_lds(lds_view):
                a_vecs = []
                for rk in range_constexpr(reg_k):
                    rk_vecs = []
                    col_base = arith.index(rk * WMMA_K) + base8
                    for rm in range_constexpr(reg_m):
                        row = (
                            wave_m * arith.index(reg_m * WMMA_M)
                            + arith.index(rm * WMMA_M)
                            + lane16
                        )
                        lds_idx = row * arith.index(BLOCK_K) + col_base
                        a_raw = vector.load_op(v8_in_ty, lds_view, [lds_idx])
                        rk_vecs.append(a_raw)
                    a_vecs.append(rk_vecs)
                return a_vecs

            def _do_compute(accs_in, b_vecs, a_vecs):
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

            # Prologue
            a_addrs_0 = _compute_a_load_addrs(arith.index(0))
            a_result_0 = _issue_a_loads(a_addrs_0)
            _llvm.inline_asm(
                res=None,
                operands_=[],
                asm_string="s_wait_loadcnt 0x0",
                constraints="",
                has_side_effects=True,
            )
            _store_a_to_lds(a_result_0, lds_a_view)
            _llvm.inline_asm(
                res=None,
                operands_=[],
                asm_string="s_wait_dscnt 0x0\ns_barrier_signal -1\ns_barrier_wait -1",
                constraints="",
                has_side_effects=True,
            )

            if num_k_tiles == 1:
                b_vecs = _load_b_tile(arith.index(0))
                a_vecs = _load_a_from_lds(lds_a_view)
                accs = _do_compute(accs, b_vecs, a_vecs)
            else:
                for kt in range(num_k_tiles - 1):
                    k_tile_idx = kt
                    b_vecs = _load_b_tile(k_tile_idx)
                    a_vecs = _load_a_from_lds(lds_a_view)
                    k_base_next = (kt + arith.index(1)) * arith.index(BLOCK_K)
                    a_addrs_next = _compute_a_load_addrs(k_base_next)
                    a_result_next = _issue_a_loads(a_addrs_next)
                    accs = _do_compute(accs, b_vecs, a_vecs)
                    _llvm.inline_asm(
                        res=None,
                        operands_=[],
                        asm_string="s_wait_dscnt 0x0\ns_barrier_signal -1\ns_barrier_wait -1",
                        constraints="",
                        has_side_effects=True,
                    )
                    _llvm.inline_asm(
                        res=None,
                        operands_=[],
                        asm_string="s_wait_loadcnt 0x0",
                        constraints="",
                        has_side_effects=True,
                    )
                    _store_a_to_lds(a_result_next, lds_a_view)
                    _llvm.inline_asm(
                        res=None,
                        operands_=[],
                        asm_string="s_wait_dscnt 0x0\ns_barrier_signal -1\ns_barrier_wait -1",
                        constraints="",
                        has_side_effects=True,
                    )

                b_vecs = _load_b_tile(arith.index(num_k_tiles - 1))
                a_vecs = _load_a_from_lds(lds_a_view)
                accs = _do_compute(accs, b_vecs, a_vecs)

            # ========== Store results ==========
            # Each wave iterates over its reg_n columns. For each column (rn):
            #   1. Barrier (reuse LDS)
            #   2. All waves write their WMMA tiles for this rn to LDS
            #      LDS layout: [BLOCK_M][16] f32, where BLOCK_M is partitioned
            #      by wave_m: rows [wave_m*reg_m*16 .. (wave_m+1)*reg_m*16)
            #      Each wave_n writes to the SAME rows but at its own lane16 col.
            #      WAIT: all wave_n values write to the same lane16 positions!
            #      They overlap because lane16 is per-lane, not per-wave.
            #
            # LDS transpose won't work without conditionals. Use scalar stores.
            c_layout_n = arith.index(N)
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
            A: lambda: Textra.memref(S, S, _in_elem_ty()),
            B_shuf: lambda: Textra.memref(S, _in_elem_ty()),
            C: lambda: Textra.memref(S, S, _out_elem_ty()),
        ):
            c1 = arith.index(1)
            total_blocks = arith.index(grid_m * grid_n)
            bk = arith.index(THREADS_PER_BLOCK)
            flir.gpu_ext.LaunchFuncOp(
                ["wmma_gemm_v5", "wmma_gemm_v5_kernel"],
                grid_size=(total_blocks, c1, c1),
                block_size=(bk, c1, c1),
                kernel_operands=[A, B_shuf, C],
            )

    return _WmmaGemmV5(), BLOCK_M, BLOCK_N, BLOCK_K
