#!/usr/bin/env python3
"""WMMA GEMM kernel v17 for RDNA4 (gfx12xx, wave32).

True double-buffered LDS A with NO explicit wait instructions.
Relies on SSA data dependencies for natural scheduling.

Architecture:
  - A goes through LDS with double buffering (ping-pong)
  - B is pre-shuffled and loaded directly from GMEM
  - NO explicit _wait_vmem() calls — the compiler inserts waits naturally
    based on data flow: A GMEM load values → ds_store, B GMEM load values → WMMA
  - This allows the compiler to optimally interleave loads, WMMAs, and LDS stores

Computes C[M,N] = A[M,K] @ B_shuffled
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
import _mlir.extras.types as Textra


WMMA_M = 16
WMMA_N = 16
WMMA_K = 16


def _unwrap(v):
    while hasattr(v, "_value"):
        v = v._value
    return v


def preshuffle_b_wmma(B_kn):
    import torch

    K, N = B_kn.shape
    assert K % 16 == 0 and N % 16 == 0
    B_reshaped = B_kn.reshape(K // 16, 2, 8, N // 16, 16)
    return B_reshaped.permute(3, 0, 1, 4, 2).contiguous()


def create_wmma_gemm_v17_module(
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
    a_k_pad=8,
):
    BLOCK_M = WMMA_M * reg_m * waves_m
    BLOCK_N = WMMA_N * reg_n * waves_n
    BLOCK_K = WMMA_K * reg_k
    NUM_WAVES = waves_m * waves_n
    WAVE_SIZE = 32
    THREADS_PER_BLOCK = NUM_WAVES * WAVE_SIZE

    assert reg_k == 2

    A_TILE_ELEMS = BLOCK_M * BLOCK_K
    A_LOAD_VEC = 8
    NUM_A_LOADS = A_TILE_ELEMS // (THREADS_PER_BLOCK * A_LOAD_VEC)

    BLOCK_K_PAD_A = BLOCK_K + a_k_pad
    LDS_A_SINGLE = BLOCK_M * BLOCK_K_PAD_A
    LDS_A_TOTAL = LDS_A_SINGLE * 2

    N0_total = N // 16
    K0_total = K // 16
    B_KPACK = 8
    B_STRIDE_NLANE = B_KPACK
    B_STRIDE_KLANE = 16 * B_KPACK
    B_STRIDE_K0 = 2 * 16 * B_KPACK
    B_STRIDE_N0 = K0_total * B_STRIDE_K0

    gpu_arch = get_rocm_arch()
    S = ir.ShapedType.get_dynamic_size()

    assert M % BLOCK_M == 0
    assert N % BLOCK_N == 0
    assert K % BLOCK_K == 0

    num_k_tiles = K // BLOCK_K
    assert num_k_tiles % 2 == 0
    outer_k_iters = num_k_tiles // 2

    grid_m = M // BLOCK_M
    grid_n = N // BLOCK_N
    is_bf16 = in_dtype == "bf16"

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

    class _WmmaGemmV17(flir.MlirModule):
        GPU_MODULE_NAME = "wmma_gemm_v17"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}">']

        def init_gpu_module(self):
            _state["lds_a"] = allocator.allocate_array(_in_elem_ty(), LDS_A_TOTAL)
            allocator.finalize()

        @flir.kernel
        def wmma_gemm_v17_kernel(
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

            tid = flir.thread_idx("x")
            pid = flir.block_idx("x")

            c32 = arith.index(32)
            c16 = arith.index(16)
            c8 = arith.index(8)
            c2 = arith.index(2)
            wave_id = tid // c32
            lane = tid % c32
            lane16 = lane % c16
            klane = lane // c16
            base8 = klane * c8

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

            n0_base = tile_n0 // c16 + wave_n * arith.index(reg_n)

            def _gmem_load_a(k_base):
                a_regs = []
                for al in range_constexpr(NUM_A_LOADS):
                    a_lin = tid * arith.index(A_LOAD_VEC) + arith.index(
                        al * THREADS_PER_BLOCK * A_LOAD_VEC
                    )
                    a_load_row = a_lin // arith.index(BLOCK_K)
                    a_load_col = a_lin % arith.index(BLOCK_K)
                    g_row = tile_m0 + a_load_row
                    g_col = k_base + a_load_col
                    elem_off = g_row * arith.index(K) + g_col
                    f32_off = elem_off // c2
                    a_raw = buffer_ops.buffer_load(
                        a_rsrc, f32_off, vec_width=4, dtype=ir.F32Type.get()
                    )
                    a_vec = vector.bitcast(v8_in_ty, a_raw)
                    a_regs.append(a_vec)
                return a_regs

            def _store_a_to_lds(a_regs, lds_offset):
                for al in range_constexpr(NUM_A_LOADS):
                    a_lin = tid * arith.index(A_LOAD_VEC) + arith.index(
                        al * THREADS_PER_BLOCK * A_LOAD_VEC
                    )
                    a_load_row = a_lin // arith.index(BLOCK_K)
                    a_load_col = a_lin % arith.index(BLOCK_K)
                    lds_idx = (
                        lds_offset
                        + a_load_row * arith.index(BLOCK_K_PAD_A)
                        + a_load_col
                    )
                    vector.store(a_regs[al], lds_a_view, [lds_idx])

            def _load_a_from_lds_rk(rk, lds_offset):
                rk_vecs = []
                col_base = arith.index(rk * WMMA_K) + base8
                for rm in range_constexpr(reg_m):
                    row = (
                        wave_m * arith.index(reg_m * WMMA_M)
                        + arith.index(rm * WMMA_M)
                        + lane16
                    )
                    lds_idx = lds_offset + row * arith.index(BLOCK_K_PAD_A) + col_base
                    a_raw = vector.load_op(v8_in_ty, lds_a_view, [lds_idx])
                    rk_vecs.append(a_raw)
                return rk_vecs

            def _load_b_tile(k_tile_idx):
                b_vecs = []
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

            def _barrier():
                _llvm.inline_asm(
                    res=None,
                    operands_=[],
                    asm_string="s_wait_dscnt 0x0\ns_wait_storecnt 0x0\ns_barrier_signal -1\ns_barrier_wait -1",
                    constraints="",
                    has_side_effects=True,
                )

            def _do_compute_rk(accs_in, a_rk_vecs, b_rk_vecs):
                new_accs = list(accs_in)
                for rm in range_constexpr(reg_m):
                    for rn in range_constexpr(reg_n):
                        idx = rm * reg_n + rn
                        new_accs[idx] = _wmma_op(
                            v8f32_ty,
                            a_rk_vecs[rm],
                            b_rk_vecs[rn],
                            new_accs[idx],
                            v8i16_ty,
                        )
                return new_accs

            zero_acc = arith.constant_vector(0.0, v8f32_ty)
            accs = [zero_acc for _ in range_constexpr(reg_m * reg_n)]

            READ_BUFS = [0, LDS_A_SINGLE]
            WRITE_BUFS = [LDS_A_SINGLE, 0]

            # ============================================================
            # PROLOGUE: load A[0] → buf0 (NO explicit wait — let compiler handle)
            # ============================================================
            a_gmem_0 = _gmem_load_a(arith.index(0))
            # Store directly — the vector.store depends on the buffer_load results,
            # so the compiler will insert s_wait_loadcnt before the ds_store
            _store_a_to_lds(a_gmem_0, arith.index(READ_BUFS[0]))
            _barrier()

            # ============================================================
            # MAIN LOOP with true double buffering
            # ============================================================
            for kt_outer in range(outer_k_iters - 1):
                for j in range_constexpr(2):
                    read_buf = arith.index(READ_BUFS[j])
                    write_buf = arith.index(WRITE_BUFS[j])
                    kt = kt_outer * arith.index(2) + arith.index(j)

                    # 1. Read A from LDS (current buffer)
                    a_rk0 = _load_a_from_lds_rk(0, read_buf)
                    a_rk1 = _load_a_from_lds_rk(1, read_buf)

                    # 2. Load B from GMEM
                    b_vecs = _load_b_tile(kt)

                    # 3. Load A_next from GMEM (for next iteration)
                    next_k_base = (kt + arith.index(1)) * arith.index(BLOCK_K)
                    a_gmem_next = _gmem_load_a(next_k_base)

                    # 4. Compute 32 WMMAs
                    # These use a_rk0/a_rk1 (from LDS) and b_vecs (from GMEM)
                    # The compiler must wait for B loads before WMMAs
                    # but A_next loads can overlap with WMMAs
                    accs = _do_compute_rk(accs, a_rk0, b_vecs[0])
                    accs = _do_compute_rk(accs, a_rk1, b_vecs[1])

                    # 5. Store A_next to LDS (other buffer)
                    # This depends on a_gmem_next values → compiler inserts wait
                    _store_a_to_lds(a_gmem_next, write_buf)
                    _barrier()

            # EPILOGUE: last 2 tiles
            kt_second_last = arith.index((outer_k_iters - 1) * 2)
            a_rk0 = _load_a_from_lds_rk(0, arith.index(READ_BUFS[0]))
            a_rk1 = _load_a_from_lds_rk(1, arith.index(READ_BUFS[0]))
            b_vecs = _load_b_tile(kt_second_last)
            next_k_base = (kt_second_last + arith.index(1)) * arith.index(BLOCK_K)
            a_gmem_last = _gmem_load_a(next_k_base)
            accs = _do_compute_rk(accs, a_rk0, b_vecs[0])
            accs = _do_compute_rk(accs, a_rk1, b_vecs[1])
            _store_a_to_lds(a_gmem_last, arith.index(WRITE_BUFS[0]))
            _barrier()

            kt_last = arith.index((outer_k_iters - 1) * 2 + 1)
            a_rk0 = _load_a_from_lds_rk(0, arith.index(READ_BUFS[1]))
            a_rk1 = _load_a_from_lds_rk(1, arith.index(READ_BUFS[1]))
            b_vecs = _load_b_tile(kt_last)
            accs = _do_compute_rk(accs, a_rk0, b_vecs[0])
            accs = _do_compute_rk(accs, a_rk1, b_vecs[1])

            # ========== Store results ==========
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
                            accs[idx], static_position=[si], dynamic_position=[]
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
                ["wmma_gemm_v17", "wmma_gemm_v17_kernel"],
                grid_size=(total_blocks, c1, c1),
                block_size=(bk, c1, c1),
                kernel_operands=[A, B_shuf, C],
            )

    return _WmmaGemmV17(), BLOCK_M, BLOCK_N, BLOCK_K
