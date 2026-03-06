#!/usr/bin/env python3
"""WMMA GEMM kernel v10 for RDNA4 (gfx12xx, wave32).

rocBLAS-style architecture:
  - Both A and B through LDS, double-buffered with XOR swap
  - Two-phase loop: each iteration splits into 2 halves (rk=0, rk=1)
  - GMEM→LDS stores for next tile interleaved BETWEEN the two WMMA groups
  - This allows GMEM loads to overlap with WMMA compute of the current tile
  - LDS padding to avoid bank conflicts

Key difference from v9: instead of doing all GMEM loads first, then all compute,
then all stores, v10 interleaves the stores between the two WMMA groups.
This mirrors rocBLAS's approach of ds_store+buffer_load pairs between WMMA halves.

Data flow:
  A[M,K] row-major -> buffer_load_b128 -> ds_store_b128 -> LDS_A[M][K_padded]
  B_T[N,K]         -> buffer_load_b128 -> ds_store_b128 -> LDS_B[N][K_padded]

Loop structure per iteration:
  Phase 1:
    - barrier + wait
    - LDS read A[rk=0], B[rk=0]
    - 16 WMMAs (all rm x rn for rk=0)

  Phase 2 (interleaved data movement):
    - LDS read A[rk=1], B[rk=1]
    - Issue GMEM loads for next tile A[kt+1], B[kt+1]
    - Store current GMEM loads → LDS next buffer (interleaved with loads)
    - XOR buffer swap
    - 16 WMMAs (all rm x rn for rk=1)
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


def create_wmma_gemm_v10_module(
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
    group_m=4,
    a_k_pad=8,
    b_k_pad=8,
):
    """Create WMMA GEMM v10 module.

    Both A and B through LDS, double-buffered, with GMEM→LDS stores
    interleaved between WMMA groups within each K-loop iteration.
    """
    BLOCK_M = WMMA_M * reg_m * waves_m  # 128
    BLOCK_N = WMMA_N * reg_n * waves_n  # 128
    BLOCK_K = WMMA_K * reg_k  # 32
    NUM_WAVES = waves_m * waves_n  # 4
    WAVE_SIZE = 32
    THREADS_PER_BLOCK = NUM_WAVES * WAVE_SIZE  # 128

    # A loading
    A_TILE_ELEMS = BLOCK_M * BLOCK_K  # 4096
    A_LOAD_VEC = 8  # 128 bits
    NUM_A_LOADS = A_TILE_ELEMS // (THREADS_PER_BLOCK * A_LOAD_VEC)  # 4

    # B loading
    B_TILE_ELEMS = BLOCK_K * BLOCK_N  # 4096
    B_LOAD_VEC = 8
    NUM_B_LOADS = B_TILE_ELEMS // (THREADS_PER_BLOCK * B_LOAD_VEC)  # 4

    # LDS layout with padding
    BLOCK_K_PAD_A = BLOCK_K + a_k_pad
    BLOCK_K_PAD_B = BLOCK_K + b_k_pad

    LDS_A_ELEMS = BLOCK_M * BLOCK_K_PAD_A  # 128*40 = 5120
    LDS_B_ELEMS = BLOCK_N * BLOCK_K_PAD_B  # 128*40 = 5120
    LDS_BUF_ELEMS = LDS_A_ELEMS + LDS_B_ELEMS  # 10240
    LDS_TOTAL_ELEMS = 2 * LDS_BUF_ELEMS  # 20480 (double-buffered)

    gpu_arch = get_rocm_arch()
    S = ir.ShapedType.get_dynamic_size()

    assert M % BLOCK_M == 0
    assert N % BLOCK_N == 0
    assert K % BLOCK_K == 0
    assert reg_k == 2, "v10 requires reg_k=2 for two-phase loop"

    num_k_tiles = K // BLOCK_K
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

    class _WmmaGemmV10(flir.MlirModule):
        GPU_MODULE_NAME = "wmma_gemm_v10"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}">']

        def init_gpu_module(self):
            _state["lds"] = allocator.allocate_array(_in_elem_ty(), LDS_TOTAL_ELEMS)
            allocator.finalize()

        @flir.kernel
        def wmma_gemm_v10_kernel(
            self: flir.T.i64,
            A: lambda: Textra.memref(S, S, _in_elem_ty()),
            B_T: lambda: Textra.memref(S, S, _in_elem_ty()),
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
            lds_view = _state["lds"](lds_base).get()

            a_rsrc = buffer_ops.create_buffer_resource(_unwrap(A), max_size=True)
            b_rsrc = buffer_ops.create_buffer_resource(_unwrap(B_T), max_size=True)
            c_rsrc = buffer_ops.create_buffer_resource(_unwrap(C), max_size=True)

            # ========================================
            # GMEM load functions
            # ========================================

            def _gmem_load_a(k_base):
                """Load A[BLOCK_M, BLOCK_K] from GMEM into registers."""
                a_regs = []
                for al in range_constexpr(NUM_A_LOADS):
                    lin = tid * arith.index(A_LOAD_VEC) + arith.index(
                        al * THREADS_PER_BLOCK * A_LOAD_VEC
                    )
                    a_row = lin // arith.index(BLOCK_K)
                    a_col = lin % arith.index(BLOCK_K)
                    g_row = tile_m0 + a_row
                    g_col = k_base + a_col
                    elem_off = g_row * arith.index(K) + g_col
                    f32_off = elem_off // c2
                    a_raw = buffer_ops.buffer_load(
                        a_rsrc, f32_off, vec_width=4, dtype=ir.F32Type.get()
                    )
                    a_vec = vector.bitcast(v8_in_ty, a_raw)
                    a_regs.append(a_vec)
                return a_regs

            def _gmem_load_b(k_base):
                """Load B_T[BLOCK_N, BLOCK_K] from GMEM into registers."""
                b_regs = []
                for bl in range_constexpr(NUM_B_LOADS):
                    lin = tid * arith.index(B_LOAD_VEC) + arith.index(
                        bl * THREADS_PER_BLOCK * B_LOAD_VEC
                    )
                    b_n = lin // arith.index(BLOCK_K)
                    b_k = lin % arith.index(BLOCK_K)
                    g_n = tile_n0 + b_n
                    g_k = k_base + b_k
                    elem_off = g_n * arith.index(K) + g_k
                    f32_off = elem_off // c2
                    b_raw = buffer_ops.buffer_load(
                        b_rsrc, f32_off, vec_width=4, dtype=ir.F32Type.get()
                    )
                    b_vec = vector.bitcast(v8_in_ty, b_raw)
                    b_regs.append(b_vec)
                return b_regs

            # ========================================
            # LDS store functions
            # ========================================

            def _store_a_to_lds(a_regs, lds_buf_off):
                for al in range_constexpr(NUM_A_LOADS):
                    lin = tid * arith.index(A_LOAD_VEC) + arith.index(
                        al * THREADS_PER_BLOCK * A_LOAD_VEC
                    )
                    a_row = lin // arith.index(BLOCK_K)
                    a_col = lin % arith.index(BLOCK_K)
                    lds_idx = lds_buf_off + a_row * arith.index(BLOCK_K_PAD_A) + a_col
                    vector.store(a_regs[al], lds_view, [lds_idx])

            def _store_b_to_lds(b_regs, lds_buf_off):
                for bl in range_constexpr(NUM_B_LOADS):
                    lin = tid * arith.index(B_LOAD_VEC) + arith.index(
                        bl * THREADS_PER_BLOCK * B_LOAD_VEC
                    )
                    b_n = lin // arith.index(BLOCK_K)
                    b_k = lin % arith.index(BLOCK_K)
                    lds_b_base = lds_buf_off + arith.index(LDS_A_ELEMS)
                    lds_idx = lds_b_base + b_n * arith.index(BLOCK_K_PAD_B) + b_k
                    vector.store(b_regs[bl], lds_view, [lds_idx])

            def _store_all_to_lds(a_regs, b_regs, lds_buf_off):
                """Store both A and B to LDS, interleaved for better pipelining."""
                # Interleave A and B stores to allow cascaded s_wait_loadcnt
                for i in range_constexpr(max(NUM_A_LOADS, NUM_B_LOADS)):
                    if i < NUM_A_LOADS:
                        lin = tid * arith.index(A_LOAD_VEC) + arith.index(
                            i * THREADS_PER_BLOCK * A_LOAD_VEC
                        )
                        a_row = lin // arith.index(BLOCK_K)
                        a_col = lin % arith.index(BLOCK_K)
                        lds_idx = (
                            lds_buf_off + a_row * arith.index(BLOCK_K_PAD_A) + a_col
                        )
                        vector.store(a_regs[i], lds_view, [lds_idx])
                    if i < NUM_B_LOADS:
                        lin = tid * arith.index(B_LOAD_VEC) + arith.index(
                            i * THREADS_PER_BLOCK * B_LOAD_VEC
                        )
                        b_n = lin // arith.index(BLOCK_K)
                        b_k = lin % arith.index(BLOCK_K)
                        lds_b_base = lds_buf_off + arith.index(LDS_A_ELEMS)
                        lds_idx = lds_b_base + b_n * arith.index(BLOCK_K_PAD_B) + b_k
                        vector.store(b_regs[i], lds_view, [lds_idx])

            def _barrier():
                _llvm.inline_asm(
                    res=None,
                    operands_=[],
                    asm_string="s_wait_dscnt 0x0\ns_barrier_signal -1\ns_barrier_wait -1",
                    constraints="",
                    has_side_effects=True,
                )

            def _wait_vmem():
                _llvm.inline_asm(
                    res=None,
                    operands_=[],
                    asm_string="s_wait_loadcnt 0x0",
                    constraints="",
                    has_side_effects=True,
                )

            # ========================================
            # LDS read functions
            # ========================================

            def _load_a_from_lds_rk(lds_buf_off, rk):
                """Load A operands for a single rk from LDS."""
                rk_vecs = []
                col_base = arith.index(rk * WMMA_K) + base8
                for rm in range_constexpr(reg_m):
                    row = (
                        wave_m * arith.index(reg_m * WMMA_M)
                        + arith.index(rm * WMMA_M)
                        + lane16
                    )
                    lds_idx = lds_buf_off + row * arith.index(BLOCK_K_PAD_A) + col_base
                    a_raw = vector.load_op(v8_in_ty, lds_view, [lds_idx])
                    rk_vecs.append(a_raw)
                return rk_vecs

            def _load_b_from_lds_rk(lds_buf_off, rk):
                """Load B operands for a single rk from LDS."""
                rk_vecs = []
                lds_b_base = lds_buf_off + arith.index(LDS_A_ELEMS)
                k_base = arith.index(rk * WMMA_K) + base8
                for rn in range_constexpr(reg_n):
                    n_row = (
                        wave_n * arith.index(reg_n * WMMA_N)
                        + arith.index(rn * WMMA_N)
                        + lane16
                    )
                    lds_idx = lds_b_base + n_row * arith.index(BLOCK_K_PAD_B) + k_base
                    b_raw = vector.load_op(v8_in_ty, lds_view, [lds_idx])
                    rk_vecs.append(b_raw)
                return rk_vecs

            # ========================================
            # Compute WMMAs for one rk
            # ========================================
            def _do_compute_rk(accs_in, a_rk_vecs, b_rk_vecs):
                """16 WMMAs for a single rk value (all rm x rn)."""
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

            # Initialize accumulators
            zero_acc = arith.constant_vector(0.0, v8f32_ty)
            accs = [zero_acc for _ in range_constexpr(reg_m * reg_n)]

            # LDS buffer offsets
            lds_buf0_off = arith.index(0)
            lds_buf1_off = arith.index(LDS_BUF_ELEMS)

            # ============================================================
            # SOFTWARE-PIPELINED K-LOOP (rocBLAS-style two-phase)
            # ============================================================
            #
            # Prologue:
            #   GMEM load tile 0 → regs → wait → store LDS buf0 → barrier
            #
            # Main loop (kt=0..N-2):
            #   Phase 1 (rk=0):
            #     - LDS read A[rk=0], B[rk=0] from buf_cur
            #     - 16 WMMAs
            #   Interleaved data movement:
            #     - LDS read A[rk=1], B[rk=1] from buf_cur
            #     - GMEM load A[kt+1], B[kt+1] → regs (async)
            #     - Store GMEM regs → LDS buf_next (compiler cascades waits)
            #     - XOR buffer swap
            #   Phase 2 (rk=1):
            #     - 16 WMMAs (using already-loaded rk=1 data)
            #   Barrier
            #
            # Epilogue: LDS read + compute last tile

            # PROLOGUE
            a_gmem = _gmem_load_a(arith.index(0))
            b_gmem = _gmem_load_b(arith.index(0))
            _wait_vmem()
            _store_a_to_lds(a_gmem, lds_buf0_off)
            _store_b_to_lds(b_gmem, lds_buf0_off)
            _barrier()

            lds_cur_off = lds_buf0_off
            lds_next_off = lds_buf1_off

            for kt in range(num_k_tiles - 1):
                # Phase 1: LDS read rk=0 + 16 WMMAs
                a_rk0 = _load_a_from_lds_rk(lds_cur_off, 0)
                b_rk0 = _load_b_from_lds_rk(lds_cur_off, 0)
                accs = _do_compute_rk(accs, a_rk0, b_rk0)

                # Interleaved: read rk=1 from LDS, then kick off next GMEM loads + stores
                a_rk1 = _load_a_from_lds_rk(lds_cur_off, 1)
                b_rk1 = _load_b_from_lds_rk(lds_cur_off, 1)

                # Issue GMEM loads for next tile (async)
                k_base_next = (kt + arith.index(1)) * arith.index(BLOCK_K)
                a_gmem = _gmem_load_a(k_base_next)
                b_gmem = _gmem_load_b(k_base_next)

                # Store GMEM data → LDS buf_next
                # (compiler inserts cascaded s_wait_loadcnt before each ds_store)
                _store_all_to_lds(a_gmem, b_gmem, lds_next_off)

                # Phase 2: 16 WMMAs for rk=1 (data already in registers)
                accs = _do_compute_rk(accs, a_rk1, b_rk1)

                # Barrier + swap
                _barrier()
                lds_cur_off, lds_next_off = lds_next_off, lds_cur_off

            # EPILOGUE
            a_rk0 = _load_a_from_lds_rk(lds_cur_off, 0)
            b_rk0 = _load_b_from_lds_rk(lds_cur_off, 0)
            accs = _do_compute_rk(accs, a_rk0, b_rk0)

            a_rk1 = _load_a_from_lds_rk(lds_cur_off, 1)
            b_rk1 = _load_b_from_lds_rk(lds_cur_off, 1)
            accs = _do_compute_rk(accs, a_rk1, b_rk1)

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
            B_T: lambda: Textra.memref(S, S, _in_elem_ty()),
            C: lambda: Textra.memref(S, S, _out_elem_ty()),
        ):
            c1 = arith.index(1)
            total_blocks = arith.index(grid_m * grid_n)
            bk = arith.index(THREADS_PER_BLOCK)
            flir.gpu_ext.LaunchFuncOp(
                ["wmma_gemm_v10", "wmma_gemm_v10_kernel"],
                grid_size=(total_blocks, c1, c1),
                block_size=(bk, c1, c1),
                kernel_operands=[A, B_T, C],
            )

    return _WmmaGemmV10(), BLOCK_M, BLOCK_N, BLOCK_K
