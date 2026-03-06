#!/usr/bin/env python3
"""WMMA GEMM kernel v23 for RDNA4 (gfx12xx, wave32).

Exact rocBLAS structure: Both A+B through LDS, double-buffered.

Per iteration:
  HALF 1: ds_load A,B from current LDS buf → compute 16 WMMAs
  HALF 2: ds_load A,B from current LDS buf (next rk offset)
           + GMEM→LDS transfer for NEXT iteration (interleaved)
           → compute 16 WMMAs → barrier → swap buffers

Key difference from v18: GMEM→LDS transfers happen BETWEEN halves,
not as a separate phase after compute. This lets the LLVM scheduler
interleave ds_stores with WMMAs.

Architecture:
- Both A and B loaded from GMEM → stored to LDS → read from LDS for WMMA
- A uses v_perm-style repacking? No, we use natural LDS layout.
- Double-buffered LDS via address XOR (like rocBLAS)
- B stored as B_T[N,K] (K contiguous) in GMEM, then to LDS

Computes C[M,N] = A[M,K] @ B_T[N,K]^T
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


def create_wmma_gemm_v23_module(
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
    b_k_pad=8,
):
    BLOCK_M = WMMA_M * reg_m * waves_m  # 128
    BLOCK_N = WMMA_N * reg_n * waves_n  # 128
    BLOCK_K = WMMA_K * reg_k  # 32
    NUM_WAVES = waves_m * waves_n  # 4
    WAVE_SIZE = 32
    THREADS_PER_BLOCK = NUM_WAVES * WAVE_SIZE  # 128

    assert reg_k == 2

    # Loading: each thread loads 8 bf16 elements per load
    A_LOAD_VEC = 8
    A_TILE_ELEMS = BLOCK_M * BLOCK_K  # 128*32 = 4096
    NUM_A_LOADS = A_TILE_ELEMS // (THREADS_PER_BLOCK * A_LOAD_VEC)  # 4

    B_LOAD_VEC = 8
    B_TILE_ELEMS = BLOCK_N * BLOCK_K  # 128*32 = 4096
    NUM_B_LOADS = B_TILE_ELEMS // (THREADS_PER_BLOCK * B_LOAD_VEC)  # 4

    BLOCK_K_PAD_A = BLOCK_K + a_k_pad  # 40
    BLOCK_K_PAD_B = BLOCK_K + b_k_pad  # 40
    LDS_A_SINGLE = BLOCK_M * BLOCK_K_PAD_A  # 128*40 = 5120
    LDS_B_SINGLE = BLOCK_N * BLOCK_K_PAD_B  # 128*40 = 5120
    LDS_SINGLE = LDS_A_SINGLE + LDS_B_SINGLE  # 10240
    LDS_TOTAL = LDS_SINGLE * 2  # 20480 elements = 40960 bytes

    gpu_arch = get_rocm_arch()
    S = ir.ShapedType.get_dynamic_size()

    assert M % BLOCK_M == 0
    assert N % BLOCK_N == 0
    assert K % BLOCK_K == 0

    num_k_tiles = K // BLOCK_K
    assert num_k_tiles >= 2

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

    class _WmmaGemmV23(flir.MlirModule):
        GPU_MODULE_NAME = "wmma_gemm_v23"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}">']

        def init_gpu_module(self):
            _state["lds"] = allocator.allocate_array(_in_elem_ty(), LDS_TOTAL)
            allocator.finalize()

        @flir.kernel
        def wmma_gemm_v23_kernel(
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
            bt_rsrc = buffer_ops.create_buffer_resource(_unwrap(B_T), max_size=True)
            c_rsrc = buffer_ops.create_buffer_resource(_unwrap(C), max_size=True)

            # LDS buffer offsets
            BUF0_A = 0
            BUF0_B = LDS_A_SINGLE
            BUF1_A = LDS_SINGLE
            BUF1_B = LDS_SINGLE + LDS_A_SINGLE

            # ============================================================
            # GMEM load functions
            # ============================================================
            def _gmem_load_and_store_to_lds(k_base, lds_a_offset, lds_b_offset):
                """Load A+B from GMEM and store directly to LDS, interleaved.

                Each pair: buffer_load → ds_store, so each ds_store only waits
                for its own buffer_load (not all loads at once).
                """
                for al in range_constexpr(NUM_A_LOADS):
                    a_lin = tid * arith.index(A_LOAD_VEC) + arith.index(
                        al * THREADS_PER_BLOCK * A_LOAD_VEC
                    )
                    a_load_row = a_lin // arith.index(BLOCK_K)
                    a_load_col = a_lin % arith.index(BLOCK_K)
                    # GMEM load
                    g_row = tile_m0 + a_load_row
                    g_col = k_base + a_load_col
                    elem_off = g_row * arith.index(K) + g_col
                    f32_off = elem_off // c2
                    a_raw = buffer_ops.buffer_load(
                        a_rsrc, f32_off, vec_width=4, dtype=ir.F32Type.get()
                    )
                    a_vec = vector.bitcast(v8_in_ty, a_raw)
                    # LDS store
                    lds_idx = (
                        arith.index(lds_a_offset)
                        + a_load_row * arith.index(BLOCK_K_PAD_A)
                        + a_load_col
                    )
                    vector.store(a_vec, lds_view, [lds_idx])

                for bl in range_constexpr(NUM_B_LOADS):
                    b_lin = tid * arith.index(B_LOAD_VEC) + arith.index(
                        bl * THREADS_PER_BLOCK * B_LOAD_VEC
                    )
                    b_load_row = b_lin // arith.index(BLOCK_K)
                    b_load_col = b_lin % arith.index(BLOCK_K)
                    # GMEM load
                    g_row = tile_n0 + b_load_row
                    g_col = k_base + b_load_col
                    elem_off = g_row * arith.index(K) + g_col
                    f32_off = elem_off // c2
                    b_raw = buffer_ops.buffer_load(
                        bt_rsrc, f32_off, vec_width=4, dtype=ir.F32Type.get()
                    )
                    b_vec = vector.bitcast(v8_in_ty, b_raw)
                    # LDS store
                    lds_idx = (
                        arith.index(lds_b_offset)
                        + b_load_row * arith.index(BLOCK_K_PAD_B)
                        + b_load_col
                    )
                    vector.store(b_vec, lds_view, [lds_idx])

            def _load_a_from_lds_rk(rk, lds_a_offset):
                rk_vecs = []
                col_base = arith.index(rk * WMMA_K) + base8
                for rm in range_constexpr(reg_m):
                    row = (
                        wave_m * arith.index(reg_m * WMMA_M)
                        + arith.index(rm * WMMA_M)
                        + lane16
                    )
                    lds_idx = (
                        arith.index(lds_a_offset)
                        + row * arith.index(BLOCK_K_PAD_A)
                        + col_base
                    )
                    a_raw = vector.load_op(v8_in_ty, lds_view, [lds_idx])
                    rk_vecs.append(a_raw)
                return rk_vecs

            def _load_b_from_lds_rk(rk, lds_b_offset):
                rk_vecs = []
                col_base = arith.index(rk * WMMA_K) + base8
                for rn in range_constexpr(reg_n):
                    row = (
                        wave_n * arith.index(reg_n * WMMA_N)
                        + arith.index(rn * WMMA_N)
                        + lane16
                    )
                    lds_idx = (
                        arith.index(lds_b_offset)
                        + row * arith.index(BLOCK_K_PAD_B)
                        + col_base
                    )
                    b_raw = vector.load_op(v8_in_ty, lds_view, [lds_idx])
                    rk_vecs.append(b_raw)
                return rk_vecs

            def _barrier():
                _llvm.inline_asm(
                    res=None,
                    operands_=[],
                    asm_string="s_wait_dscnt 0x0\ns_wait_storecnt 0x0\ns_barrier_signal -1\ns_barrier_wait -1",
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

            # Initialize accumulators
            zero_acc = arith.constant_vector(0.0, v8f32_ty)
            accs = [zero_acc for _ in range_constexpr(reg_m * reg_n)]

            # ============================================================
            # MAIN STRUCTURE (rocBLAS-style)
            # ============================================================
            #
            # Prologue:
            #   1. GMEM load tile[0] → wait → ds_store buf[0] → barrier
            #   2. GMEM load tile[1] (in flight, for use in iteration 1)
            #
            # Main loop iteration i (for i in 0..num_k_tiles-3):
            #   HALF 1 (compute rk0):
            #     ds_load A,B rk0 from LDS buf[i%2]
            #     compute 16 WMMAs
            #
            #   MID-SECTION (GMEM→LDS transfer for NEXT iteration):
            #     wait for GMEM loads (from prev iter or prologue)
            #     ds_store to buf[(i+1)%2]
            #     barrier
            #     GMEM load tile[i+2] (new loads, won't be used until iter i+1)
            #
            #   HALF 2 (compute rk1):
            #     ds_load A,B rk1 from LDS buf[i%2]
            #     compute 16 WMMAs
            #
            # Penultimate (num_k_tiles-2):
            #   HALF 1 + MID (wait + store last GMEM + barrier) + HALF 2
            #
            # Last (num_k_tiles-1):
            #   HALF 1 + HALF 2 (no GMEM loads needed)

            # --- PROLOGUE: Load tile[0] → LDS buf[0] ---
            _gmem_load_and_store_to_lds(arith.index(0), BUF0_A, BUF0_B)
            _barrier()

            # --- MAIN LOOP ---
            # Each iteration: read from current buf → compute → load+store to other buf → barrier
            # This is a simple single-buffer-ahead pipeline:
            # - Iteration i reads from buf[i%2], computes, writes tile[i+1] to buf[(i+1)%2]

            main_loop_iters = (num_k_tiles - 1) // 2

            READ_A = [BUF0_A, BUF1_A]
            READ_B = [BUF0_B, BUF1_B]
            WRITE_A = [BUF1_A, BUF0_A]
            WRITE_B = [BUF1_B, BUF0_B]

            for kt_outer in range(main_loop_iters):
                for j in range_constexpr(2):
                    rd_a = READ_A[j]
                    rd_b = READ_B[j]
                    wr_a = WRITE_A[j]
                    wr_b = WRITE_B[j]

                    # Read from LDS and compute both rk halves
                    a_rk0 = _load_a_from_lds_rk(0, rd_a)
                    b_rk0 = _load_b_from_lds_rk(0, rd_b)
                    accs = _do_compute_rk(accs, a_rk0, b_rk0)

                    a_rk1 = _load_a_from_lds_rk(1, rd_a)
                    b_rk1 = _load_b_from_lds_rk(1, rd_b)
                    accs = _do_compute_rk(accs, a_rk1, b_rk1)

                    # Load next tile from GMEM → store to other LDS buffer
                    next_k = (
                        kt_outer * arith.index(2) + arith.index(j + 1)
                    ) * arith.index(BLOCK_K)
                    _gmem_load_and_store_to_lds(next_k, wr_a, wr_b)
                    _barrier()

            # --- LAST TILE (if odd number of tiles after prologue) ---
            remainder = (num_k_tiles - 1) % 2
            if remainder > 0:
                last_rd_a = READ_A[0]  # After even main loop iters, we read from buf[0]
                last_rd_b = READ_B[0]
                a_rk0 = _load_a_from_lds_rk(0, last_rd_a)
                b_rk0 = _load_b_from_lds_rk(0, last_rd_b)
                accs = _do_compute_rk(accs, a_rk0, b_rk0)
                a_rk1 = _load_a_from_lds_rk(1, last_rd_a)
                b_rk1 = _load_b_from_lds_rk(1, last_rd_b)
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
                ["wmma_gemm_v23", "wmma_gemm_v23_kernel"],
                grid_size=(total_blocks, c1, c1),
                block_size=(bk, c1, c1),
                kernel_operands=[A, B_T, C],
            )

    return _WmmaGemmV23(), BLOCK_M, BLOCK_N, BLOCK_K
