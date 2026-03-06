#!/usr/bin/env python3
"""WMMA GEMM kernel v9 for RDNA4 (gfx12xx, wave32).

Architecture modeled after rocBLAS Tensile MT128x128x32:
  - Both A and B loaded from GMEM into LDS (double-buffered)
  - A read from LDS as contiguous 8-bf16 vectors (natural WMMA A operand layout)
  - B_T[N,K] is pre-transposed: K is contiguous in GMEM, matching LDS_B[N][K] layout
  - Both A and B LDS reads are contiguous 8-element vector loads
  - Software-pipelined K-loop: GMEM loads for kt+1 overlap with LDS reads + compute for kt
  - Each loop iteration does 32 WMMAs (reg_m=4, reg_n=4, reg_k=2)

Data flow per K-tile:
  A[M,K] row-major -> buffer_load_b128 -> ds_store_b128 -> LDS_A[BLOCK_M][BLOCK_K]
                                                           -> vector.load 8xbf16 -> WMMA A operand
  B_T[N,K] (transposed) -> buffer_load_b128 -> ds_store_b128 -> LDS_B[BLOCK_N][BLOCK_K]
                                                                -> vector.load 8xbf16 -> WMMA B operand

Computes C[M,N] = A[M,K] @ B_T[N,K]^T  (B_T = B.T.contiguous())
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
    """Unwrap ArithValue to raw MLIR Value."""
    while hasattr(v, "_value"):
        v = v._value
    return v


def create_wmma_gemm_v9_module(
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
    """Create WMMA GEMM v9 module.

    Both A and B through LDS, double-buffered, SW-pipelined K-loop.
    Takes row-major A[M,K] and pre-transposed B_T[N,K] (B_T = B.T.contiguous()).
    Computes C = A @ B_T.T
    """
    BLOCK_M = WMMA_M * reg_m * waves_m  # 128
    BLOCK_N = WMMA_N * reg_n * waves_n  # 128
    BLOCK_K = WMMA_K * reg_k  # 32
    NUM_WAVES = waves_m * waves_n  # 4
    WAVE_SIZE = 32
    THREADS_PER_BLOCK = NUM_WAVES * WAVE_SIZE  # 128

    # A loading: each thread loads some 128-bit chunks from A[M,K]
    A_TILE_ELEMS = BLOCK_M * BLOCK_K  # 128*32 = 4096 bf16
    A_LOAD_VEC = 8  # 8 bf16 = 128 bits
    NUM_A_LOADS = A_TILE_ELEMS // (THREADS_PER_BLOCK * A_LOAD_VEC)  # 4096/(128*8)=4

    # B loading: each thread loads some chunks from B[K,N]
    B_TILE_ELEMS = BLOCK_K * BLOCK_N  # 32*128 = 4096 bf16
    B_LOAD_VEC = 8  # 8 bf16 = 128 bits
    NUM_B_LOADS = B_TILE_ELEMS // (THREADS_PER_BLOCK * B_LOAD_VEC)  # 4096/(128*8)=4

    # LDS layout:
    # A: stored as [BLOCK_M][BLOCK_K_PAD_A] bf16, row-major in LDS
    #    with padding to reduce bank conflicts
    # B: stored TRANSPOSED as [BLOCK_N][BLOCK_K_PAD] bf16 in LDS
    #    with padding to avoid bank conflicts during vectorized reads
    A_K_PAD = a_k_pad
    BLOCK_K_PAD_A = BLOCK_K + A_K_PAD
    B_K_PAD = b_k_pad
    BLOCK_K_PAD = BLOCK_K + B_K_PAD

    LDS_A_ELEMS = BLOCK_M * BLOCK_K_PAD_A  # 128*40 = 5120
    LDS_B_ELEMS = BLOCK_N * BLOCK_K_PAD  # 128*40 = 5120
    LDS_BUF_ELEMS = LDS_A_ELEMS + LDS_B_ELEMS  # 9216
    LDS_TOTAL_ELEMS = 2 * LDS_BUF_ELEMS  # 18432 (double-buffered)
    # Total bytes: 18432 * 2 = 36864 bytes = 36KB (fits in 64KB LDS)

    gpu_arch = get_rocm_arch()
    S = ir.ShapedType.get_dynamic_size()

    assert M % BLOCK_M == 0
    assert N % BLOCK_N == 0
    assert K % BLOCK_K == 0

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

    class _WmmaGemmV9(flir.MlirModule):
        GPU_MODULE_NAME = "wmma_gemm_v9"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}">']

        def init_gpu_module(self):
            _state["lds"] = allocator.allocate_array(_in_elem_ty(), LDS_TOTAL_ELEMS)
            allocator.finalize()

        @flir.kernel
        def wmma_gemm_v9_kernel(
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
            klane = lane // c16  # 0 or 1
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
            # GMEM load functions (load to registers only, no LDS store)
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
            # Registers -> LDS store functions
            # ========================================

            def _store_a_to_lds(a_regs, lds_buf_off):
                """Store pre-loaded A registers to LDS (padded stride)."""
                for al in range_constexpr(NUM_A_LOADS):
                    lin = tid * arith.index(A_LOAD_VEC) + arith.index(
                        al * THREADS_PER_BLOCK * A_LOAD_VEC
                    )
                    a_row = lin // arith.index(BLOCK_K)
                    a_col = lin % arith.index(BLOCK_K)
                    lds_idx = lds_buf_off + a_row * arith.index(BLOCK_K_PAD_A) + a_col
                    vector.store(a_regs[al], lds_view, [lds_idx])

            def _store_b_to_lds(b_regs, lds_buf_off):
                """Store pre-loaded B registers to LDS (padded stride)."""
                for bl in range_constexpr(NUM_B_LOADS):
                    lin = tid * arith.index(B_LOAD_VEC) + arith.index(
                        bl * THREADS_PER_BLOCK * B_LOAD_VEC
                    )
                    b_n = lin // arith.index(BLOCK_K)
                    b_k = lin % arith.index(BLOCK_K)
                    lds_b_base = lds_buf_off + arith.index(LDS_A_ELEMS)
                    lds_idx = lds_b_base + b_n * arith.index(BLOCK_K_PAD) + b_k
                    vector.store(b_regs[bl], lds_view, [lds_idx])

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
            # LDS -> Registers: A operands
            # ========================================
            # WMMA A operand (row-of-cols): lane t loads A[t%16][(t/16)*8+i], i=0..7
            # In LDS [BLOCK_M][BLOCK_K]:
            #   row = wave_m*reg_m*16 + rm*16 + lane16
            #   col = rk*16 + klane*8 ... +7  (8 contiguous bf16)

            def _load_a_from_lds(lds_buf_off):
                """Load A operands from LDS (padded stride)."""
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
                        lds_idx = (
                            lds_buf_off + row * arith.index(BLOCK_K_PAD_A) + col_base
                        )
                        a_raw = vector.load_op(v8_in_ty, lds_view, [lds_idx])
                        rk_vecs.append(a_raw)
                    a_vecs.append(rk_vecs)
                return a_vecs

            # ========================================
            # LDS -> Registers: B operands
            # ========================================
            # WMMA B operand (col-of-rows): lane t loads B[(t/16)*8+i][t%16], i=0..7
            # B is stored TRANSPOSED in LDS as [BLOCK_N][BLOCK_K]:
            #   LDS_B[n][k] = B[k][n]
            # For WMMA tile (rk, rn):
            #   n = wave_n*reg_n*16 + rn*16 + lane16
            #   k = rk*16 + klane*8 ... +7  (8 contiguous bf16!)
            # Read as contiguous 8-element vector — same pattern as A.

            def _load_b_from_lds(lds_buf_off):
                """Load B operands from LDS (transposed layout with padding, contiguous reads)."""
                b_vecs = []
                lds_b_base = lds_buf_off + arith.index(LDS_A_ELEMS)
                for rk in range_constexpr(reg_k):
                    rk_vecs = []
                    k_base = arith.index(rk * WMMA_K) + base8
                    for rn in range_constexpr(reg_n):
                        n_row = (
                            wave_n * arith.index(reg_n * WMMA_N)
                            + arith.index(rn * WMMA_N)
                            + lane16
                        )
                        # Contiguous 8 bf16 at LDS_B[n_row][k_base..k_base+7]
                        lds_idx = lds_b_base + n_row * arith.index(BLOCK_K_PAD) + k_base
                        b_raw = vector.load_op(v8_in_ty, lds_view, [lds_idx])
                        rk_vecs.append(b_raw)
                    b_vecs.append(rk_vecs)
                return b_vecs

            # ========================================
            # Compute WMMAs
            # ========================================
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

            # LDS buffer offsets (in bf16 elements)
            lds_buf0_off = arith.index(0)
            lds_buf1_off = arith.index(LDS_BUF_ELEMS)

            # ============================================================
            # SOFTWARE-PIPELINED K-LOOP
            # ============================================================
            #
            # Key insight: separate GMEM loads from LDS stores to allow
            # GMEM loads to overlap with LDS reads + WMMA compute.
            #
            # Structure:
            #   Prologue: GMEM load tile 0 → regs, wait, store → LDS buf0, barrier
            #   Loop (kt=0..N-2):
            #     1. Issue GMEM loads for tile kt+1 → regs (async, don't wait!)
            #     2. Read tile kt from LDS buf_cur → WMMA operands
            #     3. Compute WMMAs (GMEM loads flying in background!)
            #     4. Wait for GMEM loads to complete
            #     5. Store regs → LDS buf_next
            #     6. Barrier (sync all threads)
            #     7. Swap buffers
            #   Epilogue: Read + compute last tile

            # PROLOGUE: Load first K-tile into LDS buf0
            a_gmem = _gmem_load_a(arith.index(0))
            b_gmem = _gmem_load_b(arith.index(0))
            _wait_vmem()
            _store_a_to_lds(a_gmem, lds_buf0_off)
            _store_b_to_lds(b_gmem, lds_buf0_off)
            _barrier()

            # MAIN LOOP: kt = 0 .. num_k_tiles-2
            lds_cur_off = lds_buf0_off
            lds_next_off = lds_buf1_off

            for kt in range(num_k_tiles - 1):
                # 1. Issue GMEM loads for tile kt+1 (async — no wait!)
                k_base_next = (kt + arith.index(1)) * arith.index(BLOCK_K)
                a_gmem = _gmem_load_a(k_base_next)
                b_gmem = _gmem_load_b(k_base_next)

                # 2. Read tile kt from LDS buf_cur
                a_vecs = _load_a_from_lds(lds_cur_off)
                b_vecs = _load_b_from_lds(lds_cur_off)

                # 3. Compute WMMAs (GMEM loads for kt+1 overlap in background!)
                accs = _do_compute(accs, a_vecs, b_vecs)

                # 4. Store loaded data to LDS buf_next
                # (The compiler inserts cascaded s_wait_loadcnt before each ds_store,
                #  so explicit _wait_vmem() is not needed and would serialize things.)
                _store_a_to_lds(a_gmem, lds_next_off)
                _store_b_to_lds(b_gmem, lds_next_off)

                # 6. Barrier: sync LDS
                _barrier()

                # 7. Swap buffers
                lds_cur_off, lds_next_off = lds_next_off, lds_cur_off

            # EPILOGUE: Process last K-tile
            a_vecs = _load_a_from_lds(lds_cur_off)
            b_vecs = _load_b_from_lds(lds_cur_off)
            accs = _do_compute(accs, a_vecs, b_vecs)

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
            B_T: lambda: Textra.memref(S, S, _in_elem_ty()),
            C: lambda: Textra.memref(S, S, _out_elem_ty()),
        ):
            c1 = arith.index(1)
            total_blocks = arith.index(grid_m * grid_n)
            bk = arith.index(THREADS_PER_BLOCK)
            flir.gpu_ext.LaunchFuncOp(
                ["wmma_gemm_v9", "wmma_gemm_v9_kernel"],
                grid_size=(total_blocks, c1, c1),
                block_size=(bk, c1, c1),
                kernel_operands=[A, B_T, C],
            )

    return _WmmaGemmV9(), BLOCK_M, BLOCK_N, BLOCK_K
