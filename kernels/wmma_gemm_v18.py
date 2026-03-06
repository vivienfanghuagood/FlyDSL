#!/usr/bin/env python3
"""WMMA GEMM kernel v18 for RDNA4 (gfx12xx, wave32).

rocBLAS-style architecture: BOTH A and B go through LDS with double buffering.

Key architecture:
  - A[M,K] row-major from GMEM → LDS K-contiguous → LDS read into WMMA A-operands
  - B[K,N] row-major from GMEM → LDS K-contiguous → LDS read into WMMA B-operands
  - Double-buffered LDS: 32KB ping-pong (like rocBLAS's XOR 0x8000 toggle)
  - GMEM→LDS stores interleaved with WMMA compute for latency hiding
  - 128 threads (4 waves × 32 lanes), 128×128×32 tile

LDS layout (per buffer):
  A: [BLOCK_M, BLOCK_K_PAD] = [128, 40] bf16 = 10240 bytes
  B: [BLOCK_N, BLOCK_K_PAD] = [128, 40] bf16 = 10240 bytes  (B stored as B_T[N,K])
  Total per buffer: 20480 bytes
  Two buffers: 40960 bytes (~40KB, fits in 64KB LDS)

B is loaded as B_T[N,K] (K contiguous), which is the natural GEMM layout for B^T.
For standard B[K,N], we need to load as transposed.

Actually, for simplicity and to match rocBLAS, B is stored with K as the contiguous
dimension. We load B_T[N,K] = B[K,N]^T with K contiguous. This matches the WMMA
B-operand layout where each lane needs 8 values along K.

The cooperative load for B: all 128 threads load parts of B[K_tile, N_tile].
Since B is [K,N] row-major, K is the slow dimension, N is contiguous.
We transpose during the LDS store so B appears as [N, K] in LDS (K contiguous).

Simpler approach: require B in transposed form B_T[N,K] on the host.
Or just load B[K,N] and store transposed into LDS.

Computes C[M,N] = A[M,K] @ B[K,N]
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


def create_wmma_gemm_v18_module(
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
    """Create v18: rocBLAS-style both A+B through LDS with double buffering.

    B_T[N,K] is passed — the user must transpose B before calling.
    """
    BLOCK_M = WMMA_M * reg_m * waves_m  # 128
    BLOCK_N = WMMA_N * reg_n * waves_n  # 128
    BLOCK_K = WMMA_K * reg_k  # 32
    NUM_WAVES = waves_m * waves_n  # 4
    WAVE_SIZE = 32
    THREADS_PER_BLOCK = NUM_WAVES * WAVE_SIZE  # 128

    assert reg_k == 2, "v18 requires reg_k=2"

    # A loading (cooperative): 128 threads load A[BLOCK_M, BLOCK_K]
    # Each thread loads 8 bf16 (128 bits) per load
    A_LOAD_VEC = 8  # 8 bf16 per load = 16 bytes
    A_TILE_ELEMS = BLOCK_M * BLOCK_K  # 128*32 = 4096 bf16
    NUM_A_LOADS = A_TILE_ELEMS // (THREADS_PER_BLOCK * A_LOAD_VEC)  # 4096/1024 = 4

    # B loading (cooperative): 128 threads load B_T[BLOCK_N, BLOCK_K]
    B_LOAD_VEC = 8
    B_TILE_ELEMS = BLOCK_N * BLOCK_K  # 128*32 = 4096 bf16
    NUM_B_LOADS = B_TILE_ELEMS // (THREADS_PER_BLOCK * B_LOAD_VEC)  # 4

    # LDS layout (per buffer, bf16 element counts):
    #   A: [BLOCK_M, BLOCK_K + a_k_pad] = [128, 40], K-contiguous, M rows
    #   B: [BLOCK_N, BLOCK_K + b_k_pad] = [128, 40], K-contiguous, N rows
    BLOCK_K_PAD_A = BLOCK_K + a_k_pad  # 40
    BLOCK_K_PAD_B = BLOCK_K + b_k_pad  # 40

    LDS_A_SINGLE = BLOCK_M * BLOCK_K_PAD_A  # 128*40 = 5120 elems = 10240 bytes
    LDS_B_SINGLE = BLOCK_N * BLOCK_K_PAD_B  # 128*40 = 5120 elems = 10240 bytes
    LDS_SINGLE = LDS_A_SINGLE + LDS_B_SINGLE  # 10240 elems = 20480 bytes
    LDS_TOTAL = LDS_SINGLE * 2  # 40960 bytes (~40KB for double buffer)

    gpu_arch = get_rocm_arch()
    S = ir.ShapedType.get_dynamic_size()

    assert M % BLOCK_M == 0
    assert N % BLOCK_N == 0
    assert K % BLOCK_K == 0

    num_k_tiles = K // BLOCK_K
    assert num_k_tiles % 2 == 0, f"num_k_tiles={num_k_tiles} must be even"
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

    class _WmmaGemmV18(flir.MlirModule):
        GPU_MODULE_NAME = "wmma_gemm_v18"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}">']

        def init_gpu_module(self):
            _state["lds"] = allocator.allocate_array(_in_elem_ty(), LDS_TOTAL)
            allocator.finalize()

        @flir.kernel
        def wmma_gemm_v18_kernel(
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
            bt_rsrc = buffer_ops.create_buffer_resource(_unwrap(B_T), max_size=True)
            c_rsrc = buffer_ops.create_buffer_resource(_unwrap(C), max_size=True)

            # ================================================
            # LDS offset constants
            # ================================================
            # Buffer 0: A at [0, LDS_A_SINGLE), B at [LDS_A_SINGLE, LDS_SINGLE)
            # Buffer 1: A at [LDS_SINGLE, LDS_SINGLE+LDS_A_SINGLE),
            #            B at [LDS_SINGLE+LDS_A_SINGLE, 2*LDS_SINGLE)
            BUF0_A = 0
            BUF0_B = LDS_A_SINGLE
            BUF1_A = LDS_SINGLE
            BUF1_B = LDS_SINGLE + LDS_A_SINGLE

            READ_A_BUFS = [BUF0_A, BUF1_A]
            READ_B_BUFS = [BUF0_B, BUF1_B]
            WRITE_A_BUFS = [BUF1_A, BUF0_A]
            WRITE_B_BUFS = [BUF1_B, BUF0_B]

            # ================================================
            # A: GMEM → registers (cooperative load, A[M,K] row-major)
            # ================================================
            def _gmem_load_a(k_base):
                """Load A tile from GMEM. k_base is the K-offset (element index)."""
                a_regs = []
                for al in range_constexpr(NUM_A_LOADS):
                    # Linear thread index within the tile
                    a_lin = tid * arith.index(A_LOAD_VEC) + arith.index(
                        al * THREADS_PER_BLOCK * A_LOAD_VEC
                    )
                    a_load_row = a_lin // arith.index(BLOCK_K)  # M index within tile
                    a_load_col = a_lin % arith.index(BLOCK_K)  # K index within tile
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

            # ================================================
            # B: GMEM → registers (cooperative load, B_T[N,K] row-major)
            # ================================================
            def _gmem_load_b(k_base):
                """Load B tile from GMEM. B_T[N,K], k_base is K-offset."""
                b_regs = []
                for bl in range_constexpr(NUM_B_LOADS):
                    b_lin = tid * arith.index(B_LOAD_VEC) + arith.index(
                        bl * THREADS_PER_BLOCK * B_LOAD_VEC
                    )
                    b_load_row = b_lin // arith.index(BLOCK_K)  # N index within tile
                    b_load_col = b_lin % arith.index(BLOCK_K)  # K index within tile
                    g_row = tile_n0 + b_load_row
                    g_col = k_base + b_load_col
                    elem_off = g_row * arith.index(K) + g_col
                    f32_off = elem_off // c2
                    b_raw = buffer_ops.buffer_load(
                        bt_rsrc, f32_off, vec_width=4, dtype=ir.F32Type.get()
                    )
                    b_vec = vector.bitcast(v8_in_ty, b_raw)
                    b_regs.append(b_vec)
                return b_regs

            # ================================================
            # A: Registers → LDS (K-contiguous, padded)
            # ================================================
            def _store_a_to_lds(a_regs, lds_a_offset):
                for al in range_constexpr(NUM_A_LOADS):
                    a_lin = tid * arith.index(A_LOAD_VEC) + arith.index(
                        al * THREADS_PER_BLOCK * A_LOAD_VEC
                    )
                    a_load_row = a_lin // arith.index(BLOCK_K)
                    a_load_col = a_lin % arith.index(BLOCK_K)
                    lds_idx = (
                        arith.index(lds_a_offset)
                        + a_load_row * arith.index(BLOCK_K_PAD_A)
                        + a_load_col
                    )
                    vector.store(a_regs[al], lds_view, [lds_idx])

            # ================================================
            # B: Registers → LDS (K-contiguous, padded)
            # B_T[N,K] → LDS[N, K_pad]
            # ================================================
            def _store_b_to_lds(b_regs, lds_b_offset):
                for bl in range_constexpr(NUM_B_LOADS):
                    b_lin = tid * arith.index(B_LOAD_VEC) + arith.index(
                        bl * THREADS_PER_BLOCK * B_LOAD_VEC
                    )
                    b_load_row = b_lin // arith.index(BLOCK_K)
                    b_load_col = b_lin % arith.index(BLOCK_K)
                    lds_idx = (
                        arith.index(lds_b_offset)
                        + b_load_row * arith.index(BLOCK_K_PAD_B)
                        + b_load_col
                    )
                    vector.store(b_regs[bl], lds_view, [lds_idx])

            # ================================================
            # A: LDS → WMMA A-operands
            # ================================================
            # A is stored as A[M_row, K_col] in LDS with stride BLOCK_K_PAD_A.
            # WMMA A-operand: lane t needs A[t%16][(t/16)*8+i], i=0..7
            # So lane t reads from row (wave_m_offset + rm*16 + lane%16),
            #   col = (lane/16)*8 + rk*16 ... reading 8 consecutive K values.
            # With K-contiguous layout, these 8 values are contiguous in LDS.
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

            # ================================================
            # B: LDS → WMMA B-operands
            # ================================================
            # B is stored as B_T[N_row, K_col] in LDS with stride BLOCK_K_PAD_B.
            # WMMA B-operand (col-of-rows): lane t needs B[(t/16)*8+i][t%16]
            #   = B_T[t%16][(t/16)*8+i]
            # So lane t reads from row (wave_n_offset + rn*16 + lane%16),
            #   col = (lane/16)*8 + rk*16.
            # These 8 values are contiguous in LDS (K-contiguous).
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

            # ================================================
            # Barrier: wait for LDS stores + signal + wait
            # ================================================
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
            # DOUBLE-BUFFERED PIPELINED K-LOOP
            # ============================================================
            #
            # Structure:
            #   Prologue: load A[0]+B[0] from GMEM → wait → store to LDS buf0 → barrier
            #   Main loop (outer_k_iters - 1 iterations):
            #     for j in constexpr(2):
            #       read A,B from LDS buf[j%2]
            #       start GMEM load of A_next, B_next for next K-tile
            #       compute 32 WMMAs with LDS data
            #       wait for GMEM loads
            #       store A_next,B_next to LDS buf[(j+1)%2]
            #       barrier
            #   Last iteration: read + compute, no prefetch

            # PROLOGUE: load first tile into LDS buf0
            a_gmem_0 = _gmem_load_a(arith.index(0))
            b_gmem_0 = _gmem_load_b(arith.index(0))
            _wait_vmem()
            _store_a_to_lds(a_gmem_0, BUF0_A)
            _store_b_to_lds(b_gmem_0, BUF0_B)
            _barrier()

            # MAIN LOOP
            # rocBLAS-style: split each K-tile into 2 halves (rk0 and rk1).
            # For each half:
            #   1. Load A,B from LDS for this half only
            #   2. Interleave GMEM→LDS stores with WMMA compute
            #   3. This keeps register pressure lower and allows better overlap
            for kt_outer in range(outer_k_iters - 1):
                for j in range_constexpr(2):
                    read_a_buf = READ_A_BUFS[j]
                    read_b_buf = READ_B_BUFS[j]
                    write_a_buf = WRITE_A_BUFS[j]
                    write_b_buf = WRITE_B_BUFS[j]

                    kt = kt_outer * arith.index(2) + arith.index(j)
                    next_k_base = (kt + arith.index(1)) * arith.index(BLOCK_K)

                    # Half 1: LDS read rk0, start GMEM loads, compute 16 WMMAs
                    a_rk0 = _load_a_from_lds_rk(0, read_a_buf)
                    b_rk0 = _load_b_from_lds_rk(0, read_b_buf)
                    # Issue GMEM loads — they overlap with WMMAs below
                    a_gmem_next = _gmem_load_a(next_k_base)
                    b_gmem_next = _gmem_load_b(next_k_base)
                    accs = _do_compute_rk(accs, a_rk0, b_rk0)

                    # Half 2: LDS read rk1, interleave GMEM→LDS stores, compute
                    a_rk1 = _load_a_from_lds_rk(1, read_a_buf)
                    b_rk1 = _load_b_from_lds_rk(1, read_b_buf)
                    # Store A to LDS (compiler waits for individual loads)
                    _store_a_to_lds(a_gmem_next, write_a_buf)
                    _store_b_to_lds(b_gmem_next, write_b_buf)
                    accs = _do_compute_rk(accs, a_rk1, b_rk1)
                    _barrier()

            # LAST OUTER ITERATION
            last_outer = outer_k_iters - 1
            kt_second_last = arith.index(last_outer * 2)
            next_k_base = (kt_second_last + arith.index(1)) * arith.index(BLOCK_K)

            # Half 1
            a_rk0 = _load_a_from_lds_rk(0, READ_A_BUFS[0])
            b_rk0 = _load_b_from_lds_rk(0, READ_B_BUFS[0])
            a_gmem_last = _gmem_load_a(next_k_base)
            b_gmem_last = _gmem_load_b(next_k_base)
            accs = _do_compute_rk(accs, a_rk0, b_rk0)

            # Half 2
            a_rk1 = _load_a_from_lds_rk(1, READ_A_BUFS[0])
            b_rk1 = _load_b_from_lds_rk(1, READ_B_BUFS[0])
            _store_a_to_lds(a_gmem_last, WRITE_A_BUFS[0])
            _store_b_to_lds(b_gmem_last, WRITE_B_BUFS[0])
            accs = _do_compute_rk(accs, a_rk1, b_rk1)
            _barrier()

            # Final tile: read buf1, compute, no prefetch
            a_rk0 = _load_a_from_lds_rk(0, READ_A_BUFS[1])
            b_rk0 = _load_b_from_lds_rk(0, READ_B_BUFS[1])
            accs = _do_compute_rk(accs, a_rk0, b_rk0)
            a_rk1 = _load_a_from_lds_rk(1, READ_A_BUFS[1])
            b_rk1 = _load_b_from_lds_rk(1, READ_B_BUFS[1])
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
                ["wmma_gemm_v18", "wmma_gemm_v18_kernel"],
                grid_size=(total_blocks, c1, c1),
                block_size=(bk, c1, c1),
                kernel_operands=[A, B_T, C],
            )

    return _WmmaGemmV18(), BLOCK_M, BLOCK_N, BLOCK_K
