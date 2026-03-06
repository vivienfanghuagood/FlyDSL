#!/usr/bin/env python3
"""WMMA GEMM kernel v25 for RDNA4 (gfx12xx, wave32).

Double-buffered LDS kernel with split GMEM load / LDS store phases.

Architecture:
- 128x128x32 tiles, 8 warps (256 threads), 4x2 warp layout
- Each warp: 2 M-repeats x 4 N-repeats (32x64 output per warp)
- 2 K-steps per iteration (K=32, WMMA_K=16) → 16 WMMAs per iter
- Double-buffered LDS (ping-pong): compute from buf[cur], prefetch to buf[1-cur]
- A[M,K] row-major GMEM, B_T[N,K] row-major GMEM
- Only 1 barrier per iteration (vs 2 in v24)

Key optimization: GMEM loads are split from LDS stores. The load data is held
in registers across loop iterations so WMMAs can overlap with in-flight loads.

Pipeline:
  Prologue:
    gmem_data = GMEM_load(tile[0])
    LDS_store(gmem_data, buf[0])
    barrier
    next_data = GMEM_load(tile[1])  // prefetch

  Loop kt=1..num_k_tiles-1:
    1. LDS read rk0 from buf[read] → WMMAs rk0   // overlaps with in-flight loads
    2. LDS read rk1 from buf[read] → WMMAs rk1
    3. LDS_store(next_data, buf[write])            // now waits for GMEM
    4. Barrier
    5. next_data = GMEM_load(tile[kt+1]) if not last
    6. Swap buffers

  Epilogue:
    LDS read rk0 → WMMAs → LDS read rk1 → WMMAs

LDS usage: 2 × (A_tile + B_tile) with K-padding = ~40KB

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


def create_wmma_gemm_v25_module(
    M: int,
    N: int,
    K: int,
    in_dtype="bf16",
    out_dtype="bf16",
    *,
    reg_m=2,  # M-repeats per warp (CK uses 2)
    reg_n=4,  # N-repeats per warp (CK uses 4)
    reg_k=2,  # K-steps per tile (32/16=2)
    waves_m=4,  # warps in M dimension (CK uses 4)
    waves_n=2,  # warps in N dimension (CK uses 2)
    group_m=8,
    a_k_pad=8,  # K-padding for A in LDS (bank conflict avoidance)
    b_k_pad=8,  # K-padding for B in LDS
):
    BLOCK_M = WMMA_M * reg_m * waves_m  # 16*2*4 = 128
    BLOCK_N = WMMA_N * reg_n * waves_n  # 16*4*2 = 128
    BLOCK_K = WMMA_K * reg_k  # 16*2 = 32
    NUM_WAVES = waves_m * waves_n  # 4*2 = 8
    WAVE_SIZE = 32
    THREADS_PER_BLOCK = NUM_WAVES * WAVE_SIZE  # 256

    assert reg_k == 2

    # Loading: each thread loads 8 bf16 elements per load (128 bits = buffer_load_b128)
    LOAD_VEC = 8
    A_TILE_ELEMS = BLOCK_M * BLOCK_K  # 128*32 = 4096
    NUM_A_LOADS = A_TILE_ELEMS // (THREADS_PER_BLOCK * LOAD_VEC)  # 4096/(256*8) = 2
    B_TILE_ELEMS = BLOCK_N * BLOCK_K  # 128*32 = 4096
    NUM_B_LOADS = B_TILE_ELEMS // (THREADS_PER_BLOCK * LOAD_VEC)  # 2
    TOTAL_LOADS = NUM_A_LOADS + NUM_B_LOADS  # 4

    BLOCK_K_PAD_A = BLOCK_K + a_k_pad  # 40
    BLOCK_K_PAD_B = BLOCK_K + b_k_pad  # 40
    LDS_A_SIZE = BLOCK_M * BLOCK_K_PAD_A  # 128*40 = 5120 elements
    LDS_B_SIZE = BLOCK_N * BLOCK_K_PAD_B  # 128*40 = 5120 elements
    LDS_ONE_BUF = LDS_A_SIZE + LDS_B_SIZE  # 10240 elements = 20480 bytes (~20KB)
    LDS_TOTAL = 2 * LDS_ONE_BUF  # 20480 elements = 40960 bytes (~40KB)

    gpu_arch = get_rocm_arch()
    S = ir.ShapedType.get_dynamic_size()

    assert M % BLOCK_M == 0
    assert N % BLOCK_N == 0
    assert K % BLOCK_K == 0

    num_k_tiles = K // BLOCK_K
    assert num_k_tiles >= 2, "Need at least 2 K-tiles for prefetch pipeline"

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

    class _WmmaGemmV25(flir.MlirModule):
        GPU_MODULE_NAME = "wmma_gemm_v25"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}">']

        def init_gpu_module(self):
            _state["lds"] = allocator.allocate_array(_in_elem_ty(), LDS_TOTAL)
            allocator.finalize()

        @flir.kernel
        def wmma_gemm_v25_kernel(
            self: flir.T.i64,
            A: lambda: Textra.memref(S, S, _in_elem_ty()),
            B_T: lambda: Textra.memref(S, S, _in_elem_ty()),
            C: lambda: Textra.memref(S, S, _out_elem_ty()),
        ):
            in_ir_ty = ir.BF16Type.get() if is_bf16 else ir.F16Type.get()
            v8_in_ty = ir.VectorType.get([8], in_ir_ty)
            v4f32_ty = ir.VectorType.get([4], ir.F32Type.get())
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

            # Swizzle workgroup mapping for L2 locality
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

            # 4x2 warp layout: wave_m in [0..3], wave_n in [0..1]
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

            # ============================================================
            # Pre-compute GMEM offsets and LDS addresses
            # ============================================================
            a_lds_info = []
            for al in range_constexpr(NUM_A_LOADS):
                a_lin = tid * arith.index(LOAD_VEC) + arith.index(
                    al * THREADS_PER_BLOCK * LOAD_VEC
                )
                a_load_row = a_lin // arith.index(BLOCK_K)
                a_load_col = a_lin % arith.index(BLOCK_K)
                lds_rel = a_load_row * arith.index(BLOCK_K_PAD_A) + a_load_col
                # Pre-compute partial GMEM offset (row component only, k_base added later)
                g_row = tile_m0 + a_load_row
                a_lds_info.append((g_row, a_load_col, lds_rel))

            b_lds_info = []
            for bl in range_constexpr(NUM_B_LOADS):
                b_lin = tid * arith.index(LOAD_VEC) + arith.index(
                    bl * THREADS_PER_BLOCK * LOAD_VEC
                )
                b_load_row = b_lin // arith.index(BLOCK_K)
                b_load_col = b_lin % arith.index(BLOCK_K)
                lds_rel = (
                    arith.index(LDS_A_SIZE)
                    + b_load_row * arith.index(BLOCK_K_PAD_B)
                    + b_load_col
                )
                g_row = tile_n0 + b_load_row
                b_lds_info.append((g_row, b_load_col, lds_rel))

            # ============================================================
            # Phase 1: Issue GMEM loads (non-blocking), return raw data
            # ============================================================
            def _gmem_load(k_base):
                """Issue buffer_loads for A+B tile. Returns list of raw v4f32."""
                raw_data = []
                for al in range_constexpr(NUM_A_LOADS):
                    g_row, a_load_col, _ = a_lds_info[al]
                    g_col = k_base + a_load_col
                    elem_off = g_row * arith.index(K) + g_col
                    f32_off = elem_off // c2
                    a_raw = buffer_ops.buffer_load(
                        a_rsrc, f32_off, vec_width=4, dtype=ir.F32Type.get()
                    )
                    raw_data.append(a_raw)

                for bl in range_constexpr(NUM_B_LOADS):
                    g_row, b_load_col, _ = b_lds_info[bl]
                    g_col = k_base + b_load_col
                    elem_off = g_row * arith.index(K) + g_col
                    f32_off = elem_off // c2
                    b_raw = buffer_ops.buffer_load(
                        bt_rsrc, f32_off, vec_width=4, dtype=ir.F32Type.get()
                    )
                    raw_data.append(b_raw)

                return raw_data  # [a0, a1, b0, b1] — 4 x v4f32

            # ============================================================
            # Phase 2: Store loaded data to LDS
            # ============================================================
            def _lds_store(raw_data, buf_offset):
                """Store previously loaded data to LDS at buf_offset."""
                for al in range_constexpr(NUM_A_LOADS):
                    _, _, lds_rel = a_lds_info[al]
                    a_vec = vector.bitcast(v8_in_ty, raw_data[al])
                    lds_idx = buf_offset + lds_rel
                    vector.store(a_vec, lds_view, [lds_idx])

                for bl in range_constexpr(NUM_B_LOADS):
                    _, _, lds_rel = b_lds_info[bl]
                    b_vec = vector.bitcast(v8_in_ty, raw_data[NUM_A_LOADS + bl])
                    lds_idx = buf_offset + lds_rel
                    vector.store(b_vec, lds_view, [lds_idx])

            # ============================================================
            # LDS read helpers — read from buf_offset
            # ============================================================
            def _load_a_from_lds(rk, buf_offset):
                """Load A WMMA operands from LDS for K-step rk."""
                vecs = []
                col_base = arith.index(rk * WMMA_K) + base8
                for rm in range_constexpr(reg_m):
                    row = (
                        wave_m * arith.index(reg_m * WMMA_M)
                        + arith.index(rm * WMMA_M)
                        + lane16
                    )
                    lds_idx = buf_offset + row * arith.index(BLOCK_K_PAD_A) + col_base
                    a_raw = vector.load_op(v8_in_ty, lds_view, [lds_idx])
                    vecs.append(a_raw)
                return vecs

            def _load_b_from_lds(rk, buf_offset):
                """Load B WMMA operands from LDS for K-step rk."""
                vecs = []
                col_base = arith.index(rk * WMMA_K) + base8
                for rn in range_constexpr(reg_n):
                    row = (
                        wave_n * arith.index(reg_n * WMMA_N)
                        + arith.index(rn * WMMA_N)
                        + lane16
                    )
                    lds_idx = (
                        buf_offset
                        + arith.index(LDS_A_SIZE)
                        + row * arith.index(BLOCK_K_PAD_B)
                        + col_base
                    )
                    b_raw = vector.load_op(v8_in_ty, lds_view, [lds_idx])
                    vecs.append(b_raw)
                return vecs

            def _barrier():
                _llvm.inline_asm(
                    res=None,
                    operands_=[],
                    asm_string="s_wait_dscnt 0x0\ns_wait_storecnt 0x0\ns_barrier_signal -1\ns_barrier_wait -1",
                    constraints="",
                    has_side_effects=True,
                )

            def _do_wmma_block(accs_in, a_vecs, b_vecs):
                """Compute all WMMAs for one K-step (reg_m x reg_n WMMAs)."""
                new_accs = list(accs_in)
                for rm in range_constexpr(reg_m):
                    for rn in range_constexpr(reg_n):
                        idx = rm * reg_n + rn
                        new_accs[idx] = _wmma_op(
                            v8f32_ty,
                            a_vecs[rm],
                            b_vecs[rn],
                            new_accs[idx],
                            v8i16_ty,
                        )
                return new_accs

            # ============================================================
            # Initialize accumulators
            # ============================================================
            zero_acc = arith.constant_vector(0.0, v8f32_ty)
            accs = [zero_acc for _ in range_constexpr(reg_m * reg_n)]

            # ============================================================
            # DOUBLE-BUFFERED PIPELINE WITH SPLIT LOAD/STORE
            #
            # The key: GMEM loads are issued at the top of the loop,
            # then WMMAs execute (overlap), then wait+store+barrier.
            # No OOB prefetch needed — load is always for a valid tile.
            # ============================================================

            c_lds_buf_stride = arith.index(LDS_ONE_BUF)
            c_zero = arith.index(0)

            # --- PROLOGUE ---
            # Load tile[0] → store to buf[0] → barrier
            prologue_data = _gmem_load(c_zero)
            _lds_store(prologue_data, c_zero)
            _barrier()

            # --- MAIN LOOP: kt=0..num_k_tiles-2 ---
            # Each iteration:
            #   1. Issue GMEM loads for tile[kt+1] (always valid)
            #   2. Compute from buf[read] (tile[kt])
            #   3. Store loaded data to buf[write]
            #   4. Barrier
            #   5. Swap
            read_off = c_zero
            write_off = c_lds_buf_stride

            for kt in range(num_k_tiles - 1):
                # 1. Issue GMEM loads for next tile (non-blocking)
                # kt ranges 0..num_k_tiles-2, so kt+1 ranges 1..num_k_tiles-1
                # All valid tile indices.
                next_k = (kt + arith.index(1)) * arith.index(BLOCK_K)
                next_data = _gmem_load(next_k)

                # 2. Compute from current read buffer
                a_rk0 = _load_a_from_lds(0, read_off)
                b_rk0 = _load_b_from_lds(0, read_off)
                accs = _do_wmma_block(accs, a_rk0, b_rk0)

                a_rk1 = _load_a_from_lds(1, read_off)
                b_rk1 = _load_b_from_lds(1, read_off)
                accs = _do_wmma_block(accs, a_rk1, b_rk1)

                # 3. Store loaded data to write buffer
                _lds_store(next_data, write_off)

                # 4. Barrier: writes visible
                _barrier()

                # 5. Swap buffers
                read_off, write_off = write_off, read_off

            # --- EPILOGUE: Last tile already in LDS at read_off ---
            a_rk0 = _load_a_from_lds(0, read_off)
            b_rk0 = _load_b_from_lds(0, read_off)
            accs = _do_wmma_block(accs, a_rk0, b_rk0)

            a_rk1 = _load_a_from_lds(1, read_off)
            b_rk1 = _load_b_from_lds(1, read_off)
            accs = _do_wmma_block(accs, a_rk1, b_rk1)

            # ============================================================
            # Store results to GMEM
            # ============================================================
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
                ["wmma_gemm_v25", "wmma_gemm_v25_kernel"],
                grid_size=(total_blocks, c1, c1),
                block_size=(bk, c1, c1),
                kernel_operands=[A, B_T, C],
            )

    return _WmmaGemmV25(), BLOCK_M, BLOCK_N, BLOCK_K
