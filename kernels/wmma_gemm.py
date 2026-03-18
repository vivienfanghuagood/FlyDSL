#!/usr/bin/env python3
"""WMMA GEMM kernel for RDNA4 (gfx12xx, wave32).

4-warp LDS kernel inspired by Triton's 93 TFLOPS approach.

Architecture:
- 128x128x32 tiles, 4 warps (128 threads), 2x2 warp layout
- Each warp: 4 M-repeats x 4 N-repeats (64x64 output per warp)
- 2 K-steps per iteration (K=32, WMMA_K=16) -> 32 WMMAs per iter
- Double-buffered LDS (ping-pong): compute from buf[cur], prefetch to buf[1-cur]
- A[M,K] row-major GMEM, B_T[N,K] row-major GMEM
- K-padding on LDS stores for bank conflict avoidance

LDS layout (per buffer):
  A tile: 128 rows x (32+pad) cols x 2B, stored row-major
  B tile: 128 rows x (32+pad) cols x 2B, stored row-major
  Total per buffer: ~20KB, double-buffered: ~40KB

Pipeline: split GMEM load / LDS store with double buffering

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


def create_wmma_gemm_module(
    M: int,
    N: int,
    K: int,
    in_dtype="bf16",
    out_dtype="bf16",
    *,
    reg_m=4,  # M-repeats per warp
    reg_n=4,  # N-repeats per warp
    reg_k=2,  # K-steps per tile (32/16=2)
    waves_m=2,  # warps in M dimension
    waves_n=2,  # warps in N dimension
    group_m=8,
    a_k_pad=8,  # K-padding for A in LDS (bank conflict avoidance)
    b_k_pad=8,  # K-padding for B in LDS
):
    BLOCK_M = WMMA_M * reg_m * waves_m  # 16*4*2 = 128
    BLOCK_N = WMMA_N * reg_n * waves_n  # 16*4*2 = 128
    BLOCK_K = WMMA_K * reg_k  # 16*2 = 32
    NUM_WAVES = waves_m * waves_n  # 2*2 = 4
    WAVE_SIZE = 32
    THREADS_PER_BLOCK = NUM_WAVES * WAVE_SIZE  # 128

    assert reg_k >= 2 and reg_k % 2 == 0

    # Loading: each thread loads 8 bf16 elements per load (128 bits = buffer_load_b128)
    LOAD_VEC = 8
    A_TILE_ELEMS = BLOCK_M * BLOCK_K  # 128*32 = 4096
    NUM_A_LOADS = A_TILE_ELEMS // (THREADS_PER_BLOCK * LOAD_VEC)  # 4096/(128*8) = 4
    B_TILE_ELEMS = BLOCK_N * BLOCK_K  # 128*32 = 4096
    NUM_B_LOADS = B_TILE_ELEMS // (THREADS_PER_BLOCK * LOAD_VEC)  # 4
    TOTAL_LOADS = NUM_A_LOADS + NUM_B_LOADS  # 8

    # LDS layout with K-padding for bank conflict avoidance
    BLOCK_K_PAD_A = BLOCK_K + a_k_pad  # 40
    BLOCK_K_PAD_B = BLOCK_K + b_k_pad  # 40
    LDS_A_SIZE = BLOCK_M * BLOCK_K_PAD_A  # 128*40 = 5120 elements
    LDS_B_SIZE = BLOCK_N * BLOCK_K_PAD_B  # 128*40 = 5120 elements
    LDS_ONE_BUF = LDS_A_SIZE + LDS_B_SIZE  # 10240 elements = 20KB
    LDS_TOTAL = 2 * LDS_ONE_BUF  # 20480 elements = 40KB

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
            return rocdl.wmma_f32_16x16x16_bf16(result_type, [a_i16, b_i16, arith.unwrap(acc)])
        else:
            return rocdl.wmma_f32_16x16x16_f16(
                result_type,
                [arith.unwrap(a_vec), arith.unwrap(b_vec), arith.unwrap(acc)],
            )

    allocator = SmemAllocator(None, arch=gpu_arch)
    _state = {}

    class _WmmaGemm(flir.MlirModule):
        GPU_MODULE_NAME = "wmma_gemm"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}">']

        def init_gpu_module(self):
            _state["lds"] = allocator.allocate_array(_in_elem_ty(), LDS_TOTAL)
            allocator.finalize()

        @flir.kernel
        def wmma_gemm_kernel(
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

            # 2x2 warp layout
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
                a_lin = tid * arith.index(LOAD_VEC) + arith.index(al * THREADS_PER_BLOCK * LOAD_VEC)
                a_load_row = a_lin // arith.index(BLOCK_K)
                a_load_col = a_lin % arith.index(BLOCK_K)
                lds_rel = a_load_row * arith.index(BLOCK_K_PAD_A) + a_load_col
                g_row = tile_m0 + a_load_row
                a_lds_info.append((g_row, a_load_col, lds_rel))

            b_lds_info = []
            for bl in range_constexpr(NUM_B_LOADS):
                b_lin = tid * arith.index(LOAD_VEC) + arith.index(bl * THREADS_PER_BLOCK * LOAD_VEC)
                b_load_row = b_lin // arith.index(BLOCK_K)
                b_load_col = b_lin % arith.index(BLOCK_K)
                lds_rel = arith.index(LDS_A_SIZE) + b_load_row * arith.index(BLOCK_K_PAD_B) + b_load_col
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
                    a_raw = buffer_ops.buffer_load(a_rsrc, f32_off, vec_width=4, dtype=ir.F32Type.get())
                    raw_data.append(a_raw)

                for bl in range_constexpr(NUM_B_LOADS):
                    g_row, b_load_col, _ = b_lds_info[bl]
                    g_col = k_base + b_load_col
                    elem_off = g_row * arith.index(K) + g_col
                    f32_off = elem_off // c2
                    b_raw = buffer_ops.buffer_load(bt_rsrc, f32_off, vec_width=4, dtype=ir.F32Type.get())
                    raw_data.append(b_raw)

                return raw_data  # [a0, a1, a2, a3, b0, b1, b2, b3] -- 8 x v4f32

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
            # LDS read helpers -- row-major with K-padding
            # ============================================================
            def _load_a_from_lds(rk, buf_offset):
                """Load A WMMA operands from LDS for K-step rk."""
                vecs = []
                col_base = arith.index(rk * WMMA_K) + base8
                for rm in range_constexpr(reg_m):
                    row = wave_m * arith.index(reg_m * WMMA_M) + arith.index(rm * WMMA_M) + lane16
                    lds_idx = buf_offset + row * arith.index(BLOCK_K_PAD_A) + col_base
                    a_raw = vector.load_op(v8_in_ty, lds_view, [lds_idx])
                    vecs.append(a_raw)
                return vecs

            def _load_b_from_lds(rk, buf_offset):
                """Load B WMMA operands from LDS for K-step rk."""
                vecs = []
                col_base = arith.index(rk * WMMA_K) + base8
                for rn in range_constexpr(reg_n):
                    row = wave_n * arith.index(reg_n * WMMA_N) + arith.index(rn * WMMA_N) + lane16
                    lds_idx = buf_offset + arith.index(LDS_A_SIZE) + row * arith.index(BLOCK_K_PAD_B) + col_base
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

            def _do_compute_rk(accs_in, rk, buf_offset):
                """Compute all WMMAs for one K-step.

                Pattern: load all B first, then for each A load 1 A -> 4 WMMAs.
                This keeps register pressure low: only 4 B + 1 A + 16 accs live.
                """
                new_accs = list(accs_in)
                # Load all B operands for this K-step first
                b_vecs = _load_b_from_lds(rk, buf_offset)
                # Then load A one at a time and do reg_n WMMAs per A
                for rm in range_constexpr(reg_m):
                    a_vec = _load_a_single_from_lds(rk, rm, buf_offset)
                    for rn in range_constexpr(reg_n):
                        idx = rm * reg_n + rn
                        new_accs[idx] = _wmma_op(
                            v8f32_ty,
                            a_vec,
                            b_vecs[rn],
                            new_accs[idx],
                            v8i16_ty,
                        )
                return new_accs

            def _load_a_single_from_lds(rk, rm_val, buf_offset):
                """Load a single A WMMA operand from LDS for K-step rk, repeat rm_val."""
                col_base = arith.index(rk * WMMA_K) + base8
                row = wave_m * arith.index(reg_m * WMMA_M) + arith.index(rm_val * WMMA_M) + lane16
                lds_idx = buf_offset + row * arith.index(BLOCK_K_PAD_A) + col_base
                return vector.load_op(v8_in_ty, lds_view, [lds_idx])

            # ============================================================
            # Initialize accumulators -- 4x4 = 16 accumulators
            # ============================================================
            zero_acc = arith.constant_vector(0.0, v8f32_ty)
            accs = [zero_acc for _ in range_constexpr(reg_m * reg_n)]

            # ============================================================
            # DOUBLE-BUFFERED PIPELINE WITH SPLIT LOAD/STORE
            # ============================================================

            c_lds_buf_stride = arith.index(LDS_ONE_BUF)
            c_zero = arith.index(0)

            # --- PROLOGUE ---
            prologue_data = _gmem_load(c_zero)
            _lds_store(prologue_data, c_zero)
            _barrier()

            # --- MAIN LOOP: kt=0..num_k_tiles-2 ---
            read_off = c_zero
            write_off = c_lds_buf_stride

            for kt in range(num_k_tiles - 1):
                # 1. Issue GMEM loads for next tile
                next_k = (kt + arith.index(1)) * arith.index(BLOCK_K)
                next_data = _gmem_load(next_k)

                # 2. Compute from current read buffer -- all K-steps
                for rk in range_constexpr(reg_k):
                    accs = _do_compute_rk(accs, rk, read_off)

                # 3. Store loaded data to write buffer
                _lds_store(next_data, write_off)

                # 4. Barrier
                _barrier()

                # 5. Swap buffers
                read_off, write_off = write_off, read_off

            # --- EPILOGUE: Last tile already in LDS at read_off ---
            for rk in range_constexpr(reg_k):
                accs = _do_compute_rk(accs, rk, read_off)

            # ============================================================
            # Store results to GMEM
            # ============================================================
            c_layout_n = arith.index(N)
            for rm in range_constexpr(reg_m):
                for rn in range_constexpr(reg_n):
                    idx = rm * reg_n + rn
                    wmma_m_off = wave_m * arith.index(reg_m * WMMA_M) + arith.index(rm * WMMA_M)
                    wmma_n_off = wave_n * arith.index(reg_n * WMMA_N) + arith.index(rn * WMMA_N)
                    for si in range_constexpr(8):
                        g_row = tile_m0 + wmma_m_off + base8 + arith.index(si)
                        g_col = tile_n0 + wmma_n_off + lane16
                        val = vector.extract(accs[idx], static_position=[si], dynamic_position=[])
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
                ["wmma_gemm", "wmma_gemm_kernel"],
                grid_size=(total_blocks, c1, c1),
                block_size=(bk, c1, c1),
                kernel_operands=[A, B_T, C],
            )

    return _WmmaGemm(), BLOCK_M, BLOCK_N, BLOCK_K
