#!/usr/bin/env python3
"""WMMA GEMM kernel v24 for RDNA4 (gfx12xx, wave32).

CK-style 8-warp LDS kernel targeting low VGPR count (~160) for 3 waves/SIMD.

Architecture:
- 128x128x32 tiles, 8 warps (256 threads), 4x2 warp layout
- Each warp: 2 M-repeats x 4 N-repeats (32x64 output per warp)
- 2 K-steps per iteration (K=32, WMMA_K=16) → 16 WMMAs per iter
- Single-buffered LDS with barriers (like CK CompV3)
- A[M,K] row-major GMEM, B_T[N,K] row-major GMEM
- Prefetch: GMEM loads for iter N+1 issued before WMMAs of iter N

VGPR budget target: <170
- 8 accumulators x 8 regs = 64 VGPRs (accum)
- 12 ds_load destinations x 4 regs = 48 VGPRs (LDS operands)
- 4 buffer_load x 4 regs = 16 VGPRs (GMEM data)
- ~30 VGPRs (addresses/temps)
- Total: ~158 VGPRs

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


def create_wmma_gemm_v24_module(
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

    # With 256 threads and 2 loads each, each load is 8 elements:
    # Total A = 256*2*8 = 4096 = BLOCK_M*BLOCK_K ✓
    # Total B = 256*2*8 = 4096 = BLOCK_N*BLOCK_K ✓

    BLOCK_K_PAD_A = BLOCK_K + a_k_pad  # 40
    BLOCK_K_PAD_B = BLOCK_K + b_k_pad  # 40
    LDS_A_SIZE = BLOCK_M * BLOCK_K_PAD_A  # 128*40 = 5120 elements
    LDS_B_SIZE = BLOCK_N * BLOCK_K_PAD_B  # 128*40 = 5120 elements
    LDS_TOTAL = LDS_A_SIZE + LDS_B_SIZE  # 10240 elements = 20480 bytes (~20KB)

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

    class _WmmaGemmV24(flir.MlirModule):
        GPU_MODULE_NAME = "wmma_gemm_v24"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}">']

        def init_gpu_module(self):
            _state["lds"] = allocator.allocate_array(_in_elem_ty(), LDS_TOTAL)
            allocator.finalize()

        @flir.kernel
        def wmma_gemm_v24_kernel(
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
            # GMEM load helpers (separated into load + store phases)
            # ============================================================
            # With 256 threads and 2 loads per thread, each load is 8 bf16 elems.
            # Total A = 256*2*8 = 4096 = BLOCK_M*BLOCK_K
            # Total B = 256*2*8 = 4096 = BLOCK_N*BLOCK_K

            # Pre-compute A and B LDS store addresses (constant per thread)
            a_lds_addrs = []
            for al in range_constexpr(NUM_A_LOADS):
                a_lin = tid * arith.index(LOAD_VEC) + arith.index(
                    al * THREADS_PER_BLOCK * LOAD_VEC
                )
                a_load_row = a_lin // arith.index(BLOCK_K)
                a_load_col = a_lin % arith.index(BLOCK_K)
                lds_idx = a_load_row * arith.index(BLOCK_K_PAD_A) + a_load_col
                a_lds_addrs.append((a_load_row, a_load_col, lds_idx))

            b_lds_addrs = []
            for bl in range_constexpr(NUM_B_LOADS):
                b_lin = tid * arith.index(LOAD_VEC) + arith.index(
                    bl * THREADS_PER_BLOCK * LOAD_VEC
                )
                b_load_row = b_lin // arith.index(BLOCK_K)
                b_load_col = b_lin % arith.index(BLOCK_K)
                lds_idx = (
                    arith.index(LDS_A_SIZE)
                    + b_load_row * arith.index(BLOCK_K_PAD_B)
                    + b_load_col
                )
                b_lds_addrs.append((b_load_row, b_load_col, lds_idx))

            def _gmem_to_lds(k_base):
                """Load A+B tile from GMEM and store to LDS in one shot."""
                for al in range_constexpr(NUM_A_LOADS):
                    a_load_row, a_load_col, lds_idx = a_lds_addrs[al]
                    g_row = tile_m0 + a_load_row
                    g_col = k_base + a_load_col
                    elem_off = g_row * arith.index(K) + g_col
                    f32_off = elem_off // c2
                    a_raw = buffer_ops.buffer_load(
                        a_rsrc, f32_off, vec_width=4, dtype=ir.F32Type.get()
                    )
                    a_vec = vector.bitcast(v8_in_ty, a_raw)
                    vector.store(a_vec, lds_view, [lds_idx])

                for bl in range_constexpr(NUM_B_LOADS):
                    b_load_row, b_load_col, lds_idx = b_lds_addrs[bl]
                    g_row = tile_n0 + b_load_row
                    g_col = k_base + b_load_col
                    elem_off = g_row * arith.index(K) + g_col
                    f32_off = elem_off // c2
                    b_raw = buffer_ops.buffer_load(
                        bt_rsrc, f32_off, vec_width=4, dtype=ir.F32Type.get()
                    )
                    b_vec = vector.bitcast(v8_in_ty, b_raw)
                    vector.store(b_vec, lds_view, [lds_idx])

            # ============================================================
            # LDS read helpers
            # ============================================================
            def _load_a_from_lds(rk):
                """Load A WMMA operands from LDS for K-step rk."""
                vecs = []
                col_base = arith.index(rk * WMMA_K) + base8
                for rm in range_constexpr(reg_m):
                    row = (
                        wave_m * arith.index(reg_m * WMMA_M)
                        + arith.index(rm * WMMA_M)
                        + lane16
                    )
                    lds_idx = row * arith.index(BLOCK_K_PAD_A) + col_base
                    a_raw = vector.load_op(v8_in_ty, lds_view, [lds_idx])
                    vecs.append(a_raw)
                return vecs

            def _load_b_from_lds(rk):
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
                        arith.index(LDS_A_SIZE)
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

            def _wait_vmem():
                _llvm.inline_asm(
                    res=None,
                    operands_=[],
                    asm_string="s_wait_loadcnt 0x0",
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
            # Initialize accumulators: reg_m * reg_n = 2*4 = 8 tiles
            # = 8 * 8 VGPRs = 64 VGPRs total
            # ============================================================
            zero_acc = arith.constant_vector(0.0, v8f32_ty)
            accs = [zero_acc for _ in range_constexpr(reg_m * reg_n)]

            # ============================================================
            # PIPELINE: CK CompV3 style
            #
            # Prologue:
            #   GMEM load tile[0] → wait → LDS store → barrier
            #
            # Main loop (tiles 1..num_k_tiles-1):
            #   1. GMEM load tile[kt+1] (prefetch, non-blocking)
            #   2. LDS load A,B rk0 from current tile
            #   3. WMMAs rk0
            #   4. LDS load A,B rk1 from current tile
            #   5. WMMAs rk1
            #   6. Wait for GMEM prefetch → LDS store → barrier
            #
            # Epilogue (last tile, already in LDS):
            #   LDS load rk0 → WMMAs → LDS load rk1 → WMMAs
            # ============================================================

            # --- PROLOGUE: Load tile[0] into LDS ---
            _gmem_to_lds(arith.index(0))
            _barrier()  # wait for GMEM→LDS stores to be visible

            # --- MAIN LOOP: tiles 0 .. num_k_tiles-2 ---
            # Two-barrier single-buffer pipeline (matches CK CompV3):
            #
            # Per iteration:
            #   1. ds_load rk0 → WMMA rk0
            #   2. ds_load rk1 → WMMA rk1
            #   3. Barrier #1 (all waves done reading LDS)
            #   4. GMEM→LDS for next tile
            #   5. Barrier #2 (new data visible in LDS)
            for kt in range(num_k_tiles - 1):
                # Compute with current tile
                a_rk0 = _load_a_from_lds(0)
                b_rk0 = _load_b_from_lds(0)
                accs = _do_wmma_block(accs, a_rk0, b_rk0)

                a_rk1 = _load_a_from_lds(1)
                b_rk1 = _load_b_from_lds(1)
                accs = _do_wmma_block(accs, a_rk1, b_rk1)

                # Barrier: all waves done reading → safe to overwrite LDS
                _barrier()

                # Load next tile
                next_k = (kt + arith.index(1)) * arith.index(BLOCK_K)
                _gmem_to_lds(next_k)

                # Barrier: new data visible
                _barrier()

            # --- EPILOGUE: Last tile already in LDS ---
            a_rk0 = _load_a_from_lds(0)
            b_rk0 = _load_b_from_lds(0)
            accs = _do_wmma_block(accs, a_rk0, b_rk0)

            a_rk1 = _load_a_from_lds(1)
            b_rk1 = _load_b_from_lds(1)
            accs = _do_wmma_block(accs, a_rk1, b_rk1)

            # ============================================================
            # Store results to GMEM
            # ============================================================
            # Each wave covers reg_m*WMMA_M rows x reg_n*WMMA_N cols = 32x64
            # Each WMMA output: lane t holds D[(t/16)*8+i][t%16], i=0..7
            # So lane covers 8 consecutive rows at column lane16
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
                ["wmma_gemm_v24", "wmma_gemm_v24_kernel"],
                grid_size=(total_blocks, c1, c1),
                block_size=(bk, c1, c1),
                kernel_operands=[A, B_T, C],
            )

    return _WmmaGemmV24(), BLOCK_M, BLOCK_N, BLOCK_K
