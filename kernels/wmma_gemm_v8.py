#!/usr/bin/env python3
"""WMMA GEMM kernel v8 for RDNA4 (gfx12xx, wave32).

Key improvement: LDS double-buffered A + preshuffle B + software pipelining.

This kernel combines:
  - Original (non-shuffled) A loaded from GMEM into LDS via buffer_load
  - Pre-shuffled B loaded directly from GMEM into WMMA registers
  - Double-buffered LDS for A (ping-pong) to overlap loads with compute
  - Software-pipelined K-loop for maximum latency hiding

The software pipeline overlaps three hardware units simultaneously:
  1. VMEM unit: GMEM loads for A[kt+1] and B[kt+1]
  2. DS unit: LDS reads for A[kt]
  3. Matrix unit: WMMA compute with A[kt] and B[kt]

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
# Constants
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
# Host-side pre-shuffle for B (same as v6)
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


def create_wmma_gemm_v8_module(
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
):
    """Create WMMA GEMM v8 module.

    LDS double-buffered A + preshuffle B + software-pipelined K-loop.
    """
    # Derived constants
    BLOCK_M = WMMA_M * reg_m * waves_m
    BLOCK_N = WMMA_N * reg_n * waves_n
    BLOCK_K = WMMA_K * reg_k
    NUM_WAVES = waves_m * waves_n
    WAVE_SIZE = 32
    THREADS_PER_BLOCK = NUM_WAVES * WAVE_SIZE

    # A loading: each thread loads A_LOAD_VEC bf16 elements
    A_TILE_ELEMS = BLOCK_M * BLOCK_K
    A_LOAD_VEC = 8  # 8 bf16 = 128 bits = 1 buffer_load_b128
    NUM_A_LOADS = A_TILE_ELEMS // (THREADS_PER_BLOCK * A_LOAD_VEC)
    assert NUM_A_LOADS >= 1, f"Need at least 1 A load, got {NUM_A_LOADS}"

    # Double-buffered LDS for A
    LDS_A_ELEMS_PER_BUF = BLOCK_M * BLOCK_K
    LDS_A_TOTAL_ELEMS = 2 * LDS_A_ELEMS_PER_BUF  # two buffers

    gpu_arch = get_rocm_arch()
    S = ir.ShapedType.get_dynamic_size()

    assert M % BLOCK_M == 0, f"M={M} must be multiple of BLOCK_M={BLOCK_M}"
    assert N % BLOCK_N == 0, f"N={N} must be multiple of BLOCK_N={BLOCK_N}"
    assert K % BLOCK_K == 0, f"K={K} must be multiple of BLOCK_K={BLOCK_K}"

    num_k_tiles = K // BLOCK_K
    grid_m = M // BLOCK_M
    grid_n = N // BLOCK_N
    is_bf16 = in_dtype == "bf16"

    # B preshuffle constants: B_shuf[N0, K0, KLane, NLane, KPack]
    N0_total = N // 16
    K0_total = K // 16
    B_KPACK = 8
    B_STRIDE_NLANE = B_KPACK  # 8
    B_STRIDE_KLANE = 16 * B_KPACK  # 128
    B_STRIDE_K0 = 2 * 16 * B_KPACK  # 256
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

    class _WmmaGemmV8(flir.MlirModule):
        GPU_MODULE_NAME = "wmma_gemm_v8"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}">']

        def init_gpu_module(self):
            # Allocate double-buffered LDS for A
            _state["lds_a"] = allocator.allocate_array(_in_elem_ty(), LDS_A_TOTAL_ELEMS)
            allocator.finalize()

        @flir.kernel
        def wmma_gemm_v8_kernel(
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
            lds_a_view = _state["lds_a"](lds_base).get()

            a_rsrc = buffer_ops.create_buffer_resource(_unwrap(A), max_size=True)
            b_rsrc = buffer_ops.create_buffer_resource(_unwrap(B_shuf), max_size=True)
            c_rsrc = buffer_ops.create_buffer_resource(_unwrap(C), max_size=True)

            # ========================================
            # A: GMEM -> LDS functions
            # ========================================
            # Each thread loads NUM_A_LOADS chunks of 8 bf16 from A[M,K]
            # and stores them to LDS in row-major order: LDS[BLOCK_M][BLOCK_K]

            def _load_a_to_lds(k_base, lds_buf_offset):
                """Load A tile from GMEM and store to LDS at buf_offset."""
                for al in range_constexpr(NUM_A_LOADS):
                    # Compute which elements this thread loads
                    a_lin = tid * arith.index(A_LOAD_VEC) + arith.index(
                        al * THREADS_PER_BLOCK * A_LOAD_VEC
                    )
                    a_load_row = a_lin // arith.index(BLOCK_K)
                    a_load_col = a_lin % arith.index(BLOCK_K)

                    # Global A address
                    g_row = tile_m0 + a_load_row
                    g_col = k_base + a_load_col

                    # Load 8 bf16 = 4 f32 from GMEM
                    elem_off = g_row * arith.index(K) + g_col
                    f32_off = elem_off // arith.index(2)  # bf16 offset -> f32 offset
                    a_raw = buffer_ops.buffer_load(
                        a_rsrc, f32_off, vec_width=4, dtype=ir.F32Type.get()
                    )
                    a_vec = vector.bitcast(v8_in_ty, a_raw)

                    # Store to LDS
                    lds_idx = (
                        lds_buf_offset + a_load_row * arith.index(BLOCK_K) + a_load_col
                    )
                    vector.store(a_vec, lds_a_view, [lds_idx])

            def _barrier():
                """LDS barrier using inline asm (avoids global_inv from gpu.barrier)."""
                _llvm.inline_asm(
                    res=None,
                    operands_=[],
                    asm_string="s_wait_dscnt 0x0\ns_barrier_signal -1\ns_barrier_wait -1",
                    constraints="",
                    has_side_effects=True,
                )

            def _wait_vmem():
                """Wait for all VMEM loads."""
                _llvm.inline_asm(
                    res=None,
                    operands_=[],
                    asm_string="s_wait_loadcnt 0x0",
                    constraints="",
                    has_side_effects=True,
                )

            # ========================================
            # A: LDS -> Registers (WMMA format)
            # ========================================
            def _load_a_from_lds(lds_buf_offset):
                """Read A operands from LDS in WMMA layout.

                WMMA A operand: lane t reads A[t%16][(t/16)*8 + i], i=0..7
                In LDS layout [BLOCK_M][BLOCK_K]:
                  row = wave_m * reg_m * 16 + rm * 16 + lane16
                  col = rk * 16 + klane * 8  (8 contiguous elements)
                """
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
                        lds_idx = lds_buf_offset + row * arith.index(BLOCK_K) + col_base
                        a_raw = vector.load_op(v8_in_ty, lds_a_view, [lds_idx])
                        rk_vecs.append(a_raw)
                    a_vecs.append(rk_vecs)
                return a_vecs

            # ========================================
            # B: GMEM -> Registers (preshuffle)
            # ========================================
            def _load_b_tile(k_tile_idx):
                """Load B operands from pre-shuffled GMEM."""
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

            # ========================================
            # Initialize accumulators
            # ========================================
            zero_acc = arith.constant_vector(0.0, v8f32_ty)
            accs = [zero_acc for _ in range_constexpr(reg_m * reg_n)]

            # LDS buffer offsets (in bf16 elements)
            lds_buf0_off = arith.index(0)
            lds_buf1_off = arith.index(LDS_A_ELEMS_PER_BUF)

            # ============================================================
            # SOFTWARE-PIPELINED K-LOOP
            # ============================================================

            # PROLOGUE: Load A[0] into LDS buf0, load B[0] into registers
            _load_a_to_lds(arith.index(0), lds_buf0_off)
            _wait_vmem()
            _barrier()

            # Load B[0] from GMEM
            b_vecs = _load_b_tile(arith.index(0))

            # MAIN LOOP: kt = 0 .. num_k_tiles-2
            # Each iteration:
            #   - A[kt] is already in LDS buf_cur
            #   - B[kt] is already in b_vecs
            #   - Issue GMEM loads for A[kt+1] into LDS buf_next
            #   - Issue GMEM loads for B[kt+1] into b_next
            #   - Read A[kt] from LDS buf_cur
            #   - Compute with A[kt], B[kt]
            #   - Wait for A[kt+1] loads, barrier, swap buffers

            # We use the fact that lds_buf_offset alternates between 0 and LDS_A_ELEMS_PER_BUF
            # Use a flag variable that the auto-lowering carries through scf.for
            lds_cur_off = lds_buf0_off
            lds_next_off = lds_buf1_off

            for kt in range(num_k_tiles - 1):
                # 1. Issue GMEM loads for A[kt+1] into LDS buf_next (async)
                k_base_next = (kt + arith.index(1)) * arith.index(BLOCK_K)
                _load_a_to_lds(k_base_next, lds_next_off)

                # 2. Issue GMEM loads for B[kt+1] (async)
                b_next = _load_b_tile(kt + arith.index(1))

                # 3. Read A[kt] from LDS buf_cur
                a_vecs = _load_a_from_lds(lds_cur_off)

                # 4. Compute WMMAs with A[kt] and B[kt]
                accs = _do_compute(accs, a_vecs, b_vecs)

                # 5. Wait for A[kt+1] GMEM loads to complete
                _wait_vmem()

                # 6. Barrier: ensure all LDS reads (step 3) complete, then
                #    ensure A[kt+1] LDS writes (step 1) are visible
                _barrier()

                # 7. Swap buffers and B vectors
                lds_cur_off, lds_next_off = lds_next_off, lds_cur_off
                b_vecs = b_next

            # EPILOGUE: Process last K-tile
            a_vecs = _load_a_from_lds(lds_cur_off)
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
            B_shuf: lambda: Textra.memref(S, _in_elem_ty()),
            C: lambda: Textra.memref(S, S, _out_elem_ty()),
        ):
            c1 = arith.index(1)
            total_blocks = arith.index(grid_m * grid_n)
            bk = arith.index(THREADS_PER_BLOCK)
            flir.gpu_ext.LaunchFuncOp(
                ["wmma_gemm_v8", "wmma_gemm_v8_kernel"],
                grid_size=(total_blocks, c1, c1),
                block_size=(bk, c1, c1),
                kernel_operands=[A, B_shuf, C],
            )

    return _WmmaGemmV8(), BLOCK_M, BLOCK_N, BLOCK_K
