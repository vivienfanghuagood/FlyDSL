#!/usr/bin/env python3
"""WMMA GEMM kernel v16 for RDNA4 (gfx12xx, wave32).

Double-buffered LDS A with range_constexpr(2) inner unroll + preshuffle B.

Key architecture:
  - A goes through LDS with double buffering (2 LDS buffers, ping-pong)
  - B is pre-shuffled and loaded directly from GMEM
  - Inner loop uses range_constexpr(2) to hardcode buf0/buf1 offsets at compile time
  - This avoids FlyDSL's peeling problem with runtime buffer swap variables
  - Only 1 barrier per K-tile (vs v13's 2 barriers) since writes and reads
    use different buffers

Outer loop structure (outer iterations = num_k_tiles // 2):
  range_constexpr(2): j=0 uses buf0 for read, buf1 for write
                       j=1 uses buf1 for read, buf0 for write

Each inner iteration:
  1. Read A from LDS (current buffer)
  2. Load A_next from GMEM
  3. Load B from GMEM (preshuffle)
  4. Compute 32 WMMAs
  5. Wait for GMEM loads
  6. Store A_next to LDS (other buffer)
  7. Barrier (sync before next iteration reads the other buffer)

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


def create_wmma_gemm_v16_module(
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
    BLOCK_M = WMMA_M * reg_m * waves_m  # 128
    BLOCK_N = WMMA_N * reg_n * waves_n  # 128
    BLOCK_K = WMMA_K * reg_k  # 32
    NUM_WAVES = waves_m * waves_n  # 4
    WAVE_SIZE = 32
    THREADS_PER_BLOCK = NUM_WAVES * WAVE_SIZE  # 128

    assert reg_k == 2, "v16 requires reg_k=2"

    # A loading (cooperative)
    A_TILE_ELEMS = BLOCK_M * BLOCK_K  # 128*32 = 4096
    A_LOAD_VEC = 8  # load 8 bf16 per thread per load
    NUM_A_LOADS = A_TILE_ELEMS // (THREADS_PER_BLOCK * A_LOAD_VEC)  # 4

    # A LDS layout: double-buffered, padded
    BLOCK_K_PAD_A = BLOCK_K + a_k_pad  # 40
    LDS_A_SINGLE = BLOCK_M * BLOCK_K_PAD_A  # 5120 elements = 10240 bytes
    LDS_A_TOTAL = LDS_A_SINGLE * 2  # 10240 elements = 20480 bytes (two buffers)

    # B preshuffle constants
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
    assert num_k_tiles % 2 == 0, (
        f"num_k_tiles={num_k_tiles} must be even for double buffering"
    )
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

    class _WmmaGemmV16(flir.MlirModule):
        GPU_MODULE_NAME = "wmma_gemm_v16"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}">']

        def init_gpu_module(self):
            _state["lds_a"] = allocator.allocate_array(_in_elem_ty(), LDS_A_TOTAL)
            allocator.finalize()

        @flir.kernel
        def wmma_gemm_v16_kernel(
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

            n0_base = tile_n0 // c16 + wave_n * arith.index(reg_n)

            # ========================================
            # A: GMEM → registers (cooperative load)
            # ========================================
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

            # ========================================
            # A: Registers → LDS (padded, to specific buffer)
            # ========================================
            def _store_a_to_lds(a_regs, lds_offset):
                """Store A regs to LDS at given buffer offset."""
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

            # ========================================
            # A: LDS → registers for WMMA (from specific buffer)
            # ========================================
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

            # ========================================
            # B: GMEM → registers (preshuffle, per-wave)
            # ========================================
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

            # Buffer offsets (compile-time constants)
            LDS_BUF_OFFSETS = [0, LDS_A_SINGLE]  # buf0 = 0, buf1 = LDS_A_SINGLE

            # ============================================================
            # DOUBLE-BUFFERED PIPELINED K-LOOP
            # ============================================================
            #
            # The key insight: with two LDS buffers, we can overlap:
            #   - GMEM→LDS write to buf[next] for the NEXT K-tile
            #   - LDS read + WMMA compute from buf[curr] for the CURRENT K-tile
            #
            # Structure:
            #   Prologue: load A[0]→buf0, load A[1]→buf1, barrier
            #   Main loop (outer_k_iters - 1 iterations):
            #     for j in constexpr(2):
            #       read A from buf[j]    (data already there from prev iter)
            #       load B from GMEM
            #       start loading A_next from GMEM
            #       compute 32 WMMAs
            #       wait for A_next GMEM loads
            #       store A_next → buf[j]  (overwrite buf we just read)
            #       barrier                (1 barrier per K-tile)
            #   Last iteration: read + compute without prefetch
            #
            # FlyDSL peeling note: the outer loop uses range(outer_k_iters - 1).
            # FlyDSL will peel iteration 0 but that's fine - the peeled iteration
            # processes the same way as any iteration, and accs is the only
            # loop-carried state.

            # ============================================================
            # TRUE DOUBLE-BUFFERED PIPELINE: 1 barrier per K-tile
            # ============================================================
            #
            # Data flow with alternating buffers:
            #   Prologue: load A[0] → buf0, barrier
            #   j=0: READ buf0 (A[0]), compute with B[0],
            #         load A[1] GMEM → wait → WRITE buf1, barrier
            #   j=1: READ buf1 (A[1]), compute with B[1],
            #         load A[2] GMEM → wait → WRITE buf0, barrier
            #   j=0: READ buf0 (A[2]), compute with B[2],
            #         load A[3] GMEM → wait → WRITE buf1, barrier
            #   ...
            #
            # Read buffer and write buffer alternate: [0,1,0,1,...]
            # Read buf for iteration i: i%2
            # Write buf for iteration i: (i+1)%2
            #
            # With range_constexpr(2):
            #   j=0: read buf[0], write buf[1]
            #   j=1: read buf[1], write buf[0]
            #
            # Total: num_k_tiles - 1 "pipelined" iterations + 1 final iteration
            # The outer loop covers (num_k_tiles - 1) total inner iterations.
            # With range_constexpr(2), each outer runs 2 inner iterations.
            # Need (num_k_tiles - 1) to be divisible by 2 (i.e. num_k_tiles odd).
            # But num_k_tiles=128 (even), so num_k_tiles-1=127 (odd). Not divisible by 2.
            #
            # Alternative: run outer_k_iters outer iterations of 2 tiles each,
            # process tiles 0..2*outer_k_iters-1 = 0..127.
            # Prologue loads tile 0 into buf0.
            # Each inner iteration i (0-indexed): reads buf[i%2], computes tile i,
            #   loads tile i+1 from GMEM and writes to buf[(i+1)%2].
            # Last inner iteration: reads buf[(num_k_tiles-1)%2], computes, no write.
            #
            # With outer_k_iters = num_k_tiles // 2 = 64:
            # Process 128 tiles: 0..127
            # Prologue: tile 0 → buf0
            # 127 pipelined iterations: tiles 0..126 each load tile i+1
            # 1 final: tile 127 compute only
            # 127 not divisible by 2, so we do 63 outer + 1 remainder.
            #
            # Actually simpler: just use range(num_k_tiles) with single iteration.
            # But FlyDSL with range() might interact.
            #
            # Let's use: outer loop = range(num_k_tiles // 2 - 1) for pipelined part
            # + epilogue of 2 tiles

            READ_BUFS = [0, LDS_A_SINGLE]  # j=0 reads buf0, j=1 reads buf1
            WRITE_BUFS = [LDS_A_SINGLE, 0]  # j=0 writes buf1, j=1 writes buf0

            # PROLOGUE: load A[0] → buf0
            a_gmem_0 = _gmem_load_a(arith.index(0))
            _wait_vmem()
            _store_a_to_lds(a_gmem_0, arith.index(READ_BUFS[0]))
            _barrier()

            # MAIN LOOP: each outer iteration processes 2 K-tiles
            # Iteration i processes tiles [2*i, 2*i+1]
            # j=0: read buf0 (tile 2*i), compute, load tile 2*i+1 → write buf1, barrier
            # j=1: read buf1 (tile 2*i+1), compute, load tile 2*(i+1) → write buf0, barrier
            for kt_outer in range(outer_k_iters - 1):
                for j in range_constexpr(2):
                    read_buf = arith.index(READ_BUFS[j])
                    write_buf = arith.index(WRITE_BUFS[j])
                    kt = kt_outer * arith.index(2) + arith.index(j)

                    # Read A from LDS (data from prologue or previous write)
                    a_rk0 = _load_a_from_lds_rk(0, read_buf)
                    a_rk1 = _load_a_from_lds_rk(1, read_buf)

                    # Load B from GMEM
                    b_vecs = _load_b_tile(kt)

                    # Load A_next from GMEM (for the NEXT inner iteration)
                    next_k_base = (kt + arith.index(1)) * arith.index(BLOCK_K)
                    a_gmem_next = _gmem_load_a(next_k_base)

                    # Compute 32 WMMAs (can overlap with A_next GMEM load)
                    accs = _do_compute_rk(accs, a_rk0, b_vecs[0])
                    accs = _do_compute_rk(accs, a_rk1, b_vecs[1])

                    # Wait for A_next and store to the OTHER buffer
                    _wait_vmem()
                    _store_a_to_lds(a_gmem_next, write_buf)
                    _barrier()

            # LAST OUTER ITERATION: 2 tiles, prefetch only between them
            # Tile second-to-last: read buf0, compute, prefetch last tile → buf1, barrier
            last_outer = outer_k_iters - 1
            kt_second_last = arith.index(last_outer * 2)
            a_rk0 = _load_a_from_lds_rk(0, arith.index(READ_BUFS[0]))
            a_rk1 = _load_a_from_lds_rk(1, arith.index(READ_BUFS[0]))
            b_vecs = _load_b_tile(kt_second_last)
            next_k_base = (kt_second_last + arith.index(1)) * arith.index(BLOCK_K)
            a_gmem_last = _gmem_load_a(next_k_base)
            accs = _do_compute_rk(accs, a_rk0, b_vecs[0])
            accs = _do_compute_rk(accs, a_rk1, b_vecs[1])
            _wait_vmem()
            _store_a_to_lds(a_gmem_last, arith.index(WRITE_BUFS[0]))
            _barrier()

            # Final tile: read buf1, compute, no more prefetch
            kt_last = arith.index(last_outer * 2 + 1)
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
                ["wmma_gemm_v16", "wmma_gemm_v16_kernel"],
                grid_size=(total_blocks, c1, c1),
                block_size=(bk, c1, c1),
                kernel_operands=[A, B_shuf, C],
            )

    return _WmmaGemmV16(), BLOCK_M, BLOCK_N, BLOCK_K
