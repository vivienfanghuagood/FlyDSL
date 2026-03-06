#!/usr/bin/env python3
"""Optimized WMMA GEMM kernel v4 for RDNA4 (gfx12xx, wave32).

Key improvements over v2:
  1. Vectorized C stores via LDS transpose epilogue:
     - WMMA D layout is "col-of-rows": lane t holds D[(t/16)*8+i][t%16], i=0..7
       Each lane has 8 values in the SAME column, DIFFERENT rows.
     - Scalar stores can't coalesce (non-contiguous in memory).
     - Solution: Write f32 accumulators to LDS in tile layout, then read back
       in row-major order so adjacent threads read adjacent columns.
     - Convert f32 -> bf16 and pack 4 bf16 -> 2×i32 for buffer_store_b64.
  2. Larger BLOCK_K=64: 4 WMMA-K steps per tile load, better compute density.
  3. Same preshuffle-B + LDS-A architecture as v2.

Architecture: "preshuffle B + LDS A" (same as v2)
  - B is pre-shuffled in global memory for direct GMEM loading
  - A goes through LDS with barrier-protected single buffer

Tile dimensions:
  BLOCK_M = 128 (same as v2)
  BLOCK_N = 128 (same as v2)
  BLOCK_K = 64  (was 32 in v2)
  WMMAs per K tile = 4*4*4 = 64 (was 32)

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
# Kernel configuration
# =============================================================================

WMMA_M = 16
WMMA_N = 16
WMMA_K = 16

# Register tiling: each wave handles REG_M x REG_N WMMA tiles
REG_M = 4  # 4 WMMA tiles vertically per wave
REG_N = 4  # 4 WMMA tiles horizontally per wave

# K-unroll: multiple WMMA-K steps per tile load
REG_K = 4  # 4 WMMA-K steps per K tile (was 2 in v2)

# Waves per workgroup arranged as WAVES_M x WAVES_N
WAVES_M = 2
WAVES_N = 2
NUM_WAVES = WAVES_M * WAVES_N  # 4

WAVE_SIZE = 32

# Derived block tile dimensions
BLOCK_M = WMMA_M * REG_M * WAVES_M  # 16*4*2 = 128
BLOCK_N = WMMA_N * REG_N * WAVES_N  # 16*4*2 = 128
BLOCK_K = WMMA_K * REG_K  # 16*4 = 64

THREADS_PER_BLOCK = NUM_WAVES * WAVE_SIZE  # 128

# A tile: cooperative global->LDS load
A_TILE_ELEMS = BLOCK_M * BLOCK_K  # 128 * 64 = 8192
# Each thread loads 8 bf16 = 16 bytes at a time (dwordx4)
A_LOAD_VEC = 8  # 8 bf16 = 16 bytes
NUM_A_LOADS = A_TILE_ELEMS // (THREADS_PER_BLOCK * A_LOAD_VEC)  # 8192 / (128*8) = 8

# B tile: each wave loads its own B operands directly from GMEM
NUM_B_LOADS_PER_THREAD = REG_K * REG_N  # 4 * 4 = 16

# LDS: only A (B is loaded from GMEM)
LDS_A_ELEMS = BLOCK_M * BLOCK_K  # 8192

# C epilogue LDS: store WMMA results for vectorized output
# Each workgroup's tile is BLOCK_M x BLOCK_N = 128x128
# We process one WMMA N-column (16 cols) at a time through LDS
# LDS for epilogue: BLOCK_M * 16 * sizeof(f32) = 128 * 16 * 4 = 8192 bytes
# But we reuse the A LDS buffer (which is 8192 * 2 = 16384 bytes for bf16)
# Actually A LDS = 8192 elements * 2 bytes = 16384 bytes
# C epilogue per pass = 128 * 16 * 4 = 8192 bytes < 16384, fits!
C_EPILOGUE_COLS = WMMA_N  # 16 columns per pass
C_EPILOGUE_ELEMS = BLOCK_M * C_EPILOGUE_COLS  # 128*16 = 2048 f32 elements
# We do REG_N * WAVES_N = 8 passes to cover all N columns

# L2 cache swizzle group size
GROUP_M = 8


def _unwrap(v):
    """Unwrap ArithValue to raw MLIR Value."""
    while hasattr(v, "_value"):
        v = v._value
    return v


# =============================================================================
# Host-side pre-shuffle for B (same as v2)
# =============================================================================


def preshuffle_b_wmma(B_kn, *, block_n=BLOCK_N, block_k=BLOCK_K):
    """Pre-shuffle B[K,N] into WMMA-friendly layout for direct GMEM loading.

    Input:  B[K, N] in row-major (standard PyTorch layout)
    Output: B_shuffled[N0, K0_total, KLane, NLane, KPack] in bf16/f16

    Where:
      N0 = N / 16
      K0_total = K / 16  (total WMMA K tiles)
      KLane = 2
      NLane = 16
      KPack = 8

    B_shuffled[n0, k0, klane, nlane, kpack] = B[k0*16 + klane*8 + kpack, n0*16 + nlane]
    """
    import torch

    K, N = B_kn.shape
    assert K % 16 == 0, f"K={K} must be multiple of 16"
    assert N % 16 == 0, f"N={N} must be multiple of 16"

    N0 = N // 16
    K0 = K // 16
    KLane = 2
    NLane = 16
    KPack = 8

    # Reshape B[K, N] -> B[K0, KLane, KPack, N0, NLane]
    B_reshaped = B_kn.reshape(K0, KLane, KPack, N0, NLane)
    # Permute to (N0, K0, KLane, NLane, KPack)
    B_shuffled = B_reshaped.permute(3, 0, 1, 4, 2).contiguous()
    return B_shuffled


# =============================================================================
# Kernel module
# =============================================================================


def create_wmma_gemm_v4_module(
    M: int, N: int, K: int, in_dtype="bf16", out_dtype="bf16"
):
    """Create WMMA GEMM v4 module with vectorized C stores.

    Args:
        M, N, K: matrix dimensions (must be multiples of BLOCK_M/N/K)
        in_dtype: "bf16" or "f16"
        out_dtype: "f32" or "bf16"

    B must be pre-shuffled using preshuffle_b_wmma() before calling this kernel.
    """
    gpu_arch = get_rocm_arch()
    S = ir.ShapedType.get_dynamic_size()

    assert M % BLOCK_M == 0, f"M={M} must be multiple of BLOCK_M={BLOCK_M}"
    assert N % BLOCK_N == 0, f"N={N} must be multiple of BLOCK_N={BLOCK_N}"
    assert K % BLOCK_K == 0, f"K={K} must be multiple of BLOCK_K={BLOCK_K}"

    num_k_tiles = K // BLOCK_K
    grid_m = M // BLOCK_M
    grid_n = N // BLOCK_N
    is_bf16 = in_dtype == "bf16"

    # B preshuffle constants
    N0 = N // 16
    K0_total = K // 16
    KLANE = 2
    NLANE = 16
    KPACK = 8
    B_STRIDE_KPACK = 1
    B_STRIDE_NLANE = KPACK  # 8
    B_STRIDE_KLANE = NLANE * KPACK  # 128
    B_STRIDE_K0 = KLANE * NLANE * KPACK  # 256
    B_STRIDE_N0 = K0_total * B_STRIDE_K0  # K0_total * 256

    def _in_elem_ty():
        return Textra.bf16() if is_bf16 else Textra.f16()

    def _out_elem_ty():
        return Textra.f32() if out_dtype == "f32" else Textra.bf16()

    def _wmma_op(result_type, a_vec, b_vec, acc, v8i16_ty):
        """Execute the correct WMMA op based on dtype, with bf16->i16 bitcast."""
        if is_bf16:
            a_i16 = vector.bitcast(v8i16_ty, a_vec)
            b_i16 = vector.bitcast(v8i16_ty, b_vec)
            return rocdl.wmma_f32_16x16x16_bf16(
                result_type,
                [a_i16, b_i16, arith.unwrap(acc)],
            )
        else:
            return rocdl.wmma_f32_16x16x16_f16(
                result_type,
                [arith.unwrap(a_vec), arith.unwrap(b_vec), arith.unwrap(acc)],
            )

    allocator = SmemAllocator(None, arch=gpu_arch)
    _state = {}

    class _WmmaGemmV4(flir.MlirModule):
        GPU_MODULE_NAME = "wmma_gemm_v4"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}">']

        def init_gpu_module(self):
            # LDS for A tile loading
            _state["lds_a"] = allocator.allocate_array(_in_elem_ty(), LDS_A_ELEMS)
            # LDS for C epilogue (reuse space after A is no longer needed)
            # We need BLOCK_M * 16 f32 = 2048 f32 = 8192 bytes per pass
            # A LDS is 8192 bf16 = 16384 bytes, so C epilogue fits
            _state["lds_c"] = allocator.allocate_array(Textra.f32(), C_EPILOGUE_ELEMS)
            allocator.finalize()

        @flir.kernel
        def wmma_gemm_v4_kernel(
            self: flir.T.i64,
            A: lambda: Textra.memref(S, S, _in_elem_ty()),
            B_shuf: lambda: Textra.memref(S, _in_elem_ty()),  # 1D flat pre-shuffled
            C: lambda: Textra.memref(S, S, _out_elem_ty()),
        ):
            # ---- Types ----
            in_ir_ty = ir.BF16Type.get() if is_bf16 else ir.F16Type.get()
            v8_in_ty = ir.VectorType.get([8], in_ir_ty)
            v8f32_ty = T.vec(8, T.f32)
            i16_ty = ir.IntegerType.get_signless(16)
            v8i16_ty = ir.VectorType.get([8], i16_ty)
            i32_ty = ir.IntegerType.get_signless(32)
            i64_ty = ir.IntegerType.get_signless(64)
            ptr_ty = ir.Type.parse("!llvm.ptr")
            v4i32_ty = ir.VectorType.get([4], ir.IntegerType.get_signless(32))

            # ---- Thread / block IDs ----
            tid = flir.thread_idx("x")
            pid = flir.block_idx("x")

            c32 = arith.index(32)
            c16 = arith.index(16)
            c8 = arith.index(8)
            wave_id = tid // c32
            lane = tid % c32
            lane16 = lane % c16
            base8 = (lane // c16) * c8  # 0 for lanes 0-15, 8 for lanes 16-31

            # ---- L2 cache swizzle (grouped block scheduling) ----
            effective_group_m = min(GROUP_M, grid_m)
            c_grid_n = arith.index(grid_n)
            c_group_m = arith.index(effective_group_m)
            num_pid_in_group = c_group_m * c_grid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * c_group_m
            group_size_m = c_group_m

            pid_in_group = pid % num_pid_in_group
            bid_m = first_pid_m + (pid_in_group % group_size_m)
            bid_n = pid_in_group // group_size_m

            # Wave position in WAVES_M x WAVES_N grid
            c_wn = arith.index(WAVES_N)
            wave_m = wave_id // c_wn
            wave_n = wave_id % c_wn

            # Global tile origins
            tile_m0 = bid_m * arith.index(BLOCK_M)
            tile_n0 = bid_n * arith.index(BLOCK_N)

            # ---- LDS setup ----
            lds_base = allocator.get_base()
            lds_a_view = _state["lds_a"](lds_base).get()
            lds_c_view = _state["lds_c"](lds_base).get()

            # ---- Buffer resources ----
            a_rsrc = buffer_ops.create_buffer_resource(_unwrap(A), max_size=True)
            b_rsrc = buffer_ops.create_buffer_resource(_unwrap(B_shuf), max_size=True)
            c_rsrc = buffer_ops.create_buffer_resource(_unwrap(C), max_size=True)

            # ---- Pre-compute A LDS store addresses (invariant across K) ----
            a_lds_addrs = []
            for al in range_constexpr(NUM_A_LOADS):
                a_lin = tid * c8 + arith.index(al * THREADS_PER_BLOCK * A_LOAD_VEC)
                a_load_row = a_lin // arith.index(BLOCK_K)
                a_load_col = a_lin % arith.index(BLOCK_K)
                a_lds_addrs.append(a_load_row * arith.index(BLOCK_K) + a_load_col)

            # ---- Inline asm for batched A global loads ----
            a_struct_ty = _llvm.StructType.get_literal([v4i32_ty] * NUM_A_LOADS)
            asm_a_load_lines = [f"s_clause {NUM_A_LOADS - 1}"]
            for i in range_constexpr(NUM_A_LOADS):
                asm_a_load_lines.append(
                    f"global_load_b128 ${i}, ${i + NUM_A_LOADS}, off"
                )
            asm_a_load_str = "\n".join(asm_a_load_lines)
            a_out_constraints = ",".join(["=&v"] * NUM_A_LOADS)
            a_in_constraints = ",".join(["v"] * NUM_A_LOADS)
            asm_a_constraints = f"{a_out_constraints},{a_in_constraints}"

            # ---- Extract base pointer for A inline asm ----
            elem_bytes = 2  # bf16 or f16
            a_base_i64 = _unwrap(
                _std_arith.IndexCastOp(
                    i64_ty,
                    _unwrap(
                        _std_memref.ExtractAlignedPointerAsIndexOp(_unwrap(A)).result
                    ),
                ).result
            )

            # ---- Helper: compute A global load addresses for a given k_base ----
            def _compute_a_load_addrs(k_base):
                addrs = []
                for al in range_constexpr(NUM_A_LOADS):
                    a_lin = tid * c8 + arith.index(al * THREADS_PER_BLOCK * A_LOAD_VEC)
                    a_load_row = a_lin // arith.index(BLOCK_K)
                    a_load_col = a_lin % arith.index(BLOCK_K)
                    g_a_row = tile_m0 + a_load_row
                    g_a_col = k_base + a_load_col
                    byte_off = (g_a_row * arith.index(K) + g_a_col) * arith.index(
                        elem_bytes
                    )
                    byte_off_i64 = _unwrap(
                        _std_arith.IndexCastOp(
                            i64_ty, _unwrap(arith.unwrap(byte_off))
                        ).result
                    )
                    addr_i64 = _unwrap(
                        _std_arith.AddIOp(a_base_i64, byte_off_i64).result
                    )
                    addr_ptr = _unwrap(_llvm.IntToPtrOp(ptr_ty, addr_i64).result)
                    addrs.append(addr_ptr)
                return addrs

            # ---- Helper: issue batched A loads ----
            def _issue_a_loads(addrs):
                return _llvm.inline_asm(
                    a_struct_ty,
                    addrs,
                    asm_a_load_str,
                    asm_a_constraints,
                    has_side_effects=True,
                )

            # ---- Helper: extract A load results and store to LDS ----
            def _store_a_to_lds(asm_result, lds_view):
                for al in range_constexpr(NUM_A_LOADS):
                    pos_attr = ir.DenseI64ArrayAttr.get([al])
                    v4i32_val = _llvm.ExtractValueOp(
                        v4i32_ty, asm_result, pos_attr
                    ).result
                    bf16_vec = vector.bitcast(v8_in_ty, v4i32_val)
                    vector.store(bf16_vec, lds_view, [a_lds_addrs[al]])

            # ---- Helper: load B tile from GMEM via buffer loads ----
            def _load_b_tile(k_tile_idx):
                """Load all B operands for this workgroup's tile from GMEM.

                Returns: b_vecs[rk][rn] = vector<8xbf16> for each (rk, rn)
                """
                b_vecs = []
                n0_base = tile_n0 // c16 + wave_n * arith.index(REG_N)
                klane = lane // c16

                for rk in range_constexpr(REG_K):
                    rk_vecs = []
                    k0 = k_tile_idx * arith.index(REG_K) + arith.index(rk)

                    for rn in range_constexpr(REG_N):
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

            # ---- Helper: load A operands from LDS for WMMA ----
            def _load_a_from_lds(lds_view):
                """Load A operands from LDS for all (rk, rm) combinations.

                Returns: a_vecs[rk][rm] = vector<8xbf16>
                """
                a_vecs = []
                for rk in range_constexpr(REG_K):
                    rk_vecs = []
                    col_base = arith.index(rk * WMMA_K) + base8
                    for rm in range_constexpr(REG_M):
                        row = (
                            wave_m * arith.index(REG_M * WMMA_M)
                            + arith.index(rm * WMMA_M)
                            + lane16
                        )
                        lds_idx = row * arith.index(BLOCK_K) + col_base
                        a_raw = vector.load_op(v8_in_ty, lds_view, [lds_idx])
                        rk_vecs.append(a_raw)
                    a_vecs.append(rk_vecs)
                return a_vecs

            # ---- Helper: execute WMMA compute ----
            def _do_compute(accs_in, b_vecs, a_vecs):
                """Execute REG_K * REG_M * REG_N WMMAs."""
                new_accs = list(accs_in)
                for rk in range_constexpr(REG_K):
                    for rm in range_constexpr(REG_M):
                        for rn in range_constexpr(REG_N):
                            idx = rm * REG_N + rn
                            new_accs[idx] = _wmma_op(
                                v8f32_ty,
                                a_vecs[rk][rm],
                                b_vecs[rk][rn],
                                new_accs[idx],
                                v8i16_ty,
                            )
                return new_accs

            # ========== Initialize accumulators ==========
            zero_acc = arith.constant_vector(0.0, v8f32_ty)
            accs = [zero_acc for _ in range_constexpr(REG_M * REG_N)]

            # ========================================================
            # K LOOP with single LDS buffer for A
            # ========================================================

            # ---- Prologue: load A tile 0 into LDS ----
            a_addrs_0 = _compute_a_load_addrs(arith.index(0))
            a_result_0 = _issue_a_loads(a_addrs_0)
            _llvm.inline_asm(
                res=None,
                operands_=[],
                asm_string="s_wait_loadcnt 0x0",
                constraints="",
                has_side_effects=True,
            )
            _store_a_to_lds(a_result_0, lds_a_view)
            _llvm.inline_asm(
                res=None,
                operands_=[],
                asm_string="s_wait_dscnt 0x0\ns_barrier_signal -1\ns_barrier_wait -1",
                constraints="",
                has_side_effects=True,
            )

            if num_k_tiles == 1:
                b_vecs = _load_b_tile(arith.index(0))
                a_vecs = _load_a_from_lds(lds_a_view)
                accs = _do_compute(accs, b_vecs, a_vecs)
            else:
                for kt in range(num_k_tiles - 1):
                    k_tile_idx = kt

                    # Load B from GMEM for current tile
                    b_vecs = _load_b_tile(k_tile_idx)

                    # Load A from LDS
                    a_vecs = _load_a_from_lds(lds_a_view)

                    # Prefetch A for next tile (overlap with compute)
                    k_base_next = (kt + arith.index(1)) * arith.index(BLOCK_K)
                    a_addrs_next = _compute_a_load_addrs(k_base_next)
                    a_result_next = _issue_a_loads(a_addrs_next)

                    # Compute
                    accs = _do_compute(accs, b_vecs, a_vecs)

                    # Barrier: ensure all waves done reading LDS
                    _llvm.inline_asm(
                        res=None,
                        operands_=[],
                        asm_string="s_wait_dscnt 0x0\ns_barrier_signal -1\ns_barrier_wait -1",
                        constraints="",
                        has_side_effects=True,
                    )

                    # Wait for next A loads
                    _llvm.inline_asm(
                        res=None,
                        operands_=[],
                        asm_string="s_wait_loadcnt 0x0",
                        constraints="",
                        has_side_effects=True,
                    )

                    # Store next A to LDS
                    _store_a_to_lds(a_result_next, lds_a_view)

                    # Barrier: ensure all waves done writing LDS
                    _llvm.inline_asm(
                        res=None,
                        operands_=[],
                        asm_string="s_wait_dscnt 0x0\ns_barrier_signal -1\ns_barrier_wait -1",
                        constraints="",
                        has_side_effects=True,
                    )

                # Epilogue: compute last tile
                b_vecs = _load_b_tile(arith.index(num_k_tiles - 1))
                a_vecs = _load_a_from_lds(lds_a_view)
                accs = _do_compute(accs, b_vecs, a_vecs)

            # ========== Store results ==========
            # Use scalar stores (same as v2) for now.
            # The vectorized LDS epilogue will be added in a follow-up
            # once we confirm the BLOCK_K=64 improvement.
            #
            # D layout: col-of-rows => lane t holds D[(t/16)*8+i][t%16]
            out_elem_bytes = 2 if out_dtype == "bf16" else 4
            c_layout_n = arith.index(N)
            for rm in range_constexpr(REG_M):
                for rn in range_constexpr(REG_N):
                    idx = rm * REG_N + rn
                    wmma_m_off = wave_m * arith.index(REG_M * WMMA_M) + arith.index(
                        rm * WMMA_M
                    )
                    wmma_n_off = wave_n * arith.index(REG_N * WMMA_N) + arith.index(
                        rn * WMMA_N
                    )

                    # D layout: col-of-rows => D[base8+i][lane16]
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
                ["wmma_gemm_v4", "wmma_gemm_v4_kernel"],
                grid_size=(total_blocks, c1, c1),
                block_size=(bk, c1, c1),
                kernel_operands=[A, B_shuf, C],
            )

    return _WmmaGemmV4()
