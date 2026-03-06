#!/usr/bin/env python3
"""Optimized WMMA GEMM kernel v2 for RDNA4 (gfx12xx, wave32).

Key improvement over v1: B loads directly from GMEM via buffer loads
(no LDS for B), eliminating the ds_load_u16 + ds_load_u16_d16_hi WAW
hazard that serialized B reads in v1.

Architecture: "preshuffle B + LDS A" (inspired by CK's wp_pipeline)
  - B is pre-shuffled in global memory so each thread's WMMA operand
    (8 bf16 values) is contiguous → single buffer_load_dwordx4 per tile
  - A goes through LDS with ping-pong double buffering
  - sched_group_barrier hints interleave VMEM reads, DS reads, and WMMA

Pre-shuffle B layout for WMMA wave32 (gfx1201):
  WMMA B operand: lane t loads B[(t/16)*8 + i][t%16] for i=0..7
  Layout: (N0, K0, KLane, NLane, KPack) where:
    N0 = N/16         (WMMA N tiles)
    K0 = K/16         (WMMA K tiles per REG_K step)
    KLane = 2         (t/16: lanes 0-15 vs 16-31)
    NLane = 16        (t%16: column within WMMA tile)
    KPack = 8         (8 bf16 values contiguous per thread)
  Strides: NLane=KPack=8, KLane=NLane*KPack=128, K0=KLane*NLane*KPack=256,
           N0=K0_total*256

WMMA data layout for wave32 (verified on gfx1201):
  A operand: "row-of-cols" -- lane t loads A[t%16][(t/16)*8 + i], i=0..7
  B operand: "col-of-rows" -- lane t loads B[(t/16)*8 + i][t%16], i=0..7
  D result:  "col-of-rows" -- lane t holds D[(t/16)*8 + i][t%16], i=0..7

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
REG_K = 2  # 2 WMMA-K steps per K tile

# Waves per workgroup arranged as WAVES_M x WAVES_N
WAVES_M = 2
WAVES_N = 2
NUM_WAVES = WAVES_M * WAVES_N  # 4

WAVE_SIZE = 32

# Derived block tile dimensions
BLOCK_M = WMMA_M * REG_M * WAVES_M  # 16*4*2 = 128
BLOCK_N = WMMA_N * REG_N * WAVES_N  # 16*4*2 = 128
BLOCK_K = WMMA_K * REG_K  # 16*2 = 32

THREADS_PER_BLOCK = NUM_WAVES * WAVE_SIZE  # 128

# A tile: cooperative global->LDS load
A_TILE_ELEMS = BLOCK_M * BLOCK_K  # 128 * 32 = 4096
# Each thread loads 8 bf16 = 16 bytes at a time (dwordx4)
A_LOAD_VEC = 8  # 8 bf16 = 16 bytes
NUM_A_LOADS = A_TILE_ELEMS // (THREADS_PER_BLOCK * A_LOAD_VEC)  # 4096 / (128*8) = 4

# B tile: each wave loads its own B operands directly from GMEM
# Per wave per K-step: REG_N * 32 threads need 8 bf16 each (but 32 threads in wave)
# Per wave per K-step: 32 threads * 1 load = 32 buffer_load_dwordx4
# Per wave for full tile: REG_K * REG_N buffer loads per thread
# Each thread loads 1 B WMMA operand per (rk, rn) combination
NUM_B_LOADS_PER_THREAD = REG_K * REG_N  # 2 * 4 = 8

# LDS: only A (single buffer for now, will add ping-pong)
LDS_A_ELEMS = BLOCK_M * BLOCK_K  # 4096

# L2 cache swizzle group size
GROUP_M = 8

# Number of inline-asm batched global loads for A
TOTAL_A_GLOBAL_LOADS = NUM_A_LOADS  # 4


def _unwrap(v):
    """Unwrap ArithValue to raw MLIR Value."""
    while hasattr(v, "_value"):
        v = v._value
    return v


# =============================================================================
# Host-side pre-shuffle for B
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

    WMMA B operand mapping (wave32):
      Lane t needs B[(t/16)*8 + i][t%16] for i=0..7
      - t/16 = KLane index (0 or 1)
      - t%16 = NLane index
      - i = KPack index (0..7)

    So B_shuffled[n0, k0, klane, nlane, kpack] = B[k0*16 + klane*8 + kpack, n0*16 + nlane]
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

    # Reshape B[K, N] -> B[K0, 16, N0, 16] -> B[K0, KLane, KPack, N0, NLane]
    B_reshaped = B_kn.reshape(K0, KLane, KPack, N0, NLane)
    # Permute to (N0, K0, KLane, NLane, KPack)
    B_shuffled = B_reshaped.permute(3, 0, 1, 4, 2).contiguous()
    return B_shuffled


def preshuffle_b_wmma_tiled(B_kn, *, block_n=BLOCK_N, block_k=BLOCK_K):
    """Pre-shuffle B with block-level tiling for better locality.

    Layout: (N_blk, K_blk, N0_local, K0_local, KLane, NLane, KPack)
    where:
      N_blk = N / BLOCK_N          (block tile along N)
      K_blk = K / BLOCK_K          (block tile along K)
      N0_local = BLOCK_N / 16      (WMMA N tiles within block)
      K0_local = BLOCK_K / 16 = REG_K  (WMMA K tiles within block)
      KLane = 2, NLane = 16, KPack = 8

    Each thread loads B_shuffled[n_blk, k_blk, rn, rk, klane, nlane, :]
    as 8 contiguous bf16 values.

    We flatten the inner 5 dims so the kernel sees a simple 1D buffer per
    (n_blk, k_blk) tile.
    """
    import torch

    K, N = B_kn.shape
    assert K % block_k == 0, f"K={K} must be multiple of block_k={block_k}"
    assert N % block_n == 0, f"N={N} must be multiple of block_n={block_n}"

    # First do the global preshuffle
    B_s = preshuffle_b_wmma(B_kn)
    # B_s shape: (N0, K0, KLane, NLane, KPack) where N0=N/16, K0=K/16

    N0 = N // 16
    K0 = K // 16
    N0_per_block = block_n // 16  # REG_N * WAVES_N
    K0_per_block = block_k // 16  # REG_K
    N_blk = N0 // N0_per_block
    K_blk = K0 // K0_per_block

    # Reshape: (N0, K0, 2, 16, 8) -> (N_blk, N0_local, K_blk, K0_local, 2, 16, 8)
    B_tiled = B_s.reshape(N_blk, N0_per_block, K_blk, K0_per_block, 2, 16, 8)
    # Permute to: (N_blk, K_blk, N0_local, K0_local, KLane, NLane, KPack)
    B_tiled = B_tiled.permute(0, 2, 1, 3, 4, 5, 6).contiguous()

    return B_tiled


# =============================================================================
# Kernel module
# =============================================================================


def create_wmma_gemm_v2_module(
    M: int, N: int, K: int, in_dtype="bf16", out_dtype="f32"
):
    """Create WMMA GEMM v2 module with B from GMEM.

    Args:
        M, N, K: matrix dimensions (must be multiples of BLOCK_M/N/K)
        in_dtype: "bf16" or "f16"
        out_dtype: "f32" or "bf16"

    B must be pre-shuffled using preshuffle_b_wmma() before calling this kernel.
    The kernel takes B_shuffled as a flat 1D memref.
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
    # Stride in elements for each dimension of B_shuffled (N0, K0, KLane, NLane, KPack)
    B_STRIDE_KPACK = 1
    B_STRIDE_NLANE = KPACK  # 8
    B_STRIDE_KLANE = NLANE * KPACK  # 128
    B_STRIDE_K0 = KLANE * NLANE * KPACK  # 256
    B_STRIDE_N0 = K0_total * B_STRIDE_K0  # K0_total * 256

    # Per-wave B tile: each wave handles REG_N WMMA N-tiles
    # Wave's N range: wave_n * REG_N * 16 .. (wave_n+1) * REG_N * 16
    # For each (rk, rn), one buffer_load_dwordx4 per thread

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

    class _WmmaGemmV2(flir.MlirModule):
        GPU_MODULE_NAME = "wmma_gemm_v2"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}">']

        def init_gpu_module(self):
            # Single LDS buffer for A (B is loaded from GMEM)
            _state["lds_a"] = allocator.allocate_array(_in_elem_ty(), LDS_A_ELEMS)
            allocator.finalize()

        @flir.kernel
        def wmma_gemm_v2_kernel(
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

            # ---- LDS setup (single buffer for A) ----
            lds_base = allocator.get_base()
            lds_a_view = _state["lds_a"](lds_base).get()

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
                """Compute NUM_A_LOADS flat pointers for A global loads."""
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

            # ---- Helper: issue batched A loads (no wait) ----
            def _issue_a_loads(addrs):
                """Issue NUM_A_LOADS batched global_load_b128 via inline asm."""
                return _llvm.inline_asm(
                    a_struct_ty,
                    addrs,
                    asm_a_load_str,
                    asm_a_constraints,
                    has_side_effects=True,
                )

            # ---- Helper: extract A load results and store to LDS ----
            def _store_a_to_lds(asm_result, lds_view):
                """Extract A results from asm, bitcast, store to LDS."""
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

                B_shuffled layout: (N0, K0, KLane, NLane, KPack)
                Thread t in wave w needs:
                  n0 = (tile_n0 + wave_n*REG_N*16 + rn*16) / 16
                     = tile_n0/16 + wave_n*REG_N + rn
                  k0 = k_tile_idx * REG_K + rk
                  klane = lane / 16  (= t/16 within wave)
                  nlane = lane % 16  (= t%16 within wave)
                  kpack = 0..7 (contiguous, loaded as one dwordx4)

                Element offset = n0 * B_STRIDE_N0 + k0 * B_STRIDE_K0
                               + klane * B_STRIDE_KLANE + nlane * B_STRIDE_NLANE + 0
                """
                b_vecs = []
                # Base n0 for this wave
                n0_base = tile_n0 // c16 + wave_n * arith.index(REG_N)
                # klane = lane / 16 (0 or 1)
                klane = lane // c16

                for rk in range_constexpr(REG_K):
                    rk_vecs = []
                    k0 = k_tile_idx * arith.index(REG_K) + arith.index(rk)

                    for rn in range_constexpr(REG_N):
                        n0 = n0_base + arith.index(rn)

                        # Compute flat element offset into B_shuffled
                        elem_off = (
                            n0 * arith.index(B_STRIDE_N0)
                            + k0 * arith.index(B_STRIDE_K0)
                            + klane * arith.index(B_STRIDE_KLANE)
                            + lane16 * arith.index(B_STRIDE_NLANE)
                        )

                        # buffer_load with dtype=f32, vec_width=4 gives 16 bytes (dwordx4)
                        # buffer_load internally converts offset to bytes: offset * sizeof(dtype)
                        # Our elem_off is in bf16 elements (2 bytes each)
                        # For f32 dtype, buffer_load does: byte_off = offset * 4
                        # We need: byte_off = elem_off * 2
                        # So: f32_offset = elem_off / 2
                        f32_off = elem_off // arith.index(2)
                        b_raw = buffer_ops.buffer_load(
                            b_rsrc,
                            f32_off,
                            vec_width=4,
                            dtype=ir.F32Type.get(),
                        )
                        # Bitcast 4xf32 (16 bytes) -> 8xbf16 or 8xf16
                        b_vec = vector.bitcast(v8_in_ty, b_raw)
                        rk_vecs.append(b_vec)
                    b_vecs.append(rk_vecs)
                return b_vecs

            # ---- Helper: load A operands from LDS for WMMA ----
            def _load_a_from_lds(lds_view):
                """Load A operands from LDS for all (rk, rm) combinations.

                A LDS layout: [BLOCK_M, BLOCK_K] row-major
                WMMA A operand: lane t loads A[t%16][(t/16)*8 + i], i=0..7
                  = contiguous 8 elements along K at row lane16, col base8..base8+7

                For wave_m's tiles:
                  row = wave_m * REG_M * 16 + rm * 16 + lane16
                  col = rk * 16 + base8

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
                """Execute REG_K * REG_M * REG_N WMMAs.

                b_vecs[rk][rn], a_vecs[rk][rm] -> accs[rm*REG_N + rn]
                """
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

            # ---- Inline asm barrier (no global_inv) ----
            def _asm_barrier():
                """Workgroup barrier via inline asm (avoids global_inv from gpu.barrier())."""
                _llvm.inline_asm(
                    res=None,
                    operands_=[],
                    asm_string="s_barrier_signal -1\ns_barrier_wait -1",
                    constraints="",
                    has_side_effects=True,
                )

            # ---- Scheduling hints ----
            def _hot_loop_schedule():
                """Emit sched_group_barrier hints to interleave VMEM, DS, WMMA.

                Instruction counts per iteration:
                  - 8 buffer_load_b128 (B from GMEM)
                  - 8 ds_load_b128 (A from LDS)
                  - 32 v_wmma (compute)
                  - 4 global_load_b128 (next A from GMEM)
                  - 4 ds_store_b128 (next A to LDS)
                """
                num_b_vmem = REG_K * REG_N  # 8
                num_a_dsrd = REG_K * REG_M  # 8
                num_a_vmem = NUM_A_LOADS  # 4
                num_a_dswr = NUM_A_LOADS  # 4

                # Preload: issue some B VMEM reads and A DS reads before first WMMA
                rocdl.sched_vmem(2)  # Start 2 B loads
                rocdl.sched_dsrd(2)  # Start 2 A LDS reads

                # Interleave: groups of 4 WMMAs with 1 B VMEM + 1 A DSRD
                remaining_vmem = num_b_vmem - 2  # 6
                remaining_dsrd = num_a_dsrd - 2  # 6
                wmma_groups = 8  # 32 / 4

                for gi in range_constexpr(wmma_groups):
                    rocdl.sched_mfma(4)  # 4 WMMAs
                    if gi < remaining_vmem:
                        rocdl.sched_vmem(1)
                    if gi < remaining_dsrd:
                        rocdl.sched_dsrd(1)
                    # Tail: interleave A GMEM loads and LDS writes
                    if gi >= wmma_groups - num_a_vmem:
                        rocdl.sched_vmem(1)  # A GMEM load
                    if gi >= wmma_groups - num_a_dswr:
                        rocdl.sched_dswr(1)  # A LDS write

                rocdl.sched_barrier(0)

            # ========== Initialize accumulators ==========
            zero_acc = arith.constant_vector(0.0, v8f32_ty)
            accs = [zero_acc for _ in range_constexpr(REG_M * REG_N)]

            # ========================================================
            # SW-PIPELINED K LOOP with ping-pong LDS for A
            # ========================================================

            # ---- Prologue: load A tile 0 into LDS ----
            a_addrs_0 = _compute_a_load_addrs(arith.index(0))
            a_result_0 = _issue_a_loads(a_addrs_0)
            # Wait for A loads
            _llvm.inline_asm(
                res=None,
                operands_=[],
                asm_string="s_wait_loadcnt 0x0",
                constraints="",
                has_side_effects=True,
            )
            _store_a_to_lds(a_result_0, lds_a_view)
            # Barrier after prologue A store — use inline asm to avoid global_inv
            _llvm.inline_asm(
                res=None,
                operands_=[],
                asm_string="s_wait_dscnt 0x0\ns_barrier_signal -1\ns_barrier_wait -1",
                constraints="",
                has_side_effects=True,
            )

            if num_k_tiles == 1:
                # Single tile: load B, load A from LDS, compute
                b_vecs = _load_b_tile(arith.index(0))
                a_vecs = _load_a_from_lds(lds_a_view)
                accs = _do_compute(accs, b_vecs, a_vecs)
            else:
                # ---- Main loop: dynamic range() -> scf.for ----
                # Use single LDS buffer (pong) since scf.for can't alternate
                # between two memrefs across iterations. We use a single buffer
                # with barrier-protected write-then-read pattern.
                # (Ping-pong would require loop-carried memref which scf.for
                #  doesn't support directly.)
                #
                # Pattern per iteration:
                #   1. Load B from GMEM (buffer loads)
                #   2. Load A from LDS (ds_load_b128)
                #   3. Compute WMMAs
                #   4. If not last: barrier, prefetch next A, wait, store to LDS, barrier

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

                    # Note: scheduling hints disabled — LLVM's scheduler produces
                    # better interleaving on its own (64→68 TFLOPS improvement)
                    pass

                    # Barrier: ensure all waves done reading LDS
                    # Use inline asm to avoid global_inv from gpu.barrier()
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

                    # Store next A to LDS (same buffer)
                    _store_a_to_lds(a_result_next, lds_a_view)

                    # Barrier: ensure all waves done writing LDS
                    _llvm.inline_asm(
                        res=None,
                        operands_=[],
                        asm_string="s_wait_dscnt 0x0\ns_barrier_signal -1\ns_barrier_wait -1",
                        constraints="",
                        has_side_effects=True,
                    )

                # ---- Epilogue: compute last tile ----
                b_vecs = _load_b_tile(arith.index(num_k_tiles - 1))
                a_vecs = _load_a_from_lds(lds_a_view)
                accs = _do_compute(accs, b_vecs, a_vecs)

            # ========== Store results (via buffer stores) ==========
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
                            accs[idx], static_position=[si], dynamic_position=[]
                        )
                        if out_dtype == "bf16":
                            val = arith.trunc_f(ir.BF16Type.get(), val)
                        # Use buffer store for better throughput
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
                ["wmma_gemm_v2", "wmma_gemm_v2_kernel"],
                grid_size=(total_blocks, c1, c1),
                block_size=(bk, c1, c1),
                kernel_operands=[A, B_shuf, C],
            )

    return _WmmaGemmV2()
