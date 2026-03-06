"""WMMA MoE GEMM kernels for RDNA4 (gfx12xx, wave32).

Implements Mixture-of-Experts GEMM with two stages:
  Stage 1: Fused gate+up projection with SiLU activation
    out[t, slot, :] = SiLU(X[t] @ W_gate[e]) * (X[t] @ W_up[e])
  Stage 2: Down projection with topk-weighted reduction
    Y[t] += weight[t,slot] * A2[t,slot] @ W_down[e]

Uses WMMA instructions (v_wmma_f32_16x16x16_bf16) with f32 accumulation.
BF16 input/output only (RDNA4 WMMA constraint).

The kernel reads tokens via sorted_token_ids from MoE routing, supporting
the same routing format as the MFMA-based moe_gemm_2stage.py.
"""

import os
import functools

import flydsl
from flydsl.dialects.ext import (
    flir,
    arith,
    gpu,
    buffer_ops,
    vector,
    rocdl,
    scf,
    memref,
    llvm,
)
from flydsl.dialects.ext.python_control_flow import range_constexpr
from flydsl.runtime.device import get_rocm_arch
from flydsl.lang.ir.types import T as I
from flydsl.kernels.kernels_common import stream_ptr_to_async_token

from _mlir import ir
import _mlir.extras.types as T


def _unwrap(v):
    while hasattr(v, "_value"):
        v = v._value
    return v


# =============================================================================
# Host-side pre-shuffle for WMMA
# =============================================================================


def preshuffle_w_wmma(W):
    """Pre-shuffle weight W[N, K] for WMMA B operand layout.

    Same as preshuffle_b_wmma but operates on W[N,K] (weight matrix).
    W is stored as [N, K], which when used as B in GEMM is B[K,N]^T.
    We need B[K,N] for preshuffle_b, so we transpose first.

    Output shape: [N0, K0, KLane, NLane, KPack]
    """
    import torch

    # W is [N, K], we need B[K, N] for preshuffle
    B_kn = W.t().contiguous()
    K, N = B_kn.shape
    assert K % 16 == 0 and N % 16 == 0, f"K={K}, N={N} must be multiples of 16"
    N0 = N // 16
    K0 = K // 16
    B_reshaped = B_kn.reshape(K0, 2, 8, N0, 16)
    B_shuffled = B_reshaped.permute(3, 0, 1, 4, 2).contiguous()
    return B_shuffled


def preshuffle_a_wmma(A_mk):
    """Pre-shuffle A[M,K] for WMMA A operand layout."""
    import torch

    M, K = A_mk.shape
    assert M % 16 == 0 and K % 16 == 0
    M0 = M // 16
    K0 = K // 16
    A_reshaped = A_mk.reshape(M0, 16, K0, 2, 8)
    A_shuffled = A_reshaped.permute(0, 2, 3, 1, 4).contiguous()
    return A_shuffled


# =============================================================================
# Stage 1: Gate+Up projection with SiLU
# =============================================================================


@functools.lru_cache(maxsize=64)
def compile_wmma_moe_gemm1(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int = 16,
    tile_n: int = 128,
    tile_k: int = 32,
    doweight_stage1: bool = False,
    in_dtype: str = "bf16",
    out_dtype: str = "bf16",
):
    """Compile WMMA MoE Stage 1 kernel for RDNA4.

    Stage1 computes: out[t, slot] = SiLU(X[t] @ W_gate[e]) * (X[t] @ W_up[e])

    Args:
        model_dim: Input dimension (K).
        inter_dim: Hidden dimension per expert (N for gate/up each).
        experts: Number of experts.
        topk: Number of experts per token.
        tile_m: M tile size (tokens per block). Should be small for MoE.
        tile_n: N tile size (output dim per block).
        tile_k: K tile size.
        doweight_stage1: Apply topk weights in stage1 output.
        in_dtype: "bf16" (only bf16 supported on RDNA4 WMMA).
        out_dtype: "bf16" (output dtype).

    Returns:
        Compiled executable:
          exe(out, x, w, scale_x, scale_w,
              sorted_token_ids, expert_ids, sorted_weights, max_token_ids,
              tokens, inter, k, num_expert_blocks, stream_ptr)
    """
    assert in_dtype == "bf16", f"WMMA MoE only supports bf16, got {in_dtype}"
    assert out_dtype in ("bf16",), f"out_dtype must be 'bf16', got {out_dtype}"

    gpu_arch = get_rocm_arch()

    WMMA_M = 16
    WMMA_N = 16
    WMMA_K = 16
    WAVE_SIZE = 32

    assert tile_m % WMMA_M == 0, f"tile_m ({tile_m}) must be multiple of {WMMA_M}"
    assert tile_n % WMMA_N == 0, f"tile_n ({tile_n}) must be multiple of {WMMA_N}"
    assert tile_k % WMMA_K == 0, f"tile_k ({tile_k}) must be multiple of {WMMA_K}"

    reg_m = tile_m // WMMA_M  # WMMA tiles along M per block
    reg_n = tile_n // WMMA_N  # WMMA tiles along N per block
    reg_k = tile_k // WMMA_K

    # For MoE with small tile_m, use fewer waves
    # Typical: tile_m=16 -> 1 wave on M, tile_n=128 -> multiple waves on N
    if reg_m >= 2 and reg_n >= 2:
        waves_m, waves_n = min(reg_m, 2), min(reg_n, 2)
    elif reg_n >= 4:
        waves_m, waves_n = 1, 4
    elif reg_n >= 2:
        waves_m, waves_n = 1, 2
    else:
        waves_m, waves_n = 1, 1

    NUM_WAVES = waves_m * waves_n
    THREADS_PER_BLOCK = NUM_WAVES * WAVE_SIZE
    wave_reg_m = reg_m // waves_m
    wave_reg_n = reg_n // waves_n

    num_k_tiles = model_dim // tile_k

    DYN = ir.ShapedType.get_dynamic_size()

    # Preshuffle stride constants for W
    # W is preshuffled as [N0, K0, KLane=2, NLane=16, KPack=8]
    K0_total = model_dim // 16
    W_KPACK = 8
    W_STRIDE_NLANE = W_KPACK
    W_STRIDE_KLANE = 16 * W_KPACK
    W_STRIDE_K0 = 2 * 16 * W_KPACK
    # W_STRIDE_N0 depends on K dimension of that expert's weight

    # For stage1, W1 has shape [E, 2*inter_dim, model_dim]
    # After preshuffle per expert: [E, 2*inter_dim//16, K0, 2, 16, 8]
    # Flattened: expert_offset = expert_id * (2*inter_dim//16) * K0 * 2 * 16 * 8
    w1_n = 2 * inter_dim
    W1_N0_per_expert = w1_n // 16
    W1_STRIDE_N0 = K0_total * W_STRIDE_K0

    module_name = f"wmma_moe1_bf16_t{tile_m}x{tile_n}x{tile_k}"

    class _MOE1(flir.MlirModule):
        GPU_MODULE_NAME = module_name
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}">']

        @flir.kernel
        def moe_gemm1(
            self: flir.T.i64,
            arg_out: lambda: T.memref(DYN, T.bf16()),
            arg_x: lambda: T.memref(DYN, T.bf16()),
            arg_w: lambda: T.memref(DYN, T.bf16()),
            arg_scale_x: lambda: T.memref(DYN, T.f32()),
            arg_scale_w: lambda: T.memref(DYN, T.f32()),
            arg_sorted_token_ids: lambda: T.memref(DYN, T.i32()),
            arg_expert_ids: lambda: T.memref(DYN, T.i32()),
            arg_sorted_weights: lambda: T.memref(DYN, T.f32()),
            arg_max_token_ids: lambda: T.memref(DYN, T.i32()),
            tokens_in: lambda: T.index(),
            inter_in: lambda: T.index(),
            k_in: lambda: T.index(),
            size_expert_ids_in: lambda: T.index(),
        ):
            bf16 = ir.BF16Type.get()
            f32 = ir.F32Type.get()
            i16_ty = ir.IntegerType.get_signless(16)
            i32 = ir.IntegerType.get_signless(32)
            v8bf16_ty = ir.VectorType.get([8], bf16)
            v8i16_ty = ir.VectorType.get([8], i16_ty)
            v8f32_ty = I.vec(8, I.f32)

            tid = flir.thread_idx("x")

            # Grid: blockIdx.x -> N tiles, blockIdx.y -> M tiles (expert blocks)
            by = flir.block_idx("x")  # N tile
            bx = flir.block_idx("y")  # M tile (expert block)

            c32 = arith.index(32)
            c16 = arith.index(16)
            c8 = arith.index(8)
            wave_id = tid // c32
            lane = tid % c32
            lane16 = lane % c16
            klane = lane // c16

            # Block validity check
            bx_m = bx * arith.index(tile_m)
            maxids_rsrc = buffer_ops.create_buffer_resource(
                _unwrap(arg_max_token_ids),
                max_size=False,
                num_records_bytes=arith.constant(4, type=i32),
            )
            max_token_id_i32 = buffer_ops.buffer_load(
                maxids_rsrc,
                arith.index(0),
                vec_width=1,
                dtype=i32,
            )
            bx_m_i32 = arith.index_cast(i32, bx_m)
            blk_valid = arith.cmpu(bx_m_i32, max_token_id_i32, "ult")

            _if_blk = scf.IfOp(blk_valid)
            with _if_blk.then():
                # Buffer resources
                sorted_rsrc = buffer_ops.create_buffer_resource(
                    _unwrap(arg_sorted_token_ids),
                    max_size=True,
                )
                sorted_w_rsrc = buffer_ops.create_buffer_resource(
                    _unwrap(arg_sorted_weights),
                    max_size=True,
                )
                expert_rsrc = buffer_ops.create_buffer_resource(
                    _unwrap(arg_expert_ids),
                    max_size=True,
                )
                x_rsrc = buffer_ops.create_buffer_resource(
                    _unwrap(arg_x), max_size=True
                )
                w_rsrc = buffer_ops.create_buffer_resource(
                    _unwrap(arg_w), max_size=True
                )
                out_rsrc = buffer_ops.create_buffer_resource(
                    _unwrap(arg_out), max_size=True
                )

                # Get expert ID for this block
                expert_i32 = buffer_ops.buffer_load(
                    expert_rsrc, bx, vec_width=1, dtype=i32
                )
                expert_idx = arith.index_cast(ir.IndexType.get(), expert_i32)

                # Wave layout
                c_wn = arith.index(waves_n)
                wave_m = wave_id // c_wn
                wave_n = wave_id % c_wn

                # N tile base (in output space: 2*inter_dim, first half = gate, second = up)
                tile_n0 = by * arith.index(tile_n)

                # Expert weight base offset (in preshuffled element space)
                # W1 preshuffled: [E, N0_per_expert, K0, KLane=2, NLane=16, KPack=8]
                expert_w_base = expert_idx * arith.index(
                    W1_N0_per_expert * K0_total * W_STRIDE_K0
                )

                mask24 = arith.constant(0xFFFFFF, type=i32)
                tokens_i32 = arith.index_cast(i32, tokens_in)

                # Load X row for each M tile row via sorted_token_ids
                # For each row in [0, tile_m), decode token_id from sorted_ids
                def _load_x_row(m_local, k_tile_idx):
                    """Load one A tile's worth of data for row m_local at k_tile k_tile_idx.

                    Returns list of reg_k lists of wave_reg_m a_vecs.
                    We load from X[token_id, k] where token_id comes from sorted_token_ids.
                    X is NOT preshuffled - it's a plain [tokens, model_dim] bf16 tensor.
                    We need to load 8 bf16 values per lane for WMMA.
                    """
                    sorted_row = bx_m + m_local
                    fused_i = buffer_ops.buffer_load(
                        sorted_rsrc,
                        sorted_row,
                        vec_width=1,
                        dtype=i32,
                    )
                    t_raw = arith.andi(fused_i, mask24)
                    t_valid = arith.cmpu(t_raw, tokens_i32, "ult")
                    t_safe = arith.select(
                        t_valid,
                        arith.index_cast(ir.IndexType.get(), t_raw),
                        arith.index(0),
                    )

                    # X row base offset
                    x_row_base = t_safe * k_in

                    # Load WMMA A operands: [KLane=2, MLane=16, KPack=8]
                    # For wave32 WMMA: lane16 selects M row, klane selects K half
                    # But X is row-major [tokens, K], not preshuffled.
                    # We need to gather 8 contiguous elements starting at:
                    #   X[token_id, k_tile*tile_k + rk*16 + klane*8]
                    # where lane16 is the M-lane (but m_local IS the M index)
                    # Actually for a single row, all lanes in the same M read the same row.
                    # The WMMA expects: lane i gets A[m_in_16tile, k_start + i_in_8], etc.
                    # This is complicated without preshuffle...

                    # Let's just load 8 bf16 values per lane from X
                    a_vecs = []
                    for rk in range_constexpr(reg_k):
                        k_base = (
                            k_tile_idx * arith.index(tile_k)
                            + arith.index(rk * 16)
                            + klane * c8
                        )
                        elem_off = x_row_base + k_base
                        f32_off = elem_off // arith.index(2)
                        a_raw = buffer_ops.buffer_load(
                            x_rsrc,
                            f32_off,
                            vec_width=4,
                            dtype=f32,
                        )
                        a_vec = vector.bitcast(v8bf16_ty, a_raw)
                        a_vecs.append(a_vec)
                    return a_vecs, t_safe, fused_i

                def _load_w_tile(k_tile_idx, n_offset):
                    """Load B (weight) operands from pre-shuffled GMEM.

                    n_offset is the absolute N offset into W1[E, 2*inter_dim, K].
                    """
                    w_vecs = []
                    n0_base = n_offset // c16 + wave_n * arith.index(wave_reg_n)
                    for rk in range_constexpr(reg_k):
                        rk_vecs = []
                        k0 = k_tile_idx * arith.index(reg_k) + arith.index(rk)
                        for rn in range_constexpr(wave_reg_n):
                            n0 = n0_base + arith.index(rn)
                            elem_off = (
                                expert_w_base
                                + n0 * arith.index(W1_STRIDE_N0)
                                + k0 * arith.index(W_STRIDE_K0)
                                + klane * arith.index(W_STRIDE_KLANE)
                                + lane16 * arith.index(W_STRIDE_NLANE)
                            )
                            f32_off = elem_off // arith.index(2)
                            w_raw = buffer_ops.buffer_load(
                                w_rsrc,
                                f32_off,
                                vec_width=4,
                                dtype=f32,
                            )
                            w_vec = vector.bitcast(v8bf16_ty, w_raw)
                            rk_vecs.append(w_vec)
                        w_vecs.append(rk_vecs)
                    return w_vecs

                def _wmma_bf16(result_type, a_vec, b_vec, acc):
                    a_i16 = vector.bitcast(v8i16_ty, a_vec)
                    b_i16 = vector.bitcast(v8i16_ty, b_vec)
                    return rocdl.wmma_f32_16x16x16_bf16(
                        result_type, [a_i16, b_i16, arith.unwrap(acc)]
                    )

                # ---- Main compute ----
                # For each M-row in the tile, compute gate and up projections
                # This is a simpler approach: iterate over M rows
                # For tile_m=16 (1 WMMA M tile), each wave handles wave_reg_n N tiles

                # Initialize gate and up accumulators
                zero_acc = arith.constant_vector(0.0, v8f32_ty)
                gate_accs = [zero_acc for _ in range_constexpr(wave_reg_m * wave_reg_n)]
                up_accs = [zero_acc for _ in range_constexpr(wave_reg_m * wave_reg_n)]

                # We need X preshuffled for WMMA. Since X is dynamic (routed tokens),
                # we can't preshuffle it on host. Instead, we load X rows into registers
                # and use them as-is - but WMMA expects a specific register layout.
                #
                # For RDNA4 WMMA bf16: A operand has 8 bf16 per lane in specific pattern.
                # Without preshuffle, we need to do a gather from X rows.
                #
                # Simpler approach for MoE: Use X as-is with natural layout.
                # Each lane loads 8 contiguous bf16 from X[token, k_offset].
                # The WMMA will interpret this as the A operand.
                # This works because the preshuffle just rearranges which lane gets which data,
                # and for a single-row M=1 case, all 16 M-lanes read from the same row.
                #
                # For tile_m=16, we have one WMMA M-tile. Lanes 0-15 each need data from
                # a different M row. With routed tokens, row i maps to sorted_token_ids[bx_m+i].
                #
                # The WMMA A layout for wave32 is:
                #   lane l (0-15): reads row l, columns [klane*8 .. klane*8+7]
                #   lane l (16-31): reads row (l-16), columns [klane*8+8 .. klane*8+15]
                # Wait - klane = lane // 16, so lane 0-15 have klane=0, lane 16-31 have klane=1.
                # So lane l reads: row = l % 16, col_start = (l//16) * 8 = klane * 8
                #
                # For X[token, k]: row = token_id, col = k_offset + klane*8
                # Each of the 16 M-rows in a WMMA tile maps to a different token.
                # Lane (lane16) determines which M-row (token) this lane processes.

                # Load X for the full M tile
                # Each lane needs its own token_id based on lane16 + wave_m position
                # But we can't per-lane index sorted_token_ids... we need to load it uniformly.
                #
                # Actually, buffer_load on sorted_token_ids with an index computed from
                # (bx_m + wave_m*wave_reg_m*16 + rm*16 + lane16) would give each lane
                # a different token_id. But that won't work because buffer_load returns
                # the same value to all lanes.
                #
                # We need EACH LANE to load its own token's X data.
                # With buffer_load, the offset is per-lane (VGPR), so:
                #   sorted_row = bx_m + wave_m*wave_reg_m*16 + rm*16 + lane16
                # This is a per-lane value! So each lane reads a different sorted_token_id.

                # K-loop: iterate over K tiles
                for kt in range_constexpr(num_k_tiles):
                    kt_idx = arith.index(kt)

                    # Load W tile for gate (first half of 2*inter_dim)
                    w_gate = _load_w_tile(kt_idx, tile_n0)
                    # Load W tile for up (second half: offset by inter_dim)
                    w_up = _load_w_tile(kt_idx, tile_n0 + arith.index(inter_dim))

                    # Load X for this K tile
                    # Each lane loads its own token's data
                    for rm in range_constexpr(wave_reg_m):
                        # M row in tile for this wave and register
                        m_local = (
                            wave_m * arith.index(wave_reg_m * WMMA_M)
                            + arith.index(rm * WMMA_M)
                            + lane16
                        )
                        sorted_row = bx_m + m_local

                        # Load token ID for this lane's row
                        fused_i = buffer_ops.buffer_load(
                            sorted_rsrc,
                            sorted_row,
                            vec_width=1,
                            dtype=i32,
                        )
                        t_raw = arith.andi(fused_i, mask24)
                        t_valid = arith.cmpu(t_raw, tokens_i32, "ult")
                        t_safe = arith.select(
                            t_valid,
                            arith.index_cast(ir.IndexType.get(), t_raw),
                            arith.index(0),
                        )
                        x_row_base = t_safe * k_in

                        for rk in range_constexpr(reg_k):
                            k_base = (
                                kt_idx * arith.index(tile_k)
                                + arith.index(rk * 16)
                                + klane * c8
                            )
                            x_off = x_row_base + k_base
                            f32_off = x_off // arith.index(2)
                            a_raw = buffer_ops.buffer_load(
                                x_rsrc,
                                f32_off,
                                vec_width=4,
                                dtype=f32,
                            )
                            a_vec = vector.bitcast(v8bf16_ty, a_raw)

                            # Compute gate and up for this A tile
                            for rn in range_constexpr(wave_reg_n):
                                acc_idx = rm * wave_reg_n + rn
                                gate_accs[acc_idx] = _wmma_bf16(
                                    v8f32_ty, a_vec, w_gate[rk][rn], gate_accs[acc_idx]
                                )
                                up_accs[acc_idx] = _wmma_bf16(
                                    v8f32_ty, a_vec, w_up[rk][rn], up_accs[acc_idx]
                                )

                # Apply SiLU(gate) * up and store
                # WMMA C output layout (wave32):
                #   lane16 (= lane % 16) -> N column
                #   klane*8 + si -> M row within the 16x16 tile
                # So for each si, we must look up the token for M-row = klane*8+si
                base8 = klane * c8

                for rm in range_constexpr(wave_reg_m):
                    m_local_base = wave_m * arith.index(
                        wave_reg_m * WMMA_M
                    ) + arith.index(rm * WMMA_M)

                    for rn in range_constexpr(wave_reg_n):
                        acc_idx = rm * wave_reg_n + rn
                        wmma_n_off = wave_n * arith.index(
                            wave_reg_n * WMMA_N
                        ) + arith.index(rn * WMMA_N)

                        for si in range_constexpr(8):
                            # M-row within WMMA tile = klane*8 + si
                            m_row = m_local_base + base8 + arith.index(si)
                            sorted_row = bx_m + m_row

                            # Look up token for this M-row
                            fused_store = buffer_ops.buffer_load(
                                sorted_rsrc,
                                sorted_row,
                                vec_width=1,
                                dtype=i32,
                            )
                            t_store_raw = arith.andi(fused_store, mask24)
                            s_store_raw = arith.shrui(
                                fused_store, arith.constant(24, type=i32)
                            )
                            t_store_valid = arith.cmpu(t_store_raw, tokens_i32, "ult")
                            t_store_idx = arith.index_cast(
                                ir.IndexType.get(), t_store_raw
                            )
                            s_store_idx = arith.index_cast(
                                ir.IndexType.get(), s_store_raw
                            )

                            gate_val = vector.extract(
                                gate_accs[acc_idx],
                                static_position=[si],
                                dynamic_position=[],
                            )
                            up_val = vector.extract(
                                up_accs[acc_idx],
                                static_position=[si],
                                dynamic_position=[],
                            )

                            # SiLU(gate) = gate * sigmoid(gate)
                            # sigmoid(x) = 1 / (1 + exp(-x))
                            # Using fast intrinsics:
                            neg_gate = gate_val * arith.constant(
                                -1.4426950408889634, type=f32
                            )
                            emu = llvm.call_intrinsic(
                                f32, "llvm.amdgcn.exp2.f32", [neg_gate], [], []
                            )
                            den = arith.constant(1.0, type=f32) + emu
                            sig = llvm.call_intrinsic(
                                f32, "llvm.amdgcn.rcp.f32", [den], [], []
                            )
                            silu_gate = gate_val * sig

                            # SiLU(gate) * up
                            result_f32 = silu_gate * up_val

                            if doweight_stage1:
                                wt_val = buffer_ops.buffer_load(
                                    sorted_w_rsrc,
                                    sorted_row,
                                    vec_width=1,
                                    dtype=f32,
                                )
                                result_f32 = result_f32 * wt_val

                            result_bf16 = arith.trunc_f(bf16, result_f32)

                            # Store to out[token_id, slot, inter_dim]
                            # out shape: [tokens, topk, inter_dim], row-major
                            # N column = lane16
                            g_col = tile_n0 + wmma_n_off + lane16

                            c_topk_idx = arith.index(topk)
                            c_inter_idx = arith.index(inter_dim)
                            out_off = (
                                t_store_idx * c_topk_idx + s_store_idx
                            ) * c_inter_idx + g_col

                            # Guard: only store if token is valid
                            _if_valid = scf.IfOp(t_store_valid)
                            with _if_valid.then():
                                buffer_ops.buffer_store(result_bf16, out_rsrc, out_off)

        @flir.jit
        def __call__(
            self: flir.T.i64,
            arg_out: lambda: T.memref(DYN, T.bf16()),
            arg_x: lambda: T.memref(DYN, T.bf16()),
            arg_w: lambda: T.memref(DYN, T.bf16()),
            arg_scale_x: lambda: T.memref(DYN, T.f32()),
            arg_scale_w: lambda: T.memref(DYN, T.f32()),
            arg_sorted_token_ids: lambda: T.memref(DYN, T.i32()),
            arg_expert_ids: lambda: T.memref(DYN, T.i32()),
            arg_sorted_weights: lambda: T.memref(DYN, T.f32()),
            arg_max_token_ids: lambda: T.memref(DYN, T.i32()),
            tokens_in: lambda: T.index(),
            inter_in: lambda: T.index(),
            k_in: lambda: T.index(),
            size_expert_ids_in: lambda: T.index(),
            stream_ptr: lambda: I.i64,
        ):
            bdx = arith.constant(THREADS_PER_BLOCK, index=True)
            c1 = arith.constant(1, index=True)
            gx = inter_in / arith.index(tile_n)
            gy = size_expert_ids_in

            stream_token = stream_ptr_to_async_token(stream_ptr)
            flir.gpu_ext.LaunchFuncOp(
                [module_name, "moe_gemm1"],
                grid_size=(gx, gy, c1),
                block_size=(bdx, c1, c1),
                kernel_operands=[
                    arg_out,
                    arg_x,
                    arg_w,
                    arg_scale_x,
                    arg_scale_w,
                    arg_sorted_token_ids,
                    arg_expert_ids,
                    arg_sorted_weights,
                    arg_max_token_ids,
                    tokens_in,
                    inter_in,
                    k_in,
                    size_expert_ids_in,
                ],
                async_dependencies=[stream_token],
            )

    m = _MOE1()
    return flydsl.compile(m)


# =============================================================================
# Stage 2: Down projection with weighted reduction
# =============================================================================


@functools.lru_cache(maxsize=64)
def compile_wmma_moe_gemm2(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int = 16,
    tile_n: int = 128,
    tile_k: int = 32,
    doweight_stage2: bool = True,
    in_dtype: str = "bf16",
    out_dtype: str = "bf16",
    accumulate: bool = False,
):
    """Compile WMMA MoE Stage 2 kernel for RDNA4.

    Stage2 computes: Y[t] += weight * A2[t,slot] @ W_down[e]
    When accumulate=False, writes to out[t, slot, model_dim] without reduction.

    Args:
        model_dim: Output dimension (N).
        inter_dim: Input dimension (K).
        experts: Number of experts.
        topk: Number of experts per token.
        tile_m, tile_n, tile_k: Tile sizes.
        doweight_stage2: Apply topk weights.
        in_dtype: "bf16".
        out_dtype: "bf16".
        accumulate: If True, atomically add to output. If False, write per-(token,slot).

    Returns:
        Compiled executable with same interface as stage1.
    """
    assert in_dtype == "bf16", f"WMMA MoE only supports bf16, got {in_dtype}"
    assert out_dtype in ("bf16",), f"out_dtype must be 'bf16', got {out_dtype}"
    # For RDNA4, atomic bf16 adds are not widely supported.
    # Use accumulate=False and do reduction separately.
    if accumulate:
        raise NotImplementedError(
            "Atomic accumulation not yet supported for WMMA MoE. "
            "Use accumulate=False with separate reduction."
        )

    gpu_arch = get_rocm_arch()

    WMMA_M = 16
    WMMA_N = 16
    WMMA_K = 16
    WAVE_SIZE = 32

    assert tile_m % WMMA_M == 0
    assert tile_n % WMMA_N == 0
    assert tile_k % WMMA_K == 0

    reg_m = tile_m // WMMA_M
    reg_n = tile_n // WMMA_N
    reg_k = tile_k // WMMA_K

    if reg_m >= 2 and reg_n >= 2:
        waves_m, waves_n = min(reg_m, 2), min(reg_n, 2)
    elif reg_n >= 4:
        waves_m, waves_n = 1, 4
    elif reg_n >= 2:
        waves_m, waves_n = 1, 2
    else:
        waves_m, waves_n = 1, 1

    NUM_WAVES = waves_m * waves_n
    THREADS_PER_BLOCK = NUM_WAVES * WAVE_SIZE
    wave_reg_m = reg_m // waves_m
    wave_reg_n = reg_n // waves_n

    num_k_tiles = inter_dim // tile_k

    DYN = ir.ShapedType.get_dynamic_size()

    # W2 preshuffle constants: W2[E, model_dim, inter_dim]
    K0_total = inter_dim // 16
    W_KPACK = 8
    W_STRIDE_NLANE = W_KPACK
    W_STRIDE_KLANE = 16 * W_KPACK
    W_STRIDE_K0 = 2 * 16 * W_KPACK
    W2_N0_per_expert = model_dim // 16
    W2_STRIDE_N0 = K0_total * W_STRIDE_K0

    module_name = f"wmma_moe2_bf16_t{tile_m}x{tile_n}x{tile_k}"

    class _MOE2(flir.MlirModule):
        GPU_MODULE_NAME = module_name
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}">']

        @flir.kernel
        def moe_gemm2(
            self: flir.T.i64,
            arg_out: lambda: T.memref(DYN, T.bf16()),
            arg_a2: lambda: T.memref(DYN, T.bf16()),
            arg_w: lambda: T.memref(DYN, T.bf16()),
            arg_scale_a2: lambda: T.memref(DYN, T.f32()),
            arg_scale_w: lambda: T.memref(DYN, T.f32()),
            arg_sorted_token_ids: lambda: T.memref(DYN, T.i32()),
            arg_expert_ids: lambda: T.memref(DYN, T.i32()),
            arg_sorted_weights: lambda: T.memref(DYN, T.f32()),
            arg_max_token_ids: lambda: T.memref(DYN, T.i32()),
            tokens_in: lambda: T.index(),
            model_in: lambda: T.index(),
            k_in: lambda: T.index(),
            size_expert_ids_in: lambda: T.index(),
        ):
            bf16 = ir.BF16Type.get()
            f32 = ir.F32Type.get()
            i16_ty = ir.IntegerType.get_signless(16)
            i32 = ir.IntegerType.get_signless(32)
            v8bf16_ty = ir.VectorType.get([8], bf16)
            v8i16_ty = ir.VectorType.get([8], i16_ty)
            v8f32_ty = I.vec(8, I.f32)

            tid = flir.thread_idx("x")
            by = flir.block_idx("x")  # N tile (model_dim)
            bx = flir.block_idx("y")  # M tile (expert block)

            c32 = arith.index(32)
            c16 = arith.index(16)
            c8 = arith.index(8)
            wave_id = tid // c32
            lane = tid % c32
            lane16 = lane % c16
            klane = lane // c16

            bx_m = bx * arith.index(tile_m)
            maxids_rsrc = buffer_ops.create_buffer_resource(
                _unwrap(arg_max_token_ids),
                max_size=False,
                num_records_bytes=arith.constant(4, type=i32),
            )
            max_token_id_i32 = buffer_ops.buffer_load(
                maxids_rsrc,
                arith.index(0),
                vec_width=1,
                dtype=i32,
            )
            bx_m_i32 = arith.index_cast(i32, bx_m)
            blk_valid = arith.cmpu(bx_m_i32, max_token_id_i32, "ult")

            _if_blk = scf.IfOp(blk_valid)
            with _if_blk.then():
                sorted_rsrc = buffer_ops.create_buffer_resource(
                    _unwrap(arg_sorted_token_ids), max_size=True
                )
                sorted_w_rsrc = buffer_ops.create_buffer_resource(
                    _unwrap(arg_sorted_weights), max_size=True
                )
                expert_rsrc = buffer_ops.create_buffer_resource(
                    _unwrap(arg_expert_ids), max_size=True
                )
                a2_rsrc = buffer_ops.create_buffer_resource(
                    _unwrap(arg_a2), max_size=True
                )
                w_rsrc = buffer_ops.create_buffer_resource(
                    _unwrap(arg_w), max_size=True
                )
                out_rsrc = buffer_ops.create_buffer_resource(
                    _unwrap(arg_out), max_size=True
                )

                expert_i32 = buffer_ops.buffer_load(
                    expert_rsrc, bx, vec_width=1, dtype=i32
                )
                expert_idx = arith.index_cast(ir.IndexType.get(), expert_i32)

                c_wn = arith.index(waves_n)
                wave_m = wave_id // c_wn
                wave_n = wave_id % c_wn
                tile_n0 = by * arith.index(tile_n)

                expert_w_base = expert_idx * arith.index(
                    W2_N0_per_expert * K0_total * W_STRIDE_K0
                )
                mask24 = arith.constant(0xFFFFFF, type=i32)
                tokens_i32 = arith.index_cast(i32, tokens_in)

                def _wmma_bf16(result_type, a_vec, b_vec, acc):
                    a_i16 = vector.bitcast(v8i16_ty, a_vec)
                    b_i16 = vector.bitcast(v8i16_ty, b_vec)
                    return rocdl.wmma_f32_16x16x16_bf16(
                        result_type, [a_i16, b_i16, arith.unwrap(acc)]
                    )

                # Initialize accumulators
                zero_acc = arith.constant_vector(0.0, v8f32_ty)
                accs = [zero_acc for _ in range_constexpr(wave_reg_m * wave_reg_n)]

                # K-loop
                for kt in range_constexpr(num_k_tiles):
                    kt_idx = arith.index(kt)

                    # Load W2 tile
                    w_vecs = []
                    n0_base = tile_n0 // c16 + wave_n * arith.index(wave_reg_n)
                    for rk in range_constexpr(reg_k):
                        rk_vecs = []
                        k0 = kt_idx * arith.index(reg_k) + arith.index(rk)
                        for rn in range_constexpr(wave_reg_n):
                            n0 = n0_base + arith.index(rn)
                            elem_off = (
                                expert_w_base
                                + n0 * arith.index(W2_STRIDE_N0)
                                + k0 * arith.index(W_STRIDE_K0)
                                + klane * arith.index(W_STRIDE_KLANE)
                                + lane16 * arith.index(W_STRIDE_NLANE)
                            )
                            f32_off = elem_off // arith.index(2)
                            w_raw = buffer_ops.buffer_load(
                                w_rsrc,
                                f32_off,
                                vec_width=4,
                                dtype=f32,
                            )
                            w_vec = vector.bitcast(v8bf16_ty, w_raw)
                            rk_vecs.append(w_vec)
                        w_vecs.append(rk_vecs)

                    # Load A2 for this K tile
                    # A2 shape: [tokens, topk, inter_dim]
                    for rm in range_constexpr(wave_reg_m):
                        m_local = (
                            wave_m * arith.index(wave_reg_m * WMMA_M)
                            + arith.index(rm * WMMA_M)
                            + lane16
                        )
                        sorted_row = bx_m + m_local

                        fused_i = buffer_ops.buffer_load(
                            sorted_rsrc,
                            sorted_row,
                            vec_width=1,
                            dtype=i32,
                        )
                        t_raw = arith.andi(fused_i, mask24)
                        s_raw = arith.shrui(fused_i, arith.constant(24, type=i32))
                        t_valid = arith.cmpu(t_raw, tokens_i32, "ult")
                        t_safe = arith.select(
                            t_valid,
                            arith.index_cast(ir.IndexType.get(), t_raw),
                            arith.index(0),
                        )
                        s_safe = arith.index_cast(ir.IndexType.get(), s_raw)

                        # A2[t, slot, k] -> row offset
                        c_topk_idx = arith.index(topk)
                        c_inter_idx = arith.index(inter_dim)
                        a2_row_base = (t_safe * c_topk_idx + s_safe) * c_inter_idx

                        for rk in range_constexpr(reg_k):
                            k_base = (
                                kt_idx * arith.index(tile_k)
                                + arith.index(rk * 16)
                                + klane * c8
                            )
                            a2_off = a2_row_base + k_base
                            f32_off = a2_off // arith.index(2)
                            a_raw = buffer_ops.buffer_load(
                                a2_rsrc,
                                f32_off,
                                vec_width=4,
                                dtype=f32,
                            )
                            a_vec = vector.bitcast(v8bf16_ty, a_raw)

                            for rn in range_constexpr(wave_reg_n):
                                acc_idx = rm * wave_reg_n + rn
                                accs[acc_idx] = _wmma_bf16(
                                    v8f32_ty, a_vec, w_vecs[rk][rn], accs[acc_idx]
                                )

                # Store output: out[t, slot, model_dim] (non-accumulate mode)
                base8 = klane * c8
                c_topk_idx = arith.index(topk)
                c_model_idx = arith.index(model_dim)

                # WMMA C output layout (wave32):
                #   lane16 -> N column, klane*8+si -> M row
                for rm in range_constexpr(wave_reg_m):
                    m_local_base = wave_m * arith.index(
                        wave_reg_m * WMMA_M
                    ) + arith.index(rm * WMMA_M)

                    for rn in range_constexpr(wave_reg_n):
                        acc_idx = rm * wave_reg_n + rn
                        wmma_n_off = wave_n * arith.index(
                            wave_reg_n * WMMA_N
                        ) + arith.index(rn * WMMA_N)

                        for si in range_constexpr(8):
                            # M-row within WMMA tile = klane*8 + si
                            m_row = m_local_base + base8 + arith.index(si)
                            sorted_row = bx_m + m_row

                            fused_store = buffer_ops.buffer_load(
                                sorted_rsrc,
                                sorted_row,
                                vec_width=1,
                                dtype=i32,
                            )
                            t_store_raw = arith.andi(fused_store, mask24)
                            s_store_raw = arith.shrui(
                                fused_store, arith.constant(24, type=i32)
                            )
                            t_store_valid = arith.cmpu(t_store_raw, tokens_i32, "ult")
                            t_store_idx = arith.index_cast(
                                ir.IndexType.get(), t_store_raw
                            )
                            s_store_idx = arith.index_cast(
                                ir.IndexType.get(), s_store_raw
                            )

                            val = vector.extract(
                                accs[acc_idx],
                                static_position=[si],
                                dynamic_position=[],
                            )

                            if doweight_stage2:
                                wt_val = buffer_ops.buffer_load(
                                    sorted_w_rsrc,
                                    sorted_row,
                                    vec_width=1,
                                    dtype=f32,
                                )
                                val = val * wt_val

                            val_bf16 = arith.trunc_f(bf16, val)

                            g_col = tile_n0 + wmma_n_off + lane16
                            # out[t, slot, model_dim]
                            out_off = (
                                t_store_idx * c_topk_idx + s_store_idx
                            ) * c_model_idx + g_col

                            _if_valid = scf.IfOp(t_store_valid)
                            with _if_valid.then():
                                buffer_ops.buffer_store(val_bf16, out_rsrc, out_off)

        @flir.jit
        def __call__(
            self: flir.T.i64,
            arg_out: lambda: T.memref(DYN, T.bf16()),
            arg_a2: lambda: T.memref(DYN, T.bf16()),
            arg_w: lambda: T.memref(DYN, T.bf16()),
            arg_scale_a2: lambda: T.memref(DYN, T.f32()),
            arg_scale_w: lambda: T.memref(DYN, T.f32()),
            arg_sorted_token_ids: lambda: T.memref(DYN, T.i32()),
            arg_expert_ids: lambda: T.memref(DYN, T.i32()),
            arg_sorted_weights: lambda: T.memref(DYN, T.f32()),
            arg_max_token_ids: lambda: T.memref(DYN, T.i32()),
            tokens_in: lambda: T.index(),
            model_in: lambda: T.index(),
            k_in: lambda: T.index(),
            size_expert_ids_in: lambda: T.index(),
            stream_ptr: lambda: I.i64,
        ):
            bdx = arith.constant(THREADS_PER_BLOCK, index=True)
            c1 = arith.constant(1, index=True)
            gx = model_in / arith.index(tile_n)
            gy = size_expert_ids_in

            stream_token = stream_ptr_to_async_token(stream_ptr)
            flir.gpu_ext.LaunchFuncOp(
                [module_name, "moe_gemm2"],
                grid_size=(gx, gy, c1),
                block_size=(bdx, c1, c1),
                kernel_operands=[
                    arg_out,
                    arg_a2,
                    arg_w,
                    arg_scale_a2,
                    arg_scale_w,
                    arg_sorted_token_ids,
                    arg_expert_ids,
                    arg_sorted_weights,
                    arg_max_token_ids,
                    tokens_in,
                    model_in,
                    k_in,
                    size_expert_ids_in,
                ],
                async_dependencies=[stream_token],
            )

    m = _MOE2()
    return flydsl.compile(m)


__all__ = [
    "compile_wmma_moe_gemm1",
    "compile_wmma_moe_gemm2",
    "preshuffle_w_wmma",
    "preshuffle_a_wmma",
]
