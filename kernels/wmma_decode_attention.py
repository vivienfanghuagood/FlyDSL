"""FlyDSL Decode Attention kernel for RDNA4 (gfx12xx, wave32).

Hybrid WMMA decode attention:
  - WMMA for Q@K^T (efficient score computation via matrix multiply)
  - Element-wise P@V accumulation (no V LDS staging needed)
  - Online softmax with exp2
  - Cross-wave merge via LDS

Design:
  - Grid: (batch, num_q_heads) — one workgroup per (batch, q_head)
  - Workgroup: num_waves waves × 32 threads
  - Each wave processes interleaved blocks of BLOCK_N=16 KV tokens
  - Q@K^T via WMMA: [16, head_dim] × [head_dim, 16] → [16, 16], 8 WMMA ops
  - Scores broadcast via LDS to all 32 lanes in wave
  - V accumulated element-wise: each thread handles head_dim/32 = 4 elements

WMMA lane mapping (wave32, v_wmma_f32_16x16x16_bf16):
  lane16 = lane % 16: selects M-row (A), N-column (B/C)
  klane  = lane // 16: 0 or 1, selects K-half
  A operand: lane loads A[lane16, klane*8 : klane*8+8] as v8bf16
  B operand: lane loads B[klane*8 : klane*8+8, lane16] as v8bf16
  C result:  lane owns C[klane*8 + si, lane16] for si in 0..7

Memory layout:
  Q:     [batch, num_q_heads, head_dim]  bf16
  K_buf: [total_kv, num_kv_heads, head_dim]  bf16
  V_buf: [total_kv, num_kv_heads, head_dim]  bf16
  O:     [batch, num_q_heads, head_dim]  bf16
"""

import os
import functools
import math as pymath

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
    math as flydsl_math,
)
from flydsl.dialects.ext.python_control_flow import range_constexpr
from flydsl.runtime.device import get_rocm_arch
from flydsl.lang.ir.types import T as I
from flydsl.kernels.kernels_common import stream_ptr_to_async_token
from flydsl.utils import SmemAllocator

from _mlir import ir
from _mlir.dialects import gpu as mlir_gpu
import _mlir.extras.types as T


WAVE_SIZE = 32
WMMA_M = 16
WMMA_N = 16
WMMA_K = 16


def _unwrap(v):
    while hasattr(v, "_value"):
        v = v._value
    return v


# =============================================================================
# Kernel compiler
# =============================================================================


@functools.lru_cache(maxsize=64)
def compile_decode_attention(
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_kv_len: int,
    num_waves: int = 4,
    block_n: int = 16,
):
    """Compile hybrid WMMA decode attention kernel.

    WMMA for Q@K^T, element-wise for P@V.

    Args:
        num_q_heads: Total number of query heads (e.g., 32)
        num_kv_heads: Number of KV heads (e.g., 8 for GQA)
        head_dim: Head dimension (e.g., 128, must be multiple of 16)
        max_kv_len: Maximum KV sequence length
        num_waves: Waves per workgroup
        block_n: KV tokens per WMMA iteration (must be 16)
    """
    gpu_arch = get_rocm_arch()
    kv_group_num = num_q_heads // num_kv_heads

    assert head_dim % WMMA_K == 0
    assert block_n == WMMA_N, "block_n must equal WMMA_N=16"

    num_k_tiles = head_dim // WMMA_K  # 128/16 = 8
    elems_per_thread = head_dim // WAVE_SIZE  # 128/32 = 4

    THREADS_PER_BLOCK = num_waves * WAVE_SIZE

    DYN = ir.ShapedType.get_dynamic_size()

    allocator = SmemAllocator(None, arch=gpu_arch)

    module_name = f"decode_attn_hybrid_h{num_q_heads}_kv{num_kv_heads}_d{head_dim}"

    class _DecodeAttn(flir.MlirModule):
        GPU_MODULE_NAME = module_name
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}">']

        def init_gpu_module(self):
            f32 = T.f32()
            # Cross-wave merge buffers
            self._smem_wave_max = allocator.allocate_array(f32, num_waves)
            self._smem_wave_sum = allocator.allocate_array(f32, num_waves)
            self._smem_wave_out = allocator.allocate_array(f32, num_waves * head_dim)
            # Score broadcast LDS (f32): 32 per wave (lane-indexed)
            self._smem_scores = allocator.allocate_array(f32, num_waves * WAVE_SIZE)
            allocator.finalize()

        @flir.kernel
        def decode_attn_kernel(
            self: flir.T.i64,
            arg_q: lambda: T.memref(DYN, T.bf16()),
            arg_k: lambda: T.memref(DYN, T.bf16()),
            arg_v: lambda: T.memref(DYN, T.bf16()),
            arg_o: lambda: T.memref(DYN, T.bf16()),
            arg_kv_indptr: lambda: T.memref(DYN, T.i32()),
            arg_kv_indices: lambda: T.memref(DYN, T.i32()),
            c_sm_scale: lambda: I.f32,
            c_num_q_heads_idx: lambda: I.index,
            c_head_dim_idx: lambda: I.index,
            c_num_kv_heads_idx: lambda: I.index,
        ):
            f32 = ir.F32Type.get()
            bf16 = ir.BF16Type.get()
            i32 = ir.IntegerType.get_signless(32)
            i16_ty = ir.IntegerType.get_signless(16)
            idx_type = ir.IndexType.get()

            v8f32_ty = I.vec(8, I.f32)
            v8bf16_ty = ir.VectorType.get([8], bf16)
            v8i16_ty = ir.VectorType.get([8], i16_ty)
            v4f32_ty = ir.VectorType.get([elems_per_thread], f32)
            v4bf16_ty = ir.VectorType.get([elems_per_thread], bf16)

            tid = flir.thread_idx("x")
            cur_batch = flir.block_idx("x")
            cur_q_head = flir.block_idx("y")

            c0 = arith.index(0)
            c1 = arith.index(1)
            c8 = arith.index(8)
            c16 = arith.index(16)
            c32 = arith.index(WAVE_SIZE)

            wave_id = tid // c32
            lane = tid % c32
            lane16 = lane % c16
            klane = lane // c16

            base8 = klane * c8  # klane * 8

            # Precompute lane predicates
            is_klane0 = klane == c0
            is_row0 = lane16 == c0

            cur_kv_head = cur_q_head // arith.index(kv_group_num)

            q_rsrc = buffer_ops.create_buffer_resource(_unwrap(arg_q), max_size=True)
            k_rsrc = buffer_ops.create_buffer_resource(_unwrap(arg_k), max_size=True)
            v_rsrc = buffer_ops.create_buffer_resource(_unwrap(arg_v), max_size=True)
            o_rsrc = buffer_ops.create_buffer_resource(_unwrap(arg_o), max_size=True)
            indptr_rsrc = buffer_ops.create_buffer_resource(
                _unwrap(arg_kv_indptr), max_size=True
            )
            indices_rsrc = buffer_ops.create_buffer_resource(
                _unwrap(arg_kv_indices), max_size=True
            )

            kv_start_i32 = buffer_ops.buffer_load(
                indptr_rsrc, cur_batch, vec_width=1, dtype=i32
            )
            kv_end_i32 = buffer_ops.buffer_load(
                indptr_rsrc, cur_batch + c1, vec_width=1, dtype=i32
            )
            seq_len_i32 = kv_end_i32 - kv_start_i32

            kv_start_idx = arith.index_cast(idx_type, kv_start_i32)
            seq_len_idx = arith.index_cast(idx_type, seq_len_i32)

            c_hdim = arith.index(head_dim)
            c_num_q = arith.index(num_q_heads)
            c_num_kv = arith.index(num_kv_heads)
            c_block_n = arith.index(block_n)
            c_num_waves = arith.index(num_waves)
            c_ept = arith.index(elems_per_thread)

            # ================================================================
            # LDS setup
            # ================================================================
            base_ptr = allocator.get_base()
            smem_max = self._smem_wave_max(base_ptr).get()
            smem_sum = self._smem_wave_sum(base_ptr).get()
            smem_out = self._smem_wave_out(base_ptr).get()
            smem_scores = self._smem_scores(base_ptr).get()

            neg_inf_val = arith.constant(float("-inf"), type=f32)
            zero_f32 = arith.constant(0.0, type=f32)
            c_log2e = arith.constant(1.4426950408889634, type=f32)

            kv_token_stride = c_num_kv * c_hdim
            kv_head_off = cur_kv_head * c_hdim

            # ================================================================
            # Load Q into WMMA A format (once, reused across all KV blocks)
            # ================================================================
            q_base = (cur_batch * c_num_q + cur_q_head) * c_hdim

            q_wmma_vecs = []  # num_k_tiles × v8i16
            for kt in range_constexpr(num_k_tiles):
                k_offset = arith.index(kt * WMMA_K) + base8
                q_off = q_base + k_offset
                q_raw = buffer_ops.buffer_load(q_rsrc, q_off, vec_width=8, dtype=bf16)
                zero_v8bf16 = arith.constant_vector(0.0, v8bf16_ty)
                q_selected = arith.select(is_row0, q_raw, zero_v8bf16)
                q_wmma_vecs.append(vector.bitcast(v8i16_ty, q_selected))

            # ================================================================
            # Initialize accumulators for element-wise V accumulation
            # Each thread handles elems_per_thread=4 elements of head_dim
            # ================================================================
            e_max = neg_inf_val
            e_sum = zero_f32
            acc = [zero_f32 for _ in range_constexpr(elems_per_thread)]

            num_blocks = (seq_len_idx + c_block_n - c1) // c_block_n  # cdiv

            for blk_idx in range(wave_id, num_blocks, c_num_waves):
                kv_start_token = blk_idx * c_block_n

                # ============================================================
                # Load K for WMMA B operand (Q@K^T)
                # lane16 selects the KV token, klane selects K-half
                # ============================================================
                kv_local_idx = kv_start_token + lane16
                kv_valid = kv_local_idx < seq_len_idx

                kv_global = kv_start_idx + kv_local_idx
                safe_global = arith.select(kv_valid, kv_global, kv_start_idx)

                token_i32 = buffer_ops.buffer_load(
                    indices_rsrc, safe_global, vec_width=1, dtype=i32
                )
                token_idx = arith.index_cast(idx_type, token_i32)
                k_base_token = token_idx * kv_token_stride + kv_head_off

                k_wmma_vecs = []
                for kt in range_constexpr(num_k_tiles):
                    k_offset = arith.index(kt * WMMA_K) + base8
                    k_off = k_base_token + k_offset
                    k_raw = buffer_ops.buffer_load(
                        k_rsrc, k_off, vec_width=8, dtype=bf16
                    )
                    zero_v8bf16 = arith.constant_vector(0.0, v8bf16_ty)
                    k_selected = arith.select(kv_valid, k_raw, zero_v8bf16)
                    k_wmma_vecs.append(vector.bitcast(v8i16_ty, k_selected))

                # ============================================================
                # WMMA Q@K^T: [16, 128] × [128, 16] → [16, 16]
                # ============================================================
                qk_acc = arith.constant_vector(0.0, v8f32_ty)
                for kt in range_constexpr(num_k_tiles):
                    qk_acc = rocdl.wmma_f32_16x16x16_bf16(
                        v8f32_ty,
                        [q_wmma_vecs[kt], k_wmma_vecs[kt], arith.unwrap(qk_acc)],
                    )

                # ============================================================
                # Extract scores from WMMA C output
                # Row 0 = klane=0, si=0 → element 0 of lanes 0-15
                # ============================================================
                score_raw = vector.extract(
                    qk_acc, static_position=[0], dynamic_position=[]
                )
                score_scaled = score_raw * c_sm_scale

                score_masked = arith.select(kv_valid, score_scaled, neg_inf_val)
                score_final = arith.select(is_klane0, score_masked, neg_inf_val)

                # Write scores to LDS for broadcast to all lanes
                score_lds_idx = wave_id * c32 + lane
                memref.store(
                    _unwrap(score_final), smem_scores, [_unwrap(score_lds_idx)]
                )

                # ============================================================
                # Read scores from LDS, compute online softmax
                # ============================================================
                blk_max = neg_inf_val
                scores_f32 = []
                for ni in range_constexpr(block_n):
                    s_idx = wave_id * c32 + arith.index(ni)
                    s_val = memref.load(smem_scores, [arith.as_value(s_idx)])
                    scores_f32.append(s_val)
                    blk_max = arith.maximum(blk_max, s_val)

                # Online softmax rescaling
                n_emax = arith.maximum(e_max, blk_max)
                rescale = flydsl_math.exp2(arith.as_value((e_max - n_emax) * c_log2e))
                e_sum = e_sum * rescale
                e_max = n_emax

                # Rescale existing accumulator
                new_acc = []
                for ei in range_constexpr(elems_per_thread):
                    new_acc.append(acc[ei] * rescale)
                acc = new_acc

                # ============================================================
                # Element-wise V accumulation
                # For each of block_n tokens, compute p and accumulate V
                # ============================================================
                for ni in range_constexpr(block_n):
                    p = flydsl_math.exp2(
                        arith.as_value((scores_f32[ni] - e_max) * c_log2e)
                    )
                    e_sum = e_sum + p

                    # Load V for token ni
                    kv_ni_local = kv_start_token + arith.index(ni)
                    kv_ni_valid = kv_ni_local < seq_len_idx
                    kv_ni_global = kv_start_idx + kv_ni_local
                    safe_ni = arith.select(kv_ni_valid, kv_ni_global, kv_start_idx)
                    token_ni_i32 = buffer_ops.buffer_load(
                        indices_rsrc, safe_ni, vec_width=1, dtype=i32
                    )
                    token_ni_idx = arith.index_cast(idx_type, token_ni_i32)
                    v_base = token_ni_idx * kv_token_stride + kv_head_off
                    v_off = v_base + lane * c_ept

                    v_vec_bf16 = buffer_ops.buffer_load(
                        v_rsrc, v_off, vec_width=elems_per_thread, dtype=bf16
                    )
                    v_vec_f32 = flir.arith.extf(v4f32_ty, arith.as_value(v_vec_bf16))

                    # acc[ei] += p * v[ei]
                    new_acc2 = []
                    for ei in range_constexpr(elems_per_thread):
                        v_f32 = vector.extract(
                            v_vec_f32, static_position=[ei], dynamic_position=[]
                        )
                        new_acc2.append(acc[ei] + p * v_f32)
                    acc = new_acc2

            # ================================================================
            # Cross-wave merge via LDS
            # ================================================================
            # Store partial output
            for ei in range_constexpr(elems_per_thread):
                out_lds_idx = wave_id * c_hdim + lane * c_ept + arith.index(ei)
                memref.store(_unwrap(acc[ei]), smem_out, [_unwrap(out_lds_idx)])

            # e_max, e_sum are identical across all lanes in a wave
            memref.store(_unwrap(e_max), smem_max, [_unwrap(wave_id)])
            memref.store(_unwrap(e_sum), smem_sum, [_unwrap(wave_id)])

            gpu.barrier()

            # Merge across waves
            merge_emax = neg_inf_val
            merge_esum = zero_f32
            merge_acc = [zero_f32 for _ in range_constexpr(elems_per_thread)]

            for w in range_constexpr(num_waves):
                w_max = memref.load(smem_max, [arith.as_value(arith.index(w))])
                w_sum = memref.load(smem_sum, [arith.as_value(arith.index(w))])

                n_merge_max = arith.maximum(merge_emax, w_max)
                old_scale = flydsl_math.exp2(
                    arith.as_value((merge_emax - n_merge_max) * c_log2e)
                )
                new_scale = flydsl_math.exp2(
                    arith.as_value((w_max - n_merge_max) * c_log2e)
                )

                merge_esum = merge_esum * old_scale + w_sum * new_scale

                new_merge_acc = []
                for ei in range_constexpr(elems_per_thread):
                    lds_idx = arith.index(w) * c_hdim + lane * c_ept + arith.index(ei)
                    w_val = memref.load(smem_out, [arith.as_value(lds_idx)])
                    new_merge_acc.append(merge_acc[ei] * old_scale + w_val * new_scale)
                merge_acc = new_merge_acc
                merge_emax = n_merge_max

            # Normalize and store output
            o_base = (cur_batch * c_num_q + cur_q_head) * c_hdim
            final_inv_sum = arith.constant(1.0, type=f32) / merge_esum

            for ei in range_constexpr(elems_per_thread):
                out_val_f32 = merge_acc[ei] * final_inv_sum
                out_val_bf16 = arith.trunc_f(bf16, out_val_f32)
                o_off = o_base + lane * c_ept + arith.index(ei)
                buffer_ops.buffer_store(out_val_bf16, o_rsrc, o_off)

        @flir.jit
        def __call__(
            self: flir.T.i64,
            arg_q: lambda: T.memref(DYN, T.bf16()),
            arg_k: lambda: T.memref(DYN, T.bf16()),
            arg_v: lambda: T.memref(DYN, T.bf16()),
            arg_o: lambda: T.memref(DYN, T.bf16()),
            arg_kv_indptr: lambda: T.memref(DYN, T.i32()),
            arg_kv_indices: lambda: T.memref(DYN, T.i32()),
            c_sm_scale: lambda: I.f32,
            c_batch: lambda: I.index,
            c_num_q_heads_idx: lambda: I.index,
            c_head_dim_idx: lambda: I.index,
            c_num_kv_heads_idx: lambda: I.index,
            stream_ptr: lambda: I.i64,
        ):
            c1 = arith.constant(1, index=True)
            bdx = arith.constant(THREADS_PER_BLOCK, index=True)
            stream_token = stream_ptr_to_async_token(stream_ptr)
            flir.gpu_ext.LaunchFuncOp(
                [module_name, "decode_attn_kernel"],
                grid_size=(c_batch, c_num_q_heads_idx, c1),
                block_size=(bdx, c1, c1),
                kernel_operands=[
                    arg_q,
                    arg_k,
                    arg_v,
                    arg_o,
                    arg_kv_indptr,
                    arg_kv_indices,
                    c_sm_scale,
                    c_num_q_heads_idx,
                    c_head_dim_idx,
                    c_num_kv_heads_idx,
                ],
                async_dependencies=[stream_token],
            )

    m = _DecodeAttn()
    return flydsl.compile(m)
