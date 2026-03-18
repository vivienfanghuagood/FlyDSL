"""FlyDSL Decode Attention kernels for RDNA4 (gfx12xx, wave32).

Three decode attention implementations:

1. compile_decode_attention() — Hybrid WMMA
   WMMA for Q@K^T (efficient score computation via matrix multiply),
   element-wise P@V accumulation (no V LDS staging needed).
   Online softmax with exp2. Cross-wave merge via LDS.

2. compile_decode_attention_elemwise() — Pure element-wise
   No WMMA. Each 32-lane wave computes Q@K dot products via element-wise
   multiply + wave-level cross-lane reduction (5 rounds of shuffle+add).
   Simpler but avoids WMMA overhead for small BLOCK_N.

3. compile_decode_attention_splitkv() — Split-KV flash decoding
   Stage 1: Grid = (batch, heads, max_kv_splits). Each workgroup processes
   a subset of KV tokens using WMMA Q@K^T + element-wise P@V.
   Outputs partial att_out (f32) and att_lse (log-sum-exp).
   Stage 2: Grid = (batch, heads). Merges partial results across splits.

All kernels share:
  - Grid: (batch, num_q_heads, ...) — one workgroup per (batch, q_head[, split])
  - Workgroup: num_waves waves x 32 threads
  - GQA support (num_q_heads // num_kv_heads grouping)
  - Online softmax with exp2
  - Cross-wave merge via LDS

Memory layout:
  Q:     [batch, num_q_heads, head_dim]  bf16
  K_buf: [total_kv, num_kv_heads, head_dim]  bf16
  V_buf: [total_kv, num_kv_heads, head_dim]  bf16
  O:     [batch, num_q_heads, head_dim]  bf16

WMMA lane mapping (wave32, v_wmma_f32_16x16x16_bf16):
  lane16 = lane % 16: selects M-row (A), N-column (B/C)
  klane  = lane // 16: 0 or 1, selects K-half
  A operand: lane loads A[lane16, klane*8 : klane*8+8] as v8bf16
  B operand: lane loads B[klane*8 : klane*8+8, lane16] as v8bf16
  C result:  lane owns C[klane*8 + si, lane16] for si in 0..7
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
# 1. Hybrid WMMA decode attention
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
            indptr_rsrc = buffer_ops.create_buffer_resource(_unwrap(arg_kv_indptr), max_size=True)
            indices_rsrc = buffer_ops.create_buffer_resource(_unwrap(arg_kv_indices), max_size=True)

            kv_start_i32 = buffer_ops.buffer_load(indptr_rsrc, cur_batch, vec_width=1, dtype=i32)
            kv_end_i32 = buffer_ops.buffer_load(indptr_rsrc, cur_batch + c1, vec_width=1, dtype=i32)
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

            q_wmma_vecs = []  # num_k_tiles x v8i16
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

                token_i32 = buffer_ops.buffer_load(indices_rsrc, safe_global, vec_width=1, dtype=i32)
                token_idx = arith.index_cast(idx_type, token_i32)
                k_base_token = token_idx * kv_token_stride + kv_head_off

                k_wmma_vecs = []
                for kt in range_constexpr(num_k_tiles):
                    k_offset = arith.index(kt * WMMA_K) + base8
                    k_off = k_base_token + k_offset
                    k_raw = buffer_ops.buffer_load(k_rsrc, k_off, vec_width=8, dtype=bf16)
                    zero_v8bf16 = arith.constant_vector(0.0, v8bf16_ty)
                    k_selected = arith.select(kv_valid, k_raw, zero_v8bf16)
                    k_wmma_vecs.append(vector.bitcast(v8i16_ty, k_selected))

                # ============================================================
                # WMMA Q@K^T: [16, 128] x [128, 16] -> [16, 16]
                # ============================================================
                qk_acc = arith.constant_vector(0.0, v8f32_ty)
                for kt in range_constexpr(num_k_tiles):
                    qk_acc = rocdl.wmma_f32_16x16x16_bf16(
                        v8f32_ty,
                        [q_wmma_vecs[kt], k_wmma_vecs[kt], arith.unwrap(qk_acc)],
                    )

                # ============================================================
                # Extract scores from WMMA C output
                # Row 0 = klane=0, si=0 -> element 0 of lanes 0-15
                # ============================================================
                score_raw = vector.extract(qk_acc, static_position=[0], dynamic_position=[])
                score_scaled = score_raw * c_sm_scale

                score_masked = arith.select(kv_valid, score_scaled, neg_inf_val)
                score_final = arith.select(is_klane0, score_masked, neg_inf_val)

                # Write scores to LDS for broadcast to all lanes
                score_lds_idx = wave_id * c32 + lane
                memref.store(_unwrap(score_final), smem_scores, [_unwrap(score_lds_idx)])

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
                    p = flydsl_math.exp2(arith.as_value((scores_f32[ni] - e_max) * c_log2e))
                    e_sum = e_sum + p

                    # Load V for token ni
                    kv_ni_local = kv_start_token + arith.index(ni)
                    kv_ni_valid = kv_ni_local < seq_len_idx
                    kv_ni_global = kv_start_idx + kv_ni_local
                    safe_ni = arith.select(kv_ni_valid, kv_ni_global, kv_start_idx)
                    token_ni_i32 = buffer_ops.buffer_load(indices_rsrc, safe_ni, vec_width=1, dtype=i32)
                    token_ni_idx = arith.index_cast(idx_type, token_ni_i32)
                    v_base = token_ni_idx * kv_token_stride + kv_head_off
                    v_off = v_base + lane * c_ept

                    v_vec_bf16 = buffer_ops.buffer_load(v_rsrc, v_off, vec_width=elems_per_thread, dtype=bf16)
                    v_vec_f32 = flir.arith.extf(v4f32_ty, arith.as_value(v_vec_bf16))

                    # acc[ei] += p * v[ei]
                    new_acc2 = []
                    for ei in range_constexpr(elems_per_thread):
                        v_f32 = vector.extract(v_vec_f32, static_position=[ei], dynamic_position=[])
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
                old_scale = flydsl_math.exp2(arith.as_value((merge_emax - n_merge_max) * c_log2e))
                new_scale = flydsl_math.exp2(arith.as_value((w_max - n_merge_max) * c_log2e))

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


# =============================================================================
# 2. Pure element-wise decode attention
# =============================================================================


@functools.lru_cache(maxsize=64)
def compile_decode_attention_elemwise(
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_kv_len: int,
    num_waves: int = 4,
    block_n: int = 1,
):
    """Compile pure element-wise decode attention kernel.

    No WMMA -- everything is element-wise dot products with wave reductions.
    Each lane: multiply 4 Q elements x 4 K elements, sum locally.
    Wave-level cross-lane reduction (5 rounds of shuffle+add).

    Args:
        num_q_heads: Total number of query heads (e.g., 32)
        num_kv_heads: Number of KV heads (e.g., 8 for GQA)
        head_dim: Head dimension (e.g., 128, must be multiple of 32)
        max_kv_len: Maximum KV sequence length
        num_waves: Waves per workgroup
        block_n: KV tokens processed per iteration (1 = simplest)
    """
    gpu_arch = get_rocm_arch()
    kv_group_num = num_q_heads // num_kv_heads

    assert head_dim % WAVE_SIZE == 0
    elems_per_thread = head_dim // WAVE_SIZE  # 128/32 = 4

    THREADS_PER_BLOCK = num_waves * WAVE_SIZE
    DYN = ir.ShapedType.get_dynamic_size()

    allocator = SmemAllocator(None, arch=gpu_arch)
    module_name = f"decode_attn_ew_h{num_q_heads}_kv{num_kv_heads}_d{head_dim}"

    class _DecodeAttn(flir.MlirModule):
        GPU_MODULE_NAME = module_name
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}">']

        def init_gpu_module(self):
            f32 = T.f32()
            # Cross-wave merge buffers
            self._smem_wave_max = allocator.allocate_array(f32, num_waves)
            self._smem_wave_sum = allocator.allocate_array(f32, num_waves)
            self._smem_wave_out = allocator.allocate_array(f32, num_waves * head_dim)
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
            idx_type = ir.IndexType.get()

            v_ept_f32_ty = ir.VectorType.get([elems_per_thread], f32)
            v_ept_bf16_ty = ir.VectorType.get([elems_per_thread], bf16)

            tid = flir.thread_idx("x")
            cur_batch = flir.block_idx("x")
            cur_q_head = flir.block_idx("y")

            c0 = arith.index(0)
            c1 = arith.index(1)
            c32 = arith.index(WAVE_SIZE)

            wave_id = tid // c32
            lane = tid % c32

            cur_kv_head = cur_q_head // arith.index(kv_group_num)

            q_rsrc = buffer_ops.create_buffer_resource(_unwrap(arg_q), max_size=True)
            k_rsrc = buffer_ops.create_buffer_resource(_unwrap(arg_k), max_size=True)
            v_rsrc = buffer_ops.create_buffer_resource(_unwrap(arg_v), max_size=True)
            o_rsrc = buffer_ops.create_buffer_resource(_unwrap(arg_o), max_size=True)
            indptr_rsrc = buffer_ops.create_buffer_resource(_unwrap(arg_kv_indptr), max_size=True)
            indices_rsrc = buffer_ops.create_buffer_resource(_unwrap(arg_kv_indices), max_size=True)

            kv_start_i32 = buffer_ops.buffer_load(indptr_rsrc, cur_batch, vec_width=1, dtype=i32)
            kv_end_i32 = buffer_ops.buffer_load(indptr_rsrc, cur_batch + c1, vec_width=1, dtype=i32)
            seq_len_i32 = kv_end_i32 - kv_start_i32

            kv_start_idx = arith.index_cast(idx_type, kv_start_i32)
            seq_len_idx = arith.index_cast(idx_type, seq_len_i32)

            c_hdim = arith.index(head_dim)
            c_num_q = arith.index(num_q_heads)
            c_num_kv = arith.index(num_kv_heads)
            c_num_waves = arith.index(num_waves)
            c_ept = arith.index(elems_per_thread)

            base_ptr = allocator.get_base()
            smem_max = self._smem_wave_max(base_ptr).get()
            smem_sum = self._smem_wave_sum(base_ptr).get()
            smem_out = self._smem_wave_out(base_ptr).get()

            neg_inf_val = arith.constant(float("-inf"), type=f32)
            zero_f32 = arith.constant(0.0, type=f32)
            c_log2e = arith.constant(1.4426950408889634, type=f32)

            kv_token_stride = c_num_kv * c_hdim
            kv_head_off = cur_kv_head * c_hdim

            # ================================================================
            # Load Q: each thread loads elems_per_thread=4 bf16 elements
            # ================================================================
            q_base = (cur_batch * c_num_q + cur_q_head) * c_hdim
            q_off = q_base + lane * c_ept
            q_vec_bf16 = buffer_ops.buffer_load(q_rsrc, q_off, vec_width=elems_per_thread, dtype=bf16)
            q_vec_f32 = flir.arith.extf(v_ept_f32_ty, arith.as_value(q_vec_bf16))

            # Extract Q elements for dot product
            q_elems = []
            for ei in range_constexpr(elems_per_thread):
                q_elems.append(vector.extract(q_vec_f32, static_position=[ei], dynamic_position=[]))

            # ================================================================
            # Initialize accumulators
            # ================================================================
            e_max = neg_inf_val
            e_sum = zero_f32
            acc = [zero_f32 for _ in range_constexpr(elems_per_thread)]

            # Process KV tokens one at a time, interleaved across waves
            for kv_idx in range(wave_id, seq_len_idx, c_num_waves):
                # Load index
                kv_global = kv_start_idx + kv_idx
                token_i32 = buffer_ops.buffer_load(indices_rsrc, kv_global, vec_width=1, dtype=i32)
                token_idx = arith.index_cast(idx_type, token_i32)
                k_base = token_idx * kv_token_stride + kv_head_off

                # Load K for this token: each lane loads 4 bf16 elements
                k_off = k_base + lane * c_ept
                k_vec_bf16 = buffer_ops.buffer_load(k_rsrc, k_off, vec_width=elems_per_thread, dtype=bf16)
                k_vec_f32 = flir.arith.extf(v_ept_f32_ty, arith.as_value(k_vec_bf16))

                # Dot product: Q . K (element-wise multiply + reduce)
                dot_local = zero_f32
                for ei in range_constexpr(elems_per_thread):
                    k_elem = vector.extract(k_vec_f32, static_position=[ei], dynamic_position=[])
                    dot_local = dot_local + q_elems[ei] * k_elem

                # Wave-level reduction (5 rounds for 32 lanes)
                dot_val = dot_local
                i32_type = ir.IntegerType.get_signless(32)
                i1_type = ir.IntegerType.get_signless(1)
                for shift in [16, 8, 4, 2, 1]:
                    shift_val = arith.constant(shift, type=i32_type)
                    width_val = arith.constant(32, type=i32_type)
                    shuf = gpu.ShuffleOp(
                        _unwrap(dot_val),
                        _unwrap(shift_val),
                        _unwrap(width_val),
                        mode="xor",
                    )
                    shuf_val = arith.ArithValue(shuf.shuffleResult)
                    dot_val = dot_val + shuf_val

                # dot_val now contains the full dot product in all lanes
                score = dot_val * c_sm_scale

                # Online softmax
                n_emax = arith.maximum(e_max, score)
                rescale = flydsl_math.exp2(arith.as_value((e_max - n_emax) * c_log2e))
                e_sum = e_sum * rescale
                e_max = n_emax

                p = flydsl_math.exp2(arith.as_value((score - e_max) * c_log2e))
                e_sum = e_sum + p

                # Rescale and accumulate V
                # Load V for this token
                v_off = k_base + lane * c_ept  # V has same layout as K
                v_vec_bf16 = buffer_ops.buffer_load(v_rsrc, v_off, vec_width=elems_per_thread, dtype=bf16)
                v_vec_f32 = flir.arith.extf(v_ept_f32_ty, arith.as_value(v_vec_bf16))

                new_acc = []
                for ei in range_constexpr(elems_per_thread):
                    v_elem = vector.extract(v_vec_f32, static_position=[ei], dynamic_position=[])
                    new_acc.append(acc[ei] * rescale + p * v_elem)
                acc = new_acc

            # ================================================================
            # Cross-wave merge via LDS
            # ================================================================
            for ei in range_constexpr(elems_per_thread):
                out_lds_idx = wave_id * c_hdim + lane * c_ept + arith.index(ei)
                memref.store(_unwrap(acc[ei]), smem_out, [_unwrap(out_lds_idx)])

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
                old_scale = flydsl_math.exp2(arith.as_value((merge_emax - n_merge_max) * c_log2e))
                new_scale = flydsl_math.exp2(arith.as_value((w_max - n_merge_max) * c_log2e))

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


# =============================================================================
# 3. Split-KV decode attention (two-stage flash decoding)
# =============================================================================


@functools.lru_cache(maxsize=64)
def compile_decode_attention_splitkv(
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_kv_splits: int,
    num_waves: int = 2,
    block_n: int = 16,
):
    """Compile split-KV WMMA decode attention kernels.

    Two-stage flash decoding:
    Stage 1: Each workgroup processes a subset of KV tokens, outputs partial
             att_out[batch, head, split, head_dim] (f32) and att_lse (f32).
    Stage 2: Merges partial results across splits using log-sum-exp rescaling.

    Args:
        num_q_heads: Total number of query heads (e.g., 32)
        num_kv_heads: Number of KV heads (e.g., 8 for GQA)
        head_dim: Head dimension (e.g., 128, must be multiple of 16)
        max_kv_splits: Maximum number of KV splits
        num_waves: Waves per workgroup for stage1
        block_n: KV tokens per WMMA iteration (must be 16)
    """
    gpu_arch = get_rocm_arch()
    kv_group_num = num_q_heads // num_kv_heads

    assert head_dim % WMMA_K == 0
    assert block_n == WMMA_N, "block_n must equal WMMA_N=16"

    num_k_tiles = head_dim // WMMA_K  # 128/16 = 8
    elems_per_thread = head_dim // WAVE_SIZE  # 128/32 = 4

    STAGE1_THREADS = num_waves * WAVE_SIZE
    STAGE2_THREADS = WAVE_SIZE  # single wave for stage2

    DYN = ir.ShapedType.get_dynamic_size()

    # Stage 1 LDS
    s1_allocator = SmemAllocator(None, arch=gpu_arch)
    # Stage 2 LDS - none needed (single wave, no cross-wave merge)
    s2_allocator = SmemAllocator(None, arch=gpu_arch)

    s1_module_name = f"decode_attn_splitkv_s1_h{num_q_heads}_kv{num_kv_heads}_d{head_dim}"
    s2_module_name = f"decode_attn_splitkv_s2_h{num_q_heads}_kv{num_kv_heads}_d{head_dim}"

    # =========================================================================
    # Stage 1: Split-KV attention
    # =========================================================================

    class _Stage1(flir.MlirModule):
        GPU_MODULE_NAME = s1_module_name
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}">']

        def init_gpu_module(self):
            f32 = T.f32()
            if num_waves > 1:
                # Cross-wave merge buffers within stage1
                self._smem_wave_max = s1_allocator.allocate_array(f32, num_waves)
                self._smem_wave_sum = s1_allocator.allocate_array(f32, num_waves)
                self._smem_wave_out = s1_allocator.allocate_array(f32, num_waves * head_dim)
            # Score broadcast LDS (f32): 32 per wave (lane-indexed)
            self._smem_scores = s1_allocator.allocate_array(f32, num_waves * WAVE_SIZE)
            s1_allocator.finalize()

        @flir.kernel
        def stage1_kernel(
            self: flir.T.i64,
            arg_q: lambda: T.memref(DYN, T.bf16()),
            arg_k: lambda: T.memref(DYN, T.bf16()),
            arg_v: lambda: T.memref(DYN, T.bf16()),
            arg_att_out: lambda: T.memref(DYN, T.f32()),  # [batch * heads * splits * hdim]
            arg_att_lse: lambda: T.memref(DYN, T.f32()),  # [batch * heads * splits]
            arg_kv_indptr: lambda: T.memref(DYN, T.i32()),
            arg_kv_indices: lambda: T.memref(DYN, T.i32()),
            arg_num_kv_splits: lambda: T.memref(DYN, T.i32()),  # [batch]
            c_sm_scale: lambda: I.f32,
            c_num_q_heads_idx: lambda: I.index,
            c_head_dim_idx: lambda: I.index,
            c_num_kv_heads_idx: lambda: I.index,
            c_max_kv_splits_idx: lambda: I.index,
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

            tid = flir.thread_idx("x")
            cur_batch = flir.block_idx("x")
            cur_q_head = flir.block_idx("y")
            split_kv_id = flir.block_idx("z")

            c0 = arith.index(0)
            c1 = arith.index(1)
            c8 = arith.index(8)
            c16 = arith.index(16)
            c32 = arith.index(WAVE_SIZE)

            wave_id = tid // c32
            lane = tid % c32
            lane16 = lane % c16
            klane = lane // c16

            base8 = klane * c8

            is_klane0 = klane == c0
            is_row0 = lane16 == c0

            cur_kv_head = cur_q_head // arith.index(kv_group_num)

            q_rsrc = buffer_ops.create_buffer_resource(_unwrap(arg_q), max_size=True)
            k_rsrc = buffer_ops.create_buffer_resource(_unwrap(arg_k), max_size=True)
            v_rsrc = buffer_ops.create_buffer_resource(_unwrap(arg_v), max_size=True)
            att_out_rsrc = buffer_ops.create_buffer_resource(_unwrap(arg_att_out), max_size=True)
            att_lse_rsrc = buffer_ops.create_buffer_resource(_unwrap(arg_att_lse), max_size=True)
            indptr_rsrc = buffer_ops.create_buffer_resource(_unwrap(arg_kv_indptr), max_size=True)
            indices_rsrc = buffer_ops.create_buffer_resource(_unwrap(arg_kv_indices), max_size=True)
            splits_rsrc = buffer_ops.create_buffer_resource(_unwrap(arg_num_kv_splits), max_size=True)

            kv_start_i32 = buffer_ops.buffer_load(indptr_rsrc, cur_batch, vec_width=1, dtype=i32)
            kv_end_i32 = buffer_ops.buffer_load(indptr_rsrc, cur_batch + c1, vec_width=1, dtype=i32)
            seq_len_i32 = kv_end_i32 - kv_start_i32
            kv_splits_i32 = buffer_ops.buffer_load(splits_rsrc, cur_batch, vec_width=1, dtype=i32)

            kv_start_idx = arith.index_cast(idx_type, kv_start_i32)
            seq_len_idx = arith.index_cast(idx_type, seq_len_i32)
            kv_splits_idx = arith.index_cast(idx_type, kv_splits_i32)

            c_hdim = arith.index(head_dim)
            c_num_q = arith.index(num_q_heads)
            c_num_kv = arith.index(num_kv_heads)
            c_block_n = arith.index(block_n)
            c_num_waves = arith.index(num_waves)
            c_ept = arith.index(elems_per_thread)
            c_max_splits = arith.index(max_kv_splits)
            # MIN_BLOCK_KV for split alignment (match Triton's _MIN_BLOCK_KV=32)
            c_min_block_kv = arith.index(32)

            # LDS setup
            base_ptr = s1_allocator.get_base()
            if num_waves > 1:
                smem_max = self._smem_wave_max(base_ptr).get()
                smem_sum = self._smem_wave_sum(base_ptr).get()
                smem_out = self._smem_wave_out(base_ptr).get()
            smem_scores = self._smem_scores(base_ptr).get()

            neg_inf_val = arith.constant(float("-inf"), type=f32)
            zero_f32 = arith.constant(0.0, type=f32)
            c_log2e = arith.constant(1.4426950408889634, type=f32)

            kv_token_stride = c_num_kv * c_hdim
            kv_head_off = cur_kv_head * c_hdim

            # Compute split range for this workgroup
            inner_cdiv = (seq_len_idx + kv_splits_idx - c1) // kv_splits_idx
            kv_len_per_split = ((inner_cdiv + c_min_block_kv - c1) // c_min_block_kv) * c_min_block_kv
            split_kv_start = kv_len_per_split * split_kv_id
            split_kv_end_raw = split_kv_start + kv_len_per_split
            split_kv_end = arith.select(split_kv_end_raw < seq_len_idx, split_kv_end_raw, seq_len_idx)

            # Early exit check: skip if this split has no work
            has_work = split_kv_end > split_kv_start

            # Load Q into WMMA A format
            q_base = (cur_batch * c_num_q + cur_q_head) * c_hdim

            q_wmma_vecs = []
            for kt in range_constexpr(num_k_tiles):
                k_offset = arith.index(kt * WMMA_K) + base8
                q_off = q_base + k_offset
                q_raw = buffer_ops.buffer_load(q_rsrc, q_off, vec_width=8, dtype=bf16)
                zero_v8bf16 = arith.constant_vector(0.0, v8bf16_ty)
                q_selected = arith.select(is_row0, q_raw, zero_v8bf16)
                q_wmma_vecs.append(vector.bitcast(v8i16_ty, q_selected))

            # Initialize accumulators
            e_max = neg_inf_val
            e_sum = zero_f32
            acc = [zero_f32 for _ in range_constexpr(elems_per_thread)]

            # Number of BLOCK_N blocks in this split's range
            split_len = split_kv_end - split_kv_start
            num_blocks = (split_len + c_block_n - c1) // c_block_n

            for blk_idx in range(wave_id, num_blocks, c_num_waves):
                kv_offset_in_split = blk_idx * c_block_n
                kv_start_token = split_kv_start + kv_offset_in_split

                # Load K for WMMA B operand
                kv_local_idx = kv_start_token + lane16
                kv_valid = kv_local_idx < split_kv_end

                kv_global = kv_start_idx + kv_local_idx
                safe_global = arith.select(kv_valid, kv_global, kv_start_idx)

                token_i32 = buffer_ops.buffer_load(indices_rsrc, safe_global, vec_width=1, dtype=i32)
                token_idx = arith.index_cast(idx_type, token_i32)
                k_base_token = token_idx * kv_token_stride + kv_head_off

                k_wmma_vecs = []
                for kt in range_constexpr(num_k_tiles):
                    k_offset = arith.index(kt * WMMA_K) + base8
                    k_off = k_base_token + k_offset
                    k_raw = buffer_ops.buffer_load(k_rsrc, k_off, vec_width=8, dtype=bf16)
                    zero_v8bf16 = arith.constant_vector(0.0, v8bf16_ty)
                    k_selected = arith.select(kv_valid, k_raw, zero_v8bf16)
                    k_wmma_vecs.append(vector.bitcast(v8i16_ty, k_selected))

                # WMMA Q@K^T
                qk_acc = arith.constant_vector(0.0, v8f32_ty)
                for kt in range_constexpr(num_k_tiles):
                    qk_acc = rocdl.wmma_f32_16x16x16_bf16(
                        v8f32_ty,
                        [q_wmma_vecs[kt], k_wmma_vecs[kt], arith.unwrap(qk_acc)],
                    )

                # Extract scores
                score_raw = vector.extract(qk_acc, static_position=[0], dynamic_position=[])
                score_scaled = score_raw * c_sm_scale
                score_masked = arith.select(kv_valid, score_scaled, neg_inf_val)
                score_final = arith.select(is_klane0, score_masked, neg_inf_val)

                # Write scores to LDS for broadcast
                score_lds_idx = wave_id * c32 + lane
                memref.store(_unwrap(score_final), smem_scores, [_unwrap(score_lds_idx)])

                # Read scores, compute online softmax
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

                new_acc = []
                for ei in range_constexpr(elems_per_thread):
                    new_acc.append(acc[ei] * rescale)
                acc = new_acc

                # Element-wise V accumulation
                for ni in range_constexpr(block_n):
                    p = flydsl_math.exp2(arith.as_value((scores_f32[ni] - e_max) * c_log2e))
                    e_sum = e_sum + p

                    kv_ni_local = kv_start_token + arith.index(ni)
                    kv_ni_valid = kv_ni_local < split_kv_end
                    kv_ni_global = kv_start_idx + kv_ni_local
                    safe_ni = arith.select(kv_ni_valid, kv_ni_global, kv_start_idx)
                    token_ni_i32 = buffer_ops.buffer_load(indices_rsrc, safe_ni, vec_width=1, dtype=i32)
                    token_ni_idx = arith.index_cast(idx_type, token_ni_i32)
                    v_base = token_ni_idx * kv_token_stride + kv_head_off
                    v_off = v_base + lane * c_ept

                    v_vec_bf16 = buffer_ops.buffer_load(v_rsrc, v_off, vec_width=elems_per_thread, dtype=bf16)
                    v_vec_f32 = flir.arith.extf(v4f32_ty, arith.as_value(v_vec_bf16))

                    new_acc2 = []
                    for ei in range_constexpr(elems_per_thread):
                        v_f32 = vector.extract(v_vec_f32, static_position=[ei], dynamic_position=[])
                        new_acc2.append(acc[ei] + p * v_f32)
                    acc = new_acc2

            # ================================================================
            # Cross-wave merge (if num_waves > 1)
            # ================================================================
            if num_waves > 1:
                # Store partial output per wave
                for ei in range_constexpr(elems_per_thread):
                    out_lds_idx = wave_id * c_hdim + lane * c_ept + arith.index(ei)
                    memref.store(_unwrap(acc[ei]), smem_out, [_unwrap(out_lds_idx)])
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
                    old_scale = flydsl_math.exp2(arith.as_value((merge_emax - n_merge_max) * c_log2e))
                    new_scale = flydsl_math.exp2(arith.as_value((w_max - n_merge_max) * c_log2e))

                    merge_esum = merge_esum * old_scale + w_sum * new_scale

                    new_merge_acc = []
                    for ei in range_constexpr(elems_per_thread):
                        lds_idx = arith.index(w) * c_hdim + lane * c_ept + arith.index(ei)
                        w_val = memref.load(smem_out, [arith.as_value(lds_idx)])
                        new_merge_acc.append(merge_acc[ei] * old_scale + w_val * new_scale)
                    merge_acc = new_merge_acc
                    merge_emax = n_merge_max

                final_emax = merge_emax
                final_esum = merge_esum
                final_acc = merge_acc
            else:
                final_emax = e_max
                final_esum = e_sum
                final_acc = acc

            # ================================================================
            # Store partial output
            # ================================================================
            att_out_base = ((cur_batch * c_num_q + cur_q_head) * c_max_splits + split_kv_id) * c_hdim
            inv_sum = arith.constant(1.0, type=f32) / final_esum

            for ei in range_constexpr(elems_per_thread):
                out_val = final_acc[ei] * inv_sum
                out_off = att_out_base + lane * c_ept + arith.index(ei)
                buffer_ops.buffer_store(arith.as_value(out_val), att_out_rsrc, out_off)

            att_lse_off = (cur_batch * c_num_q + cur_q_head) * c_max_splits + split_kv_id
            lse_val = final_emax + flydsl_math.log(arith.as_value(final_esum))
            buffer_ops.buffer_store(arith.as_value(lse_val), att_lse_rsrc, att_lse_off)

        @flir.jit
        def __call__(
            self: flir.T.i64,
            arg_q: lambda: T.memref(DYN, T.bf16()),
            arg_k: lambda: T.memref(DYN, T.bf16()),
            arg_v: lambda: T.memref(DYN, T.bf16()),
            arg_att_out: lambda: T.memref(DYN, T.f32()),
            arg_att_lse: lambda: T.memref(DYN, T.f32()),
            arg_kv_indptr: lambda: T.memref(DYN, T.i32()),
            arg_kv_indices: lambda: T.memref(DYN, T.i32()),
            arg_num_kv_splits: lambda: T.memref(DYN, T.i32()),
            c_sm_scale: lambda: I.f32,
            c_batch: lambda: I.index,
            c_num_q_heads_idx: lambda: I.index,
            c_head_dim_idx: lambda: I.index,
            c_num_kv_heads_idx: lambda: I.index,
            c_max_kv_splits_idx: lambda: I.index,
            stream_ptr: lambda: I.i64,
        ):
            c1 = arith.constant(1, index=True)
            bdx = arith.constant(STAGE1_THREADS, index=True)
            stream_token = stream_ptr_to_async_token(stream_ptr)
            flir.gpu_ext.LaunchFuncOp(
                [s1_module_name, "stage1_kernel"],
                grid_size=(c_batch, c_num_q_heads_idx, c_max_kv_splits_idx),
                block_size=(bdx, c1, c1),
                kernel_operands=[
                    arg_q,
                    arg_k,
                    arg_v,
                    arg_att_out,
                    arg_att_lse,
                    arg_kv_indptr,
                    arg_kv_indices,
                    arg_num_kv_splits,
                    c_sm_scale,
                    c_num_q_heads_idx,
                    c_head_dim_idx,
                    c_num_kv_heads_idx,
                    c_max_kv_splits_idx,
                ],
                async_dependencies=[stream_token],
            )

    # =========================================================================
    # Stage 2: Merge across splits
    # =========================================================================

    class _Stage2(flir.MlirModule):
        GPU_MODULE_NAME = s2_module_name
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}">']

        def init_gpu_module(self):
            s2_allocator.finalize()

        @flir.kernel
        def stage2_kernel(
            self: flir.T.i64,
            arg_att_out: lambda: T.memref(DYN, T.f32()),
            arg_att_lse: lambda: T.memref(DYN, T.f32()),
            arg_o: lambda: T.memref(DYN, T.bf16()),
            arg_kv_indptr: lambda: T.memref(DYN, T.i32()),
            arg_num_kv_splits: lambda: T.memref(DYN, T.i32()),
            c_num_q_heads_idx: lambda: I.index,
            c_head_dim_idx: lambda: I.index,
            c_max_kv_splits_idx: lambda: I.index,
        ):
            f32 = ir.F32Type.get()
            bf16 = ir.BF16Type.get()
            i32 = ir.IntegerType.get_signless(32)
            idx_type = ir.IndexType.get()

            tid = flir.thread_idx("x")
            cur_batch = flir.block_idx("x")
            cur_q_head = flir.block_idx("y")

            c0 = arith.index(0)
            c1 = arith.index(1)

            lane = tid  # single wave, tid = lane

            c_hdim = arith.index(head_dim)
            c_num_q = arith.index(num_q_heads)
            c_ept = arith.index(elems_per_thread)
            c_max_splits = arith.index(max_kv_splits)

            att_out_rsrc = buffer_ops.create_buffer_resource(_unwrap(arg_att_out), max_size=True)
            att_lse_rsrc = buffer_ops.create_buffer_resource(_unwrap(arg_att_lse), max_size=True)
            o_rsrc = buffer_ops.create_buffer_resource(_unwrap(arg_o), max_size=True)
            indptr_rsrc = buffer_ops.create_buffer_resource(_unwrap(arg_kv_indptr), max_size=True)
            splits_rsrc = buffer_ops.create_buffer_resource(_unwrap(arg_num_kv_splits), max_size=True)

            kv_start_i32 = buffer_ops.buffer_load(indptr_rsrc, cur_batch, vec_width=1, dtype=i32)
            kv_end_i32 = buffer_ops.buffer_load(indptr_rsrc, cur_batch + c1, vec_width=1, dtype=i32)
            seq_len_i32 = kv_end_i32 - kv_start_i32
            kv_splits_i32 = buffer_ops.buffer_load(splits_rsrc, cur_batch, vec_width=1, dtype=i32)

            seq_len_idx = arith.index_cast(idx_type, seq_len_i32)
            kv_splits_idx = arith.index_cast(idx_type, kv_splits_i32)

            neg_inf_val = arith.constant(float("-inf"), type=f32)
            zero_f32 = arith.constant(0.0, type=f32)
            c_min_block_kv = arith.index(32)

            # Compute kv_len_per_split (same formula as stage1)
            inner_cdiv = (seq_len_idx + kv_splits_idx - c1) // kv_splits_idx
            kv_len_per_split = ((inner_cdiv + c_min_block_kv - c1) // c_min_block_kv) * c_min_block_kv

            # Base offsets for this (batch, head)
            bh_base = cur_batch * c_num_q + cur_q_head
            att_out_bh = bh_base * c_max_splits  # * head_dim added per element
            att_lse_bh = bh_base * c_max_splits

            # Merge across splits
            e_max = neg_inf_val
            e_sum = zero_f32
            acc = [zero_f32 for _ in range_constexpr(elems_per_thread)]

            for s in range_constexpr(max_kv_splits):
                s_idx = arith.index(s)
                # Check if this split has work
                split_start = kv_len_per_split * s_idx
                split_end_raw = split_start + kv_len_per_split
                split_end = arith.select(split_end_raw < seq_len_idx, split_end_raw, seq_len_idx)
                split_has_work = split_end > split_start

                # Load LSE for this split
                lse_off = att_lse_bh + s_idx
                lse_val = buffer_ops.buffer_load(att_lse_rsrc, lse_off, vec_width=1, dtype=f32)
                # If no work, set lse to -inf
                lse_val = arith.select(split_has_work, lse_val, neg_inf_val)

                # Online merge
                n_e_max = arith.maximum(e_max, lse_val)
                old_scale = flydsl_math.exp(arith.as_value(e_max - n_e_max))
                exp_logic = flydsl_math.exp(arith.as_value(lse_val - n_e_max))

                e_sum = e_sum * old_scale + exp_logic

                # Load and accumulate partial output
                att_out_split_base = (att_out_bh + s_idx) * c_hdim
                new_acc = []
                for ei in range_constexpr(elems_per_thread):
                    v_off = att_out_split_base + lane * c_ept + arith.index(ei)
                    v_val = buffer_ops.buffer_load(att_out_rsrc, v_off, vec_width=1, dtype=f32)
                    v_val = arith.select(split_has_work, v_val, zero_f32)
                    new_acc.append(acc[ei] * old_scale + exp_logic * v_val)
                acc = new_acc
                e_max = n_e_max

            # Normalize and store final output
            o_base = (cur_batch * c_num_q + cur_q_head) * c_hdim
            inv_sum = arith.constant(1.0, type=f32) / e_sum

            for ei in range_constexpr(elems_per_thread):
                out_f32 = acc[ei] * inv_sum
                out_bf16 = arith.trunc_f(bf16, out_f32)
                o_off = o_base + lane * c_ept + arith.index(ei)
                buffer_ops.buffer_store(out_bf16, o_rsrc, o_off)

        @flir.jit
        def __call__(
            self: flir.T.i64,
            arg_att_out: lambda: T.memref(DYN, T.f32()),
            arg_att_lse: lambda: T.memref(DYN, T.f32()),
            arg_o: lambda: T.memref(DYN, T.bf16()),
            arg_kv_indptr: lambda: T.memref(DYN, T.i32()),
            arg_num_kv_splits: lambda: T.memref(DYN, T.i32()),
            c_batch: lambda: I.index,
            c_num_q_heads_idx: lambda: I.index,
            c_head_dim_idx: lambda: I.index,
            c_max_kv_splits_idx: lambda: I.index,
            stream_ptr: lambda: I.i64,
        ):
            c1 = arith.constant(1, index=True)
            bdx = arith.constant(STAGE2_THREADS, index=True)
            stream_token = stream_ptr_to_async_token(stream_ptr)
            flir.gpu_ext.LaunchFuncOp(
                [s2_module_name, "stage2_kernel"],
                grid_size=(c_batch, c_num_q_heads_idx, c1),
                block_size=(bdx, c1, c1),
                kernel_operands=[
                    arg_att_out,
                    arg_att_lse,
                    arg_o,
                    arg_kv_indptr,
                    arg_num_kv_splits,
                    c_num_q_heads_idx,
                    c_head_dim_idx,
                    c_max_kv_splits_idx,
                ],
                async_dependencies=[stream_token],
            )

    s1 = _Stage1()
    s2 = _Stage2()
    exe_s1 = flydsl.compile(s1)
    exe_s2 = flydsl.compile(s2)
    return exe_s1, exe_s2, max_kv_splits
