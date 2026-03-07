"""FlyDSL Pure Element-wise Decode Attention for RDNA4 (gfx12xx, wave32).

No WMMA — everything is element-wise dot products with wave reductions.

For decode attention, Q is a single vector [1, head_dim]. The Q@K^T dot product
for each KV token is just a scalar dot product, which 32 lanes can compute via:
  - Each lane: multiply 4 Q elements × 4 K elements, sum locally
  - Wave-level cross-lane reduction (5 rounds of shuffle+add)
  - Result: score in lane 0

This avoids WMMA overhead and is likely faster for small BLOCK_N.

Design:
  - Grid: (batch, num_q_heads)
  - Workgroup: num_waves × 32 threads
  - Each wave processes interleaved blocks of BLOCK_N KV tokens
  - Each thread handles head_dim/32=4 elements
  - Process BLOCK_N tokens at a time:
    1. Load K for all BLOCK_N tokens, compute dot products via wave reduction
    2. Online softmax update
    3. Load V for all BLOCK_N tokens, accumulate P@V element-wise
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


def _unwrap(v):
    while hasattr(v, "_value"):
        v = v._value
    return v


# =============================================================================
# Kernel compiler
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
            q_vec_bf16 = buffer_ops.buffer_load(
                q_rsrc, q_off, vec_width=elems_per_thread, dtype=bf16
            )
            q_vec_f32 = flir.arith.extf(v_ept_f32_ty, arith.as_value(q_vec_bf16))

            # Extract Q elements for dot product
            q_elems = []
            for ei in range_constexpr(elems_per_thread):
                q_elems.append(
                    vector.extract(q_vec_f32, static_position=[ei], dynamic_position=[])
                )

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
                token_i32 = buffer_ops.buffer_load(
                    indices_rsrc, kv_global, vec_width=1, dtype=i32
                )
                token_idx = arith.index_cast(idx_type, token_i32)
                k_base = token_idx * kv_token_stride + kv_head_off

                # Load K for this token: each lane loads 4 bf16 elements
                k_off = k_base + lane * c_ept
                k_vec_bf16 = buffer_ops.buffer_load(
                    k_rsrc, k_off, vec_width=elems_per_thread, dtype=bf16
                )
                k_vec_f32 = flir.arith.extf(v_ept_f32_ty, arith.as_value(k_vec_bf16))

                # Dot product: Q . K (element-wise multiply + reduce)
                dot_local = zero_f32
                for ei in range_constexpr(elems_per_thread):
                    k_elem = vector.extract(
                        k_vec_f32, static_position=[ei], dynamic_position=[]
                    )
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
                v_vec_bf16 = buffer_ops.buffer_load(
                    v_rsrc, v_off, vec_width=elems_per_thread, dtype=bf16
                )
                v_vec_f32 = flir.arith.extf(v_ept_f32_ty, arith.as_value(v_vec_bf16))

                new_acc = []
                for ei in range_constexpr(elems_per_thread):
                    v_elem = vector.extract(
                        v_vec_f32, static_position=[ei], dynamic_position=[]
                    )
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
