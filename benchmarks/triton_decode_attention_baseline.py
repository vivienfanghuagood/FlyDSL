#!/usr/bin/env python3
"""Standalone Triton decode attention baseline extracted from sglang.

This is the split-KV flash decoding approach for GQA decode attention.
Adapted from sglang/python/sglang/srt/layers/attention/triton_ops/decode_attention.py

Two stages:
  Stage1: Each KV-split computes partial softmax(Q @ K^T) @ V with LSE
  Stage2: Merge partial results across KV-splits using log-sum-exp

Simplified for benchmarking:
  - No paged KV cache (contiguous KV buffer per sequence)
  - No logit_cap, no xai_temperature
  - bf16 precision
"""

import math
import time
import torch
import triton
import triton.language as tl

# ============================================================================
# Stage 1: Split-KV attention
# ============================================================================

_MIN_BLOCK_KV = 32


@triton.jit
def _fwd_grouped_kernel_stage1(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale,
    kv_indptr,
    kv_indices,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)

    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        q = tl.load(Q + offs_q, mask=(mask_h[:, None]) & (mask_d[None, :]), other=0.0)
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=offs_n < split_kv_end,
                other=0,
            )
            offs_buf_k = (
                kv_loc[None, :] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_d[:, None]
            )
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(offs_n[None, :] < split_kv_end) & (mask_d[:, None]),
                other=0.0,
            )
            qk = tl.dot(q, k.to(q.dtype))
            qk *= sm_scale

            qk = tl.where(
                mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf")
            )

            offs_buf_v = (
                kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + offs_dv[None, :]
            )
            v = tl.load(
                V_Buffer + offs_buf_v,
                mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                other=0.0,
            )

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(v.dtype), v)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv[None, :]
        )
        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv[None, :]),
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // Lv
        tl.store(
            Att_Lse + offs_mid_o_1,
            e_max + tl.log(e_sum),
            mask=mask_h,
        )


# ============================================================================
# Stage 2: Merge KV-splits
# ============================================================================


@triton.jit
def _fwd_kernel_stage2(
    Mid_O,
    Mid_O_1,
    O,
    v_scale,
    kv_indptr,
    num_kv_splits,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    MAX_KV_SPLITS: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - tl.load(
        kv_indptr + cur_batch
    )
    kv_splits = tl.load(num_kv_splits + cur_batch)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = (cur_batch * stride_mid_ob + cur_head * stride_mid_oh) // Lv
    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )

    for split_kv_id in range(0, MAX_KV_SPLITS):
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            tv = tl.load(
                Mid_O + offs_v + split_kv_id * stride_mid_os, mask=mask_d, other=0.0
            )
            tlogic = tl.load(Mid_O_1 + offs_logic + split_kv_id * stride_mid_os // Lv)
            n_e_max = tl.maximum(tlogic, e_max)

            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    tl.store(
        O + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        acc / e_sum * v_scale,
        mask=mask_d,
    )


# ============================================================================
# Python wrapper
# ============================================================================


def triton_decode_attention(
    q,
    k_buffer,
    v_buffer,
    o,
    kv_indptr,
    kv_indices,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
):
    """Run sglang-style split-KV decode attention (GQA).

    Args:
        q: [batch, num_heads, head_dim] bf16
        k_buffer: [total_kv_tokens, num_kv_heads, head_dim] bf16
        v_buffer: [total_kv_tokens, num_kv_heads, head_dim] bf16
        o: [batch, num_heads, head_dim] bf16 (output)
        kv_indptr: [batch+1] int32 (CSR-style pointers into kv_indices)
        kv_indices: [total_kv_tokens] int32 (maps to positions in k/v_buffer)
        num_kv_splits: [batch] int32
        max_kv_splits: int (max across batch)
        sm_scale: float (1/sqrt(head_dim))
    """
    batch, num_heads, Lk = q.shape
    num_kv_heads = k_buffer.shape[1]
    Lv = v_buffer.shape[-1]
    kv_group_num = num_heads // num_kv_heads

    BLOCK_DMODEL = triton.next_power_of_2(Lk)
    BLOCK_DV = triton.next_power_of_2(Lv)
    BLOCK_N = 32
    # On RDNA4 (gfx12), WMMA requires M >= 16. So BLOCK_H must be >= 16
    # when tl.dot is used. For kv_group_num < 16, we pad BLOCK_H to 16.
    BLOCK_H = max(16, min(16, kv_group_num)) if kv_group_num > 1 else 1

    # Intermediate buffers
    att_out = torch.zeros(
        batch, num_heads, max_kv_splits, Lv, dtype=torch.float32, device=q.device
    )
    att_lse = torch.zeros(
        batch, num_heads, max_kv_splits, dtype=torch.float32, device=q.device
    )

    # Stage 1
    grid1 = (batch, triton.cdiv(num_heads, min(BLOCK_H, kv_group_num)), max_kv_splits)

    extra_kargs = {}
    num_stages = 1
    # ROCm tuning
    extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}

    _fwd_grouped_kernel_stage1[grid1](
        q,
        k_buffer,
        v_buffer,
        sm_scale,
        kv_indptr,
        kv_indices,
        att_out,
        att_lse,
        num_kv_splits,
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        kv_group_num=kv_group_num,
        q_head_num=num_heads,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK_N,
        BLOCK_H=BLOCK_H,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        Lk=Lk,
        Lv=Lv,
        num_warps=4,
        num_stages=num_stages,
        **extra_kargs,
    )

    # Stage 2
    grid2 = (batch, num_heads)
    _fwd_kernel_stage2[grid2](
        att_out,
        att_lse,
        o,
        1.0,
        kv_indptr,
        num_kv_splits,
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        o.stride(0),
        o.stride(1),
        MAX_KV_SPLITS=max_kv_splits,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
        num_warps=4,
        num_stages=2,
    )

    return o


# ============================================================================
# Reference implementation
# ============================================================================


def reference_gqa_attention(q, k_buffer, v_buffer, kv_indptr, kv_indices, sm_scale):
    """Reference GQA decode attention using PyTorch."""
    batch, num_heads, head_dim = q.shape
    num_kv_heads = k_buffer.shape[1]
    kv_group_num = num_heads // num_kv_heads

    o = torch.zeros_like(q)
    for b in range(batch):
        kv_start = kv_indptr[b].item()
        kv_end = kv_indptr[b + 1].item()
        indices = kv_indices[kv_start:kv_end]

        for h in range(num_heads):
            kv_h = h // kv_group_num
            q_vec = q[b, h].float()  # [head_dim]
            k_mat = k_buffer[indices, kv_h].float()  # [seq_len, head_dim]
            v_mat = v_buffer[indices, kv_h].float()  # [seq_len, head_dim]

            scores = (q_vec @ k_mat.T) * sm_scale  # [seq_len]
            attn = torch.softmax(scores, dim=-1)  # [seq_len]
            out = attn @ v_mat  # [head_dim]
            o[b, h] = out.to(q.dtype)

    return o


# ============================================================================
# Benchmark
# ============================================================================


def create_test_data(
    batch_size, num_heads, num_kv_heads, head_dim, kv_len, device="cuda"
):
    """Create test data for decode attention.

    For decode: each batch entry has 1 query token and kv_len KV tokens.
    All sequences share the same kv_len for simplicity.
    """
    # Q: [batch, num_heads, head_dim]
    q = (
        torch.randn(
            batch_size, num_heads, head_dim, device=device, dtype=torch.bfloat16
        )
        * 0.1
    )

    # KV buffer: contiguous, each sequence has kv_len tokens
    total_kv = batch_size * kv_len
    k_buffer = (
        torch.randn(
            total_kv, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16
        )
        * 0.1
    )
    v_buffer = (
        torch.randn(
            total_kv, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16
        )
        * 0.1
    )

    # kv_indptr: CSR pointers [0, kv_len, 2*kv_len, ...]
    kv_indptr = torch.arange(
        0, (batch_size + 1) * kv_len, kv_len, device=device, dtype=torch.int32
    )

    # kv_indices: identity mapping (contiguous)
    kv_indices = torch.arange(0, total_kv, device=device, dtype=torch.int32)

    # num_kv_splits: how many splits per sequence
    max_kv_splits = max(1, triton.cdiv(kv_len, _MIN_BLOCK_KV * 2))
    max_kv_splits = min(max_kv_splits, 32)  # cap
    num_kv_splits_tensor = torch.full(
        (batch_size,), max_kv_splits, device=device, dtype=torch.int32
    )

    # Output
    o = torch.zeros(
        batch_size, num_heads, head_dim, device=device, dtype=torch.bfloat16
    )

    sm_scale = 1.0 / math.sqrt(head_dim)

    return (
        q,
        k_buffer,
        v_buffer,
        o,
        kv_indptr,
        kv_indices,
        num_kv_splits_tensor,
        max_kv_splits,
        sm_scale,
    )


def test_correctness():
    """Verify Triton baseline matches reference."""
    # Qwen3-8B config
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128

    print("=== Correctness Test ===")
    for bs in [1, 4]:
        for kv_len in [128, 256, 512]:
            (
                q,
                k_buf,
                v_buf,
                o,
                kv_indptr,
                kv_indices,
                num_splits,
                max_splits,
                sm_scale,
            ) = create_test_data(bs, num_heads, num_kv_heads, head_dim, kv_len)

            # Triton
            o_triton = o.clone()
            triton_decode_attention(
                q,
                k_buf,
                v_buf,
                o_triton,
                kv_indptr,
                kv_indices,
                num_splits,
                max_splits,
                sm_scale,
            )

            # Reference
            o_ref = reference_gqa_attention(
                q, k_buf, v_buf, kv_indptr, kv_indices, sm_scale
            )

            cos_sim = torch.nn.functional.cosine_similarity(
                o_triton.flatten().float(), o_ref.flatten().float(), dim=0
            ).item()
            max_diff = (o_triton.float() - o_ref.float()).abs().max().item()
            status = "PASS" if cos_sim > 0.99 else "FAIL"
            print(
                f"  BS={bs}, KV={kv_len}: cos_sim={cos_sim:.6f}, max_diff={max_diff:.4e} {status}"
            )


def benchmark():
    """Benchmark Triton decode attention for Qwen3-8B."""
    # Qwen3-8B config
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128

    print("\n=== Triton Decode Attention Benchmark (Qwen3-8B) ===")
    print(f"num_heads={num_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}")
    print(f"{'BS':>4} | {'KV_len':>7} | {'Triton ms':>10} | {'Triton us':>10}")
    print("-" * 50)

    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    kv_lens = [256, 512, 768, 1024]

    warmup = 50
    iters = 200

    for kv_len in kv_lens:
        for bs in batch_sizes:
            (
                q,
                k_buf,
                v_buf,
                o,
                kv_indptr,
                kv_indices,
                num_splits,
                max_splits,
                sm_scale,
            ) = create_test_data(bs, num_heads, num_kv_heads, head_dim, kv_len)

            # Warmup
            for _ in range(warmup):
                triton_decode_attention(
                    q,
                    k_buf,
                    v_buf,
                    o,
                    kv_indptr,
                    kv_indices,
                    num_splits,
                    max_splits,
                    sm_scale,
                )
            torch.cuda.synchronize()

            # Benchmark
            start = time.perf_counter()
            for _ in range(iters):
                triton_decode_attention(
                    q,
                    k_buf,
                    v_buf,
                    o,
                    kv_indptr,
                    kv_indices,
                    num_splits,
                    max_splits,
                    sm_scale,
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            avg_ms = elapsed / iters * 1000
            avg_us = avg_ms * 1000

            print(f"{bs:>4} | {kv_len:>7} | {avg_ms:>10.4f} | {avg_us:>10.1f}")
        print()


if __name__ == "__main__":
    test_correctness()
    benchmark()
