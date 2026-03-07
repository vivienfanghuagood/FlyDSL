#!/usr/bin/env python3
"""Test and benchmark FlyDSL element-wise decode attention."""

import math
import time
import torch
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from kernels.wmma_decode_attention_elemwise import compile_decode_attention_elemwise


def reference_gqa_attention(q, k_buffer, v_buffer, kv_indptr, kv_indices, sm_scale):
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
            q_vec = q[b, h].float()
            k_mat = k_buffer[indices, kv_h].float()
            v_mat = v_buffer[indices, kv_h].float()
            scores = (q_vec @ k_mat.T) * sm_scale
            attn = torch.softmax(scores, dim=-1)
            out = attn @ v_mat
            o[b, h] = out.to(q.dtype)
    return o


def create_test_data(
    batch_size, num_heads, num_kv_heads, head_dim, kv_len, device="cuda"
):
    q = (
        torch.randn(
            batch_size, num_heads, head_dim, device=device, dtype=torch.bfloat16
        )
        * 0.1
    )
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
    kv_indptr = torch.arange(
        0, (batch_size + 1) * kv_len, kv_len, device=device, dtype=torch.int32
    )
    kv_indices = torch.arange(0, total_kv, device=device, dtype=torch.int32)
    o = torch.zeros(
        batch_size, num_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    sm_scale = 1.0 / math.sqrt(head_dim)
    return q, k_buffer, v_buffer, o, kv_indptr, kv_indices, sm_scale


def run_flydsl(exe, q, k_buf, v_buf, o, kv_indptr, kv_indices, sm_scale):
    batch, num_heads, head_dim = q.shape
    num_kv_heads = k_buf.shape[1]
    q_flat = q.contiguous().view(-1)
    k_flat = k_buf.contiguous().view(-1)
    v_flat = v_buf.contiguous().view(-1)
    o_flat = o.contiguous().view(-1)
    stream_ptr = torch.cuda.current_stream().cuda_stream
    exe(
        q_flat,
        k_flat,
        v_flat,
        o_flat,
        kv_indptr,
        kv_indices,
        sm_scale,
        batch,
        num_heads,
        head_dim,
        num_kv_heads,
        stream_ptr,
    )


def test_correctness():
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128

    print("=== Element-wise Decode Attention Correctness ===")
    exe = compile_decode_attention_elemwise(
        num_q_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_kv_len=1024,
        num_waves=4,
    )

    for bs in [1, 4, 8]:
        for kv_len in [128, 256, 512, 1024]:
            q, k_buf, v_buf, o, kv_indptr, kv_indices, sm_scale = create_test_data(
                bs, num_heads, num_kv_heads, head_dim, kv_len
            )
            run_flydsl(exe, q, k_buf, v_buf, o, kv_indptr, kv_indices, sm_scale)
            o_ref = reference_gqa_attention(
                q, k_buf, v_buf, kv_indptr, kv_indices, sm_scale
            )
            cos_sim = torch.nn.functional.cosine_similarity(
                o.flatten().float(), o_ref.flatten().float(), dim=0
            ).item()
            max_diff = (o.float() - o_ref.float()).abs().max().item()
            status = "PASS" if cos_sim > 0.999 else "FAIL"
            print(
                f"  BS={bs}, KV={kv_len}: cos_sim={cos_sim:.6f}, max_diff={max_diff:.4e} [{status}]"
            )


def benchmark_gpu_only():
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128

    print("\n=== Element-wise GPU-Only Timing ===")
    print(f"{'BS':>4} | {'KV':>6} | {'gpu_us':>10} | {'wall_us':>10}")
    print("-" * 45)

    warmup = 100
    iters = 500

    exe = compile_decode_attention_elemwise(
        num_q_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_kv_len=1024,
        num_waves=4,
    )

    for kv_len in [256, 512, 1024]:
        for bs in [1, 4, 8, 16, 32]:
            q, k_buf, v_buf, o, kv_indptr, kv_indices, sm_scale = create_test_data(
                bs, num_heads, num_kv_heads, head_dim, kv_len
            )

            # Warmup
            for _ in range(warmup):
                run_flydsl(exe, q, k_buf, v_buf, o, kv_indptr, kv_indices, sm_scale)
            torch.cuda.synchronize()

            # GPU timing
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(iters):
                run_flydsl(exe, q, k_buf, v_buf, o, kv_indptr, kv_indices, sm_scale)
            end_event.record()
            torch.cuda.synchronize()
            gpu_us = start_event.elapsed_time(end_event) / iters * 1000

            # Wall timing
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(iters):
                run_flydsl(exe, q, k_buf, v_buf, o, kv_indptr, kv_indices, sm_scale)
            torch.cuda.synchronize()
            wall_us = (time.perf_counter() - start) / iters * 1e6

            print(f"{bs:>4} | {kv_len:>6} | {gpu_us:>10.1f} | {wall_us:>10.1f}")
        print()


if __name__ == "__main__":
    test_correctness()
    benchmark_gpu_only()
