#!/usr/bin/env python3
"""Benchmark FlyDSL decode attention with GPU-only timing (events)."""

import math
import time
import torch
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from kernels.wmma_decode_attention import compile_decode_attention


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


def benchmark_gpu_only():
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128

    exe = compile_decode_attention(
        num_q_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_kv_len=1024,
        num_waves=4,
    )

    print("=== GPU-Only Timing (CUDA Events) ===")
    print(f"{'BS':>4} | {'KV':>6} | {'wall_us':>10} | {'gpu_us':>10}")
    print("-" * 45)

    warmup = 100
    iters = 500

    for kv_len in [256, 512, 1024]:
        for bs in [1, 4, 8, 16, 32]:
            q, k_buf, v_buf, o, kv_indptr, kv_indices, sm_scale = create_test_data(
                bs, num_heads, num_kv_heads, head_dim, kv_len
            )

            # Warmup
            for _ in range(warmup):
                run_flydsl(exe, q, k_buf, v_buf, o, kv_indptr, kv_indices, sm_scale)
            torch.cuda.synchronize()

            # GPU-only timing
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            for _ in range(iters):
                run_flydsl(exe, q, k_buf, v_buf, o, kv_indptr, kv_indices, sm_scale)
            end_event.record()
            torch.cuda.synchronize()

            gpu_ms = start_event.elapsed_time(end_event) / iters
            gpu_us = gpu_ms * 1000

            # Wall-clock timing
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(iters):
                run_flydsl(exe, q, k_buf, v_buf, o, kv_indptr, kv_indices, sm_scale)
            torch.cuda.synchronize()
            wall_ms = (time.perf_counter() - start) / iters * 1000
            wall_us = wall_ms * 1000

            print(f"{bs:>4} | {kv_len:>6} | {wall_us:>10.1f} | {gpu_us:>10.1f}")
        print()


if __name__ == "__main__":
    benchmark_gpu_only()
