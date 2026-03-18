#!/usr/bin/env python3
"""Benchmark Triton decode attention with GPU-only timing."""

import math
import time
import torch
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from benchmarks.triton_decode_attention_baseline import (
    triton_decode_attention,
    create_test_data,
    _MIN_BLOCK_KV,
)
import triton


def benchmark_gpu_only():
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128

    print("=== Triton GPU-Only Timing (CUDA Events) ===")
    print(f"{'BS':>4} | {'KV':>6} | {'wall_us':>10} | {'gpu_us':>10}")
    print("-" * 45)

    warmup = 100
    iters = 500

    for kv_len in [256, 512, 1024]:
        for bs in [1, 4, 8, 16, 32]:
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

            # GPU-only timing
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
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
            end_event.record()
            torch.cuda.synchronize()

            gpu_ms = start_event.elapsed_time(end_event) / iters
            gpu_us = gpu_ms * 1000

            # Wall-clock
            torch.cuda.synchronize()
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
            wall_ms = (time.perf_counter() - start) / iters * 1000
            wall_us = wall_ms * 1000

            print(f"{bs:>4} | {kv_len:>6} | {wall_us:>10.1f} | {gpu_us:>10.1f}")
        print()


if __name__ == "__main__":
    benchmark_gpu_only()
