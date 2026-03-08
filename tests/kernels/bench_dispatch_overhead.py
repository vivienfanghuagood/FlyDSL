#!/usr/bin/env python3
"""Measure dispatch overhead for FlyDSL vs Triton."""

import math
import time
import torch
import sys, os

# Triton
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from benchmarks.triton_decode_attention_baseline import (
    triton_decode_attention,
    create_test_data,
    _MIN_BLOCK_KV,
)

# FlyDSL
from kernels.wmma_decode_attention_elemwise import compile_decode_attention_elemwise


def main():
    bs, num_heads, num_kv_heads, head_dim, kv_len = 1, 32, 8, 128, 256
    N = 5000

    # === Triton ===
    q, k_buf, v_buf, o, kv_indptr, kv_indices, num_splits, max_splits, sm_scale = (
        create_test_data(bs, num_heads, num_kv_heads, head_dim, kv_len)
    )

    for _ in range(200):
        triton_decode_attention(
            q, k_buf, v_buf, o, kv_indptr, kv_indices, num_splits, max_splits, sm_scale
        )
    torch.cuda.synchronize()

    # Dispatch only
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        triton_decode_attention(
            q, k_buf, v_buf, o, kv_indptr, kv_indices, num_splits, max_splits, sm_scale
        )
    t1 = time.perf_counter()
    triton_dispatch_us = (t1 - t0) / N * 1e6
    torch.cuda.synchronize()

    # With sync
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        triton_decode_attention(
            q, k_buf, v_buf, o, kv_indptr, kv_indices, num_splits, max_splits, sm_scale
        )
    torch.cuda.synchronize()
    triton_total_us = (time.perf_counter() - t0) / N * 1e6

    print(f"Triton dispatch (no sync): {triton_dispatch_us:.1f} us")
    print(f"Triton total (with sync):  {triton_total_us:.1f} us")
    print(f"Triton GPU compute:        {triton_total_us - triton_dispatch_us:.1f} us")
    print()

    # === FlyDSL ===
    exe = compile_decode_attention_elemwise(
        num_q_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_kv_len=kv_len,
        num_waves=4,
    )

    q2, k_buf2, v_buf2, o2, kv_indptr2, kv_indices2, sm_scale2 = (
        q,
        k_buf,
        v_buf,
        o.clone(),
        kv_indptr,
        kv_indices,
        sm_scale,
    )
    q_flat = q2.contiguous().view(-1)
    k_flat = k_buf2.contiguous().view(-1)
    v_flat = v_buf2.contiguous().view(-1)
    o_flat = o2.contiguous().view(-1)
    stream_ptr = torch.cuda.current_stream().cuda_stream

    for _ in range(200):
        exe(
            q_flat,
            k_flat,
            v_flat,
            o_flat,
            kv_indptr2,
            kv_indices2,
            sm_scale2,
            bs,
            num_heads,
            head_dim,
            num_kv_heads,
            stream_ptr,
        )
    torch.cuda.synchronize()

    # Dispatch only
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        exe(
            q_flat,
            k_flat,
            v_flat,
            o_flat,
            kv_indptr2,
            kv_indices2,
            sm_scale2,
            bs,
            num_heads,
            head_dim,
            num_kv_heads,
            stream_ptr,
        )
    t1 = time.perf_counter()
    fly_dispatch_us = (t1 - t0) / N * 1e6
    torch.cuda.synchronize()

    # With sync
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        exe(
            q_flat,
            k_flat,
            v_flat,
            o_flat,
            kv_indptr2,
            kv_indices2,
            sm_scale2,
            bs,
            num_heads,
            head_dim,
            num_kv_heads,
            stream_ptr,
        )
    torch.cuda.synchronize()
    fly_total_us = (time.perf_counter() - t0) / N * 1e6

    print(f"FlyDSL dispatch (no sync): {fly_dispatch_us:.1f} us")
    print(f"FlyDSL total (with sync):  {fly_total_us:.1f} us")
    print(f"FlyDSL GPU compute:        {fly_total_us - fly_dispatch_us:.1f} us")
    print()
    print(f"Dispatch overhead gap:     {fly_dispatch_us - triton_dispatch_us:.1f} us")


if __name__ == "__main__":
    main()
