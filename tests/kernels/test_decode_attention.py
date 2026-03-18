#!/usr/bin/env python3
"""Test and benchmark FlyDSL decode attention kernel."""

import math
import time
import torch
import ctypes

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from kernels.wmma_decode_attention import compile_decode_attention


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
    """Create test data for decode attention."""
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


def run_flydsl_attention(
    exe, q, k_buffer, v_buffer, o, kv_indptr, kv_indices, sm_scale
):
    """Run the FlyDSL attention kernel."""
    batch, num_heads, head_dim = q.shape
    num_kv_heads = k_buffer.shape[1]

    # Flatten tensors for buffer_resource access
    q_flat = q.contiguous().view(-1)
    k_flat = k_buffer.contiguous().view(-1)
    v_flat = v_buffer.contiguous().view(-1)
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
        batch,  # c_batch
        num_heads,  # c_num_q_heads_idx
        head_dim,  # c_head_dim_idx
        num_kv_heads,  # c_num_kv_heads_idx
        stream_ptr,
    )

    return o


def test_correctness():
    """Test FlyDSL decode attention against reference."""
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128

    print("=== Compiling FlyDSL Decode Attention ===")
    exe = compile_decode_attention(
        num_q_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_kv_len=1024,
        num_waves=4,
    )
    print("Compilation successful!")

    print("\n=== Correctness Test ===")
    for bs in [1, 4]:
        for kv_len in [64, 128, 256, 512]:
            q, k_buf, v_buf, o, kv_indptr, kv_indices, sm_scale = create_test_data(
                bs, num_heads, num_kv_heads, head_dim, kv_len
            )

            # FlyDSL
            o_fly = o.clone()
            run_flydsl_attention(
                exe, q, k_buf, v_buf, o_fly, kv_indptr, kv_indices, sm_scale
            )
            torch.cuda.synchronize()

            # Reference
            o_ref = reference_gqa_attention(
                q, k_buf, v_buf, kv_indptr, kv_indices, sm_scale
            )

            cos_sim = torch.nn.functional.cosine_similarity(
                o_fly.flatten().float(), o_ref.flatten().float(), dim=0
            ).item()
            max_diff = (o_fly.float() - o_ref.float()).abs().max().item()
            status = "PASS" if cos_sim > 0.99 else "FAIL"
            print(
                f"  BS={bs}, KV={kv_len}: cos_sim={cos_sim:.6f}, max_diff={max_diff:.4e} {status}"
            )


def benchmark():
    """Benchmark FlyDSL decode attention."""
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128

    print("\n=== Compiling for benchmark ===")
    exe = compile_decode_attention(
        num_q_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_kv_len=1024,
        num_waves=4,
    )
    print("\n=== FlyDSL Decode Attention Benchmark (Qwen3-8B) ===")
    print(f"num_heads={num_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}")
    print(f"{'BS':>4} | {'KV_len':>7} | {'FlyDSL ms':>10} | {'FlyDSL us':>10}")
    print("-" * 50)

    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    kv_lens = [256, 512, 768, 1024]
    warmup = 50
    iters = 200

    for kv_len in kv_lens:
        for bs in batch_sizes:
            q, k_buf, v_buf, o, kv_indptr, kv_indices, sm_scale = create_test_data(
                bs, num_heads, num_kv_heads, head_dim, kv_len
            )

            # Warmup
            for _ in range(warmup):
                run_flydsl_attention(
                    exe, q, k_buf, v_buf, o, kv_indptr, kv_indices, sm_scale
                )
            torch.cuda.synchronize()

            # Benchmark
            start = time.perf_counter()
            for _ in range(iters):
                run_flydsl_attention(
                    exe, q, k_buf, v_buf, o, kv_indptr, kv_indices, sm_scale
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
