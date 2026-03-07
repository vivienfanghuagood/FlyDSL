#!/usr/bin/env python3
"""Test and benchmark WMMA W4A16 GEMV kernel for RDNA4 (gfx12xx).

Optimized for small-M inference (decode phase): M=16..64, large N and K.
"""

import sys
import os
import time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from flydsl.runtime.device import get_rocm_arch

device = "cuda"
gpu_arch = get_rocm_arch()


def test_correctness(M, N, K, group_size=128):
    """Test W4A16 GEMV correctness."""
    from kernels.wmma_w4a16_gemv import compile_wmma_w4a16_gemv, quantize_int4_symmetric

    print(f"\nw4a16: M={M}, N={N}, K={K}, gs={group_size}")

    torch.manual_seed(42)
    A_bf16 = (torch.randn(M, K, device=device) * 0.1).to(torch.bfloat16)
    B_f32 = torch.randn(K, N, device=device) * 0.5

    # Quantize (returns transposed layout)
    B_packed_t, scales_t = quantize_int4_symmetric(B_f32.cpu(), group_size=group_size)
    B_packed_t = B_packed_t.to(device)
    scales_t = scales_t.to(device)

    # Reference: dequantize from transposed layout and matmul
    K_dim, N_dim = B_f32.shape
    num_groups = K_dim // group_size
    # Untranspose for reference
    B_packed_kn = B_packed_t.t().contiguous()  # [K//2, N]
    scales_kn = scales_t.t().contiguous()  # [num_groups, N]

    B_deq = torch.zeros(K_dim, N_dim, device=device, dtype=torch.float32)
    for g in range(num_groups):
        k_start = g * group_size
        k_end = k_start + group_size
        for k in range(k_start, k_end):
            k_packed = k // 2
            is_high = k % 2
            if is_high:
                nibble = (B_packed_kn[k_packed] >> 4) & 0xF
            else:
                nibble = B_packed_kn[k_packed] & 0xF
            B_deq[k] = (nibble.float() - 8.0) * scales_kn[g]

    ref = A_bf16.float() @ B_deq

    # Our kernel
    c_out = torch.zeros(M, N, device=device, dtype=torch.bfloat16)
    stream_ptr = torch.cuda.current_stream().cuda_stream

    exe = compile_wmma_w4a16_gemv(
        M=M,
        N=N,
        K=K,
        tile_n=128,
        group_size=group_size,
        num_waves=4,
    )

    exe(
        c_out.flatten(),
        A_bf16.flatten(),
        B_packed_t.flatten().view(torch.float32),
        scales_t.flatten(),
        M,
        N,
        K,
        stream_ptr,
    )
    torch.cuda.synchronize()

    c_f32 = c_out.float()
    ref_f32 = ref.float()

    max_abs = (c_f32 - ref_f32).abs().max().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        c_f32.flatten(), ref_f32.flatten(), dim=0
    ).item()
    mask = ref_f32.abs() > 0.05
    if mask.any():
        rel_error = (
            ((c_f32[mask] - ref_f32[mask]).abs() / ref_f32[mask].abs()).max().item()
        )
    else:
        rel_error = 0.0

    print(f"  Max abs error: {max_abs:.2e}")
    print(f"  Cosine sim: {cos_sim:.6f}")
    print(f"  Max rel error (|ref|>0.05): {rel_error:.2e}")
    ok = cos_sim > 0.999
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def benchmark(M, N, K, group_size=128, warmup=50, iters=500):
    """Benchmark W4A16 GEMV."""
    from kernels.wmma_w4a16_gemv import compile_wmma_w4a16_gemv, quantize_int4_symmetric

    torch.manual_seed(42)
    A_bf16 = (torch.randn(M, K, device=device) * 0.1).to(torch.bfloat16)
    B_f32 = torch.randn(K, N, device=device) * 0.5
    B_packed_t, scales_t = quantize_int4_symmetric(B_f32.cpu(), group_size=group_size)
    B_packed_t = B_packed_t.to(device)
    scales_t = scales_t.to(device)

    c_out = torch.zeros(M, N, device=device, dtype=torch.bfloat16)
    stream_ptr = torch.cuda.current_stream().cuda_stream

    exe = compile_wmma_w4a16_gemv(
        M=M,
        N=N,
        K=K,
        tile_n=128,
        group_size=group_size,
        num_waves=4,
    )

    for _ in range(warmup):
        exe(
            c_out.flatten(),
            A_bf16.flatten(),
            B_packed_t.flatten().view(torch.float32),
            scales_t.flatten(),
            M,
            N,
            K,
            stream_ptr,
        )
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        exe(
            c_out.flatten(),
            A_bf16.flatten(),
            B_packed_t.flatten().view(torch.float32),
            scales_t.flatten(),
            M,
            N,
            K,
            stream_ptr,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    avg_ms = elapsed / iters * 1000

    # Bandwidth: dominated by B weight reads
    # B: N * K/2 bytes (int4), scales: N * K/gs * 4 bytes
    # A: M * K * 2 bytes (bf16, but cached), C: M * N * 2 bytes
    bytes_moved = N * K * 0.5 + N * (K // group_size) * 4 + M * K * 2 + M * N * 2
    bw_gbs = bytes_moved / avg_ms / 1e6

    return avg_ms, bw_gbs


def benchmark_pytorch_bf16(M, N, K, warmup=50, iters=500):
    A = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    B = torch.randn(K, N, device=device, dtype=torch.bfloat16)
    for _ in range(warmup):
        torch.mm(A, B)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        torch.mm(A, B)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    avg_ms = elapsed / iters * 1000
    bytes_moved = M * K * 2 + K * N * 2 + M * N * 2
    bw_gbs = bytes_moved / avg_ms / 1e6
    return avg_ms, bw_gbs


if __name__ == "__main__":
    print("=== Correctness Tests ===")
    all_ok = True
    for M, N, K in [
        (16, 128, 128),
        (16, 256, 256),
        (16, 4096, 4096),
        (32, 4096, 4096),
        (64, 4096, 4096),
    ]:
        try:
            ok = test_correctness(M, N, K)
            all_ok = all_ok and ok
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback

            traceback.print_exc()
            all_ok = False

    if not all_ok:
        print("\nSome tests FAILED, skipping benchmarks.")
        sys.exit(1)

    print("\n\n=== Benchmarks ===")
    print(
        f"{'M':>4} x {'N':>5} x {'K':>5} | {'INT4 ms':>8} | {'INT4 GB/s':>10} | {'PT bf16 ms':>10} | {'PT GB/s':>10} | {'speedup':>7}"
    )
    print("-" * 85)

    shapes = [
        (16, 4096, 4096),
        (16, 14336, 4096),
        (16, 4096, 14336),
        (32, 4096, 4096),
        (32, 14336, 4096),
        (32, 4096, 14336),
        (64, 4096, 4096),
        (64, 14336, 4096),
        (64, 4096, 14336),
    ]

    for M, N, K in shapes:
        try:
            our_ms, our_bw = benchmark(M, N, K)
            pt_ms, pt_bw = benchmark_pytorch_bf16(M, N, K)
            speedup = pt_ms / our_ms
            print(
                f"{M:>4} x {N:>5} x {K:>5} | {our_ms:>8.3f} | {our_bw:>8.1f}   | {pt_ms:>10.3f} | {pt_bw:>8.1f}   | {speedup:>6.2f}x"
            )
        except Exception as e:
            print(f"{M:>4} x {N:>5} x {K:>5} | FAILED: {e}")
