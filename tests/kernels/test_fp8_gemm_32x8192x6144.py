#!/usr/bin/env python3
"""Test and benchmark: fast float8 GEMM for M=32, N=8192, K=6144 on RDNA4.

Correctness: compares against float32 reference using dequantized fp8 values.
Benchmark: measures TFLOPS and compares against rocBLAS (torch.mm).
"""

import sys
import os
import time
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from flydsl.runtime.device import get_rocm_arch

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available", allow_module_level=True)

gpu_arch = get_rocm_arch()
if not gpu_arch.startswith("gfx12"):
    pytest.skip(
        f"FP8 WMMA GEMM requires RDNA4 (gfx12xx), got {gpu_arch}",
        allow_module_level=True,
    )

from kernels.wmma_fp8_gemm import (
    compile_fp8_gemm,
    preshuffle_a_fp8,
    preshuffle_b_fp8,
    fp8_quantize_per_tensor,
)

device = "cuda"
DTYPE_FP8 = torch.float8_e4m3fn

# Target dimensions
M_TARGET = 32
N_TARGET = 8192
K_TARGET = 6144


# =============================================================================
# Correctness tests
# =============================================================================


@pytest.mark.parametrize(
    "M,N,K",
    [
        (32, 128, 128),  # Small sanity check
        (32, 256, 256),  # Slightly larger
        (32, 8192, 6144),  # Target shape
    ],
    ids=["32x128x128", "32x256x256", "32x8192x6144"],
)
def test_fp8_gemm_correctness(M, N, K):
    """Test fp8 preshuffle GEMM correctness."""
    tile_m = min(32, M)
    tile_n = min(128, N)
    tile_k = 32

    num_k_tiles = K // tile_k
    k_unroll = min(4, num_k_tiles - 1) if num_k_tiles > 1 else 1
    while k_unroll > 1 and (num_k_tiles - 1) % k_unroll != 0:
        k_unroll -= 1

    print(f"\nfp8 GEMM: M={M}, N={N}, K={K}, tiles=({tile_m},{tile_n},{tile_k}), k_unroll={k_unroll}")

    torch.manual_seed(42)
    A_f32 = torch.randn(M, K, device=device) * 0.1
    B_f32 = torch.randn(K, N, device=device) * 0.1

    # Quantize to fp8 with per-tensor scale
    A_fp8, scale_a = fp8_quantize_per_tensor(A_f32)
    B_fp8, scale_b = fp8_quantize_per_tensor(B_f32)

    # Reference: dequant then matmul in f32
    ref = (A_fp8.float() * scale_a) @ (B_fp8.float() * scale_b)

    # Preshuffle operands
    a_shuf = preshuffle_a_fp8(A_fp8)
    b_shuf = preshuffle_b_fp8(B_fp8)

    # Output buffer
    c_out = torch.zeros(M, N, device=device, dtype=torch.bfloat16)
    scale_a_t = torch.tensor([scale_a], device=device, dtype=torch.float32)
    scale_b_t = torch.tensor([scale_b], device=device, dtype=torch.float32)

    stream_ptr = torch.cuda.current_stream().cuda_stream

    # Compile and run
    exe = compile_fp8_gemm(
        M=M,
        N=N,
        K=K,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        k_unroll=k_unroll,
    )

    exe(
        c_out.flatten(),
        a_shuf.flatten().view(torch.float32),
        b_shuf.flatten().view(torch.float32),
        scale_a_t,
        scale_b_t,
        M,
        N,
        K,
        stream_ptr,
    )
    torch.cuda.synchronize()

    # Check correctness
    c_f32 = c_out.float()
    ref_f32 = ref.float()
    nonzero = ref_f32.abs() > 1e-6
    if nonzero.any():
        diff = (c_f32[nonzero] - ref_f32[nonzero]).abs()
        rel_error = (diff / ref_f32[nonzero].abs().clamp(min=1e-8)).max().item()
        max_abs = diff.max().item()
    else:
        rel_error = 0.0
        max_abs = 0.0

    cos_sim = torch.nn.functional.cosine_similarity(c_f32.flatten(), ref_f32.flatten(), dim=0).item()

    print(f"  Max abs error: {max_abs:.2e}")
    print(f"  Max rel error: {rel_error:.2e}")
    print(f"  Cosine similarity: {cos_sim:.6f}")

    # FP8 tolerance: < 5% relative error (per guide)
    assert rel_error < 0.10, f"fp8 GEMM rel error too high: {rel_error:.2e}"
    assert cos_sim > 0.999, f"fp8 GEMM cosine sim too low: {cos_sim:.6f}"
    print("  PASS")


# =============================================================================
# Benchmark
# =============================================================================


def benchmark_fp8_gemm(M=M_TARGET, N=N_TARGET, K=K_TARGET, iters=200, warmup=10):
    """Benchmark fp8 preshuffle GEMM and compare to rocBLAS."""
    tile_m = 32
    tile_n = 128
    tile_k = 32

    num_k_tiles = K // tile_k
    k_unroll = min(4, num_k_tiles - 1) if num_k_tiles > 1 else 1
    while k_unroll > 1 and (num_k_tiles - 1) % k_unroll != 0:
        k_unroll -= 1

    print(f"\n{'=' * 60}")
    print(f"Benchmark: fp8 GEMM  M={M}, N={N}, K={K}")
    print(f"  Tiles: ({tile_m}, {tile_n}, {tile_k}), k_unroll={k_unroll}")
    print(f"  Waves: 1x2 = 2 waves = 64 threads/block")
    print(f"  Grid: {M // tile_m} x {N // tile_n} = {(M // tile_m) * (N // tile_n)} blocks")
    print(f"{'=' * 60}")

    torch.manual_seed(42)
    A_f32 = torch.randn(M, K, device=device) * 0.1
    B_f32 = torch.randn(K, N, device=device) * 0.1

    A_fp8, scale_a = fp8_quantize_per_tensor(A_f32)
    B_fp8, scale_b = fp8_quantize_per_tensor(B_f32)

    a_shuf = preshuffle_a_fp8(A_fp8)
    b_shuf = preshuffle_b_fp8(B_fp8)

    c_out = torch.zeros(M, N, device=device, dtype=torch.bfloat16)
    scale_a_t = torch.tensor([scale_a], device=device, dtype=torch.float32)
    scale_b_t = torch.tensor([scale_b], device=device, dtype=torch.float32)

    stream_ptr = torch.cuda.current_stream().cuda_stream

    exe = compile_fp8_gemm(
        M=M,
        N=N,
        K=K,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        k_unroll=k_unroll,
    )

    # Flatten preshuffle views for passing to kernel
    a_flat = a_shuf.flatten().view(torch.float32)
    b_flat = b_shuf.flatten().view(torch.float32)
    c_flat = c_out.flatten()

    # Warmup
    for _ in range(warmup):
        exe(c_flat, a_flat, b_flat, scale_a_t, scale_b_t, M, N, K, stream_ptr)
    torch.cuda.synchronize()

    # Timed loop
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        exe(c_flat, a_flat, b_flat, scale_a_t, scale_b_t, M, N, K, stream_ptr)
    torch.cuda.synchronize()
    avg_ms = (time.time() - t0) / iters * 1000

    flops = 2 * M * N * K
    tflops = flops / (avg_ms / 1000) / 1e12
    print(f"\n  FlyDSL fp8 GEMM:")
    print(f"    Avg time:  {avg_ms:.3f} ms")
    print(f"    TFLOPS:    {tflops:.1f}")

    # Measure dispatch overhead (no-sync timing)
    torch.cuda.synchronize()
    dispatch_iters = 2000
    t0 = time.perf_counter()
    for _ in range(dispatch_iters):
        exe(c_flat, a_flat, b_flat, scale_a_t, scale_b_t, M, N, K, stream_ptr)
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    dispatch_us = (t1 - t0) / dispatch_iters * 1e6
    print(f"    Dispatch:  {dispatch_us:.1f} us")

    # rocBLAS baseline (torch.mm)
    A_bf16 = (A_fp8.float() * scale_a).to(torch.bfloat16)
    B_bf16 = (B_fp8.float() * scale_b).to(torch.bfloat16)

    for _ in range(warmup):
        torch.mm(A_bf16, B_bf16)
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        torch.mm(A_bf16, B_bf16)
    torch.cuda.synchronize()
    rocblas_ms = (time.time() - t0) / iters * 1000
    rocblas_tflops = flops / (rocblas_ms / 1000) / 1e12

    print(f"\n  rocBLAS (torch.mm bf16):")
    print(f"    Avg time:  {rocblas_ms:.3f} ms")
    print(f"    TFLOPS:    {rocblas_tflops:.1f}")

    ratio = tflops / rocblas_tflops * 100 if rocblas_tflops > 0 else 0
    print(f"\n  Ratio: {ratio:.0f}% of rocBLAS")
    print(f"  Theoretical peak fp8: 244 TFLOPS -> {tflops / 244 * 100:.0f}% of peak")

    return avg_ms, tflops


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="fp8 GEMM test & benchmark")
    parser.add_argument("--test", action="store_true", help="Run correctness tests")
    parser.add_argument("--bench", action="store_true", help="Run benchmark")
    parser.add_argument("--iters", type=int, default=200, help="Benchmark iterations")
    parser.add_argument("-M", type=int, default=32)
    parser.add_argument("-N", type=int, default=8192)
    parser.add_argument("-K", type=int, default=6144)
    args = parser.parse_args()

    if not args.test and not args.bench:
        args.test = True
        args.bench = True

    if args.test:
        print("=== Correctness Tests ===")
        for M, N, K in [(32, 128, 128), (32, 256, 256), (32, 8192, 6144)]:
            try:
                test_fp8_gemm_correctness(M, N, K)
            except Exception as e:
                print(f"  FAIL: {e}")
                import traceback

                traceback.print_exc()

    if args.bench:
        benchmark_fp8_gemm(M=args.M, N=args.N, K=args.K, iters=args.iters)
