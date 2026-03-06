#!/usr/bin/env python3
"""Test WMMA Preshuffle GEMM kernel for RDNA4 (gfx12xx).

Tests correctness and performance of the WMMA preshuffle GEMM kernel
which uses the same compile_preshuffle_gemm_a8-style interface.
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
        f"WMMA preshuffle GEMM requires RDNA4 (gfx12xx), got {gpu_arch}",
        allow_module_level=True,
    )

from kernels.wmma_preshuffle_gemm import (
    compile_wmma_preshuffle_gemm,
    preshuffle_a_wmma,
    preshuffle_b_wmma,
)

device = "cuda"


# =============================================================================
# Correctness Tests
# =============================================================================

CORRECTNESS_SHAPES = [
    # (M, N, K, tile_m, tile_n, tile_k)
    (
        128,
        128,
        128,
        128,
        128,
        32,
    ),  # 1 block, 4 K-tiles (smallest valid with k_unroll=1)
    (128, 128, 256, 128, 128, 32),  # 1 block, 8 K-tiles
    (256, 256, 256, 128, 128, 32),  # 4 blocks
    (512, 512, 512, 128, 128, 32),  # medium
    (1024, 1024, 1024, 128, 128, 32),  # larger
]


@pytest.mark.parametrize(
    "M,N,K,tm,tn,tk",
    CORRECTNESS_SHAPES,
    ids=[f"{m}x{n}x{k}_t{tm}x{tn}x{tk}" for m, n, k, tm, tn, tk in CORRECTNESS_SHAPES],
)
@pytest.mark.parametrize("in_dtype", ["bf16", "fp16"])
def test_wmma_preshuffle_gemm_correctness(M, N, K, tm, tn, tk, in_dtype):
    """Test WMMA preshuffle GEMM correctness against PyTorch reference."""
    # Adjust k_unroll for small K
    num_k_tiles = K // tk
    # Need num_k_tiles > k_unroll and num_k_tiles % k_unroll == 0
    if num_k_tiles <= 1:
        if __name__ == "__main__":
            print(f"  SKIP: K={K} too small for tile_k={tk}")
            return
        pytest.skip(f"K={K} too small for tile_k={tk}")
    k_unroll = 1
    for ku in [4, 2, 1]:
        if num_k_tiles > ku and num_k_tiles % ku == 0:
            k_unroll = ku
            break

    torch_dtype = torch.bfloat16 if in_dtype == "bf16" else torch.float16

    print(
        f"\nCorrectness: M={M}, N={N}, K={K}, tile=({tm},{tn},{tk}), "
        f"dtype={in_dtype}, k_unroll={k_unroll}"
    )

    # Compile kernel
    exe = compile_wmma_preshuffle_gemm(
        M=M,
        N=N,
        K=K,
        tile_m=tm,
        tile_n=tn,
        tile_k=tk,
        in_dtype=in_dtype,
        out_dtype="bf16",
        k_unroll=k_unroll,
    )

    # Create inputs
    torch.manual_seed(42)
    A = torch.randn(M, K, device=device, dtype=torch_dtype) * 0.1
    B = torch.randn(K, N, device=device, dtype=torch_dtype) * 0.1

    # Pre-shuffle
    A_shuf = preshuffle_a_wmma(A).flatten()
    B_shuf = preshuffle_b_wmma(B).flatten()

    # Output
    C = torch.zeros(M, N, device=device, dtype=torch.bfloat16)

    # Dummy scales (unused for bf16/fp16)
    scale_a = torch.empty(0, device=device, dtype=torch.float32)
    scale_b = torch.empty(0, device=device, dtype=torch.float32)

    # Get stream pointer
    stream = torch.cuda.current_stream()
    stream_ptr = stream.cuda_stream

    # Run kernel
    exe(C.flatten(), A_shuf, B_shuf, scale_a, scale_b, M, N, K, stream_ptr)
    torch.cuda.synchronize()

    # Reference
    expected = A.float() @ B.float()

    # Check
    c_host = C.float().cpu()
    e_host = expected.cpu()
    max_abs = torch.max(torch.abs(c_host - e_host)).item()
    ref_max = torch.max(torch.abs(e_host)).item()
    rel_error = max_abs / (ref_max + 1e-8)

    print(f"  Max abs error: {max_abs:.2e}")
    print(f"  Max rel error: {rel_error:.2e}")

    tol = 0.05 if in_dtype == "bf16" else 0.02
    assert rel_error < tol, f"Error too high: rel_error={rel_error:.2e} > {tol}"
    print("  PASS")


# =============================================================================
# Performance Benchmark
# =============================================================================

BENCHMARK_SIZES = [1024, 2048, 4096]


def _run_benchmark(
    M, N, K, in_dtype="bf16", tile_m=128, tile_n=128, tile_k=32, k_unroll=4
):
    """Run a single benchmark and return (tflops, pytorch_tflops)."""
    torch_dtype = torch.bfloat16 if in_dtype == "bf16" else torch.float16

    print(f"\n{'=' * 60}")
    print(f"WMMA Preshuffle GEMM Benchmark: {M}x{N}x{K}, dtype={in_dtype}")
    print(f"  tile=({tile_m},{tile_n},{tile_k}), k_unroll={k_unroll}")
    print(f"{'=' * 60}")

    exe = compile_wmma_preshuffle_gemm(
        M=M,
        N=N,
        K=K,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        in_dtype=in_dtype,
        out_dtype="bf16",
        k_unroll=k_unroll,
    )

    torch.manual_seed(42)
    A = torch.randn(M, K, device=device, dtype=torch_dtype) * 0.01
    B = torch.randn(K, N, device=device, dtype=torch_dtype) * 0.01
    A_shuf = preshuffle_a_wmma(A).flatten()
    B_shuf = preshuffle_b_wmma(B).flatten()
    C = torch.zeros(M, N, device=device, dtype=torch.bfloat16)
    scale_a = torch.empty(0, device=device, dtype=torch.float32)
    scale_b = torch.empty(0, device=device, dtype=torch.float32)
    stream_ptr = torch.cuda.current_stream().cuda_stream

    # Warmup
    for _ in range(5):
        exe(C.flatten(), A_shuf, B_shuf, scale_a, scale_b, M, N, K, stream_ptr)
    torch.cuda.synchronize()

    # Benchmark
    num_iters = 50
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iters):
        exe(C.flatten(), A_shuf, B_shuf, scale_a, scale_b, M, N, K, stream_ptr)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / num_iters) * 1000
    flops = 2 * M * N * K
    tflops = flops / (avg_ms / 1000) / 1e12

    # Verify correctness
    expected = A.float() @ B.float()
    rel_error = torch.max(torch.abs(C.float().cpu() - expected.cpu())).item() / (
        torch.max(torch.abs(expected.cpu())).item() + 1e-8
    )

    print(f"  Avg time: {avg_ms:.3f} ms")
    print(f"  Throughput: {tflops:.2f} TFLOPS")
    print(f"  Rel error: {rel_error:.2e}")
    assert rel_error < 0.1, f"Benchmark incorrect: rel_error={rel_error:.2e}"

    # PyTorch reference
    A_ref = A.clone()
    B_ref = B.clone()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iters):
        _ = A_ref @ B_ref
    torch.cuda.synchronize()
    pt_elapsed = time.perf_counter() - start
    pt_avg_ms = (pt_elapsed / num_iters) * 1000
    pt_tflops = flops / (pt_avg_ms / 1000) / 1e12

    print(f"  PyTorch: {pt_avg_ms:.3f} ms, {pt_tflops:.2f} TFLOPS")
    print(f"  Efficiency: {tflops / pt_tflops * 100:.1f}% of PyTorch")

    return tflops, pt_tflops


def test_wmma_preshuffle_gemm_benchmark():
    """Benchmark at standard sizes."""
    results = {}
    for sz in BENCHMARK_SIZES:
        tflops, pt_tflops = _run_benchmark(sz, sz, sz)
        results[sz] = (tflops, pt_tflops)

    print(f"\n{'=' * 60}")
    print("Summary:")
    print(
        f"{'Size':>6} | {'WMMA TFLOPS':>12} | {'PyTorch TFLOPS':>14} | {'Efficiency':>10}"
    )
    print("-" * 50)
    for sz, (tf, ptf) in results.items():
        print(f"{sz:>6} | {tf:>12.2f} | {ptf:>14.2f} | {tf / ptf * 100:>9.1f}%")
    print(f"{'=' * 60}")


# =============================================================================
# CLI entry point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="WMMA Preshuffle GEMM Test")
    parser.add_argument("-m", type=int, default=4096)
    parser.add_argument("-n", type=int, default=4096)
    parser.add_argument("-k", type=int, default=4096)
    parser.add_argument("--tile_m", type=int, default=128)
    parser.add_argument("--tile_n", type=int, default=128)
    parser.add_argument("--tile_k", type=int, default=32)
    parser.add_argument("--k_unroll", type=int, default=4)
    parser.add_argument(
        "--in_dtype", type=str, default="bf16", choices=["bf16", "fp16"]
    )
    parser.add_argument(
        "--correctness", action="store_true", help="Run correctness tests only"
    )
    args = parser.parse_args()

    if args.correctness:
        for shape in CORRECTNESS_SHAPES:
            M, N, K, tm, tn, tk = shape
            for dt in ["bf16", "fp16"]:
                try:
                    test_wmma_preshuffle_gemm_correctness(M, N, K, tm, tn, tk, dt)
                except Exception as e:
                    print(f"  SKIP/FAIL: {e}")
    else:
        _run_benchmark(
            args.m,
            args.n,
            args.k,
            in_dtype=args.in_dtype,
            tile_m=args.tile_m,
            tile_n=args.tile_n,
            tile_k=args.tile_k,
            k_unroll=args.k_unroll,
        )
