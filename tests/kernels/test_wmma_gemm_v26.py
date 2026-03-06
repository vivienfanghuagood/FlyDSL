#!/usr/bin/env python3
"""Test and benchmark for WMMA GEMM v26 kernel (4-warp LDS, XOR-swizzle)."""

import sys
import os
import numpy as np
import torch
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import flydsl
from flydsl.runtime.device import get_rocm_arch

gpu_arch = get_rocm_arch()
assert gpu_arch.startswith("gfx12"), f"Need RDNA4, got {gpu_arch}"

from kernels.wmma_gemm_v26 import create_wmma_gemm_v26_module


def test_correctness(M=128, N=128, K=128):
    """Test v26 kernel correctness."""
    print(f"\n{'=' * 60}")
    print(f"v26 correctness test: M={M}, N={N}, K={K}")
    print(f"{'=' * 60}")

    mod, BLOCK_M, BLOCK_N, BLOCK_K = create_wmma_gemm_v26_module(
        M, N, K, in_dtype="bf16", out_dtype="f32"
    )
    exe = flydsl.compile(mod)

    torch.manual_seed(42)
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.1
    B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16) * 0.1
    B_T = B.t().contiguous()
    C = torch.zeros(M, N, device="cuda", dtype=torch.float32)

    expected = A.float() @ B.float()

    exe(A, B_T, C)
    torch.cuda.synchronize()

    error = torch.max(torch.abs(C - expected)).item()
    rel_error = error / (torch.max(torch.abs(expected)).item() + 1e-8)
    print(f"Max abs error: {error:.4e}, rel error: {rel_error:.4e}")
    assert rel_error < 0.01, f"Correctness failed: rel_error={rel_error}"
    print("PASSED")
    return True


def benchmark(M=4096, N=4096, K=4096):
    """Benchmark v26 kernel."""
    print(f"\n{'=' * 60}")
    print(f"v26 benchmark: M={M}, N={N}, K={K}")
    print(f"{'=' * 60}")

    mod, BLOCK_M, BLOCK_N, BLOCK_K = create_wmma_gemm_v26_module(
        M, N, K, in_dtype="bf16", out_dtype="f32"
    )
    exe = flydsl.compile(mod)

    torch.manual_seed(42)
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.1
    B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16) * 0.1
    B_T = B.t().contiguous()
    C = torch.zeros(M, N, device="cuda", dtype=torch.float32)

    # Correctness check
    expected = A.float() @ B.float()
    exe(A, B_T, C)
    torch.cuda.synchronize()
    error = torch.max(torch.abs(C - expected)).item()
    rel_error = error / (torch.max(torch.abs(expected)).item() + 1e-8)
    print(f"Correctness: max_abs_err={error:.4e}, rel_err={rel_error:.4e}")
    if rel_error > 0.01:
        print("WARNING: Correctness check failed!")

    # Warmup
    for _ in range(5):
        exe(A, B_T, C)
    torch.cuda.synchronize()

    # Benchmark
    N_RUNS = 50
    start = time.perf_counter()
    for _ in range(N_RUNS):
        exe(A, B_T, C)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_ms = elapsed / N_RUNS * 1000
    flops = 2 * M * N * K
    tflops = flops / (avg_ms / 1000) / 1e12
    print(f"Avg time: {avg_ms:.3f} ms")
    print(f"Performance: {tflops:.1f} TFLOPS")

    # PyTorch reference
    torch.cuda.synchronize()
    for _ in range(5):
        _ = A @ B
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(N_RUNS):
        _ = A @ B
    torch.cuda.synchronize()
    pt_elapsed = time.perf_counter() - start
    pt_ms = pt_elapsed / N_RUNS * 1000
    pt_tflops = flops / (pt_ms / 1000) / 1e12
    print(f"\nPyTorch: {pt_ms:.3f} ms, {pt_tflops:.1f} TFLOPS")
    print(f"v26 / PyTorch: {tflops / pt_tflops * 100:.1f}%")

    return tflops


if __name__ == "__main__":
    # Small correctness tests
    test_correctness(128, 128, 64)
    test_correctness(128, 128, 128)
    test_correctness(256, 256, 256)

    # Main benchmark
    benchmark(4096, 4096, 4096)
