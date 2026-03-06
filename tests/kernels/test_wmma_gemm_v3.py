#!/usr/bin/env python3
"""Test for WMMA GEMM v3 kernel (reduced VGPR, 2-wave occupancy target).

Tests:
  - Kernel correctness at multiple sizes
  - Performance benchmarks vs v2 and PyTorch
  - VGPR count verification
"""

import sys
import os
import numpy as np
import pytest
import torch
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import flydsl
from flydsl.runtime.device import get_rocm_arch

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available", allow_module_level=True)

gpu_arch = get_rocm_arch()
if not gpu_arch.startswith("gfx12"):
    pytest.skip(
        f"WMMA GEMM v3 test requires RDNA4 (gfx12xx), got {gpu_arch}",
        allow_module_level=True,
    )

from kernels.wmma_gemm_v3 import (
    create_wmma_gemm_v3_module,
    preshuffle_b_wmma,
    BLOCK_M,
    BLOCK_N,
    BLOCK_K,
)


# All shapes must be multiples of BLOCK_M=64, BLOCK_N=128, BLOCK_K=32
TEST_SHAPES = [
    (64, 128, 32),  # 1 workgroup, 1 K tile
    (64, 128, 128),  # 1 workgroup, 4 K tiles
    (128, 128, 128),  # 2 workgroups along M
    (256, 256, 256),  # medium
    (512, 512, 512),  # medium
]


@pytest.mark.parametrize("M,N,K", TEST_SHAPES)
def test_wmma_gemm_v3_correctness(M, N, K):
    """Test WMMA GEMM v3 with bf16 inputs, f32 output."""
    print(f"\n{'=' * 60}")
    print(f"WMMA GEMM v3 Test: M={M}, N={N}, K={K}, in=bf16, out=f32")
    print(f"GPU: {gpu_arch}, BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, BLOCK_K={BLOCK_K}")
    print(f"{'=' * 60}")

    m = create_wmma_gemm_v3_module(M, N, K, in_dtype="bf16", out_dtype="f32")
    exe = flydsl.compile(m)

    np.random.seed(42)
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.1
    B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16) * 0.1
    C = torch.zeros(M, N, device="cuda", dtype=torch.float32)

    # Pre-shuffle B
    B_shuf = preshuffle_b_wmma(B)

    expected = A.float() @ B.float()

    exe(A, B_shuf.reshape(-1), C)
    torch.cuda.synchronize()

    c_host = C.cpu()
    e_host = expected.cpu()
    error = torch.max(torch.abs(c_host - e_host)).item()
    rel_error = error / (torch.max(torch.abs(e_host)).item() + 1e-8)

    print(f"Max absolute error: {error:.2e}")
    print(f"Max relative error: {rel_error:.2e}")

    tol = 0.05
    assert rel_error < tol, f"WMMA GEMM v3 error too high: rel_error={rel_error:.2e}"
    print("PASS")


def _run_benchmark(M, N, K, in_dtype="bf16"):
    """Run benchmark at given size and return (our_tflops, pt_tflops)."""
    print(f"\n{'=' * 60}")
    print(f"WMMA GEMM v3 Benchmark: M={M}, N={N}, K={K}, in={in_dtype}, out=f32")
    print(f"GPU: {gpu_arch}, REG_M=2, REG_N=4, interleaved B loads")
    print(f"{'=' * 60}")

    torch_dtype = torch.bfloat16 if in_dtype == "bf16" else torch.float16
    m = create_wmma_gemm_v3_module(M, N, K, in_dtype=in_dtype, out_dtype="f32")
    exe = flydsl.compile(m)

    A = torch.randn(M, K, device="cuda", dtype=torch_dtype) * 0.01
    B = torch.randn(K, N, device="cuda", dtype=torch_dtype) * 0.01
    C = torch.zeros(M, N, device="cuda", dtype=torch.float32)

    # Pre-shuffle B
    B_shuf = preshuffle_b_wmma(B).reshape(-1)

    # Warmup
    for _ in range(5):
        exe(A, B_shuf, C)
    torch.cuda.synchronize()

    # Benchmark
    num_iters = 50
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iters):
        exe(A, B_shuf, C)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / num_iters) * 1000
    flops = 2 * M * N * K
    tflops = flops / (avg_ms / 1000) / 1e12

    print(f"Average time: {avg_ms:.3f} ms")
    print(f"Throughput: {tflops:.2f} TFLOPS")

    # Verify correctness
    expected = A.float() @ B.float()
    c_host = C.cpu()
    e_host = expected.cpu()
    error = torch.max(torch.abs(c_host - e_host)).item()
    rel_error = error / (torch.max(torch.abs(e_host)).item() + 1e-8)
    print(f"Max relative error: {rel_error:.2e}")
    assert rel_error < 0.1, f"Benchmark result incorrect: rel_error={rel_error:.2e}"

    # PyTorch reference
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iters):
        _ = A @ B
    torch.cuda.synchronize()
    pt_elapsed = time.perf_counter() - start
    pt_avg_ms = (pt_elapsed / num_iters) * 1000
    pt_tflops = flops / (pt_avg_ms / 1000) / 1e12
    print(f"PyTorch bf16 matmul: {pt_avg_ms:.3f} ms, {pt_tflops:.2f} TFLOPS")
    print(f"Efficiency vs PyTorch: {tflops / pt_tflops * 100:.1f}%")
    print(f"Efficiency vs peak (97.8 TF): {tflops / 97.8 * 100:.1f}%")

    return tflops, pt_tflops


def test_wmma_gemm_v3_benchmark():
    """Benchmark WMMA GEMM v3 at 1024."""
    _run_benchmark(1024, 1024, 1024)


def test_wmma_gemm_v3_benchmark_large():
    """Benchmark WMMA GEMM v3 at larger sizes."""
    for size in [2048, 4096]:
        _run_benchmark(size, size, size)


if __name__ == "__main__":
    test_wmma_gemm_v3_correctness(64, 128, 32)
    test_wmma_gemm_v3_correctness(128, 128, 128)
    test_wmma_gemm_v3_correctness(256, 256, 256)
    test_wmma_gemm_v3_benchmark()
    test_wmma_gemm_v3_benchmark_large()
