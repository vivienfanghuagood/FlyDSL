#!/usr/bin/env python3
"""Test for WMMA GEMM v2 kernel (B from GMEM, ping-pong LDS for A).

Tests:
  - Correctness of the pre-shuffle layout
  - Kernel correctness at multiple sizes
  - Performance benchmarks vs v1 and PyTorch
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
        f"WMMA GEMM v2 test requires RDNA4 (gfx12xx), got {gpu_arch}",
        allow_module_level=True,
    )

from kernels.wmma_gemm_v2 import (
    create_wmma_gemm_v2_module,
    preshuffle_b_wmma,
    BLOCK_M,
    BLOCK_N,
    BLOCK_K,
)


def test_preshuffle_layout():
    """Verify that preshuffle_b_wmma produces the correct layout."""
    K, N = 32, 32
    B = torch.arange(K * N, dtype=torch.bfloat16, device="cpu").reshape(K, N)
    B_s = preshuffle_b_wmma(B.cuda())

    # B_s shape: (N0, K0, KLane, NLane, KPack)
    N0 = N // 16
    K0 = K // 16
    assert B_s.shape == (N0, K0, 2, 16, 8), f"Bad shape: {B_s.shape}"

    # Verify: B_s[n0, k0, klane, nlane, kpack] == B[k0*16 + klane*8 + kpack, n0*16 + nlane]
    B_cpu = B.float()
    B_s_cpu = B_s.cpu().float()
    for n0 in range(N0):
        for k0 in range(K0):
            for klane in range(2):
                for nlane in range(16):
                    for kpack in range(8):
                        k_idx = k0 * 16 + klane * 8 + kpack
                        n_idx = n0 * 16 + nlane
                        expected = B_cpu[k_idx, n_idx].item()
                        actual = B_s_cpu[n0, k0, klane, nlane, kpack].item()
                        assert expected == actual, (
                            f"Mismatch at n0={n0},k0={k0},klane={klane},"
                            f"nlane={nlane},kpack={kpack}: "
                            f"expected B[{k_idx},{n_idx}]={expected}, got {actual}"
                        )
    print("PASS: preshuffle layout verified")


# All shapes must be multiples of BLOCK_M=128, BLOCK_N=128, BLOCK_K=32
TEST_SHAPES = [
    (128, 128, 32),  # 1 workgroup, 1 K tile
    (128, 128, 128),  # 1 workgroup, 4 K tiles
    (256, 256, 256),  # 4 workgroups
    (512, 512, 512),  # medium
]


@pytest.mark.parametrize("M,N,K", TEST_SHAPES)
def test_wmma_gemm_v2_correctness(M, N, K):
    """Test WMMA GEMM v2 with bf16 inputs, f32 output."""
    print(f"\n{'=' * 60}")
    print(f"WMMA GEMM v2 Test: M={M}, N={N}, K={K}, in=bf16, out=f32")
    print(f"GPU: {gpu_arch}, BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, BLOCK_K={BLOCK_K}")
    print(f"{'=' * 60}")

    m = create_wmma_gemm_v2_module(M, N, K, in_dtype="bf16", out_dtype="f32")
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
    assert rel_error < tol, f"WMMA GEMM v2 error too high: rel_error={rel_error:.2e}"
    print("PASS")


def _run_benchmark(M, N, K, in_dtype="bf16"):
    """Run benchmark at given size and return (our_tflops, pt_tflops)."""
    print(f"\n{'=' * 60}")
    print(f"WMMA GEMM v2 Benchmark: M={M}, N={N}, K={K}, in={in_dtype}, out=f32")
    print(f"GPU: {gpu_arch}, B from GMEM (preshuffle), ping-pong LDS for A")
    print(f"{'=' * 60}")

    torch_dtype = torch.bfloat16 if in_dtype == "bf16" else torch.float16
    m = create_wmma_gemm_v2_module(M, N, K, in_dtype=in_dtype, out_dtype="f32")
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

    return tflops, pt_tflops


def test_wmma_gemm_v2_benchmark():
    """Benchmark WMMA GEMM v2 at 1024."""
    _run_benchmark(1024, 1024, 1024)


def test_wmma_gemm_v2_benchmark_large():
    """Benchmark WMMA GEMM v2 at larger sizes."""
    for size in [2048, 4096]:
        _run_benchmark(size, size, size)


if __name__ == "__main__":
    test_preshuffle_layout()
    test_wmma_gemm_v2_correctness(128, 128, 32)
    test_wmma_gemm_v2_correctness(128, 128, 128)
    test_wmma_gemm_v2_correctness(256, 256, 256)
    test_wmma_gemm_v2_benchmark()
    test_wmma_gemm_v2_benchmark_large()
