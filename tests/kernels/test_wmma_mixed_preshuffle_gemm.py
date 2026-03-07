#!/usr/bin/env python3
"""Test WMMA mixed-precision preshuffle GEMM for RDNA4 (gfx12xx).

Tests three paths:
  1. fp8 + fp8: Both operands fp8_e4m3fn
  2. bf16 + fp8: A in bf16, B in fp8
  3. bf16 + int4: A in bf16, B in packed int4 with per-group scales
"""

import sys
import os
import time
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from flydsl.runtime.device import get_rocm_arch

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available", allow_module_level=True)

gpu_arch = get_rocm_arch()
if not gpu_arch.startswith("gfx12"):
    pytest.skip(
        f"WMMA mixed GEMM requires RDNA4 (gfx12xx), got {gpu_arch}",
        allow_module_level=True,
    )

from kernels.wmma_mixed_preshuffle_gemm import (
    compile_wmma_mixed_preshuffle_gemm,
    preshuffle_a_fp8,
    preshuffle_b_fp8,
    preshuffle_a_bf16,
    preshuffle_b_bf16,
    preshuffle_b_int4,
    quantize_int4_symmetric,
)

device = "cuda"
DTYPE_FP8 = torch.float8_e4m3fn


# =============================================================================
# Helpers
# =============================================================================


def fp8_quantize_per_tensor(x_f32):
    """Quantize f32 tensor to fp8_e4m3fn with per-tensor scale.

    Returns (x_fp8, scale) where x_f32 ≈ x_fp8.float() * scale.
    """
    amax = x_f32.abs().amax().clamp(min=1e-12)
    # fp8_e4m3fn max value is 448.0
    scale = amax / 448.0
    x_scaled = (x_f32 / scale).clamp(-448.0, 448.0)
    x_fp8 = x_scaled.to(DTYPE_FP8)
    return x_fp8, scale.item()


# =============================================================================
# Path 1: fp8 + fp8
# =============================================================================


@pytest.mark.parametrize(
    "M,N,K",
    [
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
    ],
    ids=["128", "256", "512"],
)
def test_fp8_fp8(M, N, K):
    """Test fp8+fp8 WMMA mixed preshuffle GEMM."""
    tile_m = min(128, M)
    tile_n = min(128, N)
    tile_k = 32
    k_unroll = min(4, (K // tile_k) - 1) if K // tile_k > 1 else 1

    # Ensure k_unroll divides (num_k_tiles - 1) or use simpler logic
    num_k_tiles = K // tile_k
    if num_k_tiles <= k_unroll:
        k_unroll = 1
    while k_unroll > 1 and (num_k_tiles - 1) % k_unroll != 0:
        k_unroll -= 1

    print(f"\nfp8+fp8: M={M}, N={N}, K={K}, tiles=({tile_m},{tile_n},{tile_k})")

    torch.manual_seed(42)
    A_f32 = torch.randn(M, K, device=device) * 0.1
    B_f32 = torch.randn(K, N, device=device) * 0.1

    # Quantize
    A_fp8, scale_a = fp8_quantize_per_tensor(A_f32)
    B_fp8, scale_b = fp8_quantize_per_tensor(B_f32)

    # Reference: use dequantized values
    ref = (A_fp8.float() * scale_a) @ (B_fp8.float() * scale_b)

    # Preshuffle
    a_shuf = preshuffle_a_fp8(A_fp8)
    b_shuf = preshuffle_b_fp8(B_fp8)

    # Output
    c_out = torch.zeros(M, N, device=device, dtype=torch.bfloat16)
    scale_a_t = torch.tensor([scale_a], device=device, dtype=torch.float32)
    scale_b_t = torch.tensor([scale_b], device=device, dtype=torch.float32)

    stream_ptr = torch.cuda.current_stream().cuda_stream

    exe = compile_wmma_mixed_preshuffle_gemm(
        M=M,
        N=N,
        K=K,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        a_dtype="fp8",
        b_dtype="fp8",
        out_dtype="bf16",
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

    # Check
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

    print(f"  Max abs error: {max_abs:.2e}")
    print(f"  Max rel error: {rel_error:.2e}")
    assert rel_error < 0.1, f"fp8+fp8 error too high: {rel_error:.2e}"
    print("  PASS")


# =============================================================================
# Path 2: bf16 + fp8
# =============================================================================


@pytest.mark.parametrize(
    "M,N,K",
    [
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
    ],
    ids=["128", "256", "512"],
)
def test_bf16_fp8(M, N, K):
    """Test bf16+fp8 WMMA mixed preshuffle GEMM."""
    tile_m = min(128, M)
    tile_n = min(128, N)
    tile_k = 32

    num_k_tiles = K // tile_k
    k_unroll = min(4, num_k_tiles - 1) if num_k_tiles > 1 else 1
    while k_unroll > 1 and (num_k_tiles - 1) % k_unroll != 0:
        k_unroll -= 1

    print(f"\nbf16+fp8: M={M}, N={N}, K={K}, tiles=({tile_m},{tile_n},{tile_k})")

    torch.manual_seed(42)
    A_bf16 = (torch.randn(M, K, device=device) * 0.1).to(torch.bfloat16)
    B_f32 = torch.randn(K, N, device=device) * 0.1

    # Quantize B to fp8
    B_fp8, scale_b = fp8_quantize_per_tensor(B_f32)

    # Reference: A_bf16.float() @ (B_fp8.float() * scale_b)
    ref = A_bf16.float() @ (B_fp8.float() * scale_b)

    # For bf16+fp8 path: A is preshuffled as fp8 (truncated from bf16)
    # The kernel truncates bf16->fp8 internally. But our current implementation
    # requires A to already be in fp8 format in the preshuffle buffer.
    # So we truncate A to fp8 on host for the preshuffle.
    A_as_fp8 = A_bf16.float().clamp(-448, 448).to(DTYPE_FP8)
    a_shuf = preshuffle_a_fp8(A_as_fp8)
    b_shuf = preshuffle_b_fp8(B_fp8)

    # Adjust reference to account for A truncation to fp8
    ref_adjusted = A_as_fp8.float() @ (B_fp8.float() * scale_b)

    c_out = torch.zeros(M, N, device=device, dtype=torch.bfloat16)
    # For bf16+fp8: scale_a = 1.0 (A is already in its native range, no scaling)
    scale_a_t = torch.tensor([1.0], device=device, dtype=torch.float32)
    scale_b_t = torch.tensor([scale_b], device=device, dtype=torch.float32)

    stream_ptr = torch.cuda.current_stream().cuda_stream

    exe = compile_wmma_mixed_preshuffle_gemm(
        M=M,
        N=N,
        K=K,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        a_dtype="bf16",
        b_dtype="fp8",
        out_dtype="bf16",
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

    c_f32 = c_out.float()
    ref_f32 = ref_adjusted.float()
    nonzero = ref_f32.abs() > 1e-6
    if nonzero.any():
        diff = (c_f32[nonzero] - ref_f32[nonzero]).abs()
        rel_error = (diff / ref_f32[nonzero].abs().clamp(min=1e-8)).max().item()
        max_abs = diff.max().item()
    else:
        rel_error = 0.0
        max_abs = 0.0

    print(f"  Max abs error: {max_abs:.2e}")
    print(f"  Max rel error: {rel_error:.2e}")
    assert rel_error < 0.1, f"bf16+fp8 error too high: {rel_error:.2e}"
    print("  PASS")


# =============================================================================
# Path 3: bf16 + int4
# =============================================================================


@pytest.mark.parametrize(
    "M,N,K,group_size",
    [
        (128, 128, 128, 128),
        (256, 256, 256, 128),
        (128, 256, 512, 128),
    ],
    ids=["128_g128", "256_g128", "128x256x512_g128"],
)
def test_bf16_int4(M, N, K, group_size):
    """Test bf16+int4 WMMA mixed preshuffle GEMM."""
    tile_m = min(128, M)
    tile_n = min(128, N)
    tile_k = 32

    num_k_tiles = K // tile_k
    k_unroll = min(4, num_k_tiles - 1) if num_k_tiles > 1 else 1
    while k_unroll > 1 and (num_k_tiles - 1) % k_unroll != 0:
        k_unroll -= 1

    print(f"\nbf16+int4: M={M}, N={N}, K={K}, gs={group_size}")

    torch.manual_seed(42)
    A_bf16 = (torch.randn(M, K, device=device) * 0.1).to(torch.bfloat16)
    B_f32 = torch.randn(K, N, device=device) * 0.5

    # Quantize B to int4 (on CPU, then move)
    B_packed_cpu, scales_cpu, zeros_cpu = quantize_int4_symmetric(
        B_f32.cpu(), group_size=group_size
    )
    B_packed = B_packed_cpu.to(device)
    scales = scales_cpu.to(device)
    zeros = zeros_cpu.to(device)

    # Dequantize B for reference (vectorized)
    num_groups = K // group_size
    B_q_full = torch.zeros(K, N, device=device, dtype=torch.float32)
    for g in range(num_groups):
        k_start = g * group_size
        k_end = k_start + group_size
        for k in range(k_start, k_end):
            k_packed = k // 2
            is_high = k % 2
            if is_high:
                nibble = (B_packed[k_packed] >> 4) & 0xF
            else:
                nibble = B_packed[k_packed] & 0xF
            B_q_full[k] = (nibble.float() - zeros[g]) * scales[g]

    ref = A_bf16.float() @ B_q_full

    # Preshuffle
    a_shuf = preshuffle_a_bf16(A_bf16)
    b_packed_shuf, scales_shuf = preshuffle_b_int4(
        B_packed, scales, K, N, group_size=group_size
    )

    c_out = torch.zeros(M, N, device=device, dtype=torch.bfloat16)
    dummy_scale = torch.empty(0, device=device, dtype=torch.float32)

    stream_ptr = torch.cuda.current_stream().cuda_stream

    exe = compile_wmma_mixed_preshuffle_gemm(
        M=M,
        N=N,
        K=K,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        a_dtype="bf16",
        b_dtype="int4",
        out_dtype="bf16",
        group_size=group_size,
        k_unroll=k_unroll,
    )

    exe(
        c_out.flatten(),
        a_shuf.flatten(),
        b_packed_shuf.flatten().view(torch.float32),
        scales_shuf.flatten(),
        dummy_scale,
        M,
        N,
        K,
        stream_ptr,
    )
    torch.cuda.synchronize()

    c_f32 = c_out.float()
    ref_f32 = ref.float()

    # For int4 quantized GEMM, use atol+rtol check since near-zero values
    # can have large relative error from quantization noise
    max_abs = (c_f32 - ref_f32).abs().max().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        c_f32.flatten(), ref_f32.flatten(), dim=0
    ).item()

    # Rel error on significant values only
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
    assert cos_sim > 0.999, f"bf16+int4 cosine sim too low: {cos_sim:.6f}"
    assert rel_error < 0.15, f"bf16+int4 rel error too high: {rel_error:.2e}"
    print("  PASS")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    print("=== Path 1: fp8 + fp8 ===")
    for M, N, K in [(128, 128, 128), (256, 256, 256)]:
        try:
            test_fp8_fp8(M, N, K)
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback

            traceback.print_exc()

    print("\n=== Path 2: bf16 + fp8 ===")
    for M, N, K in [(128, 128, 128), (256, 256, 256)]:
        try:
            test_bf16_fp8(M, N, K)
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback

            traceback.print_exc()

    print("\n=== Path 3: bf16 + int4 ===")
    for M, N, K, gs in [(128, 128, 128, 128), (256, 256, 256, 128)]:
        try:
            test_bf16_int4(M, N, K, gs)
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback

            traceback.print_exc()
