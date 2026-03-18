#!/usr/bin/env python3
"""Test and benchmark W4A16 GEMM kernel for RDNA4 (gfx12xx).

Target: C[16, 8192] = A[16, 6144] @ dequant(B_int4[6144, 8192])
Memory-bound decode inference workload.
"""

import sys
import os
import time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from flydsl.runtime.device import get_rocm_arch

device = "cuda"
gpu_arch = get_rocm_arch()


def dequantize_reference(B_packed_t, scales_t, K, N, group_size=128):
    """Dequantize int4 weights from transposed packed layout for reference.

    B_packed_t: [N, K//2] uint8 (transposed, K contiguous)
    scales_t: [N, K//group_size] f32 (transposed)
    Returns: B_deq[K, N] f32
    """
    num_groups = K // group_size
    # Untranspose
    B_packed_kn = B_packed_t.t().contiguous()  # [K//2, N]
    scales_kn = scales_t.t().contiguous()  # [num_groups, N]

    B_deq = torch.zeros(K, N, device=B_packed_t.device, dtype=torch.float32)
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
    return B_deq


def test_correctness(M, N, K, group_size=128):
    """Test W4A16 GEMM correctness with preshuffled A."""
    from kernels.wmma_w4a16_gemm import (
        compile_w4a16_gemm,
        quantize_int4_symmetric,
        preshuffle_a_bf16,
    )

    print(f"\nw4a16 GEMM: M={M}, N={N}, K={K}, gs={group_size}")

    torch.manual_seed(42)
    A_bf16 = (torch.randn(M, K, device=device) * 0.1).to(torch.bfloat16)
    B_f32 = torch.randn(K, N, device=device) * 0.5

    # Quantize B (returns transposed layout)
    B_packed_t, scales_t = quantize_int4_symmetric(B_f32.cpu(), group_size=group_size)
    B_packed_t = B_packed_t.to(device)
    scales_t = scales_t.to(device)

    # Reference: dequantize and matmul in f32
    B_deq = dequantize_reference(B_packed_t, scales_t, K, N, group_size)
    ref = A_bf16.float() @ B_deq

    # Preshuffle A for WMMA
    A_shuf = preshuffle_a_bf16(A_bf16)

    # Our kernel
    c_out = torch.zeros(M, N, device=device, dtype=torch.bfloat16)
    stream_ptr = torch.cuda.current_stream().cuda_stream

    exe = compile_w4a16_gemm(
        M=M,
        N=N,
        K=K,
        tile_n=128,
        group_size=group_size,
        num_waves=4,
    )

    exe(
        c_out.flatten(),
        A_shuf.flatten(),
        B_packed_t.flatten().view(torch.float32),
        scales_t.flatten(),
        M,
        N,
        K,
        stream_ptr,
    )
    torch.cuda.synchronize()

    # Metrics
    c_f32 = c_out.float()
    ref_f32 = ref.float()

    diff = (c_f32 - ref_f32).abs()

    # atol: max|kernel - ref| (absolute tolerance)
    atol = diff.max().item()

    # rtol: max(|kernel - ref| / |ref|) over ALL elements
    # Use a safe denominator to avoid div-by-zero
    rtol = (diff / ref_f32.abs().clamp(min=1e-12)).max().item()

    # Also compute torch.allclose-style check:
    # allclose passes if |a - b| <= atol + rtol * |b| for all elements
    # Find the tightest (atol, rtol) pair that passes:
    # With rtol=0: need atol >= max|diff| = atol above
    # With atol=0: need rtol >= max(|diff|/|ref|) = rtol above
    # Practical: what rtol makes allclose pass with atol=0.01?
    allclose_atol_001 = torch.allclose(c_f32, ref_f32, atol=0.01, rtol=0.05)
    allclose_atol_005 = torch.allclose(c_f32, ref_f32, atol=0.05, rtol=0.05)
    allclose_default = torch.allclose(c_f32, ref_f32, atol=1e-5, rtol=1e-3)

    cos_sim = torch.nn.functional.cosine_similarity(c_f32.flatten(), ref_f32.flatten(), dim=0).item()

    # Mean absolute and relative errors
    mean_abs = diff.mean().item()
    nonzero_mask = ref_f32.abs() > 1e-6
    if nonzero_mask.any():
        mean_rtol = (diff[nonzero_mask] / ref_f32[nonzero_mask].abs()).mean().item()
    else:
        mean_rtol = 0.0

    print(f"  atol (max abs error):  {atol:.6e}")
    print(f"  rtol (max rel error):  {rtol:.6e}")
    print(f"  mean abs error:        {mean_abs:.6e}")
    print(f"  mean rel error:        {mean_rtol:.6e}")
    print(f"  cosine sim:            {cos_sim:.8f}")
    print(f"  allclose(atol=1e-5, rtol=1e-3): {allclose_default}")
    print(f"  allclose(atol=0.01, rtol=0.05): {allclose_atol_001}")
    print(f"  allclose(atol=0.05, rtol=0.05): {allclose_atol_005}")

    ok = allclose_atol_005
    print(f"  {'PASS' if ok else 'FAIL'}")
    return {
        "atol": atol,
        "rtol": rtol,
        "mean_abs": mean_abs,
        "mean_rtol": mean_rtol,
        "cos_sim": cos_sim,
        "pass": ok,
    }


def benchmark(M, N, K, group_size=128, warmup=50, iters=500, tile_n=128, num_waves=4):
    """Benchmark W4A16 GEMM."""
    from kernels.wmma_w4a16_gemm import (
        compile_w4a16_gemm,
        quantize_int4_symmetric,
        preshuffle_a_bf16,
    )

    torch.manual_seed(42)
    A_bf16 = (torch.randn(M, K, device=device) * 0.1).to(torch.bfloat16)
    B_f32 = torch.randn(K, N, device=device) * 0.5
    B_packed_t, scales_t = quantize_int4_symmetric(B_f32.cpu(), group_size=group_size)
    B_packed_t = B_packed_t.to(device)
    scales_t = scales_t.to(device)

    A_shuf = preshuffle_a_bf16(A_bf16)
    c_out = torch.zeros(M, N, device=device, dtype=torch.bfloat16)
    stream_ptr = torch.cuda.current_stream().cuda_stream

    exe = compile_w4a16_gemm(
        M=M,
        N=N,
        K=K,
        tile_n=tile_n,
        group_size=group_size,
        num_waves=num_waves,
    )

    # Warmup
    for _ in range(warmup):
        exe(
            c_out.flatten(),
            A_shuf.flatten(),
            B_packed_t.flatten().view(torch.float32),
            scales_t.flatten(),
            M,
            N,
            K,
            stream_ptr,
        )
    torch.cuda.synchronize()

    # Timed
    start = time.perf_counter()
    for _ in range(iters):
        exe(
            c_out.flatten(),
            A_shuf.flatten(),
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

    # Effective bandwidth
    # B: N * K/2 bytes (int4), scales: N * K/gs * 4, A: M * K * 2, C: M * N * 2
    bytes_moved = N * K * 0.5 + N * (K // group_size) * 4 + M * K * 2 + M * N * 2
    bw_gbs = bytes_moved / avg_ms / 1e6

    # TFLOPS (for reference, not primary metric since memory-bound)
    tflops = 2 * M * N * K / (avg_ms / 1000) / 1e12

    return avg_ms, bw_gbs, tflops


def benchmark_pytorch_bf16(M, N, K, warmup=50, iters=500):
    """Benchmark PyTorch bf16 GEMM (rocBLAS) as baseline."""
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
    tflops = 2 * M * N * K / (avg_ms / 1000) / 1e12
    return avg_ms, bw_gbs, tflops


if __name__ == "__main__":
    print("=" * 70)
    print("W4A16 GEMM Kernel Test (RDNA4 gfx12xx)")
    print(f"GPU: {gpu_arch}")
    print("=" * 70)

    # ===== Correctness Tests =====
    print("\n=== Correctness Tests ===")
    all_ok = True
    results = {}
    for M, N, K in [
        (16, 128, 128),
        (16, 256, 256),
        (16, 8192, 6144),
    ]:
        try:
            r = test_correctness(M, N, K)
            results[(M, N, K)] = r
            all_ok = all_ok and r["pass"]
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback

            traceback.print_exc()
            all_ok = False
            results[(M, N, K)] = {"pass": False, "error": str(e)}

    # ===== Correctness Report =====
    print("\n")
    print("=" * 60)
    print("CORRECTNESS REPORT: w4a16_gemm_16x8192x6144")
    print("=" * 60)
    print(f"Kernel:      wmma_w4a16_gemm")
    print(f"File:        kernels/wmma_w4a16_gemm.py")
    print(f"Test:        tests/kernels/test_w4a16_gemm_16x8192x6144.py")
    print(f"GPU:         {gpu_arch} (RDNA4)")
    print(f"Precision:   bf16 activations + int4 weights (W4A16)")
    print(f"Reference:   float32 PyTorch dequant matmul")
    print()
    print(
        f"{'Shape':>16s}  {'atol':>12s}  {'rtol':>12s}  {'mean_abs':>12s}  {'mean_rtol':>12s}  {'cos_sim':>12s}  {'Status':>6s}"
    )
    print("-" * 100)
    for (M, N, K), r in results.items():
        if r.get("pass") is not None and "atol" in r:
            status = "PASS" if r["pass"] else "FAIL"
            print(
                f"{M:>4}x{N:>5}x{K:>5}  "
                f"{r['atol']:>12.2e}  "
                f"{r['rtol']:>12.2e}  "
                f"{r['mean_abs']:>12.2e}  "
                f"{r['mean_rtol']:>12.2e}  "
                f"{r['cos_sim']:>12.8f}  "
                f"{status:>6s}"
            )
        else:
            print(f"{M:>4}x{N:>5}x{K:>5}  {'ERROR':>12s}")
    print()
    verdict = "PASS" if all_ok else "FAIL"
    print(f"VERDICT: {verdict}")

    if not all_ok:
        print("\nSome tests FAILED, skipping benchmarks.")
        sys.exit(1)

    # ===== Benchmarks =====
    print("\n\n=== Benchmarks ===")
    print(
        f"{'Shape':>20s} | {'INT4 ms':>8s} | {'INT4 GB/s':>10s} | "
        f"{'INT4 TFLOPS':>11s} | {'PT bf16 ms':>10s} | {'PT GB/s':>10s} | "
        f"{'speedup':>7s}"
    )
    print("-" * 100)

    shapes = [
        (16, 8192, 6144),
    ]

    # Sweep tile_n and num_waves configurations
    configs = [
        (128, 4),  # baseline (best: 64 WGs, 4 waves each)
        (256, 8),  # larger tile (32 WGs, 8 waves each)
    ]

    bench_results = {}
    pt_ms, pt_bw, pt_tflops = benchmark_pytorch_bf16(16, 8192, 6144)
    print(f"rocBLAS bf16 baseline: {pt_ms:.3f} ms, {pt_bw:.1f} GB/s\n")

    for M, N, K in shapes:
        for tile_n, num_waves in configs:
            tag = f"tn={tile_n},w={num_waves}"
            try:
                our_ms, our_bw, our_tflops = benchmark(M, N, K, tile_n=tile_n, num_waves=num_waves)
                speedup = pt_ms / our_ms
                print(
                    f"{M:>4}x{N:>5}x{K:>5} [{tag:>12s}] | {our_ms:>8.3f} | {our_bw:>8.1f}   | "
                    f"{our_tflops:>9.3f}   | {speedup:>6.2f}x"
                )
                bench_results[(M, N, K, tag)] = {
                    "our_ms": our_ms,
                    "our_bw": our_bw,
                    "our_tflops": our_tflops,
                    "pt_ms": pt_ms,
                    "pt_bw": pt_bw,
                    "pt_tflops": pt_tflops,
                    "speedup": speedup,
                }
            except Exception as e:
                print(f"{M:>4}x{N:>5}x{K:>5} [{tag:>12s}] | FAILED: {e}")
                import traceback

                traceback.print_exc()

    # ===== Optimization Report =====
    print("\n")
    print("=" * 60)
    print("OPTIMIZATION REPORT: w4a16_gemm_16x8192x6144")
    print("=" * 60)
    print(f"Kernel:      wmma_w4a16_gemm")
    print(f"GPU:         {gpu_arch} (RDNA4), 64 CUs, wave32")
    print(f"Precision:   bf16 activations + int4 weights")
    print(f"Peak:        122 TFLOPS (bf16)")
    print(f"Peak BW:     492 GB/s")
    print()
    print("--- ROOFLINE ANALYSIS ---")
    M_t, N_t, K_t = 16, 8192, 6144
    flops = 2 * M_t * N_t * K_t
    # B dominates: N * K / 2 bytes + scales
    bytes_total = N_t * K_t // 2 + N_t * (K_t // 128) * 4 + M_t * K_t * 2 + M_t * N_t * 2
    ai = flops / bytes_total
    print(f"FLOPs:                {flops / 1e9:.3f} GFLOP")
    print(f"Bytes accessed:       {bytes_total / 1e6:.2f} MB")
    print(f"Arithmetic Intensity: {ai:.1f} FLOPs/byte")
    print(f"Ridge Point:          248 FLOPs/byte")
    print(f"Classification:       MEMORY-BOUND (AI={ai:.1f} < 248)")
    achievable_tflops = ai * 492 / 1000
    print(f"Achievable Ceiling:   {achievable_tflops:.1f} TFLOPS (limited by BW)")
    print()
    print("--- DESIGN CHOICES ---")
    print("1. A preshuffled in WMMA layout (no LDS needed)")
    print("2. B transposed [N, K//2] for K-contiguous coalesced loads")
    print("3. INT4 dequant in registers via FMA: val*scale + bias")
    print("4. Software-pipelined K-loop with unrolling")
    print("5. 4 waves along N (128 threads per workgroup)")
    print("6. Grid parallelism along N (64 workgroups for tile_n=128)")
    print()

    if bench_results:
        print("--- RESULTS ---")
        for key, r in bench_results.items():
            label = f"{key[0]}x{key[1]}x{key[2]}" if len(key) == 4 else f"{key[0]}x{key[1]}x{key[2]}"
            tag = key[3] if len(key) == 4 else ""
            print(f"Config [{tag}]:")
            print(f"  Kernel:     {r['our_ms']:.3f} ms, {r['our_bw']:.1f} GB/s, {r['our_tflops']:.3f} TFLOPS")
            print(f"  rocBLAS:    {r['pt_ms']:.3f} ms, {r['pt_bw']:.1f} GB/s")
            bw_efficiency = r["our_bw"] / 492 * 100
            print(f"  BW eff:     {bw_efficiency:.1f}% of peak (492 GB/s)")
            print(f"  vs rocBLAS: {r['speedup']:.2f}x (int4 = 4x less weight data)")
        # Best config
        best_key = min(bench_results, key=lambda k: bench_results[k]["our_ms"])
        best = bench_results[best_key]
        print(f"\n--- BEST CONFIG: {best_key[3] if len(best_key) == 4 else 'default'} ---")
        print(f"  {best['our_ms']:.3f} ms, {best['our_bw']:.1f} GB/s ({best['our_bw'] / 492 * 100:.1f}% peak BW)")
        print(f"  {best['speedup']:.2f}x vs rocBLAS bf16")
