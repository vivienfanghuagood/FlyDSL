#!/usr/bin/env python3
"""Measure kernel launch overhead on ROCm."""

import torch
import time
import triton
import triton.language as tl


@triton.jit
def _noop_kernel(x_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    tl.store(x_ptr + offs, x)


def main():
    x = torch.zeros(32, device="cuda", dtype=torch.float32)
    N = 2000

    # Warmup
    for _ in range(200):
        _noop_kernel[(1,)](x, BLOCK=32)
    torch.cuda.synchronize()

    # 1 launch
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(N):
        _noop_kernel[(1,)](x, BLOCK=32)
    end.record()
    torch.cuda.synchronize()
    gpu_us_1 = start.elapsed_time(end) / N * 1000

    # 2 launches
    start.record()
    for _ in range(N):
        _noop_kernel[(1,)](x, BLOCK=32)
        _noop_kernel[(1,)](x, BLOCK=32)
    end.record()
    torch.cuda.synchronize()
    gpu_us_2 = start.elapsed_time(end) / N * 1000

    # Wall clock 1 launch
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        _noop_kernel[(1,)](x, BLOCK=32)
    torch.cuda.synchronize()
    wall_us_1 = (time.perf_counter() - t0) / N * 1e6

    # Wall clock 2 launches
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        _noop_kernel[(1,)](x, BLOCK=32)
        _noop_kernel[(1,)](x, BLOCK=32)
    torch.cuda.synchronize()
    wall_us_2 = (time.perf_counter() - t0) / N * 1e6

    print(f"1 noop kernel:  gpu={gpu_us_1:.1f}us  wall={wall_us_1:.1f}us")
    print(f"2 noop kernels: gpu={gpu_us_2:.1f}us  wall={wall_us_2:.1f}us")
    print(
        f"Delta per extra: gpu={gpu_us_2 - gpu_us_1:.1f}us  wall={wall_us_2 - wall_us_1:.1f}us"
    )


if __name__ == "__main__":
    main()
