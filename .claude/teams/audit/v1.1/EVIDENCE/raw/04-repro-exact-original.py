"""Reproduce the EXACT methodology of mac_bench-mps_kernels.py
to see if 1024^3 fp32 matmul really is 3.29 ms."""
from __future__ import annotations

import json
import statistics
import sys
import time

import torch
import torch.nn.functional as F  # noqa: F401

# Adapted from /tmp/mac_bench-mps_kernels.py:
WARMUP = 3
N = 10


def event_timer_block(fn):
    """Mimics gpucheck's event_timer."""
    torch.mps.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def factory_matmul(M, N_, K, dtype, device):
    def build():
        a = torch.randn(M, K, device=device, dtype=dtype)
        b = torch.randn(K, N_, device=device, dtype=dtype)
        def run():
            return a @ b
        return run
    return build


def bench(M, N_, K, dtype, device):
    build = factory_matmul(M, N_, K, dtype, device)
    fn = build()
    samples = []
    for _ in range(WARMUP):
        event_timer_block(fn)
    for _ in range(N):
        samples.append(event_timer_block(fn))
    return samples


def main():
    print(f"torch={torch.__version__}, mps_avail={torch.backends.mps.is_available()}")
    print(f"WARMUP={WARMUP}, N={N}")

    # The exact ordering in the original sweep iterates:
    # for spec in [256,1024,2048,4096]:
    #   for dtype in [fp32, fp16, bf16]:
    #     for device in [mps, cpu]:
    # With ALL the prior kernels run before the next.
    # Let's just run pristine matmul to isolate.

    print("\n=== Direct (pristine, single dtype, fresh memory each call) ===")
    for sz in (256, 512, 1024, 2048):
        for dtype_name, dtype in [("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16)]:
            samples = bench(sz, sz, sz, dtype, "mps")
            med = statistics.median(samples)
            flops = 2 * sz**3
            gflops = flops / (med / 1000.0) / 1e9
            print(f"  {sz:4d}^3 {dtype_name}: med={med:.3f} ms  gflops={gflops:.0f}  raw={[round(s,3) for s in samples]}")

    print("\n=== With dtype-iteration order (mimics original sweep) ===")
    for sz in (256, 1024, 2048, 4096):
        for dtype_name, dtype in [("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16)]:
            samples = bench(sz, sz, sz, dtype, "mps")
            med = statistics.median(samples)
            flops = 2 * sz**3
            gflops = flops / (med / 1000.0) / 1e9
            print(f"  {sz:4d}^3 {dtype_name}: med={med:.3f} ms  gflops={gflops:.0f}")

    print("\n=== Fresh tensor each iter (allocator effect) ===")
    for sz in (1024,):
        for dtype_name, dtype in [("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16)]:
            samples = []
            for _ in range(WARMUP):
                a = torch.randn(sz, sz, device="mps", dtype=dtype)
                b = torch.randn(sz, sz, device="mps", dtype=dtype)
                event_timer_block(lambda: a @ b)
            for _ in range(N):
                a = torch.randn(sz, sz, device="mps", dtype=dtype)
                b = torch.randn(sz, sz, device="mps", dtype=dtype)
                samples.append(event_timer_block(lambda: a @ b))
            med = statistics.median(samples)
            flops = 2 * sz**3
            gflops = flops / (med / 1000.0) / 1e9
            print(f"  {sz:4d}^3 {dtype_name}: med={med:.3f} ms  gflops={gflops:.0f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
