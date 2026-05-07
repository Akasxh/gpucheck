"""Isolate: is 1024^3 fp32 first-call slow because of cache miss, or something else?"""
from __future__ import annotations
import time
import statistics
import torch

WARMUP = 3
N = 10


def event_timer_block(fn):
    torch.mps.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def bench_one_dtype(sz, dtype, repeat=3):
    """Bench one shape/dtype with WARMUP+N. Repeat the whole thing."""
    out = []
    for trial in range(repeat):
        a = torch.randn(sz, sz, device="mps", dtype=dtype)
        b = torch.randn(sz, sz, device="mps", dtype=dtype)
        for _ in range(WARMUP):
            event_timer_block(lambda: a @ b)
        samples = [event_timer_block(lambda: a @ b) for _ in range(N)]
        med = statistics.median(samples)
        flops = 2 * sz**3
        gflops = flops / (med / 1000.0) / 1e9
        out.append((med, gflops, samples))
    return out


def main():
    print(f"torch={torch.__version__}")

    print("\n=== Test 1: Cold-start 1024^3 fp32 (only this shape, only this dtype, fresh process equiv) ===")
    sz = 1024
    a = torch.randn(sz, sz, device="mps", dtype=torch.float32)
    b = torch.randn(sz, sz, device="mps", dtype=torch.float32)
    # Cold first call
    for i in range(15):
        ms = event_timer_block(lambda: a @ b)
        print(f"  call {i:2d}: {ms:.4f} ms")

    print("\n=== Test 2: After fp32 'sticks' at 3ms, run fp16, then back to fp32 ===")
    a16 = a.to(torch.float16)
    b16 = b.to(torch.float16)
    print("  >>> running 5 fp16 1024^3:")
    for i in range(5):
        ms = event_timer_block(lambda: a16 @ b16)
        print(f"    fp16 call {i}: {ms:.4f} ms")
    print("  >>> back to fp32 1024^3:")
    for i in range(5):
        ms = event_timer_block(lambda: a @ b)
        print(f"    fp32 call {i}: {ms:.4f} ms")

    print("\n=== Test 3: Run 2048^3 fp32 'crosstalk' check ===")
    print("  Fresh tensors, fp32 1024^3 cold, then fp32 2048^3, then fp32 1024^3 again:")
    a = torch.randn(1024, 1024, device="mps", dtype=torch.float32)
    b = torch.randn(1024, 1024, device="mps", dtype=torch.float32)
    print(f"  cold 1024 fp32: {event_timer_block(lambda: a @ b):.4f} ms")
    print(f"  cold 1024 fp32: {event_timer_block(lambda: a @ b):.4f} ms")
    print(f"  cold 1024 fp32: {event_timer_block(lambda: a @ b):.4f} ms")
    a2 = torch.randn(2048, 2048, device="mps", dtype=torch.float32)
    b2 = torch.randn(2048, 2048, device="mps", dtype=torch.float32)
    print(f"  2048 fp32: {event_timer_block(lambda: a2 @ b2):.4f} ms")
    print(f"  back 1024 fp32: {event_timer_block(lambda: a @ b):.4f} ms")
    print(f"  back 1024 fp32: {event_timer_block(lambda: a @ b):.4f} ms")
    print(f"  back 1024 fp32: {event_timer_block(lambda: a @ b):.4f} ms")
    print(f"  back 1024 fp32: {event_timer_block(lambda: a @ b):.4f} ms")
    print(f"  back 1024 fp32: {event_timer_block(lambda: a @ b):.4f} ms")

    print("\n=== Test 4: Cold-start 2048^3 fp32 (only this shape) ===")
    a = torch.randn(2048, 2048, device="mps", dtype=torch.float32)
    b = torch.randn(2048, 2048, device="mps", dtype=torch.float32)
    for i in range(10):
        ms = event_timer_block(lambda: a @ b)
        print(f"  call {i:2d}: {ms:.4f} ms")

    print("\n=== Test 5: Cold-start 1024^3 fp16 (only this shape) ===")
    a = torch.randn(1024, 1024, device="mps", dtype=torch.float16)
    b = torch.randn(1024, 1024, device="mps", dtype=torch.float16)
    for i in range(15):
        ms = event_timer_block(lambda: a @ b)
        print(f"  call {i:2d}: {ms:.4f} ms")

    return 0


if __name__ == "__main__":
    main()
