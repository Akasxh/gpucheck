"""Trigger fast-path: cold 1024 a few times, then off-shape 2048, then 1024 again."""
import os
os.environ["MTL_CAPTURE_ENABLED"] = "1"

import time
import torch

assert torch.backends.mps.is_available()


def event(fn):
    torch.mps.synchronize()
    t = time.perf_counter()
    fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t) * 1000


print(f"PID={os.getpid()}", flush=True)
a1024 = torch.randn(1024, 1024, device="mps", dtype=torch.float32)
b1024 = torch.randn(1024, 1024, device="mps", dtype=torch.float32)
a2048 = torch.randn(2048, 2048, device="mps", dtype=torch.float32)
b2048 = torch.randn(2048, 2048, device="mps", dtype=torch.float32)
torch.mps.synchronize()

print("=== TRACER_MARK_BEGIN_SLOW_PHASE ===", flush=True)
for i in range(5):
    print(f"  cold-{i}: {event(lambda: a1024 @ b1024):.3f} ms", flush=True)
print("=== TRACER_MARK_END_SLOW_PHASE ===", flush=True)

print("=== TRACER_MARK_OFFSHAPE ===", flush=True)
print(f"  offshape: {event(lambda: a2048 @ b2048):.3f} ms", flush=True)

print("=== TRACER_MARK_BEGIN_FAST_PHASE ===", flush=True)
for i in range(5):
    print(f"  fast-{i}: {event(lambda: a1024 @ b1024):.3f} ms", flush=True)
print("=== TRACER_MARK_END_FAST_PHASE ===", flush=True)
