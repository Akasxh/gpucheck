"""Repro both slow and fast path in one process.

Phase 1: cold 1024^3 fp32 (slow).
Phase 2: 2048^3 fp32 (off-shape unblock).
Phase 3: 1024^3 fp32 again (fast).
"""
import os
os.environ["MTL_CAPTURE_ENABLED"] = "1"
os.environ["MTL_DEBUG_LAYER"] = "1"
os.environ["MTL_SHADER_VALIDATION"] = "1"
os.environ["MPSGRAPH_LOG_VERBOSITY"] = "verbose"
os.environ["MPS_LOG_VERBOSITY"] = "verbose"
os.environ["METAL_DEBUG_ERROR_MODE"] = "1"

import time
import torch

assert torch.backends.mps.is_available()


def event(fn, label):
    torch.mps.synchronize()
    t = time.perf_counter()
    fn()
    torch.mps.synchronize()
    ms = (time.perf_counter() - t) * 1000
    print(f"[{label}] {ms:.3f} ms", flush=True)
    return ms


print(f"PID={os.getpid()}", flush=True)

a1024 = torch.randn(1024, 1024, device="mps", dtype=torch.float32)
b1024 = torch.randn(1024, 1024, device="mps", dtype=torch.float32)

print("==== PHASE 1 SLOW 1024 fp32 ====", flush=True)
for i in range(5):
    event(lambda: a1024 @ b1024, f"slow-{i}")

a2048 = torch.randn(2048, 2048, device="mps", dtype=torch.float32)
b2048 = torch.randn(2048, 2048, device="mps", dtype=torch.float32)
print("==== PHASE 2 OFF-SHAPE 2048 fp32 ====", flush=True)
event(lambda: a2048 @ b2048, "offshape-0")

print("==== PHASE 3 FAST 1024 fp32 ====", flush=True)
for i in range(5):
    event(lambda: a1024 @ b1024, f"fast-{i}")
