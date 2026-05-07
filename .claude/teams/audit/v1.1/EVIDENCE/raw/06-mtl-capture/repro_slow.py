"""Cold-start 1024^3 fp32 matmul — should hit slow MPSGraph kernel-pick.

Run as a fresh process. Saves nothing — we capture via `log stream`.
"""
import os
# Set Metal verbose env-vars BEFORE torch imports MPS
os.environ["MTL_CAPTURE_ENABLED"] = "1"
os.environ["MTL_DEBUG_LAYER"] = "1"
os.environ["MTL_SHADER_VALIDATION"] = "1"
os.environ["MPSGRAPH_LOG_VERBOSITY"] = "verbose"
os.environ["MPS_LOG_VERBOSITY"] = "verbose"
os.environ["METAL_DEBUG_ERROR_MODE"] = "1"
os.environ["MPSGRAPH_OPTIMIZATION_LEVEL"] = "0"

import time
import torch

assert torch.backends.mps.is_available()


def event(fn):
    torch.mps.synchronize()
    t = time.perf_counter()
    fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t) * 1000


# Cold-start: 1024^3 fp32, never touch any other shape first
print(f"PID={os.getpid()}", flush=True)
a = torch.randn(1024, 1024, device="mps", dtype=torch.float32)
b = torch.randn(1024, 1024, device="mps", dtype=torch.float32)

print("=== SLOW PATH START ===", flush=True)
for i in range(10):
    t = event(lambda: a @ b)
    print(f"  call {i:2d}: {t:.3f} ms", flush=True)
print("=== SLOW PATH END ===", flush=True)
