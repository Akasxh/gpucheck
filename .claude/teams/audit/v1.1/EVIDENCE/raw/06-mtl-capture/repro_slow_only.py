"""Cold-start 1024^3 fp32 — slow path only, no validation, with timing markers."""
import os
# NO MTL_DEBUG_LAYER (otherwise validation slows things to mask the bug)
# Keep MTL_CAPTURE_ENABLED for posterity in case we add Swift wrapper
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
a = torch.randn(1024, 1024, device="mps", dtype=torch.float32)
b = torch.randn(1024, 1024, device="mps", dtype=torch.float32)
torch.mps.synchronize()

print("=== TRACER_MARK_BEGIN_SLOW ===", flush=True)
for i in range(8):
    t = event(lambda: a @ b)
    print(f"  slow-{i}: {t:.3f} ms", flush=True)
print("=== TRACER_MARK_END_SLOW ===", flush=True)
