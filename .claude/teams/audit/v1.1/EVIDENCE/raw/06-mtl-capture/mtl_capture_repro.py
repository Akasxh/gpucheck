"""Drive MTLCaptureManager from Python via PyObjC to capture GPU trace of slow vs fast 1024^3 fp32 matmul.

Pre-req: Set `MTL_CAPTURE_ENABLED=1` and run from a bundle with `MetalCaptureEnabled=YES`
(or just attempt and report the supportsDestination diagnostic).
"""
import os
os.environ["MTL_CAPTURE_ENABLED"] = "1"
os.environ["METAL_CAPTURE_ENABLED"] = "1"
os.environ["MetalCaptureEnabled"] = "1"

import sys
import time
import objc
import Metal
import torch

CAPTURE_DIR = "/Users/cero/Code/gpucheck/.claude/teams/audit/v1.1/EVIDENCE/raw/06-mtl-capture"


def diag():
    mgr = Metal.MTLCaptureManager.sharedCaptureManager()
    print("--- Capture diagnostics ---")
    print(f"  python exe: {sys.executable}")
    print(f"  PID: {os.getpid()}")
    print(f"  shared manager: {mgr}")
    for dest, name in [
        (Metal.MTLCaptureDestinationGPUTraceDocument, "GPUTraceDocument"),
        (Metal.MTLCaptureDestinationDeveloperTools, "DeveloperTools"),
    ]:
        sup = mgr.supportsDestination_(dest)
        print(f"  supportsDestination({name}) = {sup}")
    print(f"  isCapturing: {mgr.isCapturing()}")
    print(f"  defaultCaptureScope: {mgr.defaultCaptureScope()}")
    return mgr


def get_mps_device():
    """Acquire the Metal device PyTorch is using.

    PyTorch MPS backend uses `MTLCreateSystemDefaultDevice()` internally;
    capturing on the system default device should match.
    """
    return Metal.MTLCreateSystemDefaultDevice()


def attempt_capture(mgr, dev, label, gputrace_path):
    desc = Metal.MTLCaptureDescriptor.alloc().init()
    desc.setCaptureObject_(dev)

    # Try GPU trace document first (writes a .gputrace file)
    desc.setDestination_(Metal.MTLCaptureDestinationGPUTraceDocument)
    if os.path.exists(gputrace_path):
        import shutil
        shutil.rmtree(gputrace_path)
    desc.setOutputURL_(objc.lookUpClass("NSURL").fileURLWithPath_(gputrace_path))

    err = objc.NULL
    ok, err = mgr.startCaptureWithDescriptor_error_(desc, None)
    if not ok:
        print(f"  [{label}] startCapture(GPUTraceDocument) FAILED: {err}")
        # Fallback to DeveloperTools (pipes to Xcode if running)
        desc2 = Metal.MTLCaptureDescriptor.alloc().init()
        desc2.setCaptureObject_(dev)
        desc2.setDestination_(Metal.MTLCaptureDestinationDeveloperTools)
        ok2, err2 = mgr.startCaptureWithDescriptor_error_(desc2, None)
        if not ok2:
            print(f"  [{label}] startCapture(DeveloperTools) ALSO FAILED: {err2}")
            return False
        print(f"  [{label}] capturing to DeveloperTools (live Xcode required for kernel names)")
        return True
    print(f"  [{label}] capturing to {gputrace_path}")
    return True


def event(fn):
    torch.mps.synchronize()
    t = time.perf_counter()
    fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t) * 1000


def main():
    mgr = diag()
    dev = get_mps_device()
    print(f"  metal device: {dev.name()}")

    # Tensors
    a1024 = torch.randn(1024, 1024, device="mps", dtype=torch.float32)
    b1024 = torch.randn(1024, 1024, device="mps", dtype=torch.float32)
    a2048 = torch.randn(2048, 2048, device="mps", dtype=torch.float32)
    b2048 = torch.randn(2048, 2048, device="mps", dtype=torch.float32)
    torch.mps.synchronize()

    # Phase A: SLOW path — single 1024 fp32 matmul, captured.
    slow_trace = os.path.join(CAPTURE_DIR, "slow-1024.gputrace")
    print("\n=== PHASE A: SLOW 1024^3 fp32 ===")
    if attempt_capture(mgr, dev, "slow", slow_trace):
        ms = event(lambda: a1024 @ b1024)
        print(f"  matmul ms: {ms:.3f}")
        mgr.stopCapture()
        print("  capture stopped")
    else:
        # No capture available; just measure
        ms = event(lambda: a1024 @ b1024)
        print(f"  matmul ms (uncaptured): {ms:.3f}")

    # Off-shape unblock
    print("\n=== OFF-SHAPE 2048^3 ===")
    ms = event(lambda: a2048 @ b2048)
    print(f"  matmul ms: {ms:.3f}")

    # Phase B: FAST path — single 1024 fp32 matmul (post-unblock), captured.
    fast_trace = os.path.join(CAPTURE_DIR, "fast-1024.gputrace")
    print("\n=== PHASE B: FAST 1024^3 fp32 (after off-shape) ===")
    if attempt_capture(mgr, dev, "fast", fast_trace):
        ms = event(lambda: a1024 @ b1024)
        print(f"  matmul ms: {ms:.3f}")
        mgr.stopCapture()
        print("  capture stopped")
    else:
        ms = event(lambda: a1024 @ b1024)
        print(f"  matmul ms (uncaptured): {ms:.3f}")


if __name__ == "__main__":
    main()
