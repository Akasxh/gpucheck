"""Probe: drive matmul through MPSGraph directly from Python via PyObjC,
on the same Metal device as PyTorch. Add a completion handler that dumps
the encoded compute pipeline state debug-name (which holds the kernel
name).

Strategy:
- We can't intercept PyTorch's command buffer creation
- But MPSGraph picks the same kernel based on (M, N, K, dtype) → calling
  MPSGraph.matrixMultiplicationWithPrimaryTensor:secondaryTensor: from
  Python with the same shapes will dispatch the SAME kernel as PyTorch
  via the cached MPSGraph executable, IF the cache is shared.
- Even if not shared, we'll get the kernel name MPSGraph picks for
  (1024, 1024, 1024, fp32) on cold-start vs after off-shape.

This bypasses needing Xcode-injected capture layer entirely.
"""
import os
os.environ["MTL_CAPTURE_ENABLED"] = "1"

import sys
import time
import objc
import Metal

# MetalPerformanceShadersGraph
try:
    import MetalPerformanceShadersGraph as MPSGraph
    print("MPSGraph imported", file=sys.stderr)
except Exception as e:
    print("MPSGraph import failed:", e, file=sys.stderr)
    sys.exit(2)

dev = Metal.MTLCreateSystemDefaultDevice()
print(f"Device: {dev.name()}")
queue = dev.newCommandQueue()
print(f"Queue: {queue}")

# Build an MPSGraph for matmul
graph = MPSGraph.MPSGraph.alloc().init()
print(f"Graph: {graph}")
print(f"Graph methods: {[m for m in dir(graph) if 'atrix' in m or 'mul' in m]}")
