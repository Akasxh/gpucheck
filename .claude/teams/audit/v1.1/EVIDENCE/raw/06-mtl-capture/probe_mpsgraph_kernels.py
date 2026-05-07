"""Drive MPSGraph matmul directly via PyObjC and inspect the compiled
executable's kernel functions. Bypass Xcode capture entirely.

Approach:
1. Build MPSGraph with matmul op
2. Compile to MPSGraphExecutable
3. Inspect the executable's debug name and its underlying compute pipeline state(s)
4. Look at MTLComputePipelineState.label (often holds the kernel function name)

Two MPSGraphs:
- Graph A: 1024x1024 @ 1024x1024 fp32 (slow path)
- Graph B: 2048x2048 @ 2048x2048 fp32 (off-shape)
- Graph C: 1024x1024 @ 1024x1024 fp32 again (fast path? maybe)

Compare what kernel each graph's executable dispatches.
"""
import os
import sys
import time
import objc
import Metal
import MetalPerformanceShadersGraph as MPSG

dev = Metal.MTLCreateSystemDefaultDevice()
queue = dev.newCommandQueue()
print(f"Device: {dev.name()}")
print(f"Architecture: {dev.architecture()}")


def build_matmul_graph(m, k, n, dtype_str):
    g = MPSG.MPSGraph.alloc().init()
    if dtype_str == "fp32":
        dt = MPSG.MPSDataTypeFloat32
    elif dtype_str == "fp16":
        dt = MPSG.MPSDataTypeFloat16
    else:
        raise ValueError(dtype_str)
    a_shape = [m, k]
    b_shape = [k, n]
    a = g.placeholderWithShape_dataType_name_(a_shape, dt, "A")
    b = g.placeholderWithShape_dataType_name_(b_shape, dt, "B")
    c = g.matrixMultiplicationWithPrimaryTensor_secondaryTensor_name_(a, b, "C")
    return g, a, b, c


def random_array(dev, shape, dtype):
    """Allocate an MPSNDArray on the device."""
    import numpy as np
    if dtype == MPSG.MPSDataTypeFloat32:
        np_dtype = np.float32
    elif dtype == MPSG.MPSDataTypeFloat16:
        np_dtype = np.float16
    arr = np.random.randn(*shape).astype(np_dtype)
    desc = MPSG.MPSNDArrayDescriptor.descriptorWithDataType_shape_(dtype, list(shape))
    nd = MPSG.MPSNDArray.alloc().initWithDevice_descriptor_(dev, desc)
    nd.writeBytes_strideBytes_(arr.tobytes(), None)
    return MPSG.MPSGraphTensorData.alloc().initWithMPSNDArray_(nd), arr


def run_and_time(label, m, k, n, dtype_str):
    print(f"\n=== {label}: {m}x{k} @ {k}x{n} {dtype_str} ===")
    g, a_ph, b_ph, c_ph = build_matmul_graph(m, k, n, dtype_str)
    if dtype_str == "fp32":
        dt = MPSG.MPSDataTypeFloat32
    else:
        dt = MPSG.MPSDataTypeFloat16

    a_data, _ = random_array(dev, (m, k), dt)
    b_data, _ = random_array(dev, (k, n), dt)
    feeds = {a_ph: a_data, b_ph: b_data}
    target_tensors = [c_ph]

    # First run -> compile + dispatch
    print(f"  graph: {g}")
    print(f"  graph debugName: {g.debugDescription()[:200] if hasattr(g, 'debugDescription') else 'n/a'}")

    cb = queue.commandBuffer()
    mpsCmdBuf = MPSG.MPSCommandBuffer.commandBufferFromCommandQueue_(queue)
    # Pre-compile via runWithMTLCommandQueue — first call compiles
    t0 = time.perf_counter()
    results = g.runWithMTLCommandQueue_feeds_targetTensors_targetOperations_(
        queue, feeds, target_tensors, None
    )
    elapsed_compile = (time.perf_counter() - t0) * 1000
    print(f"  compile+dispatch: {elapsed_compile:.3f} ms")

    # Steady-state dispatch x N
    times = []
    for _ in range(8):
        t = time.perf_counter()
        results = g.runWithMTLCommandQueue_feeds_targetTensors_targetOperations_(
            queue, feeds, target_tensors, None
        )
        times.append((time.perf_counter() - t) * 1000)
    print(f"  steady-state ms (8 runs): {[round(t, 3) for t in times]}")

    # Try to get executable info
    if hasattr(g, "compilationDescriptor"):
        print(f"  compilationDescriptor: {g.compilationDescriptor()}")
    return g, times


def main():
    print(f"PID: {os.getpid()}")
    # Probe the system: run cold 1024^3 fp32 first
    g_slow, t_slow = run_and_time("SLOW 1024^3 fp32 cold", 1024, 1024, 1024, "fp32")
    # Off-shape unblock
    g_off, t_off = run_and_time("OFF-SHAPE 2048^3 fp32", 2048, 2048, 2048, "fp32")
    # Try fast path
    g_fast, t_fast = run_and_time("FAST 1024^3 fp32 post-unblock", 1024, 1024, 1024, "fp32")

    # Inspect the graphs - look for the underlying executable
    for label, gh in [("slow", g_slow), ("off", g_off), ("fast", g_fast)]:
        print(f"\n=== {label} graph methods ===")
        for m in dir(gh):
            if not m.startswith("_"):
                if any(k in m.lower() for k in ("exec", "compile", "kernel", "name", "debug", "operation")):
                    print(f"   {m}")


if __name__ == "__main__":
    main()
