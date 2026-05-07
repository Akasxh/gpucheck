"""Compile MPSGraph to MPSGraphExecutable, dump executable info & full IR for slow vs fast.

Goal: extract the kernel function name MPSGraph's compiler picks for
(1024, 1024, 1024, fp32) cold vs after off-shape.

Methods tried:
- graph.debugDescription() — gives MLIR module IR with `mps.placeholder` + `mps.matmul` ops, no kernel name
- graph.compileWithDevice... — produces MPSGraphExecutable
- executable.dump() / executable.serialize() — try
- executable.feedTensors() / feedOperations() — try
"""
import os
import sys
import time
import objc
import Metal
import MetalPerformanceShadersGraph as MPSG

dev = Metal.MTLCreateSystemDefaultDevice()
queue = dev.newCommandQueue()
print(f"Device: {dev.name()} arch={dev.architecture().name() if dev.architecture() else 'unknown'}")


def build_matmul_graph(m, k, n):
    g = MPSG.MPSGraph.alloc().init()
    dt = MPSG.MPSDataTypeFloat32
    a = g.placeholderWithShape_dataType_name_([m, k], dt, "A")
    b = g.placeholderWithShape_dataType_name_([k, n], dt, "B")
    c = g.matrixMultiplicationWithPrimaryTensor_secondaryTensor_name_(a, b, "C")
    return g, a, b, c


def random_data(shape):
    import numpy as np
    arr = np.random.randn(*shape).astype(np.float32)
    desc = MPSG.MPSNDArrayDescriptor.descriptorWithDataType_shape_(MPSG.MPSDataTypeFloat32, list(shape))
    nd = MPSG.MPSNDArray.alloc().initWithDevice_descriptor_(dev, desc)
    nd.writeBytes_strideBytes_(arr.tobytes(), None)
    return MPSG.MPSGraphTensorData.alloc().initWithMPSNDArray_(nd)


def dump_executable(g, a, b, c, label):
    print(f"\n=== {label}: graph={g} ===")
    feeds_dict = {a: MPSG.MPSGraphShapedType.alloc().initWithShape_dataType_(a.shape(), a.dataType()),
                  b: MPSG.MPSGraphShapedType.alloc().initWithShape_dataType_(b.shape(), b.dataType())}
    targets = [c]

    desc = MPSG.MPSGraphCompilationDescriptor.alloc().init()
    print(f"  compilation desc methods: {[m for m in dir(desc) if not m.startswith('_') and 'set' in m][:30]}")

    # Compile to executable - need MPSGraphDevice wrapper, not raw MTLDevice
    mps_dev = MPSG.MPSGraphDevice.deviceWithMTLDevice_(dev)
    t0 = time.perf_counter()
    exe = g.compileWithDevice_feeds_targetTensors_targetOperations_compilationDescriptor_(
        mps_dev, feeds_dict, targets, None, desc
    )
    print(f"  compile time: {(time.perf_counter() - t0) * 1000:.2f} ms")
    print(f"  executable: {exe}")
    print(f"  executable type: {type(exe).__name__}")
    # Inspect executable methods
    print(f"  exe methods (filtered):")
    for m in dir(exe):
        if m.startswith("_"):
            continue
        if any(k in m.lower() for k in ("dump", "serial", "kernel", "function", "name", "debug", "desc", "spec")):
            print(f"    {m}")

    # Try debug descriptions
    if hasattr(exe, "debugDescription"):
        print(f"  exe debug:\n{exe.debugDescription()[:1200]}")
    if hasattr(g, "debugDescription"):
        print(f"  graph debug FULL:\n{g.debugDescription()}")

    # Try serializing to .mpsgraphpackage
    pkg_path = f"/Users/cero/Code/gpucheck/.claude/teams/audit/v1.1/EVIDENCE/raw/06-mtl-capture/{label}.mpsgraphpackage"
    if hasattr(exe, "serializeToMPSGraphPackageAtURL_descriptor_") or \
       hasattr(exe, "serializeToMPSGraphPackageAtURL_descriptor_error_"):
        # Newer API:
        if hasattr(exe, "serializeToMPSGraphPackageAtURL_descriptor_"):
            url = objc.lookUpClass("NSURL").fileURLWithPath_(pkg_path)
            try:
                exe.serializeToMPSGraphPackageAtURL_descriptor_(url, None)
                print(f"  serialized to {pkg_path}")
            except Exception as e:
                print(f"  serialize FAILED: {e}")
    return exe


def main():
    print(f"PID={os.getpid()}")

    # Pre-trigger fast path? No — we want SLOW first
    g_slow, a_s, b_s, c_s = build_matmul_graph(1024, 1024, 1024)
    exe_slow = dump_executable(g_slow, a_s, b_s, c_s, "slow-1024-cold")

    # Run it on the GPU (cold) to measure
    a_data = random_data((1024, 1024))
    b_data = random_data((1024, 1024))
    feeds = {a_s: a_data, b_s: b_data}

    # Time the slow graph
    cb = MPSG.MPSCommandBuffer.commandBufferFromCommandQueue_(queue)
    times = []
    for i in range(8):
        t = time.perf_counter()
        g_slow.runWithMTLCommandQueue_feeds_targetTensors_targetOperations_(queue, feeds, [c_s], None)
        times.append((time.perf_counter() - t) * 1000)
    print(f"  slow runtime ms: {[round(t, 2) for t in times]}")

    # Off-shape unblock
    g_off, a_o, b_o, c_o = build_matmul_graph(2048, 2048, 2048)
    a_o_data = random_data((2048, 2048))
    b_o_data = random_data((2048, 2048))
    feeds_o = {a_o: a_o_data, b_o: b_o_data}
    g_off.runWithMTLCommandQueue_feeds_targetTensors_targetOperations_(queue, feeds_o, [c_o], None)

    # Now compile fresh 1024 graph after off-shape - is it the SAME or DIFFERENT?
    g_fast, a_f, b_f, c_f = build_matmul_graph(1024, 1024, 1024)
    exe_fast = dump_executable(g_fast, a_f, b_f, c_f, "fast-1024-after-off")

    # Run fast
    a_f_data = random_data((1024, 1024))
    b_f_data = random_data((1024, 1024))
    feeds_f = {a_f: a_f_data, b_f: b_f_data}
    fast_times = []
    for i in range(8):
        t = time.perf_counter()
        g_fast.runWithMTLCommandQueue_feeds_targetTensors_targetOperations_(queue, feeds_f, [c_f], None)
        fast_times.append((time.perf_counter() - t) * 1000)
    print(f"  fast runtime ms: {[round(t, 2) for t in fast_times]}")

    # Direct comparison of the two executables
    print(f"\n=== DIFF: slow exe vs fast exe ===")
    print(f"  slow exe ptr: {exe_slow}")
    print(f"  fast exe ptr: {exe_fast}")
    print(f"  slow == fast: {exe_slow == exe_fast}")
    print(f"  slow type: {type(exe_slow).__name__}")
    print(f"  fast type: {type(exe_fast).__name__}")


if __name__ == "__main__":
    main()
