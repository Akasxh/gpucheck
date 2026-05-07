"""Probe the actual MTLComputePipelineState dispatched per command-buffer.

Strategy: Install a `MTLCommandBuffer` completion handler that walks all
encoded compute-pipeline-states and prints their `label` (Metal exposes
this as the function name when set, or the kernel function name).

This requires that we run the matmul through a command buffer we own, and
inspect the encoded compute commands. MPSGraph wraps the encoder, so we
need to use `encodeToCommandBuffer:inputsArray:resultsArray:executionDescriptor:`
on the executable, then snoop the buffer.

Direct Metal pipeline-state inspection from outside the encoder isn't
exposed by the Metal API — but we CAN read `MTLCommandBuffer.kernelStartTime`
+ `kernelEndTime` (post-completion) and try to retrieve any printed
debug-info from MTL_DEBUG_LAYER.

Best pragma route: dump GPU function-table via the MTLDevice's
newComputePipelineStateWithFunction interceptor — but that requires
swizzling MTLDevice. Instead: install a per-CommandBuffer
addCompletedHandler that prints the buffer's debug info.

Realistic deliverable from this script:
- Slow vs fast steady-state timing diff (already known)
- The MPSCommandBuffer's predefined behavior diff (if any)
- The MTLCommandBuffer.label (which MPSGraph sets to descriptive names)
"""
import os
import sys
import time
import objc
import Metal
import MetalPerformanceShadersGraph as MPSG

dev = Metal.MTLCreateSystemDefaultDevice()
queue = dev.newCommandQueue()
mps_dev = MPSG.MPSGraphDevice.deviceWithMTLDevice_(dev)
print(f"Device: {dev.name()} arch={dev.architecture().name() if dev.architecture() else 'unknown'}")


def build_matmul_graph(m, k, n):
    g = MPSG.MPSGraph.alloc().init()
    a = g.placeholderWithShape_dataType_name_([m, k], MPSG.MPSDataTypeFloat32, "A")
    b = g.placeholderWithShape_dataType_name_([k, n], MPSG.MPSDataTypeFloat32, "B")
    c = g.matrixMultiplicationWithPrimaryTensor_secondaryTensor_name_(a, b, "C")
    return g, a, b, c


def random_data(shape):
    import numpy as np
    arr = np.random.randn(*shape).astype(np.float32)
    desc = MPSG.MPSNDArrayDescriptor.descriptorWithDataType_shape_(MPSG.MPSDataTypeFloat32, list(shape))
    nd = MPSG.MPSNDArray.alloc().initWithDevice_descriptor_(dev, desc)
    nd.writeBytes_strideBytes_(arr.tobytes(), None)
    return MPSG.MPSGraphTensorData.alloc().initWithMPSNDArray_(nd)


def compile_to_executable(g, a, b, c):
    feeds_dict = {a: MPSG.MPSGraphShapedType.alloc().initWithShape_dataType_(a.shape(), a.dataType()),
                  b: MPSG.MPSGraphShapedType.alloc().initWithShape_dataType_(b.shape(), b.dataType())}
    desc = MPSG.MPSGraphCompilationDescriptor.alloc().init()
    return g.compileWithDevice_feeds_targetTensors_targetOperations_compilationDescriptor_(
        mps_dev, feeds_dict, [c], None, desc
    )


def run_via_executable_and_inspect(exe, a_data, b_data, label):
    """Use executable.encodeToCommandBuffer to run with a buffer we own."""
    inputs = [a_data, b_data]

    cb = MPSG.MPSCommandBuffer.commandBufferFromCommandQueue_(queue)
    print(f"\n--- {label}: cb={cb} (MPSCommandBuffer)")
    print(f"  cb methods (sample): {[m for m in dir(cb) if not m.startswith('_') and ('label' in m.lower() or 'commit' in m.lower() or 'name' in m.lower() or 'kernel' in m.lower())][:30]}")

    exec_desc = MPSG.MPSGraphExecutableExecutionDescriptor.alloc().init()
    out = exe.encodeToCommandBuffer_inputsArray_resultsArray_executionDescriptor_(
        cb, inputs, None, exec_desc
    )
    cb.commit()
    cb.waitUntilCompleted()

    # After completion, we can inspect kernel start/end times
    print(f"  underlying MTLCommandBuffer:")
    mtl_cb = cb.commandBuffer() if hasattr(cb, 'commandBuffer') else cb
    print(f"    label: {mtl_cb.label() if hasattr(mtl_cb, 'label') and mtl_cb.label() else '(none)'}")
    if hasattr(mtl_cb, 'kernelStartTime'):
        print(f"    kernelStartTime: {mtl_cb.kernelStartTime()}")
    if hasattr(mtl_cb, 'kernelEndTime'):
        print(f"    kernelEndTime: {mtl_cb.kernelEndTime()}")
    if hasattr(mtl_cb, 'GPUStartTime'):
        print(f"    GPUStartTime: {mtl_cb.GPUStartTime()}")
    if hasattr(mtl_cb, 'GPUEndTime'):
        gpu_ms = (mtl_cb.GPUEndTime() - mtl_cb.GPUStartTime()) * 1000.0
        print(f"    GPU exec ms: {gpu_ms:.3f}")
    if hasattr(mtl_cb, 'logs'):
        print(f"    logs: {mtl_cb.logs()}")
    return out


def main():
    print(f"PID={os.getpid()}")

    # Phase 1: SLOW — compile + run 1024^3 fp32 cold
    g_slow, a_s, b_s, c_s = build_matmul_graph(1024, 1024, 1024)
    exe_slow = compile_to_executable(g_slow, a_s, b_s, c_s)
    a_slow_data = random_data((1024, 1024))
    b_slow_data = random_data((1024, 1024))

    # Run a few times to get steady-state, inspect last
    print("\n=== SLOW 1024^3 fp32 cold ===")
    for i in range(5):
        run_via_executable_and_inspect(exe_slow, a_slow_data, b_slow_data, f"slow-{i}")

    # Phase 2: OFF-SHAPE — 2048^3 fp32 (un-blocker)
    g_off, a_o, b_o, c_o = build_matmul_graph(2048, 2048, 2048)
    exe_off = compile_to_executable(g_off, a_o, b_o, c_o)
    a_off_data = random_data((2048, 2048))
    b_off_data = random_data((2048, 2048))
    print("\n=== OFF-SHAPE 2048^3 fp32 ===")
    run_via_executable_and_inspect(exe_off, a_off_data, b_off_data, "off-0")

    # Phase 3: FAST — 1024^3 fp32 again (post unblock)
    g_fast, a_f, b_f, c_f = build_matmul_graph(1024, 1024, 1024)
    exe_fast = compile_to_executable(g_fast, a_f, b_f, c_f)
    a_fast_data = random_data((1024, 1024))
    b_fast_data = random_data((1024, 1024))
    print("\n=== FAST 1024^3 fp32 post-unblock ===")
    for i in range(5):
        run_via_executable_and_inspect(exe_fast, a_fast_data, b_fast_data, f"fast-{i}")


if __name__ == "__main__":
    main()
