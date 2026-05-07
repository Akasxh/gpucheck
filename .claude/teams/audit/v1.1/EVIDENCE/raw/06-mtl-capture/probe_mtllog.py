"""Use MTLLogState to capture shader logging during slow vs fast matmul.

MTLLogState is Apple's shader-side os_log facility. We can attach a log
state to the MTLCommandQueue and read back log messages emitted by the
GPU shader. If MPSGraph's shaders emit os_log calls, we'd capture
function names. Otherwise this just proves they don't.
"""
import os
import time
import objc
import Metal
import MetalPerformanceShadersGraph as MPSG

dev = Metal.MTLCreateSystemDefaultDevice()


def make_logging_queue():
    """Build an MTLCommandQueue with logging enabled."""
    log_desc = Metal.MTLLogStateDescriptor.alloc().init()
    if hasattr(log_desc, 'setBufferSize_'):
        log_desc.setBufferSize_(1 << 20)  # 1 MiB
    if hasattr(log_desc, 'setLevel_'):
        log_desc.setLevel_(0)  # debug
    err = objc.NULL
    log_state, err = dev.newLogStateWithDescriptor_error_(log_desc, None)
    if err is not None:
        print(f"  newLogState error: {err}")
        return dev.newCommandQueue()
    print(f"  log_state: {log_state}")

    # Attach handler
    def log_handler(submessage, fileinfo, line, msg):
        print(f"    [GPU LOG] {submessage}: {msg}")

    if hasattr(log_state, 'addLogHandler_'):
        log_state.addLogHandler_(log_handler)
        print("  log handler installed")

    qd = Metal.MTLCommandQueueDescriptor.alloc().init()
    qd.setLogState_(log_state)
    err = objc.NULL
    queue, err = dev.newCommandQueueWithDescriptor_error_(qd, None) if hasattr(dev, 'newCommandQueueWithDescriptor_error_') else (dev.newCommandQueueWithDescriptor_(qd), None)
    if err is not None:
        print(f"  queue error: {err}")
    print(f"  queue with logState: {queue}")
    return queue


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


def main():
    print(f"PID={os.getpid()} dev={dev.name()}")
    queue = make_logging_queue()

    # Build & run slow
    g_s, a_s, b_s, c_s = build_matmul_graph(1024, 1024, 1024)
    a_data = random_data((1024, 1024))
    b_data = random_data((1024, 1024))

    print("\n--- SLOW 1024^3 ---")
    for i in range(3):
        t = time.perf_counter()
        g_s.runWithMTLCommandQueue_feeds_targetTensors_targetOperations_(queue, {a_s: a_data, b_s: b_data}, [c_s], None)
        ms = (time.perf_counter() - t) * 1000
        print(f"  run {i}: {ms:.3f} ms")

    # Off-shape unblocker
    g_o, a_o, b_o, c_o = build_matmul_graph(2048, 2048, 2048)
    a_o_data = random_data((2048, 2048))
    b_o_data = random_data((2048, 2048))
    print("\n--- OFF 2048^3 ---")
    g_o.runWithMTLCommandQueue_feeds_targetTensors_targetOperations_(queue, {a_o: a_o_data, b_o: b_o_data}, [c_o], None)

    # Fast
    g_f, a_f, b_f, c_f = build_matmul_graph(1024, 1024, 1024)
    a_f_data = random_data((1024, 1024))
    b_f_data = random_data((1024, 1024))
    print("\n--- FAST 1024^3 ---")
    for i in range(3):
        t = time.perf_counter()
        g_f.runWithMTLCommandQueue_feeds_targetTensors_targetOperations_(queue, {a_f: a_f_data, b_f: b_f_data}, [c_f], None)
        ms = (time.perf_counter() - t) * 1000
        print(f"  run {i}: {ms:.3f} ms")


if __name__ == "__main__":
    main()
