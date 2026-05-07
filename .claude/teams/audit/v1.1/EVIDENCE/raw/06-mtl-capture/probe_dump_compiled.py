"""Try `dumpCompiledProducts` on the executable - this should write the
compiled Metal shader binaries to disk somewhere.
"""
import os
import sys
import time
import objc
import Metal
import MetalPerformanceShadersGraph as MPSG

dev = Metal.MTLCreateSystemDefaultDevice()
mps_dev = MPSG.MPSGraphDevice.deviceWithMTLDevice_(dev)
queue = dev.newCommandQueue()

# To capture during dumpCompiledProducts, set MPSGRAPH_DUMP env-var
os.environ.setdefault("MPSGRAPH_DUMP", "1")
os.environ.setdefault("MPSGRAPH_DUMP_FILE_PATH", "/Users/cero/Code/gpucheck/.claude/teams/audit/v1.1/EVIDENCE/raw/06-mtl-capture/dump")
os.environ.setdefault("MTL_DUMP_PIPELINES", "1")
os.environ.setdefault("MPS_KERNEL_DUMP", "1")
os.environ.setdefault("MPSGRAPH_RUNTIME_TRACE", "1")
os.environ.setdefault("MPSGRAPH_RUNTIME_DEBUG", "1")
os.environ.setdefault("MPSGRAPH_DUMP_GPU_KERNELS", "1")
os.environ.setdefault("MPSGRAPH_VERBOSE_LOGGING", "1")
os.environ.setdefault("MPSGRAPH_PRINT_LOG", "1")

print("env probes:")
for k in sorted(os.environ):
    if "MPS" in k.upper() or "MTL" in k.upper():
        print(f"  {k}={os.environ[k]}")

g = MPSG.MPSGraph.alloc().init()
a = g.placeholderWithShape_dataType_name_([1024, 1024], MPSG.MPSDataTypeFloat32, "A")
b = g.placeholderWithShape_dataType_name_([1024, 1024], MPSG.MPSDataTypeFloat32, "B")
c = g.matrixMultiplicationWithPrimaryTensor_secondaryTensor_name_(a, b, "C")

feeds = {a: MPSG.MPSGraphShapedType.alloc().initWithShape_dataType_(a.shape(), a.dataType()),
         b: MPSG.MPSGraphShapedType.alloc().initWithShape_dataType_(b.shape(), b.dataType())}

desc = MPSG.MPSGraphCompilationDescriptor.alloc().init()
exe = g.compileWithDevice_feeds_targetTensors_targetOperations_compilationDescriptor_(mps_dev, feeds, [c], None, desc)
print(f"\nexe: {exe}")
print(f"functionNames: {exe.functionNames()}")
print(f"\nCalling dumpCompiledProducts() ...")
try:
    exe.dumpCompiledProducts()
    print("  dumped (check stdout / cwd / fs)")
except Exception as e:
    print(f"  dumpCompiledProducts() raised: {e}")

print(f"\nCalling dump() ...")
try:
    exe.dump()
    print("  dump() returned")
except Exception as e:
    print(f"  dump() raised: {e}")

print(f"\nCalling dumpModuleWithEV_(0) ...")
try:
    out = exe.dumpModuleWithEV_(0)
    print(f"  dumpModuleWithEV: {out}")
except Exception as e:
    print(f"  dumpModuleWithEV: {e}")

# Try reflection data
try:
    refl = exe.getFunctionReflectionData()
    print(f"\nreflection data: {refl}")
    if refl:
        print(f"  type: {type(refl).__name__}")
        if isinstance(refl, list):
            for r in refl:
                print(f"  item: {r}")
        elif hasattr(refl, 'allKeys'):
            for k in refl.allKeys():
                print(f"  {k}: {refl.objectForKey_(k)}")
except Exception as e:
    print(f"  reflection FAILED: {e}")

# inspect specializeWithDevice
import numpy as np
arr_a = np.random.randn(1024, 1024).astype(np.float32)
arr_b = np.random.randn(1024, 1024).astype(np.float32)
desc_a = MPSG.MPSNDArrayDescriptor.descriptorWithDataType_shape_(MPSG.MPSDataTypeFloat32, [1024, 1024])
nd_a = MPSG.MPSNDArray.alloc().initWithDevice_descriptor_(dev, desc_a)
nd_a.writeBytes_strideBytes_(arr_a.tobytes(), None)
td_a = MPSG.MPSGraphTensorData.alloc().initWithMPSNDArray_(nd_a)
nd_b = MPSG.MPSNDArray.alloc().initWithDevice_descriptor_(dev, desc_a)
nd_b.writeBytes_strideBytes_(arr_b.tobytes(), None)
td_b = MPSG.MPSGraphTensorData.alloc().initWithMPSNDArray_(nd_b)

print("\nRun once to force specialize+dispatch ...")
exe_desc = MPSG.MPSGraphExecutableExecutionDescriptor.alloc().init()
out = exe.runWithMTLCommandQueue_inputsArray_resultsArray_executionDescriptor_(queue, [td_a, td_b], None, exe_desc)

# Re-check reflection
print("After run:")
try:
    refl = exe.getFunctionReflectionData()
    print(f"  reflection: {refl}")
except Exception as e:
    print(f"  reflection: {e}")

# specialize
print("\nspecializeWithDevice_inputShapes... ")
try:
    err = objc.NULL
    spec = exe.specializeWithDevice_inputTypes_compilationDescriptor_(
        mps_dev,
        [MPSG.MPSGraphShapedType.alloc().initWithShape_dataType_([1024, 1024], MPSG.MPSDataTypeFloat32),
         MPSG.MPSGraphShapedType.alloc().initWithShape_dataType_([1024, 1024], MPSG.MPSDataTypeFloat32)],
        desc,
    )
    print(f"  specialized: {spec}")
except Exception as e:
    print(f"  specializeWithDevice failed: {e}")

# Inspect optimizedBytecode
print("\noptimizedBytecode_entryPoint_compilationDescriptor_('main', desc) ...")
try:
    bc = exe.optimizedBytecode_entryPoint_compilationDescriptor_("main", desc)
    print(f"  bytecode: {bc}")
    if bc:
        print(f"  length: {len(bytes(bc))}")
        # Save it and grep
        path = "/Users/cero/Code/gpucheck/.claude/teams/audit/v1.1/EVIDENCE/raw/06-mtl-capture/optimized-bytecode-slow.bin"
        with open(path, "wb") as f:
            f.write(bytes(bc))
        print(f"  saved to {path}")
except Exception as e:
    print(f"  optimizedBytecode FAILED: {e}")
