---
name: metal-shader-profiling
description: Profile and validate Metal compute shaders (`.metal` files, MTLComputePipelineState kernels) on macOS using `xcrun metal`, `xcrun metal-tt` (timeline trace), `MTLCaptureManager` GPU traces, and Xcode Instruments via the `xctrace` command-line driver. Covers shader-level correctness checks (printf, assertions in Metal), per-threadgroup occupancy analysis, memory-bandwidth and ALU-utilization counters, and capturing a `.gputrace` bundle suitable for Xcode inspection. Use this skill IMMEDIATELY when the user asks to profile a Metal kernel, debug a `.metal` shader, capture a GPU trace on macOS, run `xctrace record --template "Metal System Trace"`, or investigate why an MPS-dispatched op is slow on Apple Silicon. Companion to `mps-kernel-debugging`. macOS-only with Xcode Command Line Tools installed.
when-to-use: macOS with `xcode-select -p` populated and `xcrun metal` available; user is debugging a custom Metal shader, an MPSGraph node, or a torch.mps op that maps to Metal; user wants Instruments-level GPU counters.
disable-model-invocation: false
---

# metal-shader-profiling

You are profiling Metal compute kernels on macOS — either custom `.metal` files compiled to `.metallib`, or the auto-generated Metal kernels that `torch.mps` emits via MPSGraph. This skill covers the three Apple-supported profiling paths and explicitly avoids reverse-engineering closed Metal internals.

## When to apply this skill

Apply when:
- The user wants to profile a Metal kernel: keywords include `metal-tt`, `xcrun metal`, `MTLCaptureManager`, `gputrace`, `Instruments`, `xctrace`, `Metal System Trace`.
- An `mps-kernel-debugging` session has identified a slow op and needs counter-level data to attribute the cost.
- The user is writing a custom MTL shader and needs printf/assert during development.
- The user wants to compare CUDA Nsight output to a Metal-equivalent profile (note: counters do not have 1:1 names).

Do NOT apply when:
- The session is on Linux/Windows (Metal is Apple-only).
- Xcode Command Line Tools are not installed (`xcode-select -p` returns empty). Tell the user to run `xcode-select --install`.
- The user wants Nsight Compute / Nsight Systems output — that is CUDA-only; redirect to a CUDA profiling skill.

## Procedure

### Step 1: Verify toolchain

```bash
xcode-select -p              # must return /Applications/Xcode.app/... or /Library/Developer/CommandLineTools
xcrun --find metal           # path to metal compiler
xcrun --find metal-tt        # path to Metal timeline-trace tool (Xcode 14+)
xcrun --find xctrace         # path to Instruments CLI
```

If any are missing, stop and tell the user the exact `xcode-select` or App-Store install step needed.

### Step 2: Compile the shader (custom Metal path)

```bash
xcrun -sdk macosx metal -c kernel.metal -o kernel.air
xcrun -sdk macosx metallib kernel.air -o kernel.metallib
```

For debugging, add `-frecord-sources` so source locations are embedded in the metallib:

```bash
xcrun -sdk macosx metal -c kernel.metal -frecord-sources -o kernel.air
```

This is required for source-level annotations in Xcode's GPU debugger.

### Step 3: Capture a GPU trace via `xctrace` (best for systemic profiling)

```bash
xctrace record \
  --template "Metal System Trace" \
  --launch -- /path/to/host/binary \
  --output trace.trace
```

Open `trace.trace` in Xcode > Instruments. Counters available: GPU utilization, ALU pipeline busy %, texture/buffer bandwidth, vertex/fragment/compute breakdown, command buffer scheduling latency.

For an attached process (e.g., a long-running pytest run):

```bash
xctrace record --template "Metal System Trace" --attach <PID> --time-limit 30s --output trace.trace
```

### Step 4: Capture a `.gputrace` bundle programmatically (best for kernel-level inspection)

In the host (Swift or Objective-C) wrapping the Metal kernel:

```swift
let captureManager = MTLCaptureManager.shared()
let captureDescriptor = MTLCaptureDescriptor()
captureDescriptor.captureObject = device       // MTLDevice
captureDescriptor.destination = .gpuTraceDocument
captureDescriptor.outputURL = URL(fileURLWithPath: "/tmp/kernel.gputrace")
try captureManager.startCapture(with: captureDescriptor)
// ... dispatch the kernel ...
captureManager.stopCapture()
```

For Python users invoking torch.mps: this requires writing a small Swift driver around the kernel of interest. The torch.mps backend does not currently expose a Python-level capture hook (verify against PyTorch master before claiming otherwise). If the user needs Python-only capture, **defer**: tell them this is currently a manual Swift-driver step and offer to scaffold the driver.

### Step 5: Use shader printf for correctness debugging

In `.metal` source:

```metal
#include <metal_stdlib>
using namespace metal;

kernel void my_kernel(...) {
    if (gid.x == 0) {
        os_log_default.log("debug: input[0] = %f", input[0]);
    }
}
```

`os_log_default.log` requires `-fmetal-enable-logging` at compile time and is read via `log stream --predicate 'subsystem == "com.apple.metal"'` in a separate terminal. This is Metal 3.0+ (macOS 14+).

### Step 6: Read counters via `metal-tt`

```bash
xcrun metal-tt --command "compile-and-run" --metallib kernel.metallib --entry-point my_kernel
```

`metal-tt` outputs a timeline-trace file with per-threadgroup execution intervals. Useful for diagnosing serialization bottlenecks (multiple threadgroups not running concurrently). Less detailed than Instruments but scriptable for CI.

### Step 7: Interpret the four most common findings

1. **Low ALU utilization (<30%) but full GPU busy**: memory-bound kernel. Check threadgroup memory size; consider tiled access pattern.
2. **High GPU busy but high command-buffer latency**: too many small dispatches. Batch via `MTLIndirectCommandBuffer`.
3. **Imbalanced threadgroup occupancy**: pick threadgroup size that divides evenly into 32 (the SIMD width on Apple GPUs). Common good sizes: 32, 64, 128, 256.
4. **Throttling under sustained load**: laptop chassis hits thermal limit. Verify by running `sudo powermetrics --samplers gpu_power -i 500` in parallel and checking GPU frequency drops.

## What NOT to do

- Do not invoke `metal-validation` from inside a CI runner that lacks GPU access — Metal needs a real GPU; software fallback does not exist.
- Do not assume `xctrace` templates are stable across Xcode versions — re-verify template names with `xctrace list templates` per Xcode major release.
- Do not promise CUDA Nsight feature parity. Apple Metal counters are a subset.

## References

- Apple — Metal Programming Guide: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
- `xctrace` man page: `man xctrace`
- WWDC 2022 "Profile Metal apps with Instruments": https://developer.apple.com/videos/play/wwdc2022/10043/
- `MTLCaptureManager` reference: https://developer.apple.com/documentation/metal/mtlcapturemanager
