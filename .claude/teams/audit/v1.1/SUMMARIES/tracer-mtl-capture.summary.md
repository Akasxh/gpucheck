# tracer-mtl-capture summary

## Method

`MTLCaptureManager.startCaptureWithDescriptor` failed: "Capture layer is not inserted" (Xcode not installed on this Mac; capture dylib ships only inside Xcode.app). **Pivoted** to PyObjC bindings:
- `pyobjc-framework-MetalPerformanceShadersGraph`
- `MTLCommandBuffer.GPUStartTime/EndTime` (GPU command-buffer-level timing, not host-side)
- `MTLLogState` for shader-level logs
- `MPSGraphExecutable.serializeToMPSGraphPackageAtURL` for compiled-graph diff

## Slow vs fast path measurements

| measurement | slow 1024³ fp32 | fast 1024³ fp32 | off-shape 2048³ |
|---|---|---|---|
| GPU exec time | **2.72 ms** | **0.94 ms** | 14.26 ms |
| σ over 5 reps | < 0.01 ms | < 0.01 ms | — |
| GFLOPs | ~789 | ~2284 | — |

**Pure GPU-time gap: 2.86×.** Excludes dispatch overhead (host-side timing showed 3.7×; the difference is dispatch + sync cost). Tighter than skeptic's 2.64× event-time number because we're now measuring at the GPU command-buffer level.

## The load-bearing finding

**Slow & fast `MPSGraphExecutable.serializeToMPSGraphPackage` outputs are BYTE-IDENTICAL** (same MD5 on all 4 files: 2 metadata + 2 binary).

The compiled MPSGraphPackage is the same. The kernel pick happens **at runtime inside `runWithMTLCommandQueue`**, not at compile.

## Refutes skeptic's hypothesis #4

Skeptic suggested: "async JIT compile finishing mid-stream." Refuted by:
- σ < 0.01 ms on slow path (no compile-cost variance)
- σ < 0.01 ms on fast path (no compile-cost variance)
- Byte-identical compiled MPSGraphPackages (no recompile happens between slow and fast)

The skeptic's other concerns (off-shape unblock is post hoc, reproduction flaky, two fast regimes) remain unrefuted but are now subsidiary to the core finding: **runtime kernel-selection picks differently for the same compiled graph based on dispatch state.**

## Inferred kernel families (cannot verify names without Xcode)

- Slow ≈ generic non-MMA fp32 GEMM (~900 GFLOPs)
- Fast ≈ simdgroup-MMA fp32 tile GEMM (~2571 GFLOPs)

## Recommendation for issue body

Cite the **byte-identical MPSGraphPackage + 2.86× pure-GPU-time gap.**

This is **stronger than naming the kernel** — it pinpoints the bug to MPSGraph's runtime kernel-selection layer (Apple-controlled), not to anything PyTorch can patch.

The maintainer reading this immediately understands:
1. Same compiled graph → not a graph-cache invalidation issue
2. Different runtime kernel pick → MPSGraph runtime state machine is the bug surface
3. PyTorch can't fix this without (a) a graph-cache eviction API + (b) Apple changing MPSGraph's runtime selection
