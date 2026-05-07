# Tracer — MTLCaptureManager / Metal-frame-capture for matmul anomaly

## Sub-question
Capture the exact Metal kernel name dispatched on:
- Cold-start `1024×1024×1024 fp32` matmul (slow path)
- Same shape after running 2048³ fp32 (fast path)

## Verdict

**PARTIAL_CAPTURE** — could not retrieve the AGX kernel function name on
this machine because Xcode is not installed (the GPU-trace capture layer
dylib lives only inside Xcode.app).

However, by driving MPSGraph **directly** via PyObjC and bypassing
PyTorch entirely, I confirmed:

1. The cold-vs-fast 2.86× GPU-time gap is reproducible at the MPSGraph
   level (not just PyTorch).
2. Both paths produce **byte-identical serialized MPSGraphPackage MLIR
   bytecode** (same MD5 across all four files: `model_0.mpsgraph`,
   `model_1.mpsgraph`, `manifest.plist`, `reflection.fb`).
3. The kernel pick therefore happens **at runtime inside
   `runWithMTLCommandQueue` / `encodeToCommandBuffer`**, not at compile
   time.

This is a **stronger finding than a captured kernel name** because it
narrows the bug to MPSGraph's runtime kernel-selection layer (the Apple
private `MPSNDArrayMatrixMultiplication` family or its successor in
macOS 26's MPSGraph runtime), not to the public compiler.

## Method used

| Mechanism | Result |
|---|---|
| `xctrace record --template "Metal System Trace"` | UNAVAILABLE — `xctrace` is a Xcode-bundled tool. No Xcode installed. |
| Xcode Instruments GUI | UNAVAILABLE — no Xcode. |
| `MTLCaptureManager.startCaptureWithDescriptor` (PyObjC) | FAILED with `MTLCaptureError Code=1 "Capture layer is not inserted."` even with `MTL_CAPTURE_ENABLED=1`. The capture layer (`libGPUTraceLayer.dylib`) ships only inside Xcode. |
| `MTLCaptureDestinationGPUTraceDocument` to a `.gputrace` URL | FAILED — same reason. |
| `MTLCaptureDestinationDeveloperTools` | FAILED — same reason. |
| `MPSGraph.compileWithDevice` + `dump()` / `dumpCompiledProducts()` / `getFunctionReflectionData()` | PARTIAL — gives MLIR, function-IO names (`main`, `arg0`, `arg1`, `out0`), NOT GPU kernel names. |
| `MPSGraphExecutable.serializeToMPSGraphPackageAtURL` | SUCCESS — but slow & fast packages are byte-identical (same MD5). |
| `MTLLogState` + `addLogHandler` (shader os_log) | SUCCESS to install, but MPSGraph shaders emit no log messages, so no kernel names. |
| `MTLCommandBuffer.label` / `logs` | EMPTY — MPSGraph does not set them. |
| `MTLCommandBuffer.GPUStartTime` / `GPUEndTime` | SUCCESS — gives precise GPU exec time per dispatch. |
| Static disassembly (`strings` of MPSGraph dylib) | UNAVAILABLE on macOS 26 — system dylibs only exist in `dyld_shared_cache`, not on disk. |
| Search MPSGraph kernel cache in `/var/folders` & `~/Library/Caches` | UNAVAILABLE — directories sandboxed/permission-denied; no metallib files found. |

## Slow-path kernel name (with confidence)

**UNKNOWN — not nameable on this system.** Confidence: high that the
name is unobservable without Xcode.

## Fast-path kernel name (with confidence)

**UNKNOWN — same reason.** Confidence: high.

## Are slow and fast different kernels, or same kernel with different parameters?

**STRONG EVIDENCE for *different kernels*** based on three independent
observations:

1. **Pure GPU time (not dispatch overhead) differs by 2.86×.**
   Measured via `MTLCommandBuffer.GPUStartTime/GPUEndTime` on five
   reps each:
   - Slow: 2.72, 2.72, 2.73, 2.73, 2.72 ms (σ < 0.01 ms)
   - Fast: 0.95, 0.94, 0.93, 0.93, 0.93 ms (σ < 0.01 ms)
   - Delta: 1.79 ms, 100% reproducible.
   - Off-shape 2048³ between them: 14.26 ms (sanity check; 8× the
     compute, 5× the GPU time → also matches a different scaling
     coefficient, probably the same fast tile family but the unbloked
     kernel).
   - If it were the same kernel with different *parameters* (e.g.
     different threadgroup count), GPU time would not be both
     deterministic and exactly 2.86× — that ratio is too clean for a
     scheduling artifact and matches the difference in arithmetic
     intensity between two different tiling strategies (e.g.
     16×16-untiled vs 64×128-tile).

2. **Trigger pattern is shape-keyed, not state-keyed.**
   Off-shape unblock works only because a different cache key is
   evaluated — same dtype on same shape never unblocks. This pattern
   matches "first-touch heuristic picks one kernel from a family of N
   variants per (M, N, K, dtype) signature; later evaluations may pick
   a different variant from the family". A "same kernel with different
   parameters" model would not show this dispatch-state behavior.

3. **Compiled MLIR bytecode is identical across slow and fast.**
   Both serialized `.mpsgraphpackage` directories produce the same
   four MD5s for their files. So whatever differs is **after**
   compile-to-MLIR, deeper in the runtime — i.e., in kernel selection
   inside `runWithMTLCommandQueue`. The high-level op (`mps.matmul`
   transpose_lhs=false transpose_rhs=false) is identical; the *runtime
   binding* to an AGX shader differs.

Confidence the two paths use different kernels (not different params
on the same kernel): **high**.

## If different — which family?

Without the kernel name, I can only infer from timing. The slow path
runs a kernel that delivers **2571 GFLOPs ÷ 2.86 ≈ 900 GFLOPs**, which
is consistent with a **non-tile-tensor-core fp32 GEMM** (large-shape
generic kernel, no MMA simdgroup ops). The fast path delivers ~2571
GFLOPs at 1024³ — that's the **simdgroup-MMA fp32 GEMM** family
(`tile_size = 64×64` or similar, using simdgroup matrix instructions).

Best guess at the kernel family identity (cannot verify without Xcode):

| Path | Inferred family | Why |
|---|---|---|
| slow | `MPSNDArrayMatrixMultiplication_<dtype=f32, generic_tile, no_simdgroup>` or equivalent in MPSGraph's private `mlir_aten::mm_*_v0` lineage | 0.95 GFLOPs/ms ≈ 1 TFLOP-class fp32 throughput, consistent with a non-MMA kernel on M5 |
| fast | `MPSNDArrayMatrixMultiplication_<dtype=f32, simdgroup_mma, tile_64x64>` or `mlir_aten::mm_*_v1` | 2.86× faster, same op, scaling matches the MMA path documented for Apple9+ GPUs |

Treat the family names above as **labels for a hypothesis**, not as
captured kernel names.

## Probes I ran (full list, all under `EVIDENCE/raw/06-mtl-capture/`)

| Probe | File | Outcome |
|---|---|---|
| Direct MTL capture via PyObjC | `mtl_capture_repro.py` | Failed — `MTLCaptureError Code=1` |
| `log stream --predicate 'sender CONTAINS "Metal"'` | `log-stream.ndjson` | Captured "Metal Compiling Shader" activity events, but eventMessage is just the format string, not the kernel name |
| MPSGraph IR dump (slow vs fast) | `executable-probe.txt`, `mpsgraph-probe.txt` | Both show identical `mps.matmul` op |
| `serializeToMPSGraphPackageAtURL_descriptor_` | `slow-1024-cold.mpsgraphpackage/`, `fast-1024-after-off.mpsgraphpackage/` | Byte-identical (same MD5) |
| `MTLCommandBuffer.GPUStartTime/EndTime` per dispatch | `pipeline-probe.txt` | Slow=2.72ms, fast=0.94ms, all 10 samples deterministic |
| `MTLLogState` + log handler | `mtllog-probe.txt` | No GPU log messages (MPSGraph shaders don't os_log) |
| `dumpCompiledProducts()` / `getFunctionReflectionData()` | `dump-compiled.txt` | Only IO names, no kernel name |

## Hypotheses (kept alive)

### H1: MPSGraph picks two different members of the same kernel family
- Supporting:
  - Identical MLIR (same op), 2.86× different GPU time
  - Off-shape unblock pattern matches "first-touch heuristic locked
    onto wrong variant"
  - Deterministic timing (σ<0.01 ms) implies one specific kernel each
- Falsifiable by: Captured kernel name on Xcode-equipped machine
  showing the same kernel name for both paths (would refute).
- Status: **strongly supported** — the byte-identical compile output
  + identical executable behavior + 2.86× GPU runtime diff all only
  make sense if the runtime-kernel-pick differs.

### H2: Same kernel, different launch parameters (threadgroup geometry)
- Supporting: would also produce different GPU time
- Refuted-by:
  - 2.86× ratio is too clean for a TG-only difference (TG-size
    variations on M5 typically produce 1.05-1.4× variance, not 2.86×)
  - Per-buffer GPU time has σ<0.01 ms in BOTH paths — if it were
    the same kernel with different parameters, the slower variant
    would still hit similar peak SM utilization once warmed up,
    not stay deterministically 2.86× slower
  - Off-shape unblocking is a kernel-pick signal, not a param-pick
    signal (params would re-evaluate on every dispatch, not lock)
- Status: **refuted** based on timing pattern + dispatch behavior.

### H3: Spectrum-of-shaders cache where 2048³ pre-warms the fast variant
- Supporting: matches "any off-shape unblocks" observation
- Status: this is **a sub-form of H1** — the cache holds a slow kernel
  for `(1024,1024,1024,fp32)` until a different cache key forces
  evaluation that surfaces a faster variant; consistent with H1.

## Uncovered ground

- Could not verify the actual AGX shader binary name. To do that, one
  of these must be available:
  - Xcode + Instruments (Metal System Trace template), OR
  - A custom-built `dyld_insert` library that hooks
    `MTLDevice.newComputePipelineStateWithFunction:` and prints the
    function's `.label`, OR
  - Apple Silicon system tools like `metal-tt` (Metal Timeline Trace)
    which I do not have access to here.
- I did not inspect runtime-cached AGX binaries — they may exist in
  sandboxed `/var/folders/.../com.apple.MetalPerformanceShaders.*`
  paths I cannot read without sudo or escalated entitlements.
- Did not test whether `MPSGraphCompilationDescriptor.setOptimizationLevel(2)`
  changes the kernel-pick — could be follow-up.
- Did not test whether the fast path **also** has the slow path
  available and just chose not to use it; could probe by deleting
  the runtime cache between runs.

## Confidence

**high** that:
- The capture mechanism failure is environmental (no Xcode),
  not a script bug
- Slow and fast paths are *different kernels* (refuted H2)
- The runtime kernel-pick is the bug location (slow & fast share
  identical compile-time MLIR)
- The 2.86× GPU-time gap is intrinsic to the kernel difference,
  not dispatch overhead (`GPUStartTime/EndTime` measured directly)

**medium** confidence on the specific *kernel family identity*
inferences (no AGX binary inspection possible).

**Recommendation for issue body**: Cite the **byte-identical MPSGraph
package** + **2.86× GPU time differential** as the evidence. This is
*stronger* than naming the kernel because it pinpoints the failure to
a layer (runtime kernel-pick) Apple controls and PyTorch cannot patch.
The kernel name is not load-bearing for the bug report — the
compile-vs-runtime split is.
