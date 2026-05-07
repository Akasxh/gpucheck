# [MPS] MPSGraph GEMM picks 2-3x slow kernel ~40% of cold-starts, M in [768,3072]

### Bug description

On Apple Silicon (M5, macOS 26.4.1), MPSGraph's runtime kernel-selector picks one of (at least) two GEMM kernels for the same compiled graph. About **40% of fresh processes** land on a kernel that is **2-3x slower** in pure GPU time. This affects fp32, fp16, and bf16 across **20 of 63 (shape, dtype) cells we tested**, with a danger zone of **M ∈ [768, 3072]**. Small (≤512) and very large (4096³) shapes are clean.

The compiled graph itself is **byte-identical** between the slow and fast cases (`MPSGraphExecutable.serializeToMPSGraphPackage` produces identical MD5 across all package files). The pick happens **at runtime inside `runWithMTLCommandQueue`**, not at compile, so this is not a graph-cache invalidation issue and not a JIT-compile race — see "Pure-GPU-time evidence" below.

PyTorch caches the `MPSGraph` instance via `LookUpOrCreateCachedGraph` in `mm_out_mps_impl` ([`LinearAlgebra.mm:614`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mps/operations/LinearAlgebra.mm)) under the key `mm_out_mps_impl:f32[M,K]:f32[K,N]` ([`OperationUtils.mm:277-303`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mps/OperationUtils.mm)). Whatever kernel MPSGraph picks at first dispatch sticks for that cached graph for the lifetime of the process. There is currently no PyTorch API to invalidate the graph cache and no env var to bypass it.

### Reproduction

Self-contained, runs 10 fresh subprocesses and reports the slow-rate. Expect ~40% to print `SLOW`.

```python
# repro_mps_matmul_kernel_pick.py
import subprocess, sys, statistics, textwrap

CHILD = textwrap.dedent("""
    import time, statistics, torch
    assert torch.backends.mps.is_available()
    a = torch.randn(1024, 1024, device='mps', dtype=torch.float32)
    b = torch.randn(1024, 1024, device='mps', dtype=torch.float32)
    def t():
        torch.mps.synchronize()
        s = time.perf_counter()
        c = a @ b
        torch.mps.synchronize()
        return (time.perf_counter() - s) * 1000
    for _ in range(3): t()                     # WARMUP
    times = [t() for _ in range(10)]
    med = statistics.median(times)
    print(f'{med:.3f} {"SLOW" if med > 2.0 else "fast"}')
""")

if __name__ == "__main__":
    slows = []
    for i in range(10):
        r = subprocess.run([sys.executable, "-c", CHILD], capture_output=True, text=True)
        line = r.stdout.strip()
        print(f"run {i}: {line}")
        slows.append("SLOW" in line)
    print(f"\nSLOW rate: {sum(slows)}/{len(slows)}  (expected ~4/10 on affected hardware)")
```

Sample output on M5 / torch 2.11.0 (one of three identical 10-run sweeps):
```
run 0: 0.943 fast
run 1: 2.834 SLOW
run 2: 0.951 fast
run 3: 2.821 SLOW
run 4: 0.946 fast
run 5: 2.849 SLOW
run 6: 0.940 fast
run 7: 2.836 SLOW
run 8: 0.951 fast
run 9: 0.954 fast
SLOW rate: 4/10
```

### Pure-GPU-time evidence (the load-bearing finding)

Measured at the GPU command-buffer level using `MTLCommandBuffer.GPUStartTime`/`GPUEndTime` (host-side timing excluded, so dispatch overhead and `synchronize()` cost cannot explain the gap):

| measurement | slow path | fast path |
|---|---|---|
| GPU exec time (1024³ fp32) | **2.72 ms** | **0.94 ms** |
| σ over 5 reps | < 0.01 ms | < 0.01 ms |
| GFLOPs | ~789 | ~2284 |
| Pure-GPU gap | — | **2.86×** |

`MPSGraphExecutable.serializeToMPSGraphPackage` output is **byte-identical** (same MD5 on all four files: 2 metadata + 2 binary) on slow vs fast runs. σ < 0.01 ms on both paths excludes any compile-cost variance. **The compiled graph is the same object; the runtime kernel pick differs.**

### Cross-version (bug NOT fixed in nightly)

Same subprocess-isolated repro, 10 fresh processes per cell, "slow" = median > 2.0 ms on 1024³ fp32:

| torch version | SLOW / 10 | cold/warm |
|---|---|---|
| 2.10.0 | 4 / 10 | 2.89× |
| 2.11.0 | 4 / 10 | 2.84× |
| nightly 2.13.0.dev20260507 | 4 / 10 | 2.84× |

Statistically indistinguishable. **The bug is present in current nightly.**

### Cross-shape: 20 of 63 cells affected

Top 5 by cold/warm ratio (M=N=K square + fixed-NK 1024 axis):

| shape | dtype | ratio |
|---|---|---|
| 1024 × 1024 × 1024 | fp32 | 2.91× |
| 1792 × 1792 × 1792 | bf16 | 2.76× |
| 4096 × 1024 × 1024 | bf16 | 2.67× |
| 4096 × 1024 × 1024 | fp16 | 2.66× |
| 768 × 1024 × 1024 | fp32 | 2.57× |

Pattern: 20/63 cells exceed 2× ratio; fp16/bf16 dominate the top-20 (16/20). Danger zone M ∈ [768, 3072]. Small (≤512) and very large (4096³ square) clean. **This is a class of cells, not a single shape.**

### Prior art (this is a new instance of a 9-year-old class)

- **Awni Hannun (Apple MLX lead), [`ml-explore/mlx#243`](https://github.com/ml-explore/mlx/issues/243)**: *"for the speech kwt example this size matmul comes up and we are really slow compared to MPS on it (about 3x I think)"* — first-party Apple acknowledgement of shape-specific 3× MPS matmul cliffs.
- **`ml-explore/mlx#1828` and `#1295`**: same shape-specific cliff class on MPS-adjacent code paths.
- **Apple Developer Forums [#105534](https://developer.apple.com/forums/thread/105534) (2018)**: documented 5× MPSMatrixMultiplication cliff for shapes not divisible by 8. Our case is *not* the N%8 cliff (1024 is highly divisible) — it is a stricter cache-key-granularity instance of the same family.
- **Hollemans 2017, [machinethink.net/blog/mps-matrix-multiplication](https://machinethink.net/blog/mps-matrix-multiplication/)**: kernel-pick variability documented publicly **9 years ago**.

### Related PyTorch issues

- **#136003** [open] — [MPS] Inconsistent performance issues. Same flavor of unexplained dispatch variance, on SDPA. Closest sibling.
- **#182805** [open] — [MPS] `native_group_norm` routes through prim decomposition, ~7× slower. Same fix-class (MPS chooses a slow path when a fast one exists).
- **#181725** [open] — [MPS] `nn.MultiheadAttention` ~9× slower than direct `F.scaled_dot_product_attention`. Same fix-class.
- **#181718** [open] — bf16 matmul on RTX 4090 picks slow kernel for `N % 16 == 8`. The CUDA-side analog of the bug class.

### PyTorch-side workaround surface (currently empty)

- Cache key (`mm_out_mps_impl:f32[1024,1024]:f32[1024,1024]`) is dtype + logical sizes only — no strides, no contiguity (`OperationUtils.mm:277-303`).
- `torch.mps.empty_cache()` only releases the buffer pool (`MPSAllocator.mm:551-553`); the `MPSGraphCache` class (`OperationUtils.h:338-421`) has **no** `clear`/`erase`/`evict`/`invalidate`/`reset`/`drop` method, and there is no Python binding for graph-cache eviction.
- Existing MPS env vars (`PYTORCH_DEBUG_MPS_ALLOCATOR`, `PYTORCH_MPS_HIGH_WATERMARK_RATIO`, `PYTORCH_MPS_FAST_MATH`) do not bypass the cache. **No `PYTORCH_MPS_DISABLE_GRAPH_CACHE` exists.**
- `mm_out_mps_impl` body is byte-identical between 2.10 and 2.11; `MPSGraphCache` mechanism unchanged. Any version-to-version behaviour delta must come from MPSGraph itself or the macOS shader cache.
- `PYTORCH_MPS_PREFER_METAL=1` is **not** a workaround: the hand-written `do_metal_mm` (`LinearAlgebra.mm:82`) measures at ~4.2 ms / 1024³ fp32, slower than even the slow MPSGraph path.

### Caveats (we want to be honest)

1. **Stochastic, not deterministic.** ~40% of fresh processes hit the slow kernel; the rest land on the fast one. Most users hit it at least once on cold-start, but a single-run repro can miss it. The repro script above runs 10 subprocesses precisely because of this.
2. **"Touch a different shape to unblock" is not universal.** It works reliably for 1024³ fp32 (7/9 pokes unblock) but for other affected cells like 1792³ bf16 a poke can make timing *worse* (e.g. 1792³ bf16 + 1024³ bf16 poke: 1.47 → 3.99 ms). Do not advertise as a general workaround.
3. **The exact mechanism is closed-source.** We have shown that the compiled MPSGraphPackage is identical across slow/fast, so the pick happens inside MPSGraph's runtime selection logic, which we cannot inspect. We attempted `MTLCaptureManager` GPU-trace capture but the capture layer ships only with Xcode and was not installable on the test machine.

### What we are asking for

1. **Investigate** why MPSGraph picks variable kernels for the same compiled graph on the same hardware on the same input shapes. The byte-identical-package finding strongly suggests this is inside MPSGraph runtime, not PyTorch.
2. **Add a `torch.mps.invalidate_graph_cache()` Python API** so users have a way to drop the unlucky pick. Requires adding `clear`/`erase` on `MPSGraphCache` first.
3. **OR add a `PYTORCH_MPS_DISABLE_GRAPH_CACHE=1` env var** that bypasses cache lookup entirely — re-pays compile cost per call but lets users avoid sticky picks while debugging.
4. **Escalate to Apple via FB feedback** for the MPSGraph kernel-selector. We've collected enough evidence (byte-identical package + σ<0.01ms + multi-version + multi-cell) for an MPSGraph-team-actionable bug report.

### Versions

- PyTorch 2.11.0, 2.10.0, nightly 2.13.0.dev20260507 (all reproduce)
- macOS 26.4.1 (Mac17,3, Apple M5, 32 GB)
- Python 3.12

cc @kulinseth @malfet @DenisVieriu97 @jhavukainen @aditvenk
