---
name: mps-kernel-debugging
description: Debug and validate PyTorch kernels running on the Apple-Silicon MPS (Metal Performance Shaders) backend. Covers device selection (`torch.device("mps")`), MPS-specific tolerance shifts vs CUDA, `torch.mps.synchronize()` event timing, op-coverage gaps (which torch ops fall back to CPU on MPS), and reproducing numerical mismatches between CUDA and MPS for the same kernel. Use this skill IMMEDIATELY when the user asks to run gpucheck or any pytest GPU test against MPS, says "this works on CUDA but fails on Mac", mentions `mps_dispatch`, asks why an op is slow on Apple Silicon, or wants to add an MPS device row to a parametric test matrix. Apple-Silicon-only: do NOT trigger on Linux/CUDA-only sessions or on Intel Macs.
when-to-use: macOS Apple Silicon, PyTorch >=2.0 with MPS backend; gpucheck `decorators.parametrize_gpu` invoked with `devices=["mps"]`; mismatch between CUDA and MPS reference outputs.
disable-model-invocation: false
---

# mps-kernel-debugging

You are extending gpucheck to handle the PyTorch MPS backend on Apple Silicon. This skill encapsulates the four real-world hazards that bite anyone porting CUDA-validated kernels to MPS:

1. **Op-coverage gaps**: not every `torch.*` op has an MPS implementation. Calling an unsupported op silently triggers `PYTORCH_ENABLE_MPS_FALLBACK=1` CPU fallback (or raises `NotImplementedError` if the env var is unset). This destroys timing benchmarks and changes numerics.
2. **Tolerance shifts vs CUDA**: MPS uses Metal's float math, which has different rounding for `tanh`, `exp`, reductions, and matmul accumulation than NVIDIA SASS. gpucheck's CUDA-calibrated tolerance tables under-tolerate MPS by ~1.5-3× for fp16 matmul.
3. **Event timing**: there is no `torch.cuda.Event` on MPS. Use `torch.mps.synchronize()` plus `time.perf_counter_ns()` deltas. CUDA-event-based benchmarking code (`gpucheck.fixtures.gpu_benchmark`) needs an MPS branch.
4. **fp64 absence**: MPS does not support `float64`. Tests parameterized with `@dtypes(torch.float64)` must be skipped or re-cast.

## When to apply this skill

Apply when:
- A test matrix includes `devices=["cuda", "mps"]` and the MPS row fails or is slow.
- The user says: "run on Mac", "MPS dispatch", "Apple Silicon GPU", "M1/M2/M3 GPU", or invokes `gpucheck` with `--device mps`.
- A CUDA-passing kernel produces NaN, large drift, or `NotImplementedError` on MPS.
- The user is adding MPS support to gpucheck's `arch/` detection module (currently SM60-SM120 NVIDIA only).

Do NOT apply when:
- The session is on Linux or Windows.
- The user is on an Intel Mac (no MPS — fail-fast with a clear error instead).
- The kernel is CPU-only.

## Procedure

### Step 1: Verify environment

```python
import torch
assert torch.backends.mps.is_available(), "MPS backend not built into this torch"
assert torch.backends.mps.is_built(), "MPS not built"
```

If either fails, stop and tell the user: "this Mac/torch lacks MPS support — install torch >=2.0 from pytorch.org with the macOS arm64 wheel."

### Step 2: Check op coverage before running the test

For each op the kernel calls, verify it has an MPS implementation. The authoritative list lives in `torch/_torch_docs.py` and `aten/src/ATen/native/mps/operations/` in the PyTorch repo. The fast runtime check:

```python
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"  # force errors instead of silent fallback
```

If the test then fails with `NotImplementedError: ... MPS backend`, the op is unsupported. Either (a) restructure the kernel to use a supported op, or (b) explicitly skip the test with `@pytest.mark.skipif(reason="op not on MPS")`.

### Step 3: Adjust tolerances

For gpucheck's `assert_close()`, the CUDA-calibrated tolerance tables under-tolerate MPS. Apply this multiplier table:

| dtype | MPS rtol multiplier vs CUDA | MPS atol multiplier vs CUDA |
|---|---|---|
| float32 | 1.0 (no shift) | 1.0 |
| float16 | 2.0 (matmul accum differs) | 2.0 |
| bfloat16 | 1.5 | 1.5 |
| float64 | N/A — skip, MPS has no fp64 | — |

Wire this into the test by passing `device_tol_multiplier={"mps": 2.0}` to `gpucheck.assert_close()` or by extending `gpucheck.assertions.tolerances` with an `mps` column.

### Step 4: Time correctly

Replace any CUDA-event timing block with:

```python
import torch
import time

torch.mps.synchronize()
t0 = time.perf_counter_ns()
out = kernel(*args)
torch.mps.synchronize()
t1 = time.perf_counter_ns()
elapsed_ms = (t1 - t0) / 1e6
```

`torch.mps.synchronize()` is the documented API (PyTorch >=2.0). Without it, kernel launch is async and the timer captures only dispatch latency.

### Step 5: Diagnose CUDA-vs-MPS numeric drift

When a kernel passes on CUDA but fails on MPS:

1. Run both backends with `torch.manual_seed(0)` and identical inputs.
2. Compute `(cuda_out - mps_out.cpu()).abs().max()` and `.mean()`.
3. If max drift is concentrated in a few elements, the likely cause is a reduction-order difference (Metal does parallel reduction in a different tree shape). Increase `rtol` rather than chasing the diff.
4. If drift is uniform, the likely cause is a different transcendental implementation (e.g., `torch.exp` on MPS uses Metal's `metal::exp` which is `relaxed-precision`). Either accept the higher tolerance or rewrite using a Newton-step refinement.

### Step 6: Reporting

Emit gpucheck mismatch reports with `device="mps"` in the metadata so the Rich console output and JSON exporter can group MPS-specific failures separately. Update `gpucheck.reporting.console` to accept `mps` as a known device key (currently only `cuda` and `cpu` are special-cased — see `arch/detect.py`).

## Out-of-scope

- Metal shader (`.metal` file) profiling: use the sibling skill `metal-shader-profiling`.
- Multi-GPU MPS: not supported by Apple Silicon (single integrated GPU per chip).
- AMD eGPU on Intel Mac: deprecated by Apple in macOS 13+, do not attempt.

## References

- PyTorch MPS backend docs: https://pytorch.org/docs/stable/notes/mps.html
- `torch.mps` API: https://pytorch.org/docs/stable/mps.html
- Op coverage tracking issue: pytorch/pytorch#77764
- gpucheck `arch/detect.py` for the NVIDIA-only detection that needs MPS extension.
