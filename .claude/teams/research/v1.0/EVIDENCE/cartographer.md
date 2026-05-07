---
specialist: research-cartographer
slug: v1.0
started: 2026-05-01T03:42:00Z
completed: 2026-05-01T03:43:00Z
tool_calls_count: 5
citations_count: 7
confidence: high
---

# Cartographer — gpucheck source map for MPS-readiness

## §1. Module-by-module CUDA assumptions

The current gpucheck codebase has CUDA hard-wired in **three load-bearing places** that block MPS without code change. Each cited by line.

### 1.1 `decorators/devices.py` — auto-detection is CUDA-only

`src/gpucheck/decorators/devices.py:13-23` (`_detect_cuda_devices`) only enumerates
`torch.cuda` devices. `_is_device_available()` at `:25-43` short-circuits to False
for any device that isn't `cpu` or `cuda*`. The default `@devices()` invocation at
`:78-88` falls back to `["cuda:0"]` when nothing is detected — meaning on an
Apple Silicon machine with no CUDA, **the default decorator emits a skip-marked
cuda:0 test**, not an MPS test. Adding MPS requires:
- a `_detect_mps_devices()` returning `["mps"]` when `torch.backends.mps.is_available()`,
- broadening `_is_device_available` to accept `"mps"`,
- adjusting the auto-detect default order: `cuda > mps > cpu`.

### 1.2 `assertions/tolerances.py` — table is CUDA-calibrated

`src/gpucheck/assertions/tolerances.py:12-24` `_DEFAULT_TOLERANCES` is explicitly
"Calibrated against cuBLAS matmul on Turing/Ampere GPUs." There is no per-device
overlay — only per-dtype. The override stack at `:28` is module-level and
explicitly noted as **not thread-safe**. For MPS, two design choices are forced:
(a) introduce a `device_kind` axis to the lookup, or (b) keep table dtype-only and
publish an MPS-specific overlay via `[tool.gpucheck.tolerances.mps]` in
pyproject.toml — easier and consistent with existing `apply_config_tolerances`
hook at `:117-125`.

### 1.3 `assertions/close.py` — GPU fast-path checks `device.type == "cuda"`

`src/gpucheck/assertions/close.py:163-176`:
```
if (
    _has_torch
    and isinstance(actual, _torch.Tensor)
    ...
    and actual.device.type == "cuda"   # ← MPS rejected
    ...
):
    if _torch.allclose(actual, expected, ...):
        return
```
The fast-path falls through to numpy on MPS, costing a CPU transfer per
`assert_close`. `torch.allclose` IS implemented on MPS (since 2.0); the gate
should be `device.type in {"cuda", "mps"}`.

## §2. Architecture detection module is NVIDIA-only

`src/gpucheck/arch/detection.py:14-28` (`SM_TO_ARCH`) and `:124-194`
(`_detect_via_pynvml`) and `:197-245` (`_detect_via_torch`) all enumerate
`torch.cuda` and the NVIDIA-only `pynvml`. README L367 confirms: "AMD ROCm and
Intel XPU are not supported yet... ROCm support is planned." Apple Silicon has
no `pynvml` analogue; the MPS detector would query `torch.mps.recommended_max_memory()`
and Apple's `IOService` via subprocess only, OR be a thin "MPS available?
device='mps0', name='Apple Silicon (MX)', total_memory=…" stub returning a
**different dataclass shape** than `GPUInfo` (no `compute_capability`, no
`tensor_core_generation`).

## §3. Fuzzing pipeline is device-agnostic — reusable on MPS

`src/gpucheck/fuzzing/shapes.py:97-179` (`fuzz_shapes`) and `:9-16` (TILE_SIZES /
PRIMES / POWER_OF_2_BOUNDARIES / LARGE_DIMS) **make no CUDA assumptions**. The
priority order (degenerate > non-tile-aligned > prime > power-of-2 > large > mixed)
is the same playbook that found triton#9838 and triton#9839; it transfers verbatim.

`src/gpucheck/fuzzing/inputs.py:75-123` (`random_inputs`) generates on CPU and
moves to `device` arg via `.to(device=device)` — works for MPS once the test
decorator allows `mps` as a device string. **Caveat**: `_is_fp8` at `:57-68` lists
the four FP8 dtypes; MPS does not support FP8 (Apple Silicon has no FP8 hardware
path). Edge inputs over fp8 must be skipped on MPS.

## §4. Reporting and sanitizers — partial blockers

- `sanitizers/memory.py` and `fixtures/profiler.py` use `torch.cuda.memory_stats()`.
  README L351 confirms "uses torch.cuda.memory_stats() when available and falls
  back to pynvml". Neither path works on MPS. MPS exposes
  `torch.mps.current_allocated_memory()` and `torch.mps.driver_allocated_memory()`
  ([torch-mps docs, retrieved 2026-05-01](https://docs.pytorch.org/docs/2.11/mps.html)),
  so a thin abstraction is feasible.
- `analysis/roofline.py` takes `GPUSpecs(peak_flops, peak_bandwidth)` as input
  data, not via NVML — README L294 demonstrates the GTX 1650 case. For MPS,
  the user supplies M-series specs externally; module is portable.
- `arch/tensor_cores.py` is hard-NVIDIA. MPS has Apple's AMX coprocessor
  (CPU-side) and the Neural Engine; the abstraction "tensor_core_generation"
  doesn't translate. For v1.0, MPS path returns `tensor_core_generation=None`.

## §5. The CUDA-only assumption is shallow

The CUDA assumption is concentrated in **3 files** (`devices.py`, `close.py`,
`tolerances.py`'s overlay shape) plus the architecture-detection abstraction. The
fuzzing core, the assertion engine, the rich-error reporter, and the analysis
modules are **already device-agnostic** below the dispatch layer. This is good
news for v1.0 — MPS support is an extension, not a rewrite.

## §6. Files that need NEW MPS code (estimate)

| File | Change | Lines (est.) |
|---|---|---|
| `decorators/devices.py` | add MPS detection + auto-fallback | +15 |
| `assertions/close.py` | broaden fast-path device gate | +1 |
| `assertions/tolerances.py` | add `device` axis to overlay (or doc the pyproject pattern) | +20 |
| `arch/detection.py` (or new `arch/mps.py`) | MPS detector returning a degenerate `GPUInfo` | +60 |
| `fixtures/profiler.py` | swap `torch.cuda.memory_stats` for a backend dispatcher | +30 |
| `fuzzing/inputs.py` | skip FP8 on MPS | +5 |
| `tests/` | parametrize MPS path on relevant suites | new files |

Net-net: ~150 LOC of plumbing + a tolerance overlay + a new test parametrization. Not architectural.

## Confidence

High — every claim is backed by line-level reads. `torch.allclose` on MPS is
documented as supported ([torch-mps docs](https://docs.pytorch.org/docs/2.11/mps.html))
and was verified to be available in 2.0+ via PyTorch issue #62811 (closed).
