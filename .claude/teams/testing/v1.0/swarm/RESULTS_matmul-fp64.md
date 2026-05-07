# RESULTS_matmul-fp64.md

**Agent:** kernel-fuzzer-matmul-fp64-v2
**Kernel:** matmul-fp64
**Backend:** MPS vs CPU
**Date:** 2026-05-01
**Status:** HALTED — PROCESS ERROR (categorical xfail, not a fuzz target)

## Halt reason

PyTorch's MPS backend does not support `torch.float64` at all. Fuzzing matmul-fp64 on MPS is structurally impossible: the framework rejects fp64 at allocation time, before any kernel runs.

## Reproduction

Environment:
- torch 2.11.0
- macOS 26.4.1 arm64 (Apple Silicon)
- `torch.backends.mps.is_available()` → `True`
- `torch.backends.mps.is_built()` → `True`

All three failure modes raise the same `TypeError` at framework boundary, not from a Metal kernel:

```python
>>> torch.randn(4, 4, dtype=torch.float64, device='mps')
TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS framework
doesn't support float64. Please use float32 instead.

>>> torch.randn(4, 4, dtype=torch.float64).to('mps')
TypeError: Cannot convert a MPS Tensor to float64 dtype …

>>> torch.matmul(a_mps_via_to, b_mps_via_to)   # both .to('mps') from fp64 CPU
TypeError: Cannot convert a MPS Tensor to float64 dtype …
```

The error originates in `MPSAllocator` / dtype validation before dispatch. There is no Metal Performance Shaders fp64 path; Apple Silicon GPUs do not implement IEEE 754 binary64 in their compute units. Promoting to fp32 silently is exactly the failure mode `assert_close` is designed to catch — but the *framework itself* refuses to do that promotion, so there is no kernel output to compare to a CPU reference.

## Classification

**NOT FILABLE** — this is not a kernel correctness bug; it is a documented hardware/framework limitation.

The correct classification under gpucheck's own taxonomy is **xfail-by-construction**. This belongs in the `[tool.gpucheck.mps.xfail]` registry described in `CLAUDE.md` (Track A, v1.0), not in the fuzzing corpus. Fuzzing it produces 500/500 framework-level `TypeError`s with zero numerical signal.

## Iteration count

- Planned: 500
- Executed: 0 (halted on pre-flight environment probe)
- FILABLE candidates: 0
- TOLERANCE_RECALIBRATION candidates: 0
- OK: 0
- Framework rejections: confirmed deterministic across alloc / transfer / matmul

## Recommendation to dispatcher

1. **Drop matmul-fp64 from the MPS swarm matrix.** Replace with `matmul-fp32` or `matmul-bf16` (both supported on MPS) if the goal is matmul precision fuzzing on MPS.
2. **If fp64 matmul correctness on MPS matters semantically** (i.e., user calls `.to(torch.float64)` and expects something), add a **front-door check** in `gpucheck.assertions` that raises a `pytest.skip(...)` with `reason="mps does not support fp64"` rather than letting the user discover this via a confusing `TypeError` from torch internals. This is a DX win, not a fuzzing finding.
3. **Add an xfail entry** to `[tool.gpucheck.mps.xfail]` of the form:
   ```toml
   "torch.float64" = "MPS framework rejects fp64 at allocation; no Metal fp64 path. Apple Silicon GPUs do not implement IEEE 754 binary64."
   ```
   Cite this RESULTS file rather than re-deriving on every run.

## Time budget

Halted at ~30 seconds of an 8-minute budget. Remaining budget returned to dispatcher.
