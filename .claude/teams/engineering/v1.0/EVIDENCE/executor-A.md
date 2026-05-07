# executor-A.md

**Specialist**: engineering-executor (Track A — feat/track-a-mps)
**Date**: 2026-05-01
**Worktree**: `/Users/cero/Code/gpucheck-worktrees/track-a-mps`
**Branch**: `feat/track-a-mps`
**Commit SHA**: `24035aa90687dfd8463acfe4b19bcc0c915dcb85`

## Tasks executed (per PLAN.md §Track-A)

### A.1 GPUInfo extension
- File: `src/gpucheck/arch/detection.py`
- Added `backend: str = "cuda"` field. Existing CUDA call sites unchanged (default).

### A.2 Backend Protocol + CUDA + MPS implementations
- New: `src/gpucheck/backends/__init__.py` — `available_backends()`, `get_backend()`.
- New: `src/gpucheck/backends/_protocol.py` — `Backend(Protocol)`, `EventTimer(Protocol)`, both `@runtime_checkable`.
- New: `src/gpucheck/backends/cuda.py` — `CUDABackend` wraps existing `torch.cuda.*` calls.
- New: `src/gpucheck/backends/mps.py` — `MPSBackend` against `torch.mps.*`. `event_timer` is the deadlock-safe `torch.mps.synchronize()` + `time.perf_counter()` pattern (SYNTHESIS §3 / pytorch#162872). Memory stats expose `current_allocated_memory`, `driver_allocated_memory`, plus `rss` from psutil. `flush_l2` is a no-op with one-time UserWarning. `arch_info` reads chip via `sysctl machdep.cpu.brand_string` (no `xcrun`, no `task_info` — see CHARTER waivers N1/N3).

### A.3 @devices accepts MPS
- File: `src/gpucheck/decorators/devices.py`. `_detect_mps_devices`, `_detect_devices` (CUDA + MPS), `_is_device_available` recognizes `"mps"`.
- File: `src/gpucheck/decorators/parametrize.py`. Renamed import from `_detect_cuda_devices` → `_detect_devices` so `parametrize_gpu(devices=None)` auto-includes MPS.

### A.4 gpu_benchmark deadlock-safe MPS path
- File: `src/gpucheck/fixtures/benchmark.py`. `_BenchmarkRunner.__call__` now branches `cuda_avail` vs `mps_avail` vs no-GPU-skip. New helpers `_run_cuda` and `_run_mps`. The MPS loop uses `time.perf_counter()` between `torch.mps.synchronize()` calls — DOES NOT use `Event.synchronize()`. `flush_l2=True` on MPS emits a UserWarning and is a no-op.

### A.5 assert_close GPU fast-path widened
- File: `src/gpucheck/assertions/close.py`. Fast-path condition changed from `actual.device.type == "cuda"` to `in ("cuda", "mps")`. Test `test_assert_close_mps_fast_path_no_cpu_transfer` patches `_to_numpy` to a tripwire and verifies it isn't called when MPS tensors compare equal.

### A.6 PROVISIONAL MPS tolerance overlay
- File: `src/gpucheck/assertions/tolerances.py`. `_MPS_TOLERANCE_MULTIPLIERS` dict (float32/float16/bfloat16 = 2.0; float64 = 1.0; FP8 = 2.0; tf32 = 1.0). `compute_tolerance(..., device_type="mps")` applies the multiplier AFTER k_dim sqrt scaling. Marked PROVISIONAL in code comment.

### A.7 [tool.gpucheck.mps.xfail] config + registry
- File: `pyproject.toml`. New `[tool.gpucheck.mps.xfail] ops = [...]` with the 12 SYNTHESIS §7 entries.
- File: `src/gpucheck/assertions/tolerances.py`. New `mps_xfail_from_config`, `apply_mps_xfail_config`, `register_mps_xfail`, `reset_mps_xfail`, `is_mps_xfailed`, `mps_xfail_list`.
- File: `src/gpucheck/plugin.py`. `pytest_configure` now reads `pyproject.toml` (via stdlib `tomllib`, fallback `tomli`) and applies both tolerance + xfail config. Silent on parse failure (best-effort).
- Top-level re-exports added in `src/gpucheck/__init__.py` and `src/gpucheck/assertions/__init__.py`.

### A.8 [mps] / [apple] extras + torch>=2.6 floor
- File: `pyproject.toml`. New extras `mps = ["torch>=2.6"]` and `apple = ["gpucheck[mps]"]`. Comment cites SYNTHESIS §3 / pytorch#162872 floor rationale.

### A.9 Tests added (4 files)
- `tests/test_backends.py` — Backend Protocol contract; AST-introspected proof that `MPSBackend.event_timer` does NOT call `Event.synchronize()` (the deadlock pattern).
- `tests/test_devices_mps.py` — `_detect_mps_devices`, `@devices("mps")`, `@devices("cuda:0", "mps")`.
- `tests/test_assert_close_mps.py` — `compute_tolerance` mps overlay, `assert_close` MPS fast-path tripwire test, MPS fp16 within 2x overlay.
- `tests/test_mps_xfail.py` — config parser, registry replacement semantics, all 12 SYNTHESIS §7 entries auto-loaded at session start.

### Misc
- `src/gpucheck/__init__.py`: `__version__ = "1.0.0rc1"` (per docs AUDIT A.5 #34/35).

## Verification

Track-A verifier evidence in `verifier-A.md`. Final state:
- `pytest`: **147 passed, 4 skipped** (baseline 117 → +30 net new).
- `ruff check src/ tests/`: **All checks passed!**
- `mypy src/`: **Success: no issues found in 38 source files**

## Verdict

PASS. All A.1–A.9 acceptance criteria met. Branch ready for Phase 3 merge (after C lands first per Phase 3 ordering recommendation).
