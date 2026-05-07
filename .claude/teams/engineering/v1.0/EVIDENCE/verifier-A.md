# verifier-A.md

**Specialist**: engineering-verifier (Track A — feat/track-a-mps)
**Date**: 2026-05-01
**Branch**: `feat/track-a-mps` @ `24035aa`

## Commands run (FRESH output)

### 1. pytest

```
$ cd /Users/cero/Code/gpucheck-worktrees/track-a-mps
$ PYTHONPATH=/Users/cero/Code/gpucheck-worktrees/track-a-mps/src \
  /Users/cero/Code/gpucheck/.venv/bin/pytest --tb=short -q

147 passed, 4 skipped, 6 warnings in 0.14s
```

Baseline (release/v1.0 @ a9a9d44): 117 passed, 3 skipped.
Delta: +30 passing tests (the 4th skip is the new `test_devices_mps`
parametrization for the still-unavailable cuda:0 slot).

### 2. ruff (E/F/W/I/N/UP/B/A/SIM/TCH)

```
$ /Users/cero/Code/gpucheck/.venv/bin/ruff check src/ tests/

All checks passed!
```

### 3. mypy strict

```
$ PYTHONPATH=/Users/cero/Code/gpucheck-worktrees/track-a-mps/src \
  /Users/cero/Code/gpucheck/.venv/bin/mypy src/

Success: no issues found in 38 source files
```

## Track A acceptance checks

| Acceptance criterion | Status |
|---|---|
| `from gpucheck.backends import Backend, available_backends, get_backend` | PASS (test_backends.py) |
| `MPSBackend.event_timer` does not call `Event.synchronize()` | PASS (AST-introspected test) |
| `MPSBackend.synchronize()` returns without raising on MPS hardware | PASS |
| `@devices("mps")` parametrizes the test | PASS (test_devices_mps.py) |
| `gpu_benchmark` MPS path exists & uses device sync | PASS (source review + branch coverage) |
| `assert_close` fast-path widened to MPS | PASS (tripwire test) |
| `compute_tolerance(..., device_type="mps")` applies 2x multiplier | PASS (4 tests) |
| `[tool.gpucheck.mps.xfail]` block populated with 12 entries | PASS (test_mps_xfail.py) |
| `is_mps_xfailed("conv2d.large_channels")` returns True | PASS |
| `[mps]` / `[apple]` extras in pyproject.toml | PASS (manual review) |
| `__version__ == "1.0.0rc1"` | PASS |

## Verdict

**PASS** on all 3 quality gates. Track A branch is ready for Phase 3 merge.
