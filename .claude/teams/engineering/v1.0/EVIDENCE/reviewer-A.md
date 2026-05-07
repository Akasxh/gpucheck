# reviewer-A.md

**Specialist**: engineering-reviewer (Track A)
**Date**: 2026-05-01
**Branch**: `feat/track-a-mps` @ `24035aa`

## Stage 1 — Spec compliance vs CHARTER + PLAN

| Plan task | Compliant? | Notes |
|---|---|---|
| A.1 GPUInfo backend field | YES | `backend: str = "cuda"` default keeps CUDA back-compat |
| A.2 Backend Protocol | YES | Protocol is additive; existing CUDA call sites unchanged |
| A.3 @devices("mps") | YES | New `_detect_mps_devices`, `_detect_devices`; `_is_device_available("mps")` works |
| A.4 gpu_benchmark MPS path | YES | `_run_mps` uses `torch.mps.synchronize()` + `time.perf_counter()`; AST test verifies no `Event.synchronize` |
| A.5 assert_close fast-path widened | YES | `actual.device.type in ("cuda", "mps")`; tripwire test confirms |
| A.6 MPS tolerance overlay | YES | `device_type="mps"` doubles atol/rtol; PROVISIONAL marked in code |
| A.7 [tool.gpucheck.mps.xfail] config | YES | 12 SYNTHESIS §7 entries; parsed at `pytest_configure` |
| A.8 [mps] / [apple] extras | YES | torch>=2.6 floor cited in pyproject |
| A.9 Tests | YES | 4 new test files, +30 passing tests |

## Stage 2 — Code quality

### Strengths
1. Backend Protocol is a clean structural type — no inheritance forced on call sites.
2. The deadlock-avoidance is verified at three layers: code comment, AST-introspected test, and the actual implementation. Defense in depth against future refactors.
3. PROVISIONAL marker on tolerance multipliers honors SYNTHESIS §7's calibration caveat.
4. xfail registry uses replacement semantics (apply_mps_xfail_config clears existing entries) — predictable behavior.

### Concerns

- **CONCERN**: `psutil` is imported lazily but not declared as an extra. Acceptable since the `mem_stats` `rss` key is best-effort (we wrap the import in try/except). Document this in v1.1 as an optional extra.
- **NIT**: `MPSBackend.flush_l2` mutates the module-global `_FLUSH_L2_WARNED`. Test `test_mps_backend_flush_l2_is_noop_with_warning` resets it explicitly. A `warnings.simplefilter("always")` context manager would be cleaner; current approach works.
- **NIT**: `_detect_apple_chip()` shells out to sysctl with a 2s timeout. On a chrooted CI without sysctl, it returns "" silently — fine for arch_info().

### REQUEST_CHANGES?

No. The two concerns above are documentation/refactor-quality items appropriate for v1.1. Track-A ships.

## Verdict

**APPROVED** for Phase 3 merge (after Track-C lands per the C → A → B → D order).
