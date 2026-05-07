# DIFF_LOG — gpucheck v1.0 Phase 2

Schema: `| ts | track | file | +/- | hunk | intent |`

| ts | track | file | +/- | hunk | intent |
|---|---|---|---|---|---|
| 2026-05-01 | A | src/gpucheck/backends/__init__.py | +95 | new | Backend Protocol package public surface (available_backends, get_backend) |
| 2026-05-01 | A | src/gpucheck/backends/_protocol.py | +66 | new | runtime_checkable Backend & EventTimer Protocol definitions |
| 2026-05-01 | A | src/gpucheck/backends/cuda.py | +103 | new | CUDABackend wraps existing torch.cuda calls (additive) |
| 2026-05-01 | A | src/gpucheck/backends/mps.py | +213 | new | MPSBackend with deadlock-safe event_timer (pytorch#162872) |
| 2026-05-01 | A | src/gpucheck/arch/detection.py | +12/-1 | GPUInfo | add backend: str = "cuda" field; expand docstring |
| 2026-05-01 | A | src/gpucheck/decorators/devices.py | +52/-15 | _detect | _detect_mps_devices; _detect_devices; _is_device_available recognizes "mps" |
| 2026-05-01 | A | src/gpucheck/decorators/parametrize.py | +2/-2 | import | _detect_cuda_devices -> _detect_devices |
| 2026-05-01 | A | src/gpucheck/fixtures/benchmark.py | +98/-23 | __call__ | branch CUDA vs MPS; new _run_cuda + _run_mps |
| 2026-05-01 | A | src/gpucheck/assertions/close.py | +18/-7 | fast-path | widen GPU fast-path to MPS; pass device_type to compute_tolerance |
| 2026-05-01 | A | src/gpucheck/assertions/tolerances.py | +120/-3 | overlay | _MPS_TOLERANCE_MULTIPLIERS; device_type kwarg; xfail registry + loader |
| 2026-05-01 | A | src/gpucheck/assertions/__init__.py | +14/-3 | exports | re-export is_mps_xfailed, mps_xfail_list, register_mps_xfail |
| 2026-05-01 | A | src/gpucheck/__init__.py | +14/-2 | LAZY_MAP | xfail helpers + Backend / available_backends / get_backend |
| 2026-05-01 | A | src/gpucheck/plugin.py | +35 | pytest_configure | _load_pyproject_config reads tomllib + applies tolerance + xfail |
| 2026-05-01 | A | pyproject.toml | +52/-3 | extras+config | [mps] [apple] extras; mypy psutil/tomllib overrides; [tool.gpucheck.mps.xfail] block (12 entries) |
| 2026-05-01 | A | tests/test_backends.py | +137 | new | Backend Protocol contract + AST-introspected proof of no Event.synchronize |
| 2026-05-01 | A | tests/test_devices_mps.py | +55 | new | @devices("mps") parametrization |
| 2026-05-01 | A | tests/test_assert_close_mps.py | +97 | new | MPS overlay + fast-path tripwire |
| 2026-05-01 | A | tests/test_mps_xfail.py | +85 | new | 12-entry config-loader contract |
| 2026-05-01 | B | src/gpucheck/fuzzing/strides.py | +257 | new | 7-category corpus + StrideStrategy |
| 2026-05-01 | B | src/gpucheck/fuzzing/__init__.py | +10/-0 | exports | fuzz_strides, fuzz_strides_for_category, StrideStrategy, STRIDE_CATEGORIES |
| 2026-05-01 | B | src/gpucheck/decorators/parametrize.py | +85/-17 | parametrize | new stride_categories= kwarg branch |
| 2026-05-01 | B | tests/test_fuzz_strides.py | +151 | new | 17 tests covering each stride category contract |
| 2026-05-01 | B | tests/test_fuzz_strides_hypothesis.py | +37 | new | 3 Hypothesis-based property tests |
| 2026-05-01 | B | tests/test_parametrize_gpu_strides.py | +44 | new | parametrize_gpu(stride_categories=) wiring |
| 2026-05-01 | C | src/gpucheck/assertions/tolerances.py | +18/-5 | ContextVar | _tolerance_overrides => ContextVar; tolerance_context uses set/reset |
| 2026-05-01 | C | src/gpucheck/sanitizers/race.py | +56/-3 | TM-E1 | _CUDA_HOME_ALLOWLIST; realpath check; warning on rejection |
| 2026-05-01 | C | tests/test_tolerance_thread_safety.py | +131 | new | 4-thread Barrier-coordinated race test + nesting + exception safety |
| 2026-05-01 | C | tests/test_race_cuda_home_allowlist.py | +136 | new | TM-E1 mitigation: 6 tests for path validation |
| 2026-05-01 | D | src/gpucheck/reporting/html.py | +266 | new | Self-contained HTML dashboard generator |
| 2026-05-01 | D | src/gpucheck/reporting/__init__.py | +2 | exports | HTMLReporter |
| 2026-05-01 | D | src/gpucheck/sanitizers/determinism.py | +160 | new | assert_deterministic + @requires_determinism + DeterminismError |
| 2026-05-01 | D | src/gpucheck/sanitizers/__init__.py | +9 | exports | determinism helpers |
| 2026-05-01 | D | tests/test_reporting_console.py | +120 | new | 10 tests for ConsoleReporter |
| 2026-05-01 | D | tests/test_reporting_json.py | +101 | new | 6 tests for JSONReporter incl compare_runs |
| 2026-05-01 | D | tests/test_reporting_ci.py | +124 | new | 8 tests for GH annotations + JUnit XML + PR comment |
| 2026-05-01 | D | tests/test_reporting_html.py | +124 | new | 8 tests for HTMLReporter |
| 2026-05-01 | D | tests/test_determinism.py | +103 | new | 8 tests for determinism sanitizer |
| 2026-05-01 | D | .github/workflows/ci.yml | +25/-4 | CFG-2+DEP-1 | permissions: contents: read; uv sync --frozen install path |
| 2026-05-01 | D | uv.lock | +1220 | new | committed lockfile (DEP-1) |

Track A: 18 file rows (~1310 LOC added, ~70 deleted)
Track B: 6 file rows (~580 LOC added, ~17 deleted)
Track C: 4 file rows (~340 LOC added, ~8 deleted)
Track D: 11 file rows (~2250 LOC added, ~4 deleted, of which 1220 LOC is uv.lock)

Total: 39 file rows across 4 tracks; 4 commits on 4 independent branches.
