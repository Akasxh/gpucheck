# Planner — testing / v1.0

Adopted persona: `~/.claude/agents/testing/testing-planner.md`. This is the
strategic decomposition that PROPERTY_PLAN.md, MUTATION_REPORT.md, and
SWARM_PLAN.md will operationalise. It also functions as a contract on
engineering-lead: every importable target named here is what the four
tracks must ship.

## Coverage gap summary

- Files analyzed: 23 source files in `src/gpucheck/`, 7 test modules in `tests/`
- Files with NEW surface (no current coverage): 7 (Tracks A/B/C/D combined)
- Files with structural refactor (`tolerances.py` → ContextVar): 1
- Estimated coverage delta from this plan: ~70% → ~85% line, with mutation score ≥75% on consequential modules

## Test plan

### P0 targets (must test) — release-blocking correctness

| # | Target (file:function/class) | Test type | Generator | Dependencies | Rationale |
|---|---|---|---|---|---|
| A1 | `gpucheck.arch.backend:Backend.detect()` (Protocol) | property + unit | property + writer | fixture: `mock_torch_mps`, `mock_pynvml` | Pure function contract — same env → same result; no I/O side effects. |
| A2 | `gpucheck.arch.backend:Backend.synchronize()` | property + unit | property + writer | `mps_available` fixture | Idempotent: call twice ≡ call once. Must not raise on bare device. |
| A3 | `gpucheck.arch.backend:Backend.event_timer()` | property | property | CUDA-event mock + MPS-event real | Returns `(start_event, end_event, elapsed_ms_callable)`; elapsed_ms is non-negative monotonic. **MUST avoid pytorch#162872 deadlock pattern** — use `torch.mps.synchronize()` device-level sync, NOT per-event sync. |
| A4 | `gpucheck.arch.backend:Backend.mem_stats()` | property | property | per-backend | Returns `MemStats(allocated, reserved, free, total)`; totals coherent: `allocated <= reserved <= total` and `allocated + free <= total + epsilon`. |
| A5 | `gpucheck.arch.backend:Backend.arch_info()` | unit | writer | mock_pynvml, mock_torch_mps | Returns `ArchInfo(family, generation, sm_major, sm_minor, tensor_cores)`; matches actual device when run on real hardware. |
| A6 | `gpucheck.assertions.close:assert_close` CPU vs MPS parity | property | property | `cpu_tensor_pair` strategy + MPS execution | Same shapes, same dtype, same RNG seed → MPS result close to CPU result within `compute_tolerance(dtype) × mps_multiplier`. Tolerance multiplier monotone in dtype precision (FP32 < FP16 < BF16 multipliers). |
| A7 | `gpucheck.assertions.tolerances:compute_tolerance` MPS overlay | property | property | mocked backend | Overlay returns ≥ base CUDA tolerance for same dtype; never tighter. Monotone in `k_dim` (sqrt scaling). |
| A8 | xfail registry: pyproject `[tool.gpucheck.mps.xfail]` | unit | writer | YAML/TOML fixture | Registry parses; SYNTHESIS table top-12 each map to exactly one entry; entries enforced via pytest hook. |
| B1 | `gpucheck.fuzzing.strides:fuzz_strides()` determinism | property | property | none | Same `seed` → identical corpus across 100 invocations. |
| B2 | `gpucheck.fuzzing.strides:fuzz_strides()` shape-compatibility | property | property | hypothesis ShapeStrategy | Every (shape, stride) pair satisfies `len(shape) == len(stride)` and `0 < stride[i] <= prod(shape[i+1:])` (post-broadcast); never produces a stride that segfaults `as_strided`. |
| B3 | `gpucheck.fuzzing.strides:fuzz_strides()` 7-class coverage | unit | writer | none | One run with `n=200, seed=0` produces ≥1 example of each: contiguous, transpose-2D, transpose-3D, broadcast-zero-stride, slice-step-2, sub-tensor-offset, negative-stride. |
| B4 | `gpucheck.fuzzing.inputs:gpu_tensors` (extended) | property | property | torch | Output tensor has expected shape AND stride matching the spec; values are finite (no NaN/Inf unless explicitly requested). |
| C1 | `gpucheck.assertions.tolerances` ContextVar LIFO | property | property | none | Nested `tolerance_context` push/pop is LIFO; after exit, outer value restored. |
| C2 | `gpucheck.assertions.tolerances` ThreadPoolExecutor isolation | property | property | `concurrent.futures` | Override set inside thread A is invisible to thread B. (`ContextVar` semantics — copies-on-task per PEP 567.) |
| C3 | `gpucheck.assertions.tolerances` asyncio task isolation | property | property | `asyncio` | Override set inside `async def task_a()` does not leak across `await` boundary into a sibling `task_b()` started by `asyncio.gather`. |
| C4 | `gpucheck.assertions.tolerances` exception-safety | property | property | none | If body raises inside `tolerance_context`, the override is still popped. |
| D1 | `gpucheck.analysis.determinism:assert_deterministic` hermetic | property | property | `random`, `numpy`, `torch` RNG | Calling `assert_deterministic(fn)` does not mutate `random.getstate()`, `np.random.get_state()`, or `torch.get_rng_state()`. |
| D2 | `gpucheck.reporting.html:render_dashboard` deterministic | property | property | sample baseline JSONs | Calling twice with same input produces byte-identical HTML (timestamps either omitted or sourced from fixture). |
| D3 | `gpucheck.reporting.html:render_dashboard` schema-tolerant | property | property | malformed JSONs | Either rejects with informative error OR emits dashboard with "missing data" placeholder; never raises uncaught exception. |

### P1 targets (should test) — security + reliability

| # | Target (file:function/class) | Test type | Generator | Dependencies | Rationale |
|---|---|---|---|---|---|
| S1 | `gpucheck.sanitizers.race:_find_compute_sanitizer` (TM-E1) | property | property | tmp_path, env-monkeypatch | After fix: `realpath(CUDA_HOME)` must match an allowlist (`/usr/local/cuda`, `/opt/nvidia/cuda`, `/opt/cuda`); attacker-controlled path rejected. |
| S2 | `.github/workflows/ci.yml` permissions block (CFG-2) | unit | writer | YAML reader | Workflow YAML parses; top-level `permissions:` includes `contents: read` and excludes `write-all`. |
| S3 | `gpucheck.reporting.json:load_baseline` schema (AR-2) | property | property | hypothesis JSON strategy | Malformed/oversize/path-traversed inputs rejected; valid baselines round-trip. |
| S4 | `gpucheck.fixtures.benchmark:gpu_benchmark` MPS event API | unit | writer | real MPS or mock | Must NOT call `event.synchronize()` in a way that triggers pytorch#162872 deadlock. Test guards by introspecting calls. |
| S5 | `gpucheck.arch.tensor_cores:has_tensor_cores` `NVIDIA_TF32_OVERRIDE` (OWASP-A05-1) | unit | writer | env-monkeypatch | Reject non-{`0`,`1`} values, log warning, fall back to default. |

### P2 targets (nice to have) — polish

| # | Target | Test type | Rationale |
|---|---|---|---|
| P2-1 | `gpucheck.reporting.console:render_diff` | unit | currently zero coverage; visual inspection only |
| P2-2 | `gpucheck.reporting.ci:emit_junit_xml` | unit | XML schema validation |
| P2-3 | `gpucheck.analysis.bottleneck` | unit | exists but under-tested |

## Fixture requirements (consumed by `testing-fixture`)

- `mps_available` — pytest fixture that yields the MPS backend OR `pytest.skip("MPS not available")`. Lazy import of `torch.mps`.
- `mock_torch_mps` — `MagicMock` matching `torch.mps.{is_available, synchronize, empty_cache, current_allocated_memory, event.Event}` API surface. Used when MPS hardware absent.
- `mock_pynvml` — `MagicMock` matching `pynvml.nvml{Init, DeviceGetHandleByIndex, DeviceGetName, ShutDown, ...}`.
- `cpu_tensor_pair(shape, dtype, seed)` — factory yielding `(reference_cpu_tensor, candidate_cpu_tensor)` for parity tests.
- `seeded_rng(seed)` — yields a context where `random`, `numpy.random`, `torch` are all seeded; restores original state on exit. Used by D1.
- `temp_baseline_json(content)` — writes a JSON baseline to `tmp_path` and returns its path. Used by S3.
- `clean_tolerance_context` — autouse fixture that pops any leaked tolerance overrides between tests. Critical for parallel runs (xdist).
- `cuda_home_sandbox` — fixture that creates a fake CUDA_HOME tree under `tmp_path` with a stub binary, used by S1.

## Mutation testing targets (consumed by `testing-mutator`)

P0-marked code where a surviving mutant indicates a real test gap:

- `src/gpucheck/arch/backend.py` (NEW) — Backend Protocol resolution
- `src/gpucheck/arch/backend_mps.py` (NEW) — `mem_stats`, `event_timer`, `synchronize`
- `src/gpucheck/assertions/tolerances.py` — ContextVar refactor + MPS overlay arithmetic
- `src/gpucheck/assertions/close.py` — `assert_close` MPS code path
- `src/gpucheck/fuzzing/strides.py` (NEW) — stride generator
- `src/gpucheck/analysis/determinism.py` (NEW) — `assert_deterministic`
- `src/gpucheck/sanitizers/race.py` — `_find_compute_sanitizer` (security-critical, TM-E1)
- `src/gpucheck/reporting/html.py` (NEW) — dashboard renderer (only rendering logic, not styling)

Skipped from mutation:
- `src/gpucheck/reporting/console.py` — Rich-based UI; mutations on style strings produce equivalent mutants
- Decorator factories — `@dtypes`, `@shapes` — mutations there produce trivially-equivalent mutants

## Dependency ordering (Phase B execution sequence)

Track A is gating: **Track A must merge first**, then B/C/D can land in parallel.

1. **Track A (MPS backend)** — fixtures `mps_available` and `mock_torch_mps` first; then property tests A1–A8; then Backend Protocol unit tests; then xfail registry tests.
2. **Track B (strides)** — depends on Track A only for backend selection; otherwise independent.
3. **Track C (ContextVar)** — independent of A/B/D, but `clean_tolerance_context` autouse fixture must land **before** any other test that touches `tolerance_context` (otherwise A/B parallel tests poison each other).
4. **Track D (determinism + HTML)** — independent of A/B/C; D1 needs `seeded_rng` fixture first.

## Estimated effort

- Total targets: 30 (19 P0 + 5 P1 + 3 P2 + 3 ad-hoc smoke checks)
- Estimated new test files: 7
  - `tests/test_backend_props.py` (Track A)
  - `tests/test_assert_close_mps_props.py` (Track A)
  - `tests/test_xfail_registry.py` (Track A)
  - `tests/test_strides_props.py` (Track B)
  - `tests/test_tolerance_contextvar_props.py` (Track C)
  - `tests/test_determinism_props.py` (Track D)
  - `tests/test_html_dashboard_props.py` (Track D)
- Estimated new test functions: ~55 properties + ~30 unit tests
- Plus security regression tests appended to existing files: `tests/test_arch.py`, `tests/test_ci.py` (new), `tests/security/test_race_path_injection.py` (new)

## Risk register

| Risk | Mitigation |
|---|---|
| MPS hardware not in CI | Tests gated by `mps_available` fixture; mock used in CI; real MPS only via Akash's M-machine swarm |
| pytorch#162872 deadlock during event-timer property test | Test must use `torch.mps.synchronize()` (device-level), never `event.synchronize()`. Property test verifies the contract. |
| Flaky property tests at low example counts | `@settings(max_examples=200, deadline=1000)` for fast properties; `max_examples=50` for MPS-touching tests; deterministic seeds via `@settings(derandomize=True)` for CI runs. |
| Mutation testing wall-clock | Per-track scope, max 1000 mutants per module, 30s per-mutant timeout. Run only in nightly CI or on local M-machine. |
| ContextVar test poisoning between tests | `clean_tolerance_context` autouse fixture in `conftest.py` |

## Verdict

PLANNED — 30 targets across P0/P1/P2, 7 new test files, 4 new fixture
groups, 8 mutation targets. Plan binds engineering-lead's 4 tracks to
specific importable names and test contracts. Ready for
testing-property + testing-mutator + testing-fixture detailing.
