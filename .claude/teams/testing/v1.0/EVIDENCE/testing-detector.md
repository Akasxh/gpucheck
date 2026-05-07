# Detector — testing / v1.0

Adopted persona: `~/.claude/agents/testing/testing-detector.md`. Phase 2
plan-prep — detection runs without executing coverage tooling because the
testing-lead dispatch is plan-only (no test generation in this round).

## Project profile

| Field | Value |
|---|---|
| Primary language | Python |
| Secondary languages | NONE |
| Python target | `>=3.10` (per `pyproject.toml:9`) |
| Build backend | `hatchling` (per `pyproject.toml:[build-system]`) |
| Package manager (preferred) | `uv` (matches user global preferences); `pip` fallback OK |
| Test framework | `pytest` (per `pyproject.toml:[tool.pytest.ini_options]`, 117 tests passing on `release/v1.0`) |
| PBT framework | `hypothesis` (declared as optional dep at `pyproject.toml:40`, floor 6.0) — already wired into `src/gpucheck/fuzzing/shapes.py:182 ShapeStrategy` |
| Coverage tool | `pytest-cov` / `coverage.py` (declared in `[dev]` extras) |
| Mutation tool | NOT INSTALLED (must add `mutmut` for Track A/B/C/D mutation plan) |
| Mocking library | `unittest.mock` (stdlib) — observed in `tests/test_arch.py`, `tests/test_assertions.py` |
| CI system | GitHub Actions (`.github/workflows/ci.yml`) — CPU-only, no GPU CI |
| Test directory | `tests/` at repo root |
| Test file naming | `test_<module>.py` (e.g. `test_assertions.py`) |
| Assertion style | bare `assert` (idiomatic pytest); `pytest.raises` for error paths |
| Fixture pattern | module-level `tests/conftest.py` (small) + ad-hoc per-test fixtures via `@pytest.fixture` |
| Hypothesis settings | none configured project-wide; opportunity to standardise via `tests/conftest.py` |

## Source-tree structure (test targets)

```
src/gpucheck/
  __init__.py        plugin.py        py.typed
  arch/              compatibility.py detection.py tensor_cores.py
  assertions/        close.py         tolerances.py reporting.py
  decorators/        (parametric markers, @require_arch, etc.)
  fixtures/          benchmark.py     gpu.py        profiler.py
  fuzzing/           shapes.py        inputs.py     strategies.py
  reporting/         ci.py            console.py    json.py
  sanitizers/        memory.py        race.py
  analysis/          (regression, roofline, bottleneck)
```

Existing tests:

```
tests/
  conftest.py
  gpu_integration/        (probably gated by torch.cuda.is_available)
  test_analysis.py        test_arch.py        test_assertions.py
  test_ci.py              test_decorators.py  test_fuzzing.py
```

NOT yet covered (per CLAUDE.md "Known Weaknesses"):
- `reporting/{console,json,ci}.py` — zero test coverage
- `fixtures/{benchmark,gpu,profiler}.py` — sparse coverage
- `sanitizers/race.py` and `sanitizers/memory.py` — design-stage coverage only

## Coverage baseline (estimated from CLAUDE.md + file-level read)

- Total source lines (rough): ~2.4k LoC across `src/gpucheck/`
- Estimated current line coverage: ~70% (covered modules: assertions, fuzzing, arch, decorators, analysis, plugin)
- Estimated current branch coverage: not measured (need `pytest --cov-branch`)
- Modules at ≤20% coverage: `reporting/*`, `sanitizers/race.py`, `sanitizers/memory.py`, `fixtures/profiler.py`
- 117 tests pass on `release/v1.0` per dispatch context; all CPU-only (CI-locked)

## Engineering deliverables that this plan binds (per dispatch contract)

The four parallel tracks engineering-lead is implementing:

| Track | Owner module(s) | New surface this plan must cover |
|---|---|---|
| **Track A — MPS backend** | `src/gpucheck/arch/backend.py` (NEW), `src/gpucheck/arch/backend_mps.py` (NEW), `src/gpucheck/arch/backend_cuda.py` (NEW), refactor of `arch/detection.py`, MPS overlay in `assertions/tolerances.py` | Backend Protocol invariants; `assert_close` CPU/MPS parity; xfail registry per the SYNTHESIS xfail list |
| **Track B — stride/contiguity fuzzing** | `src/gpucheck/fuzzing/strides.py` (NEW), extension of `fuzzing/inputs.py:gpu_tensors` | `fuzz_strides()` determinism + 7 stride classes |
| **Track C — ContextVar tolerance overrides** | refactor `assertions/tolerances.py:_tolerance_overrides` from list to `contextvars.ContextVar` | LIFO push/pop; thread-isolation; asyncio task-isolation |
| **Track D — determinism + HTML dashboard** | `src/gpucheck/analysis/determinism.py` (NEW `assert_deterministic`), `src/gpucheck/reporting/html.py` (NEW dashboard) | hermetic determinism (no global RNG mutation); identical input → identical HTML output |

## PBT framework readiness

`hypothesis>=6.0` is already a declared optional dep. If engineering bumps the
floor (recommend `>=6.100` per security FINDINGS DEP-4), property tests can use
modern strategies (`st.binary(min_size=...)`, `st.deferred`, `target()`).

## Mutation framework

`mutmut` is NOT installed. Plan adds it under `[dev]`:

```toml
# pyproject.toml [project.optional-dependencies] dev addition:
"mutmut>=2.5",
```

Run command (proposed): `mutmut run --paths-to-mutate src/gpucheck/<module> --runner "pytest -x -q tests/<focused>"`

## Conventions to preserve

- File naming: `test_<module>.py`; new property test files use `test_<module>_props.py` to keep them visually separate.
- Import style: absolute (`from gpucheck.fuzzing.shapes import fuzz_shapes`), already canonical in existing tests.
- Type strictness: mypy strict — every new test fixture must be typed.
- No bare `except`; observed nowhere in current `tests/`.
- Lazy GPU imports: tests must import `torch` inside fixtures or via `pytest.importorskip`, NEVER at module top.

## Verdict

DETECTED — Python 3.10+, pytest+hypothesis stack confirmed, mutmut to be
added as `[dev]` extra, project ready for the property + mutation plan.
Profile suffices to direct the planner downstream.
