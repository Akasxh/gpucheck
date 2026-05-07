# Mutator — testing / v1.0

Adopted persona: `~/.claude/agents/testing/testing-mutator.md`. Phase 2
plan-prep — this file specifies the mutmut config, target list, mutator
selection, expected kill-rate baseline, and dispatch plan. Mutation runs
do NOT execute in this dispatch (no `mutmut run`); they execute post-merge
in each track's worktree.

## Framework

- Python language → `mutmut`
- Version pin (proposed): `mutmut>=2.5,<3` (latest 2.x with stable CLI; 3.x adds breaking flag changes)
- Install: add to `pyproject.toml [project.optional-dependencies] dev`:
  ```toml
  "mutmut>=2.5,<3",
  ```

## mutmut configuration

Add to `pyproject.toml`:

```toml
[tool.mutmut]
paths_to_mutate = "src/gpucheck/"
runner = "pytest -x -q --no-header --tb=no"
backup = false
tests_dir = "tests/"
# Mutators we want active. mutmut auto-runs all by default; we narrow to
# the high-signal set:
# - boundary: <  ->  <=, > -> >= (off-by-one)
# - replace_constant: numeric and string constant replacement
# - swap_operator: + -> -, * -> /, and/or
# - keyword: True -> False, return -> pass
# - comparison: == -> !=
# Equivalent-mutant suspects to skip:
# - expression-statement deletion in __init__ (often equivalent)
exclude = [
    "src/gpucheck/__init__.py",            # lazy-import shim, all-equivalent mutants
    "src/gpucheck/reporting/console.py",   # Rich style strings, equivalent mutants
    "src/gpucheck/decorators/markers.py",  # marker plumbing, equivalent mutants
]
```

Per-target invocation pattern:

```bash
# Track A — MPS backend
mutmut run \
  --paths-to-mutate src/gpucheck/arch/backend.py,src/gpucheck/arch/backend_mps.py \
  --runner "pytest -x -q tests/test_backend_props.py tests/test_assert_close_mps_props.py"

# Track B — strides
mutmut run \
  --paths-to-mutate src/gpucheck/fuzzing/strides.py \
  --runner "pytest -x -q tests/test_strides_props.py tests/test_inputs_props.py"

# Track C — ContextVar tolerances
mutmut run \
  --paths-to-mutate src/gpucheck/assertions/tolerances.py \
  --runner "pytest -x -q tests/test_tolerance_contextvar_props.py tests/test_tolerance_props.py"

# Track D — determinism + HTML
mutmut run \
  --paths-to-mutate src/gpucheck/analysis/determinism.py,src/gpucheck/reporting/html.py \
  --runner "pytest -x -q tests/test_determinism_props.py tests/test_html_dashboard_props.py"

# Cross-cutting — assertions/close.py
mutmut run \
  --paths-to-mutate src/gpucheck/assertions/close.py \
  --runner "pytest -x -q tests/test_assertions.py tests/test_assert_close_mps_props.py"

# Security regression — race.py path-injection guard
mutmut run \
  --paths-to-mutate src/gpucheck/sanitizers/race.py \
  --runner "pytest -x -q tests/security/test_race_path_injection.py"
```

## Targets per track (priority-ordered)

### Track A — MPS backend (estimated 240 mutants)

| File | LoC est | Mutants est | Property test coverage | Threshold | Equivalent-mutant suspects |
|---|---|---|---|---|---|
| `src/gpucheck/arch/backend.py` (Protocol + factory) | 80 | 30 | A1, A5 | ≥80% | abstract method bodies (`...`) |
| `src/gpucheck/arch/backend_mps.py` | 220 | 110 | A2, A3a/b/c, A4 | ≥75% | `torch.mps` thin wrappers |
| `src/gpucheck/arch/backend_cuda.py` (refactor) | 180 | 70 | inherited from existing test_arch.py | ≥70% | pynvml call-throughs |
| `src/gpucheck/arch/detection.py` (refactored to delegate) | 50 | 30 | existing | ≥85% | (none, dense logic) |

Most-consequential mutants:
- `>` → `>=` in `mem_stats` coherence guards (would silently wrap allocation reporting)
- `True` → `False` in MPS-availability cache
- Stripping of `torch.mps.synchronize()` call (regresses pytorch#162872 deadlock fix)
- Constant `2.0` → `1.0` in MPS tolerance multiplier

### Track B — strides (estimated 90 mutants)

| File | LoC est | Mutants est | Property test coverage | Threshold |
|---|---|---|---|---|
| `src/gpucheck/fuzzing/strides.py` (NEW) | 180 | 90 | B1, B2, B3, B4 | ≥85% (security: stride bugs cause segfault) |

Most-consequential mutants:
- Off-by-one on stride bounds (B2 catches via `as_strided` validity)
- `seed = None` → `seed = 0` (B1 catches via determinism property)
- Removing one stride class from generator → B3 fails totality property

### Track C — ContextVar (estimated 60 mutants)

| File | LoC est | Mutants est | Property test coverage | Threshold |
|---|---|---|---|---|
| `src/gpucheck/assertions/tolerances.py` | 130 | 60 | A7a/b/c, C1, C2, C3, C4 | ≥85% |

Most-consequential mutants:
- `ContextVar` → `list` (regresses thread isolation — C2/C3 catch)
- `try/finally` removed around `ContextVar.set/.reset` (C4 catches)
- `min` → `max` in tolerance scaling (A7b catches via monotonicity)
- `sqrt(k_dim / 128)` → `sqrt(k_dim * 128)` (A7b catches)

### Track D — determinism + HTML (estimated 110 mutants)

| File | LoC est | Mutants est | Property test coverage | Threshold |
|---|---|---|---|---|
| `src/gpucheck/analysis/determinism.py` (NEW) | 90 | 40 | D1 | ≥85% |
| `src/gpucheck/reporting/html.py` (NEW) | 200 | 70 | D2, D3 | ≥70% (HTML structure has many equivalent mutants) |

Most-consequential mutants:
- Saving global RNG state vs not saving (D1 catches)
- `runs=3` → `runs=1` (D1 weakens but still passes; advisory)
- Including `time.time()` in HTML output (D2 catches)
- Skipping malformed-input branch (D3 catches)

### Cross-cutting — `assertions/close.py` (estimated 80 mutants)

| File | LoC est | Mutants est | Property test coverage | Threshold |
|---|---|---|---|---|
| `src/gpucheck/assertions/close.py` | 200 | 80 | A6a/b + existing tests | ≥75% |

### Security — `sanitizers/race.py` (estimated 40 mutants)

| File | LoC est | Mutants est | Property test coverage | Threshold |
|---|---|---|---|---|
| `src/gpucheck/sanitizers/race.py:_find_compute_sanitizer` | 30 (function) | 15 | S1 | **≥90%** (security-critical) |
| Rest of `race.py` | 200 | 80 | existing test_arch.py + new tests | ≥70% |

Most-consequential mutants for security target:
- Allowlist `startswith` → `==` (would over-match; S1 catches)
- `realpath` → `abspath` (would not resolve symlinks; S1 catches with symlink fixture)
- Removing the allowlist check entirely (S1 catches)

## Expected kill rate baseline (per track)

| Track | Module(s) | Mutant count est | Killed est | Survived est | Equivalent est | Score est |
|---|---|---|---|---|---|---|
| A | arch/backend*.py | 240 | 192 | 24 | 24 | ~80% |
| B | fuzzing/strides.py | 90 | 80 | 5 | 5 | ~89% |
| C | assertions/tolerances.py | 60 | 54 | 3 | 3 | ~90% |
| D | analysis/determinism.py + reporting/html.py | 110 | 80 | 20 | 10 | ~73% |
| Cross | assertions/close.py | 80 | 64 | 8 | 8 | ~80% |
| Sec | sanitizers/race.py | 95 | 76 | 9 | 10 | ~80% |
| **Aggregate** | | **675** | **546** | **69** | **60** | **~81%** |

These estimates assume:
- Property tests as specified in `EVIDENCE/testing-property.md` land
- Existing 117 tests still pass post-refactor
- Equivalent-mutant rate ~9% (typical for Python with type hints)

## Dispatch plan (post-merge of Track A)

The four tracks merge to `release/v1.0` in this order:
1. Track A (MPS backend) — gating
2. Tracks B, C, D in parallel
3. Cross-cutting + security after all four

Mutation runs in each track's worktree:

```
~/Code/gpucheck-worktrees/
  fuzz-track-a-mutate/    -- Track A mutmut
  fuzz-track-b-mutate/    -- Track B mutmut
  fuzz-track-c-mutate/    -- Track C mutmut
  fuzz-track-d-mutate/    -- Track D mutmut
```

Each worktree runs:
```bash
cd ~/Code/gpucheck-worktrees/fuzz-track-<X>-mutate/
git checkout release/v1.0
pip install -e ".[dev]"
mutmut run --paths-to-mutate <files> --runner "pytest -x -q <tests>" 2>&1 | tee mutate-track-<X>.log
mutmut html  # produces html/index.html
mutmut results > mutate-track-<X>.summary.txt
```

Wall-clock estimates (per track, on M-series Mac):
- Track A: 240 mutants × ~8s/mutant = ~32 min
- Track B: 90 mutants × ~2s = ~3 min
- Track C: 60 mutants × ~1s = ~1 min
- Track D: 110 mutants × ~3s = ~5.5 min
- Cross: 80 × ~6s = ~8 min
- Sec: 95 × ~3s = ~5 min

**Total**: ~55 min if sequential, ~32 min if Tracks B/C/D run in parallel after A.

## Equivalent-mutant policy

When a mutant survives:
1. Inspect the mutation manually (`mutmut show <id>`).
2. If the mutation is provably equivalent (e.g., `x * 1` → `x * 1.0`), mark it `mutmut mark-equivalent <id>` and document in `MUTATION_REPORT.md` Appendix.
3. If genuinely surviving but on edge-case code, add a targeted unit test to kill it.
4. If on critical path (Track A `event_timer` deadlock pattern, S1 path-injection, C2/C3 isolation), this is a **HIGH-severity gap**: testing-skeptic must escalate.

## Mutation score thresholds (binding for Phase 3 merge)

Per protocol comprehensive-tier `>=75%`. Per-track stricter where security/correctness load-bearing:

| Track | Min mutation score for Phase 3 merge |
|---|---|
| A (MPS backend) | 75% (correctness-critical) |
| B (strides) | 80% (segfault risk) |
| C (ContextVar) | 85% (concurrency-critical) |
| D (determinism + HTML) | 70% advisory; HTML is rendering, mutations are noisy |
| Cross (close.py) | 75% |
| Sec (race.py path-injection function) | **90% strict** |
| Aggregate (all of `src/gpucheck/`) | 75% |

## Surviving mutant escalation matrix

| Surviving location | Severity | Action |
|---|---|---|
| Path-injection guard `realpath`/allowlist | HIGH | block merge until killed |
| `event_timer` MPS sync pattern | HIGH | block merge |
| ContextVar isolation (C2/C3 misses) | HIGH | block merge |
| Stride bounds in `as_strided` | HIGH | block merge |
| Tolerance arithmetic (k_dim, dtype monotonicity) | MEDIUM | document + add test next iteration |
| HTML rendering (non-structural) | LOW | accept, list in `MUTATION_REPORT.md` |
| Equivalent mutants | N/A | mark and document |

## Verdict

MUTATION_TESTED (planned) — 6 dispatch groups, ~675 mutants total,
~81% baseline kill rate target, hard 90% floor on the path-injection
guard. mutmut not yet installed; install command added to dev extras.
Run plan dispatches in parallel worktrees post-merge, ~32-55 min wall.
