# MUTATION_REPORT — gpucheck v1.0 Phase 2

**Slug**: testing/v1.0
**Owner**: testing-lead
**Date**: 2026-05-01
**Source**: `EVIDENCE/testing-mutator.md` (full target list + per-track scores)
**Status**: PLANNED — no `mutmut run` in this dispatch; runs post-merge

This file is the mutmut configuration + targets + run plan for the four
engineering tracks. Mutation runs in each track's dedicated worktree
under `~/Code/gpucheck-worktrees/fuzz-track-<X>-mutate/`, dispatched
in parallel after Track A merges to `release/v1.0`.

---

## Tooling

- Tool: `mutmut>=2.5,<3` (Python language, language-recommended)
- Install: add to `pyproject.toml [project.optional-dependencies] dev`:
  ```toml
  "mutmut>=2.5,<3",
  ```

## `pyproject.toml` config block (engineering adds verbatim)

```toml
[tool.mutmut]
paths_to_mutate = "src/gpucheck/"
runner = "pytest -x -q --no-header --tb=no"
backup = false
tests_dir = "tests/"
exclude = [
    "src/gpucheck/__init__.py",            # lazy-import shim
    "src/gpucheck/reporting/console.py",   # Rich UI strings (equivalent mutants)
    "src/gpucheck/decorators/markers.py",  # marker plumbing
]
```

Mutators we rely on (mutmut's defaults are appropriate):
- `boundary` — `<` ↔ `<=`, `>` ↔ `>=` (off-by-one)
- `replace_constant` — numeric and string constant replacement
- `swap_operator` — `+` ↔ `-`, `*` ↔ `/`, `and` ↔ `or`
- `keyword` — `True` ↔ `False`, `None` removal
- `comparison` — `==` ↔ `!=`

Mutators we disable (high equivalent-mutant rate):
- statement deletion in `__init__.py` files
- f-string literal replacement (rendering-only modules)

---

## Targets per track

### Track A — MPS backend (~240 mutants)

| Module | LoC | Mutants est | Property tests covering | Threshold |
|---|---|---|---|---|
| `src/gpucheck/arch/backend.py` (NEW Protocol + factory) | 80 | 30 | A1, A5, A8, A9 | ≥80% |
| `src/gpucheck/arch/backend_mps.py` (NEW) | 220 | 110 | A2, A3a/b/c, A4, A6 | ≥75% |
| `src/gpucheck/arch/backend_cuda.py` (refactor) | 180 | 70 | existing test_arch.py | ≥70% |
| `src/gpucheck/arch/detection.py` (refactored) | 50 | 30 | existing | ≥85% |

Most-consequential mutants (block-merge if surviving):
- Removal of `torch.mps.synchronize()` device-level call inside `event_timer` (regresses pytorch#162872 fix)
- `True` → `False` in MPS-availability cache
- `> 0` → `>= 0` in `mem_stats` `allocated/reserved/free/total` coherence
- `2.0` → `1.0` for the MPS tolerance multiplier scalar

### Track B — Strides (~90 mutants)

| Module | LoC | Mutants est | Property tests | Threshold |
|---|---|---|---|---|
| `src/gpucheck/fuzzing/strides.py` (NEW) | 180 | 90 | B1, B2, B3, B4 | ≥80% |

Most-consequential:
- Off-by-one in stride bounds (B2 catches via valid `as_strided`)
- `seed = None` → `seed = 0` (B1 catches via determinism)
- Class skip in generator → B3 totality fails

### Track C — ContextVar tolerances (~60 mutants)

| Module | LoC | Mutants est | Property tests | Threshold |
|---|---|---|---|---|
| `src/gpucheck/assertions/tolerances.py` (refactor) | 130 | 60 | A7a/b/c, C1-C5 | ≥85% |

Most-consequential:
- `ContextVar` ↔ `list` (C2/C3 catch — would re-introduce thread bleed)
- `try/finally` removal around `ContextVar.set/reset` (C4 catches)
- `min` ↔ `max` in tolerance scaling (A7b catches)
- `sqrt(k_dim / 128)` typo (A7b catches)

### Track D — Determinism + HTML (~110 mutants)

| Module | LoC | Mutants est | Property tests | Threshold |
|---|---|---|---|---|
| `src/gpucheck/analysis/determinism.py` (NEW) | 90 | 40 | D1 | ≥85% |
| `src/gpucheck/reporting/html.py` (NEW) | 200 | 70 | D2-D4 | ≥70% (advisory) |

Most-consequential:
- Skipping global RNG state save (D1 catches)
- `time.time()` injection (D2 catches)
- Malformed-input branch removal (D3 catches)
- Invalid-HTML emission (D4 catches via html5lib)

### Cross-cutting — `assertions/close.py` (~80 mutants)

| Module | LoC | Mutants est | Property tests | Threshold |
|---|---|---|---|---|
| `src/gpucheck/assertions/close.py` | 200 | 80 | A6a/b + existing | ≥75% |

### Security — `sanitizers/race.py` (path-injection guard, ~95 mutants)

| Module / Function | LoC | Mutants est | Property tests | Threshold |
|---|---|---|---|---|
| `_find_compute_sanitizer` (after fix) | 30 | 15 | S1 | **≥90% strict** |
| Rest of `race.py` | 200 | 80 | existing test_arch.py + new tests | ≥70% |

Most-consequential (block-merge if surviving):
- Allowlist `startswith` ↔ `==` (S1 catches with realpath fixture)
- `realpath` ↔ `abspath` (S1 catches with symlink fixture)
- Allowlist check removal (S1 catches)

---

## Aggregate baseline

| Track | Mutants | Killed (target) | Survived (max) | Equivalent (est) | Score (target) |
|---|---|---|---|---|---|
| A | 240 | 192 | 24 | 24 | 80% |
| B | 90 | 80 | 5 | 5 | 89% |
| C | 60 | 54 | 3 | 3 | 90% |
| D | 110 | 80 | 20 | 10 | 73% |
| Cross | 80 | 64 | 8 | 8 | 80% |
| Sec | 95 | 76 | 9 | 10 | 80% (90% on _find_compute_sanitizer) |
| **Aggregate** | **675** | **546** | **69** | **60** | **~81%** |

---

## Run plan

Each track gets a worktree post-merge:

```
~/Code/gpucheck-worktrees/
  fuzz-track-a-mutate/   # Track A: arch/backend*.py
  fuzz-track-b-mutate/   # Track B: fuzzing/strides.py
  fuzz-track-c-mutate/   # Track C: assertions/tolerances.py
  fuzz-track-d-mutate/   # Track D: analysis/determinism.py + reporting/html.py
  fuzz-cross-mutate/     # cross-cutting: assertions/close.py
  fuzz-sec-mutate/       # security: sanitizers/race.py
```

Each worktree:

```bash
cd ~/Code/gpucheck-worktrees/fuzz-track-<X>-mutate/
git fetch && git checkout release/v1.0 && git pull
pip install -e ".[dev]"  # mutmut included
mutmut run \
  --paths-to-mutate <files> \
  --runner "pytest -x -q <focused tests>" \
  2>&1 | tee mutate-<X>.log
mutmut html  # produces html/index.html
mutmut results > mutate-<X>.summary.txt
```

**Per-track invocations** (verbatim):

```bash
# Track A
mutmut run --paths-to-mutate src/gpucheck/arch/backend.py,src/gpucheck/arch/backend_mps.py,src/gpucheck/arch/backend_cuda.py,src/gpucheck/arch/detection.py \
  --runner "pytest -x -q tests/test_backend_props.py tests/test_assert_close_mps_props.py tests/test_arch.py tests/test_xfail_registry.py tests/test_xfail_plugin_hook.py"

# Track B
mutmut run --paths-to-mutate src/gpucheck/fuzzing/strides.py \
  --runner "pytest -x -q tests/test_strides_props.py tests/test_inputs_props.py"

# Track C
mutmut run --paths-to-mutate src/gpucheck/assertions/tolerances.py \
  --runner "pytest -x -q tests/test_tolerance_contextvar_props.py tests/test_tolerance_props.py tests/test_assertions.py"

# Track D
mutmut run --paths-to-mutate src/gpucheck/analysis/determinism.py,src/gpucheck/reporting/html.py \
  --runner "pytest -x -q tests/test_determinism_props.py tests/test_html_dashboard_props.py"

# Cross
mutmut run --paths-to-mutate src/gpucheck/assertions/close.py \
  --runner "pytest -x -q tests/test_assertions.py tests/test_assert_close_mps_props.py"

# Security
mutmut run --paths-to-mutate src/gpucheck/sanitizers/race.py \
  --runner "pytest -x -q tests/security/test_race_path_injection.py tests/test_arch.py"
```

---

## Wall-clock estimates (M-series Mac)

| Track | Mutants | Avg time/mutant | Wall-clock |
|---|---|---|---|
| A | 240 | ~8s | ~32 min |
| B | 90 | ~2s | ~3 min |
| C | 60 | ~1s | ~1 min |
| D | 110 | ~3s | ~5.5 min |
| Cross | 80 | ~6s | ~8 min |
| Sec | 95 | ~3s | ~5 min |
| **Sequential total** | **675** | — | **~55 min** |
| **Parallel (B/C/D after A)** | — | — | **~32 min** |

---

## Equivalent-mutant policy

When a mutant survives:

1. `mutmut show <id>` to inspect the diff.
2. If equivalent (e.g., `1 * x` → `1.0 * x`), `mutmut mark-equivalent <id>` and document in this file's appendix.
3. If on critical-path code (Track A `event_timer` deadlock pattern, S1 path-injection allowlist, C2/C3 isolation, B2 stride bounds), this is HIGH severity — block merge until killed.
4. Otherwise, add a targeted unit test to kill the survivor.

---

## Surviving mutant escalation matrix

| Surviving location | Severity | Action |
|---|---|---|
| `_find_compute_sanitizer` allowlist / realpath | HIGH | block merge until killed |
| `event_timer` device-sync pattern | HIGH | block merge until killed |
| ContextVar isolation (C2/C3) | HIGH | block merge until killed |
| `as_strided` stride bounds | HIGH | block merge until killed |
| MPS tolerance arithmetic | MEDIUM | document + add test next iteration |
| HTML rendering (non-structural) | LOW | accept, list in appendix |
| Equivalent mutants | N/A | mark and document |

---

## Phase 3 merge gate

For Track <X> to merge to `release/v1.0`:

1. `mutate-<X>.summary.txt` mutation score ≥ track threshold.
2. No HIGH-severity surviving mutant per the escalation matrix.
3. `EVIDENCE/testing-mutator.md` updated with actual numbers (post-run).
4. `evaluator.md` (this session's, plus the post-run evaluator) PASS.

For aggregate (final pre-release sign-off):
- Aggregate mutation score ≥75%.
- Per-track minimums met (see table above).
- All HIGH-severity survivors killed.

---

## Open questions

- Q1: Does mutmut handle `ContextVar` correctly under pytest-xdist? Validate empirically in Track C worktree.
- Q2: HTML rendering 70% threshold may be too lax. Consider tightening to 75% if D4 (`valid_html5`) catches more than expected.
- Q3: Should `compatibility.py` and `tensor_cores.py` be added to the run plan? Currently scoped out as "existing coverage adequate"; revisit if Track A diff touches them.

---

## File pointers

- `EVIDENCE/testing-mutator.md` — full per-target rationale, equivalent-mutant heuristics
- `EVIDENCE/testing-skeptic.md` — adversarial review (notes mutation thresholds may need tightening)
- `EVIDENCE/testing-property.md` — property tests that drive the kill rate
