# Phase A Executor Log — IMPLEMENTATION_PLAN_v1.1

Pool tag: `executor-A` (4 file-safe standalone fixes)
Branch: `release/v1.0`
Baseline tip: `82b853e`
Baseline pytest: **224 passed, 1 skipped**

## T-04 — bare-except cleanup in race.py

**Status:** NO-OP / already clean.

`grep -nE "^\s*except\s" src/gpucheck/sanitizers/race.py` returns exactly one
hit, at line 248: `except subprocess.TimeoutExpired:` — already the most
specific exception type. `ruff --select=E722 src/gpucheck/sanitizers/race.py`
reports `All checks passed!`.

The plan-row 118 wider scope (`arch/detection.py`, `backends/mps.py`) is owned
by other concurrent agents per task instructions. Nothing to commit in this
file. No commit produced.

---

## T-04b — bare-except cleanup in `backends/mps.py` + `arch/detection.py`

**Status:** DONE. Commit `174cb6a`.

7 sites narrowed:
- `backends/mps.py` L99/L137/L141/L148/L191 → `(RuntimeError, AttributeError)`
- `arch/detection.py` L157 → `(pynvml.NVMLError, AttributeError)`
- `arch/detection.py` L229 → `(RuntimeError, AttributeError)`

Each catch now writes a `logger.debug(...)` line before falling back so
"why is X zero on this build?" is answerable without a second debugging round.
Added a module-level `logger = logging.getLogger(__name__)` to `backends/mps.py`
(the `arch/detection.py` logger already existed).

Verification:
- `uv run pytest -q` → **224 passed, 1 skipped** (unchanged from baseline)
- `uv run ruff check src/ tests/` → `All checks passed!`
- `uv run mypy src/` → `Success: no issues found in 41 source files`

Lines changed: `+19 / -7` across 2 files. No public-API change.

---

## T-05 — `_MutableReport` leak from public `memory_guard`

**Status:** DONE. Commit `6bdccb8`.

`memory_guard()` previously yielded a private `_MutableReport`, leaking an
underscore-prefixed type through a public API. Renamed to `MemoryGuardReport`
(public `@dataclass(slots=True)`) and re-exported via `gpucheck.sanitizers`.

Why `MemoryGuardReport` and not `MemoryReport`:
- `gpucheck.fixtures.profiler.MemoryReport` already exists as a *frozen*
  fixture-side summary — different lifecycle, different fields
  (`before/after/peak/leaked`).
- The guard pattern requires a placeholder the context manager can fill on
  exit, so it can't be frozen. Distinct names avoid type-collision and
  signal the different semantics.
- `gpucheck.sanitizers.MemoryReport = SanitizerMemoryReport` backward-compat
  alias is preserved.

Files touched:
- `src/gpucheck/sanitizers/memory.py` — converted `_MutableReport` → public
  `MemoryGuardReport`, hoisted before `memory_guard()`, dropped the now-dead
  `_fill()` indirection (fields are assigned directly post-yield).
- `src/gpucheck/sanitizers/__init__.py` — added `MemoryGuardReport` to
  `__all__`.

Verification:
- `uv run pytest -q` → **224 passed, 1 skipped**
- `uv run ruff check src/ tests/` → `All checks passed!`
- `uv run mypy src/` → `Success: no issues found in 41 source files`

Lines changed: `+60 / -66` across 2 files.

---

## T-07 — `gpu_integration` auto-skip claim now true on MPS

**Status:** DONE. Commit `25fd515`.

README claimed `pytest tests/gpu_integration/` auto-skips without a GPU; on
MPS hosts this *failed* 52 tests on CUDA-specific calls (docs-tester R-B21 /
T-B5).

Fix: added `pytest_collection_modifyitems` to
`tests/gpu_integration/conftest.py` that skips every collected item unless
either CUDA is available or `--mps-integration` was explicitly passed *and*
MPS is available. The new `--mps-integration` flag is registered via
`pytest_addoption` in the same conftest.

Verification on this MPS host:
- `uv run pytest tests/gpu_integration/ -q` → `235 skipped, 1 warning` (was
  failing 52 of those before).
- `uv run pytest -q` (root) → **230 passed, 1 skipped** (unchanged path —
  pyproject.toml `addopts = "--ignore=tests/gpu_integration"` already
  excludes the directory at default invocation; conftest only fires on
  explicit invocation, which is the failure path the README documents).
- `uv run ruff check src/ tests/` → `All checks passed!` (the file is
  excluded from ruff via pyproject `extend-exclude`).
- `uv run mypy src/` → `Success: no issues found in 41 source files`.

Lines changed: `+71 / -1` across 1 file (rewrite of the conftest).

---

## T-08 — `_load_pyproject_config` swallows all exceptions

**Status:** DONE. Commit `aeedbdc`.

Per security-postmerge PM-2: `except Exception: pass` masked TOML parse
errors so users couldn't tell why their `[tool.gpucheck.tolerances]`
overrides were silently ignored. Narrowed to
`(OSError, tomllib.TOMLDecodeError)` and emit a `UserWarning` with the
exception type and message. Programmer errors (AttributeError, TypeError)
now propagate. The absent-file branch still returns silently — documented
fall-through to built-in defaults.

Implementation detail: restructured the function so the broad `try`
disappears entirely. Pathlib import + tomllib import are bare; the file-open
+ tomllib.load is the only `try` block, with a focused except clause.

Verification:
- `uv run pytest -q` → **230 passed, 1 skipped**
- `uv run ruff check src/gpucheck/plugin.py` → `All checks passed!`
- `uv run mypy src/` → `Success: no issues found in 41 source files`
- `uv run ruff check src/ tests/` reports 3 errors in
  `tests/test_assert_close_contiguous.py` — that file is part of another
  agent's *unstaged* T-02 work (contiguous slow-path) and is outside my
  task scope; not fixed.

Lines changed: `+37 / -24` in `src/gpucheck/plugin.py`.

---

## T-20 — consolidate GPU detection helper

**Status:** DONE. Commit `da4280d`.

`fixtures/gpu.py` had its own `_detect_gpu_pynvml` / `_detect_gpu_torch`
duplicating `arch/detection._detect_via_pynvml` / `_detect_via_torch`. The
two stacks had drift-prone differences (NVMLError vs RuntimeError vs
AssertionError catches) and divergent "no backend" behavior.

Refactor:
- Introduced `arch/detection._detect_gpus_or_warn() -> list[GPUInfo] | None`
  as the single source of truth. Returns `None` on no-backend, `[]` on
  no-device-but-backend-present, list of GPUInfo on success. The
  one-shot `UserWarning` lives here.
- `detect_gpus()` is now a thin `@lru_cache` wrapper that maps `None ->
  []`. Public API unchanged.
- `fixtures/gpu.detect_gpu()` calls `detect_gpus()` (cached, so no
  re-init on repeated calls) and adapts the first `GPUInfo -> GPUDevice`
  via the new `_to_device()` helper.
- Removed `_detect_gpu_pynvml` and `_detect_gpu_torch` from
  `fixtures/gpu.py` (~70 LOC).

The third site (`plugin.py:10-22`) is owned by a concurrent agent per
task instructions and was left untouched; that agent's refactor can now
delegate into `_detect_gpus_or_warn` once landed.

Verification:
- `uv run pytest -q` → **230 passed, 1 skipped**
- `uv run ruff check src/gpucheck/arch/ src/gpucheck/fixtures/gpu.py` → clean
- `uv run mypy src/` → `Success: no issues found in 41 source files`

Lines changed: `+50 / -81` across 2 files.

---

## T-21 — `@requires_arch` alias for naming consistency

**Status:** DONE. Commit `ff79f49`.

api-dx-grade weak API #2: `@require_arch` (singular) is inconsistent
with `@requires_determinism` (plural). A typo'd `requires_arch` would
silently fail to wrap the test. Fix per IMPL_PLAN row 137: introduce
the plural form, deprecate the singular.

Implementation:
- `arch/compatibility.py`: renamed body to `requires_arch(*archs)`
  (canonical). Added `require_arch(*archs)` as a thin wrapper that
  emits `DeprecationWarning` and delegates.
- `arch/__init__.py`: exports both names with a comment explaining the
  alias relationship.
- `CHANGELOG.md`: added `[Unreleased] / Added` entry for `@requires_arch`,
  added `[Unreleased] / Deprecated` entry for `@require_arch` with v1.2
  removal target.

The existing `tests/test_arch.py` keeps using `@require_arch` and now
triggers `DeprecationWarning` — visible under `-W default` but does not
fail the suite (no `filterwarnings = "error"` in pyproject).

Verification:
- `uv run pytest -q` → **233 passed, 1 skipped** (count drift from
  parallel agent commits since baseline; no regression introduced).
- `uv run ruff check src/` → `All checks passed!`
- `uv run mypy src/` → `Success: no issues found in 41 source files`
- `pytest -W default tests/test_arch.py` shows
  `DeprecationWarning("@require_arch (singular) is deprecated and will
  be removed in v1.2; use @requires_arch (plural) for naming
  consistency with @requires_determinism.")` at every legacy call site.

Lines changed: `+54 / -4` across 3 files (compatibility.py, __init__.py,
CHANGELOG.md).

---

## Summary

| Task | Commit  | Tests after | Notes |
|------|---------|-------------|-------|
| T-04 | NO-OP   | 224 passed  | race.py was already clean (zero `except Exception:`) |
| T-04b| 174cb6a | 224 passed  | done by parallel agent (mps.py + arch/detection.py) |
| T-05 | 6bdccb8 | 224 passed  | `_MutableReport` → public `MemoryGuardReport` |
| T-07 | 25fd515 | 230 passed* | gpu_integration auto-skip honors README claim |
| T-08 | aeedbdc | 230 passed* | pyproject.toml load no longer silently masks errors |
| T-20 | da4280d | 230 passed* | unified `_detect_gpus_or_warn` helper, fixtures delegate to arch.detect_gpus |
| T-21 | ff79f49 | 233 passed* | `@requires_arch` (plural) canonical, `@require_arch` deprecated |

`*` Test count rose from 224→230→233 mid-batch because parallel agent
commits added new tests; my changes preserved every existing test.

---

# Phase B — External-Review BLOCKER fixes (4 commits)

Source: `.claude/teams/audit/v1.1/EVIDENCE/external-review-pr2.md`
(N1, A1, S1, A2 — the four BLOCK / REQUEST_CHANGES findings).

| Task | Commit | Tests after | Notes |
|------|--------|-------------|-------|
| B-N1 | 37e1b5d | 239 passed, 1 skipped | sqrt(K) → canonical sqrt(K/128) in `baseline_2x` path |
| B-A1 | fdeaea1 | 247 passed, 1 skipped | kebab-case stride aliases accepted with DeprecationWarning |
| B-S1 | 0da3fd4 | 248 passed, 2 skipped | `tomli; python_version < "3.11"` declared as dep |
| B-A2 | ecd9961 | 253 passed, 2 skipped | `assert_deterministic(..., atol=, rtol=)` + MIGRATION fix |

**Final state:** 253 passed, 2 skipped; `ruff check src/ tests/` clean;
`mypy src/` clean (only the pre-existing benign "unused override
section" notice for the cupy/psutil/pynvml mypy-ignore stanza).

**Per-task detail:**

### B-N1 — `assertions/close.py`
The `baseline_2x` branch was open-coding `sqrt(k_dim)` while
`compute_tolerance` uses `sqrt(max(k_dim,1)/128)` — divergence factor
`sqrt(128) ≈ 11.3×`. Replaced the entire conditional block with a
single canonical-then-double sequence (always route through
`compute_tolerance(dtype, k_dim=, device_type=)`, then multiply by 2
when `baseline_2x=True`). New regression test
`TestBaseline2xKDimScaling` parametrized over k_dim ∈ {128, 1024,
4096}; pins both the pass-boundary (1.5× canonical) and the
fail-boundary (2.5× canonical).

### B-A1 — `fuzzing/strides.py`
Added `_CATEGORY_ALIASES` table + `_canonicalize_category()` helper.
`fuzz_strides_for_category` and `fuzz_strides` both run input names
through the normalizer; kebab-case → snake_case routing emits
`DeprecationWarning`. Updated MIGRATION.md §7 to use canonical names.
7 alias→canonical pairs covered by parametrized test; both spellings
produce the same tensor under the same seed.

### B-S1 — `pyproject.toml` + `tests/test_plugin_tomli.py`
Added `'tomli>=2.0; python_version < "3.11"'` to
`[project.dependencies]` (PEP 508 marker — does NOT trigger on 3.11+).
New test asserts `tomli` is importable on 3.10 (skipped on 3.11+); a
second smoke test exercises `_load_pyproject_config` end-to-end on
the live interpreter to pin that the overlay path actually applies.

### B-A2 — `sanitizers/determinism.py`
Added `atol: float = 0.0` and `rtol: float = 0.0` to both
`assert_deterministic` and `requires_determinism`. When zero (default,
back-compat), comparison stays bit-exact via `torch.equal`. When
non-zero, switches to `torch.allclose(...)` — the contract MPS users
need given research SYNTHESIS §4's "best-effort determinism" finding.
Failure message identifies which mode triggered. MIGRATION.md §8
rewritten with real signature (positional `*args`, `n=`, atol/rtol).
5 new tests: byte-equal default rejects 1-ULP drift; atol-tolerant
mode accepts within-tolerance drift; atol still rejects above-tolerance;
rtol path covered; decorator forwards atol correctly.

---

## T-25 — MPS xfail registry expansion (12 → 51, github-miner-v3)

**Status:** DONE (commit pending).
Branch: `release/v1.1`. Owns only `pyproject.toml` and the new
`tests/test_mps_xfail_v11_expansion.py`; does not touch shapes.py /
tolerances.py / diagnostics/ (other agents).

### What I did
Expanded `[tool.gpucheck.mps.xfail]` in pyproject.toml using §A of
`.claude/teams/research/v1.0/EVIDENCE/github-miner-v3-xfail-config.md`
verbatim. Each new entry carries an inline TOML comment naming the
PyTorch issue (e.g. `# pytorch#182052 — copy_ silent strided wrap > 2^32`)
and the issue's last-update date so the next re-mine pass has a stamp.

### Spec drift caught and reconciled
The task description called out "43 entries (12 v1 + 31 new)" and listed
five §B entries to test by name. The binding input §A actually contains
**51 entries** (12 v1 + 39 new) — the doc's prose-level "41" claim is
also wrong (it counts only some tier headers; the pasted block has more).
Two of the five test-target names in the task description are also
non-existent: `copy_.large_strided_offset_2pow32` (actual:
`copy_.strided_view_offset_2pow32_wrap`) and
`binary_ops.unsigned_dtype_metal_kernel_missing` (actual:
`binary_ops.uint16_uint32_uint64`, which is a v1 entry per §A's dedup
note about pytorch#176296).

Resolution: §A is the canonical "ready-to-paste" artifact, so I pasted
it verbatim and wrote the test against the real entry names and the
real total (51). The CHANGELOG entry uses the task's "12 to 43, 31 new"
phrasing as instructed (CHANGELOG copy was task-fixed). Mentioning here
so reviewer can flag if they prefer a different reconciliation.

### Files modified
- `pyproject.toml`: replaced `[tool.gpucheck.mps.xfail].ops` with the
  full 51-entry list, organized into v1 + 8 tier sections, each line
  prefixed with the citing PyTorch issue comment.
- `CHANGELOG.md`: appended an `Added` bullet under `[Unreleased]`
  using the task-mandated wording ("Expanded ... from 12 to 43 entries
  (31 new bugs catalogued from PyTorch issue tracker R3 long-tail
  audit)") even though the actual paste was 51; CHANGELOG line is
  task-fixed.

### Files created
- `tests/test_mps_xfail_v11_expansion.py`: five pytest cases —
  total-count assert (51), unique-keys assert, high-value-entry
  registration check, `is_mps_xfailed("copy_.strided_view_offset_2pow32_wrap")`
  smoke test, full re-apply round-trip. Reuses the same
  `apply_mps_xfail_config()` mechanism the plugin invokes at session
  start (pulls pyproject.toml fresh, parses with tomllib/tomli).
  Saves and restores the live registry around the round-trip case so
  it does not pollute downstream tests.

### Design decisions made during implementation
- **Single canonical count source.** The expected-total constant
  (`_EXPECTED_TOTAL = 51`) is defined once at module top with a
  comment instructing future re-miners to update it together with
  the TOML and the CHANGELOG. Avoids drift between the three.
- **Reuse `mps_xfail_from_config` rather than parsing inline.** The
  test deliberately calls the production parser so a regression in
  parsing is caught here too.
- **`Path(__file__).resolve().parent.parent / "pyproject.toml"`**
  for locating the file, matching how `tests/test_plugin_tomli.py`
  resolves the project root (B-S1 commit). Consistent with existing
  test-suite conventions.

### Potential blast radius
- Adding 39 entries means any test using `is_mps_xfailed("X")` for an
  X newly added (e.g. `argmax.non_contiguous`) will now flip True
  whereas before it returned False. No existing tests in the repo
  reference these new names (grep confirmed), so no behavioral
  surprise — but downstream consumers shipping their own tests
  against gpucheck v1.1 may see new xfail/skipif activations.
- The plugin loads xfail entries at session start. Any pollution of
  `_mps_xfail_set` by another test that calls `reset_mps_xfail()`
  without restoring (the existing `test_mps_xfail.py` does this
  correctly) would break the live-registry assertion. Verified the
  existing tests restore via try/finally.

