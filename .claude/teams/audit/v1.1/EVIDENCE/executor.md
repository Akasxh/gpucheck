# Executor — close.py collision group (T-01, T-02, T-10)

Branch: release/v1.0
Owner: engineering-executor
Mutmut baseline at start: 169 killed mutants
Pre-existing test count at start: 225 collected (224 passed + 1 skipped)

## Task T-01 (Lazy torch import)

### What I did
Replaced the module-level `try: import torch as _torch / _has_torch` block in
`src/gpucheck/assertions/close.py` with a lazy `_torch_mod()` helper that
caches the imported module (or `None`) using a sentinel. Both call sites
inside `assert_close` (device-type detection and the GPU fast-path) now bind
`_torch = _torch_mod()` once at the top of the function and guard with
`_torch is not None`.

### Files modified
- `src/gpucheck/assertions/close.py`: lazy-import refactor; `_has_torch` removed; `_torch_mod()` is the single point of access.

### Files created
- `DIFF_LOG.md` (this iteration logs the T-01 change).
- `.claude/teams/audit/v1.1/EVIDENCE/executor.md` (this file).

### Design decisions made during implementation
- Used a sentinel (`_TORCH_UNRESOLVED = object()`) instead of `None` for the
  cache initial state, so a process where torch is genuinely absent still
  short-circuits after the first probe (caches `None` permanently).
- Kept the inner `import torch` inside the `__cuda_array_interface__`
  fallback as-is rather than routing it through `_torch_mod()`. That branch
  needed torch *and* dlpack to be present and was already lazy by virtue of
  living inside a function body — re-routing through the helper would have
  changed behaviour (it would no longer except `ImportError` locally).

### Potential blast radius
- If `gpucheck.assertions.close` is imported on a torch-less host, the
  module load path no longer raises or warns. Behaviour matches the
  previous code (which set `_has_torch = False` silently).
- The `_torch_cached` global is process-wide. If a test fixture
  monkey-patches `sys.modules['torch']` after `_torch_mod()` has run once,
  the cached value will be stale. Existing tests do not do this, but the
  verifier should confirm.

## Task T-02 (`.contiguous()` on slow path)

### What I did
Added `.contiguous()` to both torch.Tensor branches of `_to_numpy` (the
primary `hasattr(tensor, "detach")` branch and the dlpack fallback in the
`__cuda_array_interface__` block). Wrote a new parametrized test module
`tests/test_assert_close_contiguous.py` covering three stride patterns:
slice (`[:, ::2]`), transpose (`.t()`), and broadcast (`.expand`).

### Files modified
- `src/gpucheck/assertions/close.py`: two `.cpu().contiguous()` insertions; PM-4 citation comment.

### Files created
- `tests/test_assert_close_contiguous.py`: 6 parametrized cases (3 stride patterns × 2 entry points).

### Design decisions made during implementation
- Applied `.contiguous()` to the dlpack fallback as well even though only
  the primary branch is in the strict T-02 scope. The same RuntimeError
  surface exists in both code paths and the cost is negligible. Noted
  here as an opportunistic widen so the reviewer can flag if undesired.
- Used `pytest.importorskip("torch")` rather than the existing
  `_has_torch`-style guard so the module skips cleanly on torch-less
  hosts (matches the lazy-import discipline from T-01).

### Potential blast radius
- `.contiguous()` allocates a new tensor when the input is non-contiguous.
  For very large stride-fuzzed tensors this could double peak memory in
  the slow path. Existing CUDA fast-path (which bypasses `_to_numpy`)
  already handles same-shape tensors without copying, so the regression is
  bounded to mismatched / failing comparisons.

## Task T-10 (Pin numeric fields in mismatch report)

### What I did
Added `TestMismatchReportPinnedNumerics` to `tests/test_assertions.py` with
three new tests. Each test constructs simple integer-valued numpy inputs so
expected values can be hand-computed exactly, then asserts those values
appear verbatim (with `:.6e` formatter) in the rendered Rich report. Did
NOT modify `src/gpucheck/assertions/reporting.py` per task instructions.

### Files modified
- `tests/test_assertions.py`: appended 3 tests targeting the ~30 surviving
  reporting.py mutants from `EVIDENCE/mutator-survivors.md`.

### Files created
- (none)

### Design decisions made during implementation
- Used 2-D input in test 2 specifically so `np.unravel_index` is exercised
  meaningfully (with a 1-D input, any axis-mutation would be a no-op).
- For the histogram count assertion, stripped ANSI escape codes via
  `re.sub(r"\x1b\[[0-9;]*m", "", report)` before scanning digits because
  Rich's coloured output otherwise leaks digits like `33`/`31` from
  ``\x1b[33m`` and pollutes the digit-only filter.
- Selected unique-maximum diff values (4.5, 5.0) so the location index
  is unambiguous — eliminating spurious passes if `nanargmax` is mutated
  to e.g. `nanargmin` and the answer happens to coincide.

### Potential blast radius
- The histogram-count test depends on the bar-rendering loop emitting
  the count after the bar (`f"  {bucket:>22s} | {bar} {count}"`). If the
  format string is reordered (count before bar), the test would still
  isolate the digits via the line-tail extraction, but the failure
  message would be misleading. Acceptable trade-off for now.
- `5 / 6 (83.33%)` substring is whitespace-sensitive: if the table
  formatter switches columns or pads differently, the test could
  false-fail. Task scope is to test current behaviour; if reporting is
  refactored, these tests must be updated.

## Tasks T-11 / T-12 / T-13 (Mutation-killer leverage tests)

### What I did
Created `tests/test_mutation_killers.py` from scratch as the dedicated home
for the three audit-named leverage clusters. The module is segregated from
`test_assertions.py` and `test_plugin_tomli.py` so the kill-rate ratchet is a
single auditable artifact. No source files in `src/gpucheck/*` touched.

### Files modified
- `CHANGELOG.md`: appended "Mutation-killer test suite (T-11 / T-12 / T-13)"
  bullet under `[Unreleased]` → `Added`.
- `DIFF_LOG.md`: appended Iteration 7 with both file entries.

### Files created
- `tests/test_mutation_killers.py`: ~270 LOC; three pytest classes / two
  parametrized free functions / one cardinality test; total 17 test cases.
  - `TestFormatMismatchReportPinnedFields` (9 tests): canonical
    `zeros((4,4)) / eye(4)` fixture pins max-abs ("1.000000e+00"), mean-abs
    ("2.500000e-01"), mismatch count "4 / 16 (25.00%)", location "(0, 0)",
    tolerances line "atol=0.00e+00, rtol=0.00e+00", "Error Histogram" panel
    presence, bucket "[1e+0, 1e+1)" with count 4, AND a non-symmetric (2,4)
    fixture that pins (1, 3) vs (3, 1) for axis-order, plus the no-histogram
    case when arrays match.
  - `TestToleranceConfigRoundTrip` (5 tests): apply → compute → reset round
    trip; empty/missing-section no-op; explicit `None` returns for absent
    `tool.gpucheck.tolerances` key paths; malformed entry skipped (missing
    rtol, missing atol, non-dict); literal-key correctness vs upper-case
    `TOOL/GPUCHECK/TOLERANCES`.
  - `test_default_tolerance_table_value_is_pinned` and
    `test_compute_tolerance_returns_pinned_default`: parametrized over a
    hard-coded 7-pair ground-truth list mirroring `_DEFAULT_TOLERANCES`.
    Each case `assert _DEFAULT_TOLERANCES["float64"] == (1e-10, 1e-7)` etc.,
    breaking the tautology in `test_known_dtypes` that iterates the dict
    against itself.
  - `test_default_tolerances_table_size_is_pinned`: cardinality + key-set
    invariant against the ground-truth list.

### Design decisions made during implementation
- **ANSI stripping**. Rich emits `\x1b[<digits>m` styling because the report
  is exported with `styles=True`. The substring assertions need to compare
  plain text (e.g. `"4 / 16"` may be split by a `\x1b[0m` reset between
  tokens). I introduced a small `_strip_ansi` helper at module scope so
  every assertion in the report-pinning class operates on plain text.
  Existing T-10 tests in `test_assertions.py` did the same locally; I lift
  it to module scope here.
- **Two parametrized variants for the dtype table**. The audit names a
  single hard-coded parametrize, but I split into "pin the dict literal"
  (`_DEFAULT_TOLERANCES[name] == expected`) and "pin the public API"
  (`compute_tolerance(name) == expected`). Two independent assertions kill
  more mutants — for example, mutating `compute_tolerance` to consult a
  different dict (or to short-circuit through the float32 fallback) is
  caught by the second variant but not the first.
- **Pre-test `reset_config_tolerances()` in `compute_tolerance`-pinned
  cases**. Tests run in a single process; if a parallel test leaks a
  `_config_overrides` entry, the public-API parametrize would silently see
  the overlay value instead of the default. I call
  `reset_config_tolerances()` before each `compute_tolerance` lookup. The
  config-loader tests already wrap in try/finally for the same reason.
- **Cardinality test for `_DEFAULT_TOLERANCES`**. If a dict-key is dropped
  or duplicated, the ground-truth list and the per-key parametrize won't
  catch it on their own (a missing key just means one parametrize case
  errors out, but doesn't fail the *list*). The cardinality test pins both
  size and key-set, so adding/removing a dtype key without updating the
  ground-truth list trips the assertion immediately.
- **Test-1 fixture (`zeros((4,4)) / eye(4)`) over the audit's diagonal-only
  pattern.** The audit recommends mismatches at `[0,0], [1,1], [2,2],
  [3,3]` with max=1.0; that's exactly what `eye(4)` gives. The (0, 0)
  location is what `np.argmax` picks among the four tied diagonals.
  Because (0, 0) is symmetric (i == j), it could mask an axis-swap bug; I
  added a separate non-symmetric (2, 4) fixture that puts the unique
  maximum at `(1, 3)` and explicitly asserts `"(3, 1)" not in report` to
  guard the axis order.
- **Did NOT modify `test_assertions.py` `test_known_dtypes`**. The task
  rules forbid editing other test files unless absolutely necessary; the
  new hard-coded parametrize lives entirely in
  `tests/test_mutation_killers.py` and runs in addition to the existing
  tautological loop. Both assertions cover the same code, but only the new
  one kills the dict-value mutants — that's enough for the kill-rate goal.

### Potential blast radius
- **Whitespace-sensitivity**. The mismatch-count substring
  `"4 / 16 (25.00%)"` is whitespace-sensitive; a switch to a different
  Rich table layout or a `:.2f` → `:.3f` change would false-fail. Same for
  `"atol=0.00e+00, rtol=0.00e+00"` (Python's `:.2e` formatter is locked,
  but a manual respelling would break it). Acceptable trade-off: T-11's
  whole point is to be sensitive to such drift.
- **Histogram bucket-count parser**. I parse the integer count after the
  bucket label by isolating the substring after `"[1e+0, 1e+1)"` and
  splitting on the first newline. If Rich wraps the bar at a column
  boundary so the count appears on a different line than the label, this
  parser will see digits from the bar character (none — it's `█`,
  not a digit). I verified the export width (100) is wide enough that for
  this fixture (4 mismatches, single bucket) wrapping cannot occur; this
  may need re-tuning if other histogram tests are added later.
- **`reset_config_tolerances()` in `test_compute_tolerance_returns_pinned_default`**.
  This drops any overlay set by a parallel test. Pytest-xdist isolates
  workers per-process so this is safe; pytest's default sequential mode
  has no parallel test issue. If a future test writes to
  `_config_overrides` without try/finally and a parametrize case runs
  between, the reset prevents leakage. No source-side change.
- **No source files touched**: the kill-rate ratchet is achieved purely by
  test additions, so there's no source blast radius for the verifier to
  re-check.

## Task T-24 (Per-(kernel_class, dtype) MPS tolerance overlay from 5K calibration)

### What I did
Refactored `_MPS_TOLERANCE_MULTIPLIERS` from a flat dtype-only dict into a
layered overlay: the v1.0 flat table is preserved verbatim as the
DEFAULT-class fallback, and a new
`_MPS_KERNEL_DTYPE_MULTIPLIERS: dict[tuple[KernelClass, str], float]`
carries the v1.1 5K-iter calibrated multipliers (MATMUL fp32/16/bf16 =
16/20/32, CONV2D fp32/16/bf16 = 4/8/12, NORM/REDUCTION/POINTWISE = 2.0×
via the `*` dtype-wildcard rows). Threaded a new keyword-only
`kernel_class: KernelClass | None = None` parameter through
`compute_tolerance`. Added `_resolve_mps_multiplier(kernel_class,
dtype_name)` with a documented five-step resolution order so v1.0 callers
(kernel_class omitted) skip steps 1-3 and get byte-identical results.
Wrote the pinning module `tests/test_per_kernel_tolerance_overlay.py`
covering measured cells, wildcard rows, backward-compat parity, and
fallback ordering. Updated CHANGELOG with both Added and Changed entries.

### Files modified
- `src/gpucheck/assertions/tolerances.py`: introduced `KernelClass`
  `str`-Enum, `_DTYPE_WILDCARD = "*"`, `_MPS_KERNEL_DTYPE_MULTIPLIERS`
  dict (9 rows), `_resolve_mps_multiplier()` helper, and a new
  `kernel_class` kwarg on `compute_tolerance`. v1.0 callers untouched.
- `src/gpucheck/assertions/__init__.py`: re-exported `KernelClass` so the
  public API matches the spec example signature.
- `CHANGELOG.md`: appended T-24 entry to `[Unreleased] / Added` and a
  matching `[Unreleased] / Changed` entry.

### Files created
- `tests/test_per_kernel_tolerance_overlay.py`: 26 collected test cases —
  6 measured-cell parametrized + 9 wildcard-row parametrized + 7
  backward-compat + 4 fallback-ordering + 1 `tolerance_context`
  short-circuit + 1 dict-cardinality pin + supporting predicate tests.

### Design decisions made during implementation
- **`str`-Enum, not plain `Enum`**. Subclassing `str` keeps
  `KernelClass.MATMUL == "matmul"` true so callers can pass either the
  enum or the raw string interchangeably (mypy strict:
  `dict[tuple[KernelClass, str], float]` lookups work because
  `str.__hash__` is delegated). Preserves DX symmetry with how
  `gpucheck` already accepts dtype names as strings.
- **Wildcard sentinel `"*"` instead of a special enum value**. The task
  spec wrote `(KernelClass.NORM, "*")` literally, so I encoded `"*"` as
  a private module-level constant `_DTYPE_WILDCARD`. Lookups consult
  exact-dtype first, then wildcard — keeps the dict lean and lets future
  precise-dtype overrides win without re-shuffling rows.
- **`kernel_class` is keyword-only**. The spec function signature uses
  `*,` — preserves room to add more parameters later without breaking
  positional callers.
- **Multiplier values from spec, not from the calibration's "Recommended
  overlay" section**. The calibration-final.md "Shape B" recommendation
  block prescribes 20/25/40 for MATMUL and 5/10/14 for CONV2D, but the
  engineering lead's task-spec table specifies 16/20/32 and 4/8/12. The
  lead's numbers are the binding contract; spec wins over recommendation.
  Rationale recorded in the dict comment: each multiplier covers the
  measured P99 with safety margin (bf16 P99.9 of 31.91× sits exactly at
  the 32× ceiling — at-edge but covered).
- **Resolution step 3 (`(KernelClass.DEFAULT, dtype)`) is
  reserved-empty**. The spec contemplates a future per-dtype override
  under DEFAULT; I wired the lookup into `_resolve_mps_multiplier` but
  seeded zero rows for now. The fallback-ordering test pins this so any
  future addition is deliberate.

### Potential blast radius
- **`assertions/__init__.py` re-export of `KernelClass`** widens the
  public surface area beyond `tolerances.py`. The task spec example
  imports `KernelClass` (without specifying the path), so this is
  unavoidable; if the reviewer prefers `from gpucheck.assertions
  .tolerances import KernelClass` instead, drop the re-export. The
  package-root `gpucheck/__init__.py` was deliberately NOT touched (out
  of scope per task ownership boundaries).
- **`tolerance_context` short-circuit**: the current `compute_tolerance`
  checks `_tolerance_overrides` first and returns immediately, so an
  active `tolerance_context(...)` block bypasses the per-kernel-class
  overlay entirely. Pinned by
  `test_tolerance_context_overrides_short_circuit_overlay`. ISS-05
  (planner's "tolerance_context bypasses MPS overlay") is therefore
  intentional and pinned, not introduced fresh.
- **Forward-compat with `[tool.gpucheck.mps.tolerances.<class>]` config
  block**. The calibration analysis recommends a TOML schema like
  `[tool.gpucheck.mps.tolerances.matmul] fp32 = 2e-3`; T-24 only ships
  the in-code table. Wiring a config-loader is out of scope per
  file-ownership rules (pyproject.toml is forbidden). The
  `_resolve_mps_multiplier` resolution order is the natural extension
  point if a future task adds the loader.
- **No source change to `assertions/close.py`**. The fast-path call site
  `compute_tolerance(dtype, k_dim=k_dim, device_type=device_type)` still
  passes `kernel_class=None` implicitly, so v1.0 behaviour is preserved
  end-to-end. A follow-up task can add a `kernel_class=` argument to
  `assert_close` for users who want per-kernel routing in the assertion
  call itself.
