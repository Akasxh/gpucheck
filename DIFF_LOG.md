# DIFF_LOG.md — Phase A executor (close.py collision group)

Owner: engineering-executor
Branch: release/v1.0
Collision group: T-01 → T-02 → T-10 (serialized per skeptic C2)

## Iteration 1 — Task T-01: Top-level torch import defeats lazy-import contract
- **File**: `src/gpucheck/assertions/close.py`
- **Change**: Replaced module-level `try: import torch as _torch / _has_torch` block with a lazy `_torch_mod()` cached helper; rewrote `device_type` detection and the GPU fast-path to bind `_torch = _torch_mod()` at the top of `assert_close`, then guard with `_torch is not None`.
- **Reason**: CLAUDE.md Lazy-imports decision-record requires torch to never be imported at collection time. The previous `try: import torch as _torch` ran at module load, breaking the contract. Caching via a sentinel makes the lookup happen at most once per process.
- **Acceptance criterion addressed**: IMPLEMENTATION_PLAN_v1.1 T-01 — `python -c "import gpucheck.assertions.close; import sys; assert 'torch' not in sys.modules"` now passes; existing `tests/test_assertions.py` still green (32 passed).

## Iteration 2 — Task T-02: Add `.contiguous()` to `_to_numpy` slow path
- **File**: `src/gpucheck/assertions/close.py`
- **Change**: Inserted `.contiguous()` between `.cpu()` and `.numpy()` on both torch.Tensor branches of `_to_numpy` (the primary `hasattr(tensor, "detach")` branch and the dlpack fallback inside the `__cuda_array_interface__` block). Added a comment citing PM-4.
- **Reason**: torch <2.1 raises `RuntimeError: input array is not C-contiguous` when `.numpy()` is called on stride-fuzzed / sliced / transposed tensors. Preventive even on newer torch — known to fire on older PyTorch.
- **Acceptance criterion addressed**: security-postmerge PM-4 / planner T-02 — new `tests/test_assert_close_contiguous.py` exercises 3 stride patterns (slice, transpose, broadcast) × 2 entry points (`_to_numpy` directly and `assert_close` end-to-end); all 6 cases pass.

- **File**: `tests/test_assert_close_contiguous.py` (created)
- **Change**: New parametrized test module with 6 cases pinning the contiguity fix.
- **Reason**: T-02 acceptance demands a regression test that would fail without the `.contiguous()` insertion.
- **Acceptance criterion addressed**: planner T-02.

## Iteration 3 — Task T-10: Pin numeric fields in mismatch report
- **File**: `tests/test_assertions.py`
- **Change**: Added `TestMismatchReportPinnedNumerics` class with 3 tests pinning (a) max abs error / mean abs error / row labels (b) mismatch count "5 / 6 (83.33%)" + 2-D max-error location "(1, 2)" (c) histogram bucket label "[1e-3, 1e-2)" with count 3 after stripping ANSI escapes.
- **Reason**: mutator-survivors top-leverage #1 — `assertions/reporting.py` had ~30 surviving mutants because no test asserted exact numeric values from the report. Hard-coded values + 2-D index + ANSI-aware bar count force any arithmetic substitution / unravel-axis swap / bucket-formatter mutation to fail.
- **Acceptance criterion addressed**: planner T-10 / mutator-survivors top-leverage #1 — adds 3 tests (commit-message claim) without modifying `reporting.py` source.

## Iteration 4 — Task T-24: Per-(kernel_class, dtype) MPS tolerance overlay
- **File**: `src/gpucheck/assertions/tolerances.py`
- **Change**: Added `KernelClass` `str`-Enum (MATMUL/CONV2D/NORM/REDUCTION/POINTWISE/DEFAULT) and `_DTYPE_WILDCARD` sentinel ("*") used for all-dtype rows in the new overlay table.
- **Reason**: Shape B from `EVIDENCE/calibration-final.md` (per-class bucketed) demands a typed kernel taxonomy. `str`-Enum keeps `KernelClass.MATMUL == "matmul"` true so callers can pass raw strings or the enum interchangeably; the wildcard sentinel lets the NORM/REDUCTION/POINTWISE rows be dtype-agnostic.
- **Acceptance criterion addressed**: T-24 (per-(kernel,dtype) overlay) and `IMPLEMENTATION_PLAN_v1.1` §"R3 deliverable: T-24".

## Iteration 5 — Task T-24: Add per-(kernel_class, dtype) overlay table + resolver
- **File**: `src/gpucheck/assertions/tolerances.py`
- **Change**: Added `_MPS_KERNEL_DTYPE_MULTIPLIERS` dict keyed by `(KernelClass, dtype_name)` with measured 5K-iter calibration values (MATMUL fp32/16/bf16 = 16/20/32, CONV2D fp32/16/bf16 = 4/8/12, NORM/REDUCTION/POINTWISE = 2.0 on dtype wildcard "*"). Introduced internal `_resolve_mps_multiplier(kernel_class, dtype_name)` with five-step resolution order (exact → wildcard → DEFAULT-class → flat dict → hard 2.0). Threaded a new keyword-only `kernel_class: KernelClass | None = None` parameter through `compute_tolerance` so v1.0 callers stay byte-identical (kernel_class=None skips steps 1-3 of resolution).
- **Reason**: Calibration data (`drift_histogram_5k.json`) shows MATMUL drift up to 27.6× P99 / 31.9× P99.9 on bf16 and CONV2D up to 10.4× P99 / 12.3× P99.9 on bf16 — flat 2× was provably under-tight. The Shape-B class-bucketed overlay matches measured tails with ≥10% headroom; backward-compat preserved by gating the new lookup behind the optional `kernel_class` kwarg.
- **Acceptance criterion addressed**: T-24 spec — `compute_tolerance(dtype, kernel_class=KernelClass.MATMUL, device_type="mps")` returns the calibrated multiplier; omitting `kernel_class` returns the v1.0 result exactly; (NORM, "float8_e4m3fn") falls through wildcard → flat → returns 2.0.

## Iteration 6 — Task T-24: Pinning tests for the per-(kernel_class, dtype) overlay
- **File**: `tests/test_per_kernel_tolerance_overlay.py` (created)
- **Change**: New test module with 6 measured-cell parametrized cases (MATMUL fp32/16/bf16 = 16/20/32; CONV2D fp32/16/bf16 = 4/8/12), 9 wildcard cases (NORM/REDUCTION/POINTWISE × fp32/fp16/bf16 = 2.0×), 7 backward-compat cases, 4 fallback-ordering cases, plus enum/dict-size/tolerance_context invariants.
- **Reason**: Spec demands (a) per-cell pinning of the calibration overlay, (b) byte-identical v1.0 behaviour when `kernel_class` is omitted, and (c) verified fallback ordering through the wildcard / DEFAULT-class / flat-dict / hard-2.0 chain. The dict-size pin (`9` entries) prevents silent table drift.
- **Acceptance criterion addressed**: Task spec — "Each measured cell: assert ... matches expected multiplier; backward-compat: omitting kernel_class produces v1.0 result exactly; (NORM, 'float8_e4m3fn') falls through to (DEFAULT, 'float8_e4m3fn')".

- **File**: `src/gpucheck/assertions/__init__.py`
- **Change**: Re-exported `KernelClass` from `gpucheck.assertions` so callers can `from gpucheck.assertions import KernelClass, compute_tolerance` per the task-spec example signature.
- **Reason**: Public API surface for the new opt-in `kernel_class` kwarg — without exporting the enum from the assertions sub-package, the kwarg is unusable.
- **Acceptance criterion addressed**: T-24 — public API exposes `KernelClass`.

## Iteration 7 — Tasks T-11 / T-12 / T-13: Mutation-killer leverage tests
- **File**: `tests/test_mutation_killers.py` (created)
- **Change**: New test module hosting the three audit-named leverage clusters. (1) `TestFormatMismatchReportPinnedFields` — 9 tests pinning max-abs / mean-abs / mismatch-count / location / tolerances-line / histogram-presence / bucket-label / bucket-count / no-histogram-when-equal, with the audit-recommended `zeros((4,4))` / `eye(4)` fixture plus a non-symmetric (2,4) fixture that pins (i,j) axis order. (2) `TestToleranceConfigRoundTrip` — 5 tests covering apply→compute→reset round trip, empty-section no-op, `None` for absent sections, malformed-entry skip, and the literal `tool.gpucheck.tolerances` key path. (3) Parametrized `test_default_tolerance_table_value_is_pinned` and `test_compute_tolerance_returns_pinned_default` over a hard-coded 7-pair ground-truth list (kills the tautological dict-iterates-itself loop in `test_known_dtypes`); plus a cardinality test on `_DEFAULT_TOLERANCES`.
- **Reason**: Audit `EVIDENCE/mutator-survivors.md` identifies these three clusters as the highest-leverage path from 42.7 % → ~69 % kill rate on `assertions/`. Substring-loose existing tests (e.g. `"NaN" in report` happily matching `"XXNaNXX"`) and an entirely-untested config loader were the dominant gap.
- **Acceptance criterion addressed**: planner T-11 / T-12 / T-13 — mutator-survivors top-leverage #1 / #2 / #3.

- **File**: `CHANGELOG.md`
- **Change**: Appended "Mutation-killer test additions" bullet under [Unreleased] → Added.
- **Reason**: User-visible signal that v1.1 raises mutmut kill-rate floor; CHANGELOG is the only release-note artifact.
- **Acceptance criterion addressed**: planner T-11 / T-12 / T-13 (visibility).

## Iteration 7 — Task T-24: CHANGELOG entry for v1.1 overlay
- **File**: `CHANGELOG.md`
- **Change**: Appended T-24 bullet to `[Unreleased] / Added` (per-(kernel-class, dtype) MPS tolerance overlay with all six measured-cell multipliers, KernelClass enum export, and explicit backward-compat note) and a matching `[Unreleased] / Changed` block (resolution order, retiring the PROVISIONAL note that v1.0 left on `_MPS_TOLERANCE_MULTIPLIERS`).
- **Reason**: Keep-a-Changelog 1.1.0 demands a user-visible record of the new opt-in `kernel_class` kwarg, the calibrated multiplier table, and the backward-compat guarantee for the v1.0 flat-overlay path.
- **Acceptance criterion addressed**: T-24 — CHANGELOG documents the v1.1 deliverable.
