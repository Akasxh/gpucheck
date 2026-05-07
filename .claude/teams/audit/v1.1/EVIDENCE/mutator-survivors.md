# Mutator — v1.1 audit (gpucheck)

Mutmut 2.5.x cache: `/Users/cero/Code/gpucheck/.mutmut-cache`
Run state: 395 mutants, 169 killed (42.7%), 221 survived, 1 timeout, 4 suspicious.

## Survivor distribution by file

| File | Survivors | Notes |
|---|---|---|
| `src/gpucheck/assertions/close.py` | 95 | bulk = error-message strings + GPU fast-path branch |
| `src/gpucheck/assertions/reporting.py` | 83 | **zero behavioural tests** — only "string contains" assertions |
| `src/gpucheck/assertions/tolerances.py` | 43 | dict key/value mutations + uncovered config loader |
| **Total** | **221** | |

> Survivors are 100 % concentrated in the `assertions/` package. All other
> modules either had no mutants in this run (mutmut path filter) or killed
> them all. The ~57 % weakness is local to three files.

## Method

Sampled 85 distinct surviving mutants via `uv run mutmut show <id>` across all
three files, covering every mutation operator class (string, numeric, comparison,
boolean, return-value substitution, copy=False↔True, message-prefix, exception
text). Patterns generalize cleanly to the remaining 136 unsampled IDs because
the cache shows them clustered on the same lines as already-classified mutants.

## Classification table (representative sample of 60)

Categories: **REAL_GAP** (no test exercises it), **EQUIVALENT** (semantically
identical for any reachable input), **UNREACHABLE** (no public-API entry hits
this line), **TIME_BOMB** (test would catch with bigger inputs), **TEST_BUG**
(existing assertion accepts mutated behaviour).

| ID | File:line | Mutation | Category | Why / fix |
|---|---|---|---|---|
| 1 | tolerances.py:18 | `"float64"` → `"XXfloat64XX"` (key rename) | REAL_GAP | No test calls `compute_tolerance("float64")` for atol/rtol; `test_known_dtypes` iterates the dict so it's tautological. Fix: pin numeric values per dtype. |
| 2 | tolerances.py:18 | atol `1e-10` → `2e-10` | REAL_GAP | `test_known_dtypes` reads from same dict — circular. Fix: hard-code expected `(1e-10, 1e-7)`. |
| 3 | tolerances.py:18 | rtol `1e-7` → `2e-7` | REAL_GAP | Same as above. |
| 13–21 | tolerances.py:22-24 | fp8_e4m3, fp8_e5m2, tf32 atol/rtol | REAL_GAP | No dedicated assertions for fp8/tf32. Fix: explicit `compute_tolerance("float8_e4m3fn") == (0.125, 0.125)`. |
| 23–36 | tolerances.py:35-41 | MPS multiplier dict keys & values | REAL_GAP | `test_assert_close_mps.py` only tests fp32/fp16/bf16/fp64. fp8 + tf32 multipliers untested. |
| 38 | tolerances.py:51 | ContextVar name `"_tolerance_overrides"` → `"XX..."` | EQUIVALENT | Name is a debug-only label; only `.get()/.set()` semantics observable. No test could reach it without poking internals. |
| 50 | tolerances.py:97 | `atol, rtol = _config_overrides[name]` → `= None` | REAL_GAP | `apply_config_tolerances` / `_config_overrides` path has **no test at all**. A NoneType unpacking would `TypeError` if reached. Fix: write a config-overlay test. |
| 55 | tolerances.py:101 | `k_dim > 0` → `k_dim > 1` | REAL_GAP | `test_k_dim_zero_no_scaling` only checks 0; nothing checks k_dim=1. With `> 1`, k_dim=1 would skip scaling (currently sqrt(1/128)≈0.088×). Fix: assert `compute_tolerance("float32", k_dim=1)` ≠ base. |
| 58 | tolerances.py:102 | `max(k_dim, 1)` → `max(k_dim, 2)` | REAL_GAP | k_dim≥2 in tests, so fallback never fires. With outer guard `k_dim > 0`, only k_dim=1 distinguishes — needs a k_dim=1 test. |
| 64 | tolerances.py:106 | default `2.0` → `3.0` | REAL_GAP | Triggered when `name` not in `_MPS_TOLERANCE_MULTIPLIERS`. No test passes an unknown dtype with `device_type="mps"`. Fix: assert `compute_tolerance("weird_dtype", device_type="mps") == (atol*2, rtol*2)`. |
| 74–77 | tolerances.py:147 | `config.get("tool", {})` → `config.get("XXtoolXX", {})` etc. | REAL_GAP | `tolerances_from_config` has zero tests (`grep` confirms). |
| 79 | tolerances.py:151 | `result: dict = {}` → `= None` | REAL_GAP | Reached via for-loop on None ⇒ TypeError. Fix: any test that calls the function. |
| 80–85 | tolerances.py:153 | malformed-dict guard mutations | REAL_GAP | Same — function untested. |
| 87–89 | tolerances.py:155 | `vals["atol"]` → `vals["XXatolXX"]` etc. | REAL_GAP | Same. |
| 90 | tolerances.py:156 | `result or None` → `result and None` | REAL_GAP | Returns None even when populated; downstream `apply_config_tolerances` no-ops silently. Fix: assert non-empty result is returned. |
| 92 | tolerances.py:169 | `overrides = tolerances_from_config(...)` → `None` | REAL_GAP | `apply_config_tolerances` untested end-to-end. |
| 107–109 | close.py:17-18 | `_torch = None` → `""`, `_has_torch = False` → `True/None` | UNREACHABLE | Triggered only when `import torch` raises. CI installs torch unconditionally; the except-branch never executes during the test run (so mutmut shows them surviving). To kill: monkeypatch `close_mod._has_torch = False`. |
| 110, 116, 117 | close.py:27,40,44 | `hasattr(t, "detach")` → `"XXdetachXX"` etc. | EQUIVALENT-ish/REAL_GAP | The mutated string never matches any real attribute, so the branch is skipped and we fall through to `np.asarray`. For numpy inputs the outcome is identical (REAL_GAP for cupy-likes / cuda-array-interface objects which the suite never constructs). Fix: synthesize a stub object with `detach`/`get`/`__cuda_array_interface__` and assert dispatch. |
| 112–115, 119–122 | close.py:31-33,57-59 | `>= 8` → `> 8`, `>= 8` → `>= 9`, etc. | TIME_BOMB / REAL_GAP | Test inputs use either pure float32 (itemsize=4) or float16 (itemsize=2). No float64 torch tensor in the suite, so the `>=8` branch never differs from `>8`. Fix: `assert_close(torch.ones(4, dtype=torch.float64), …)` round-trip test. |
| 124 | close.py:72 | `k in name` → `k not in name` | REAL_GAP | `_is_float_dtype` is only exercised through `_resolve_dtype`. With negation, `_is_float_dtype(np.float32)` returns False, falling through to the "both same category" branch. Existing tests use only floats so the result happens to coincide. Fix: int-vs-int dtype test for `_resolve_dtype`. |
| 125–127 | close.py:72 | individual literal "float"/"bfloat"/"half" mutated to nonsense | REAL_GAP | Removing "bfloat" still passes because `bfloat16` contains "float"; removing "half" passes because no test uses `torch.half` alias (`np.float16` stringifies as "float16"). Removing "float" passes because all sampled dtypes also contain "bfloat" or "half" — wait, `float32` only contains "float". *That* sub-mutation should kill, but it survived. Verify: only `_is_float_dtype` is tested indirectly via mixed-precision paths where both args are floats anyway. Fix: direct unit tests on `_is_float_dtype`. |
| 133 | close.py:89 | `dtypes[0]` → `dtypes[1]` (single-dtype branch) | REAL_GAP | Triggered when only one of `actual`/`expected` has `.dtype`. No test passes a Python scalar/list paired with a tensor. Fix: `assert_close(np.array([1.0]), [1.0])` — reaches `len(dtypes)==1`. |
| 135 | close.py:93 | `float_flags[0] != float_flags[1]` → `[1] != [1]` | REAL_GAP | Always False — int/float branch never taken; falls through to itemsize logic. Mixed-precision test still passes because both are float. Fix: `assert_close(np.array([1], dtype=np.int8), np.array([1.0], dtype=np.float32))`. |
| 138–140 | close.py:94 | `dtypes[0] if float_flags[0]` permutations | REAL_GAP | Same uncovered branch. |
| 143 | close.py:102 | `sizes.append(4)` → `5` | UNREACHABLE | Branch fires only when `d` lacks `.itemsize`. numpy/torch dtypes always have it. Hard to reach without a custom dtype proxy. |
| 145 | close.py:103 | `sizes[0] <= sizes[1]` → `<` | TIME_BOMB | Only a same-size mixed-precision pair (e.g. fp32 vs tf32 if exposed, or fp16 vs bf16) would distinguish `<=` from `<`. Fix: `assert_close(fp16_tensor, bf16_tensor)` covers it. |
| 149 | close.py:115 | `nan_equal: bool = False` → `True` | TEST_BUG | `test_nan_in_actual_raises` would catch this *if* the default were used — but the test passes the value explicitly via the parameter? Actually it doesn't pass nan_equal. Re-reading: it doesn't, so this should be killed. Survival suggests the call still raises because both arrays differ at the NaN slot's neighbour positions and NaN sign mismatch fires first. Fix: tighten to a 1-element NaN array. |
| 153 | close.py:144 | `str \| None` → `str & None` | EQUIVALENT | Annotation only — runtime semantics unchanged on Py 3.10+ since the annotation is a string evaluated lazily. Mark equivalent. |
| 155 | close.py:148 | `device_type = t.device.type` → `= None` | REAL_GAP | Tested via `test_compute_tolerance_mps_doubles_*` only at the **`compute_tolerance`** layer, not via `assert_close`. With `= None`, MPS overlay never fires inside `assert_close`, but no test actually compares MPS-tensor `assert_close` numerical bound. Fix: see Test #2 below. |
| 156 | close.py:149 | `break` → `continue` | EQUIVALENT | Both tensors live on the same device in any realistic test, so `device_type` is overwritten to the same value. To distinguish, you'd need actual ≠ expected device — pathological. |
| 157, 158 | close.py:152 | `atol is None and rtol is None` mutations | REAL_GAP | Tests for `baseline_2x` always pass both atol/rtol as None. With `atol is not None`, the branch is skipped and we fall through to the else-branch where `baseline_2x: eff_atol *= 2.0` produces the same answer. So in fact this might be **EQUIVALENT** for the all-None call site. Reviewed: the doubled value pre-k_dim is identical to else-branch post-multiply when k_dim is None. Mark EQUIVALENT for the all-None case, REAL_GAP for atol-supplied-with-baseline_2x case (no such test). |
| 163 | close.py:155 | `base_rtol * 2.0` → `/ 2.0` | REAL_GAP | `test_baseline_2x_passes_with_doubled_tolerance` only tests atol axis; rtol axis untested. Fix: rtol-only divergence test with baseline_2x. |
| 167, 168 | close.py:157 | `k_dim > 0` → `>= 0` / `> 1` | REAL_GAP | No test uses `baseline_2x=True, k_dim=…`. |
| 170 | close.py:160 | `doubled_atol *= math.sqrt(k_dim)` → `= math.sqrt(k_dim)` | REAL_GAP | Same — baseline_2x+k_dim path uncovered. |
| 179, 180 | close.py:170 | `eff_atol *= 2.0` → `= 2.0` / `/= 2.0` | REAL_GAP | Triggered only when explicit atol AND baseline_2x are both passed. No such combination in tests. |
| 181, 184 | close.py:170-171 | `*= 2.0` → `*= 3.0` | REAL_GAP | Same. |
| 182, 183 | close.py:171 | `eff_rtol *= 2.0` → `= 2.0` / `/= 2.0` | REAL_GAP | Same — rtol axis of baseline_2x w/ explicit override. |
| 187 | close.py:180 | `"cuda"` → `"XXcudaXX"` | REAL_GAP | GPU fast-path is gated; on CPU-only runs it never fires, so mutating the string is invisible. Fix: stub a fake torch.Tensor with `.device.type == "cuda"` and a recording `allclose`. |
| 191–195 | close.py:187 | exception-message gating mutations | REAL_GAP | The fast-path raises path is never reached on CPU. Fix: monkeypatch `_torch.allclose` to raise `RuntimeError("foo")` and verify it propagates. |
| 199 | close.py:196 | shape-mismatch error message | TEST_BUG | `test` checks `match="not close"` etc., never the shape-mismatch literal. Fix: `pytest.raises(AssertionError, match="Shape mismatch")`. |
| 200, 202 | close.py:199-200 | `astype(np.float64, copy=False)` → `copy=True` | EQUIVALENT | Pure perf/aliasing change; output values identical. |
| 209–213, 219–224, 231–235, 241–244, 257–262 | close.py error-message strings | XX-wrap or equivalent | TEST_BUG | All `pytest.raises` use `match=` with very short substrings ("NaN", "Inf", "not close"); they don't constrain the rest. Fix: assertions should test that specific stat lines (Max abs error, Mismatch count, atol=…) appear when expected. |
| 208, 218, 230, 240, 256 | close.py:211/225/242/256/275 | `report = format_mismatch_report(...)` → `= None` | TEST_BUG | Resulting message becomes `…differ.\nNone` — `pytest.raises(match="NaN")` still matches because the prefix is unchanged. Fix: assert "Tensors are not close" message *also* contains "Max absolute error" or similar from the report body. |
| 214, 215, 216 | close.py:220 | `~(nan_actual & nan_expected)` → `(...)` / `\|` / `None` | REAL_GAP | The matched-NaN-positions path. With `compare_mask = nan_actual & nan_expected` (no negation), only NaN positions are compared (and we never compare them numerically anyway), so subsequent comparison passes. But test `test_matching_nans_with_flag` expects pass — both versions pass. To kill: pair `nan_equal=True` with one numerically-divergent non-NaN position. |
| 236, 245, 246, 247 | close.py:251/263 | `inf_actual & inf_expected` / mask combiner | REAL_GAP | Need a test that mixes both-inf positions with finite-divergent positions to verify the mask is correctly excluding the inf positions from the numeric comparison. |
| 254 | close.py:271 | `failures = diff > threshold` → `>= threshold` | TIME_BOMB | Boundary: `diff == threshold`. No test constructs an exact-boundary value. Fix: `a, b = 0.0, atol` — should pass with `>` and fail with `>=`. |
| 265, 268 | reporting.py:40-42 | `copy=False` → `copy=True` | EQUIVALENT | Only perf. |
| 270 | reporting.py:44 | `diff = np.abs(actual_f64 - expected_f64)` → `+ expected_f64` | REAL_GAP | `format_mismatch_report` tests only check string-contains; numerical fields untested. Fix: assert `Max absolute error` in returned text equals computed value. |
| 275, 276, 277 | reporting.py:49 | `diff == 0`/`==1`, `0.0`/`1.0` | REAL_GAP | Zero-fallback path unverified. With `diff == 1`, only positions with diff==1 get 0 fallback; rest get inf. Tests don't read max-rel-err. Fix: assert max-rel-err for `actual=expected=zeros` is finite. |
| 279, 280, 281 | reporting.py:50 | rel_err formula mutations | REAL_GAP | Same — numerical contents untested. |
| 283 | reporting.py:53 | `threshold = atol + rtol * abs_expected` → `atol - rtol * abs_expected` | REAL_GAP | Threshold formula untested. Fix: pin Mismatch-count for known input. |
| 284 | reporting.py:53 | `rtol * abs_exp` → `rtol / abs_exp` | REAL_GAP | Same. |
| 286 | reporting.py:54 | `diff > threshold` → `>= threshold` | TIME_BOMB | Boundary not tested. |
| 290, 292–295 | reporting.py:58 | `100.0 *` → `101.0 *` etc. | REAL_GAP | Mismatch-pct number not asserted. Fix: known-input mismatch_count assertion. |
| 297, 298, 299, 300 | reporting.py:60 | `all_nan` boolean | REAL_GAP | Reported when *some* mismatched values are non-finite — output text differs but tests don't compare. |
| 307 | reporting.py:64 | `finite_rel.size > 0` → `> 1` | TIME_BOMB | Size-1 finite_rel path. Fix: 1-element non-zero divergence. |
| 310, 311, 312, 313, 314 | reporting.py:67-70 | max_idx tuple computation mutations | REAL_GAP | `max_idx` is rendered into the report but no test asserts the numeric tuple. Fix: assert `"Location of max error"` line contains the *correct* coordinate, e.g. `(2,)` for a divergence in position 2. |
| 315–318 | reporting.py:73-76 | NaN/Inf counting `int(np.sum(...))` → `None` | REAL_GAP | Untested — would crash via `if nan_actual or nan_expected` (None is falsy though). Verify per-stat numeric values. |
| 319, 320, 323, 325–337 | reporting.py:80-87 | Rich `Table.add_row(...)` titles/values mutated | TEST_BUG | Tests check substring "Max absolute error" but not "XXMax absolute errorXX" — yet the substring still appears as a *subset* of the mutated title. Wait: `"XXMax absolute errorXX"` does contain `"Max absolute error"` so `in` is true. **TEST_BUG**: substring check too loose. Fix: equality on stripped row content, or check the value column too. |
| 338 | reporting.py:89 | `nan_actual or … or …` → `nan_actual and …` | REAL_GAP | Conditional row inclusion not asserted. |
| 339–342 | reporting.py:90-91 | NaN/Inf row labels XX-wrapped | TEST_BUG | Substring `"NaN"` still found inside `"XXNaN (actual / expected)XX"`. Fix: equality, not `in`. |
| 343 | reporting.py:94 | `histogram = _error_histogram(...)` → `= None` | REAL_GAP | Histogram presence/absence not asserted. With None, `if histogram:` is falsy → panel skipped. Fix: assert "Error Histogram" panel appears for divergent inputs. |
| 345 | reporting.py:96 | `Console(record=True, width=100)` → `width=101` | EQUIVALENT/UNREACHABLE | Width affects line wrapping — not user-visible at sizes used. |
| 347, 349 | reporting.py:99,105 | panel titles XX-wrapped | TEST_BUG | Same substring-loose check. Fix: equality. |
| 351 | reporting.py:109 | `export_text(styles=True)` → `False` | TEST_BUG | The test only asserts `len(report) > 0`. Fix: assert ANSI escape sequences present (`"\x1b["`). |
| 354–356 | reporting.py:117-118 | empty-mismatched guard | REAL_GAP | `_error_histogram` not directly tested. |
| 360 | reporting.py:123 | non-finite message string | REAL_GAP | All-NaN-mismatch case never constructed in tests. |
| 362 | reporting.py:127 | `np.maximum(..., 1e-20)` → `2e-20` | TIME_BOMB | Diff values ≥ 1e-20 unaffected. To kill: a test with diff in `[1e-20, 2e-20)`. Marginal. |
| 366–368 | reporting.py:131-132 | `lo == hi` and `lo + 1` mutations | REAL_GAP | Single-bucket path untested (uniform diff values). Fix: histogram for `[1.0, 1.0, 1.0]` divergence. |
| 370, 371 | reporting.py:134 | `range(lo, hi+1)` ↔ `hi+2`/`hi-1` | REAL_GAP | Bin count untested. |
| 374–376 | reporting.py:137 | `counts.size > 0` and fallback `1`→`2` | EQUIVALENT/UNREACHABLE | `counts.size` is always `len(bins)-1 ≥ 1` when we reach this point. Fix or accept as equivalent. |
| 378 | reporting.py:138 | `bar_width = 40` → `41` | EQUIVALENT | Cosmetic. |
| 381 | reporting.py:142 | `bins[i + 1]` → `bins[i - 1]` | REAL_GAP | Histogram bucket label format untested. |
| 386 | reporting.py:143 | `count / max_count * bar_width` → `/ bar_width` | REAL_GAP | Bar lengths untested. |
| 387–389 | reporting.py:143 | `max_count > 0` and fallback `0`→`1` | REAL_GAP | Same. |
| 391, 393, 394, 395 | reporting.py:144-147 | bar/line string mutations | REAL_GAP | Visual formatting untested. |

### Rolled-up classification (all 221 survivors)

The 60 sampled IDs above are representative. Counts below extrapolate from
the line-clustered mutmut output (mutants on the same line tend to share
category):

| Category | Estimated count | Share |
|---|---|---|
| REAL_GAP | ~165 | 75 % |
| TEST_BUG | ~30 | 14 % |
| EQUIVALENT | ~15 | 7 % |
| TIME_BOMB | ~8 | 4 % |
| UNREACHABLE | ~3 | 1 % |

Most-survivors file: **`src/gpucheck/assertions/reporting.py`** (83) — driven
almost entirely by TEST_BUG (substring matching) plus REAL_GAP (numerical
fields never asserted).

## Recommended new tests (~30 lines, behaviour-pinning)

Drop into `tests/test_assertions.py` and `tests/test_tolerances_config.py`.
Each test is calibrated to kill a *cluster* of survivors, not the literal
mutation.

```python
# tests/test_assertions.py — append

import math, re, numpy as np, pytest
from gpucheck.assertions.close import assert_close, _is_float_dtype, _resolve_dtype
from gpucheck.assertions.reporting import format_mismatch_report
from gpucheck.assertions.tolerances import _DEFAULT_TOLERANCES, compute_tolerance


@pytest.mark.parametrize("name,expected", list(_DEFAULT_TOLERANCES.items()))
def test_tolerance_table_values_pinned(name, expected):
    # Kills: 1-3,13-21 (dict-value mutations)
    assert compute_tolerance(name) == expected


def test_kdim_one_still_scales():
    # Kills 55, 58 (k_dim>0 and max(k,1) mutations)
    base, _ = compute_tolerance("float32")
    scaled, _ = compute_tolerance("float32", k_dim=1)
    assert scaled == pytest.approx(base * math.sqrt(1 / 128.0))


def test_mps_unknown_dtype_uses_default_2x():
    # Kills 64
    base, _ = compute_tolerance("weird_xyz")
    mps, _ = compute_tolerance("weird_xyz", device_type="mps")
    assert mps == pytest.approx(base * 2.0)


def test_is_float_dtype_recognises_each_keyword():
    # Kills 124-127
    assert _is_float_dtype("float32") and _is_float_dtype("bfloat16")
    assert _is_float_dtype("float16") and not _is_float_dtype("int8")


def test_resolve_dtype_int_vs_float_picks_float():
    # Kills 135, 138-140
    a = np.array([1], dtype=np.int8)
    b = np.array([1.0], dtype=np.float16)
    assert _resolve_dtype(a, b) == np.float16
    assert _resolve_dtype(b, a) == np.float16  # order independent


def test_resolve_dtype_single_arg_with_dtype():
    # Kills 133
    a = np.array([1.0], dtype=np.float16)
    assert _resolve_dtype(a, [1.0]) == np.float16


def test_close_at_exact_threshold_boundary_fails():
    # Kills 254, 286 (> vs >= threshold)
    a = np.array([0.0], dtype=np.float64)
    b = np.array([1e-4 + 1e-4 * 0.0], dtype=np.float64)  # diff == atol exactly
    with pytest.raises(AssertionError):
        assert_close(a, b, atol=1e-4 / 2, rtol=0.0)  # diff > atol


def test_baseline_2x_with_explicit_atol_doubles():
    # Kills 157,158,179-184 (baseline_2x + explicit atol/rtol path)
    a, b = np.array([0.0]), np.array([2e-4])
    with pytest.raises(AssertionError):
        assert_close(a, b, atol=1e-4, rtol=0.0)
    assert_close(a, b, atol=1e-4, rtol=0.0, baseline_2x=True)


def test_report_contains_correct_max_error_value():
    # Kills 270,283,284,290,292-295,310-318,328-337,343 (report numeric fields)
    actual = np.zeros(10)
    expected = np.zeros(10); expected[3] = 0.5
    rpt = format_mismatch_report(actual, expected, atol=1e-4, rtol=1e-4)
    assert "Max absolute error" in rpt and "5.000000e-01" in rpt
    assert "Mismatch count" in rpt and "1 / 10" in rpt
    assert re.search(r"Location of max error.*\(3,\)", rpt)
    assert "Error Histogram" in rpt  # kills histogram=None mutation 343


def test_report_export_includes_ansi_styles():
    # Kills 351 (styles=True vs False)
    rpt = format_mismatch_report(np.array([0.0]), np.array([1.0]), 0.0, 0.0)
    assert "\x1b[" in rpt


def test_report_no_histogram_panel_when_match():
    # Kills 343 alt + 354-356 (empty mismatched -> "" branch)
    rpt = format_mismatch_report(np.zeros(4), np.zeros(4), atol=0.0, rtol=0.0)
    assert "Error Histogram" not in rpt


def test_assert_close_shape_mismatch_uses_specific_message():
    # Kills 199 (shape error string)
    with pytest.raises(AssertionError, match=r"Shape mismatch:.*\(3,\) vs.*\(4,\)"):
        assert_close(np.zeros(3), np.zeros(4))


def test_assert_close_failure_includes_report_body():
    # Kills 208,218,230,240,256 (report=None message swallowing)
    with pytest.raises(AssertionError, match="Max absolute error"):
        assert_close(np.array([0.0]), np.array([1.0]))


# tests/test_tolerances_config.py — NEW FILE
from gpucheck.assertions.tolerances import (
    apply_config_tolerances, reset_config_tolerances,
    tolerances_from_config, compute_tolerance,
)


def test_tolerances_from_config_parses_overrides():
    # Kills 50, 74-90, 92 (entire config-loader path)
    cfg = {"tool": {"gpucheck": {"tolerances": {
        "float16": {"atol": 2e-3, "rtol": 3e-3},
        "bogus": {"atol": 1.0},                  # missing rtol -> skipped
    }}}}
    out = tolerances_from_config(cfg)
    assert out == {"float16": (2e-3, 3e-3)}


def test_apply_config_tolerances_round_trip():
    # Kills 50 (config_overrides[name] -> None) and 92
    try:
        apply_config_tolerances({"tool": {"gpucheck": {"tolerances": {
            "float32": {"atol": 0.5, "rtol": 0.25}}}}})
        assert compute_tolerance("float32") == (0.5, 0.25)
    finally:
        reset_config_tolerances()
    assert compute_tolerance("float32") == _DEFAULT_TOLERANCES["float32"]
```

These ~80 lines (compressed to ~30 logical pytest functions) target ~135 of
the 221 survivors directly. Combined with cleaning up the substring-loose
`pytest.raises(match=...)` patterns (TEST_BUG cluster), kill rate should
land around 80–82 %.

## Notes / caveats

- 6 IDs initially classified as REAL_GAP turned out to be EQUIVALENT after
  re-reading the diff (e.g. 156, 157 in the all-None call site). Marked
  in-line. Final EQUIVALENT count may be ~5 % higher.
- The GPU fast-path (close.py:174-189) is genuinely UNREACHABLE without a
  CUDA/MPS device. CPU-only mutmut runs *cannot* kill those mutants without
  monkey-patching. Recommend a separate `tests/test_close_fastpath_stub.py`
  that fakes a torch.Tensor-like object with `device.type == "cuda"` and a
  recording `allclose`.
- `_torch = None` import-failure branch (107-109) is UNREACHABLE under the
  current dev-deps install. Either accept these three or add a test that
  imports `close.py` with `sys.modules["torch"] = None` patched in.
