# mutator-survivors summary (W1)

**Sample size:** 85 of 221 surviving mutants classified; rest extrapolated by clustering on same source lines.

**Per-category counts (across 221):**
- REAL_GAP ~165 (75%) — uncovered config loader, fp8/tf32 entries, k_dim=1 boundary, baseline_2x×explicit-atol matrix, report numeric fields
- TEST_BUG ~30 (14%) — `pytest.raises(match="NaN")` is substring check, matches "XXNaNXX"; `assert "X" in report` accepts mutated row labels
- EQUIVALENT ~15 (7%) — copy=False↔True, ContextVar name string, str|None annotation, panel widths
- TIME_BOMB ~8 (4%) — > vs >= mutations need exact-threshold inputs
- UNREACHABLE ~3 (1%) — _torch=None import-fail branch, sizes.append(4) itemsize fallback

**File with most survivors:** `src/gpucheck/assertions/reporting.py` — **83 of 221 (38%)**. Has only 3 substring-only tests for format_mismatch_report; zero tests for _error_histogram.

**Top-3 highest-leverage new tests (kill 10-30+ each):**
1. `test_report_contains_correct_max_error_value` — pin numeric fields (max err, mismatch count, location, histogram presence) → kills ~30 reporting.py mutants
2. `test_apply_config_tolerances_round_trip` + `test_tolerances_from_config_parses_overrides` — covers entirely untested config loader → kills ~17 tolerances.py mutants
3. `@pytest.mark.parametrize` over _DEFAULT_TOLERANCES.items() with hard-coded expected pairs (not the dict itself) → kills ~12 dict-value mutants currently shielded by tautological iteration

**Action**: ~30 lines of new test code can drive kill rate from 42.7% → ≥80%.
