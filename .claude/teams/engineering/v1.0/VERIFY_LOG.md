# VERIFY_LOG — gpucheck v1.0 Phase 2

Schema: `| ts | track | command | exit | summary |`

| ts | track | command | exit | summary |
|---|---|---|---|---|
| 2026-05-01 | baseline | `pytest --tb=no -q` (release/v1.0 a9a9d44) | 0 | 117 passed, 3 skipped |
| 2026-05-01 | A | `pytest --tb=short -q` (track-a-mps 24035aa) | 0 | **147 passed, 4 skipped** (+30 net new) |
| 2026-05-01 | A | `ruff check src/ tests/` | 0 | All checks passed! |
| 2026-05-01 | A | `mypy src/` | 0 | Success: no issues found in 38 source files |
| 2026-05-01 | B | `pytest --tb=short -q` (track-b-strides 4ede763) | 0 | **139 passed, 4 skipped** (+22 net new) |
| 2026-05-01 | B | `ruff check src/ tests/` | 0 | All checks passed! |
| 2026-05-01 | B | `mypy src/` | 0 | Success: no issues found in 35 source files |
| 2026-05-01 | C | `pytest --tb=short -q` (track-c-thread-safety 5ddd26e) | 0 | **126 passed, 3 skipped** (+9 net new) |
| 2026-05-01 | C | `ruff check src/ tests/` | 0 | All checks passed! |
| 2026-05-01 | C | `mypy src/` | 0 | Success: no issues found in 34 source files |
| 2026-05-01 | D | `pytest --tb=short -q` (track-d-bundle 02507da) | 0 | **157 passed, 3 skipped** (+40 net new) |
| 2026-05-01 | D | `pytest --cov=gpucheck.reporting tests/test_reporting_*.py` | 0 | **98% reporting coverage** (target ≥ 90%) |
| 2026-05-01 | D | `ruff check src/ tests/` | 0 | All checks passed! |
| 2026-05-01 | D | `mypy src/` | 0 | Success: no issues found in 36 source files |

## Aggregate

- **Baseline**: 117 passed, 3 skipped (release/v1.0 @ a9a9d44)
- **Final per-track passing test counts**: A 147, B 139, C 126, D 157
- **Net new tests across all 4 branches**: ~30 + 22 + 9 + 40 = **101 net new** (some test files cross over via test reuse — true sum after dedup is ~95)
- **All 4 tracks PASS strict mypy and ruff with no `# type: ignore` rationales beyond cross-version `tomllib` import compatibility on Python 3.10**.
- Reporting module coverage: **0% → 98%** on Track D (target was ≥ 90%).
