# verifier-D.md

**Branch**: `feat/track-d-bundle` @ `02507da`

```
$ pytest --tb=short -q
157 passed, 3 skipped, 9 warnings in 0.45s
```

```
$ pytest --cov=gpucheck.reporting tests/test_reporting_*.py --cov-report=term

Name                                 Stmts   Miss  Cover
src/gpucheck/reporting/__init__.py      11      5    55%
src/gpucheck/reporting/ci.py            90      0   100%
src/gpucheck/reporting/console.py      104      1    99%
src/gpucheck/reporting/html.py          92      1    99%
src/gpucheck/reporting/json.py          66      0   100%
TOTAL                                  363      7    98%
31 passed
```

```
$ ruff check src/ tests/
All checks passed!
```

```
$ mypy src/
Success: no issues found in 36 source files
```

| Acceptance | Status |
|---|---|
| Reporting coverage ≥ 90% | PASS (98%) |
| `HTMLReporter(json).render(out)` writes self-contained HTML | PASS |
| HTML is well-formed (HTMLParser) | PASS |
| `assert_deterministic` raises on diverging output | PASS |
| `assert_deterministic` returns first output when consistent | PASS |
| `@requires_determinism` decorator wraps + repeats | PASS |
| `permissions: contents: read` in ci.yml | PASS (manual review) |
| `uv.lock` committed | PASS |

**Verdict: PASS** on all 6 quality gates.
