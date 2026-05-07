# executor-D.md

**Branch**: `feat/track-d-bundle` @ `02507da`

## D.1 Reporting test coverage 0 -> 98%
- New tests: `test_reporting_console.py` (10), `test_reporting_json.py` (6), `test_reporting_ci.py` (8) — exercise console, JSON, GitHub Actions, JUnit XML, PR-comment paths.
- Coverage measurement (with `pytest --cov=gpucheck.reporting`):
  - `reporting/ci.py` 100%
  - `reporting/console.py` 99%
  - `reporting/html.py` 99%
  - `reporting/json.py` 100%
  - `reporting/__init__.py` 55% (lazy-import scaffold; covered by side effects in normal use)
  - **Total: 98%** (target was ≥ 90%).

## D.2 reporting/html.py
- Self-contained dashboard: zero external CSS/JS, inline SVG bar chart, escapes HTML in messages, handles empty payloads.
- 8 tests covering well-formedness, summary card counts, comparison band, empty data, parent-dir creation, XSS escaping.

## D.3 sanitizers/determinism.py
- `assert_deterministic(fn, *args, n=3, seed=0, **kwargs)` and `@requires_determinism(n, seed)`.
- Seeds random + numpy + torch (CPU + CUDA + MPS where available).
- `_equal` compares torch.Tensor (same shape/dtype/device + torch.equal), tuple, list, scalar.
- `DeterminismError` surfaces the SYNTHESIS §4 best-effort caveat with an actionable failure message.
- 8 tests including torch-tensor outputs (skipped if torch unavailable).

## D.4 uv.lock committed (DEP-1)
- 1220-line lock file copied from the existing dev environment.

## D.5 .github/workflows/ci.yml hardened (CFG-2)
- Top-level `permissions: contents: read`.
- Switched install path to `uv sync --frozen --extra dev` (with fallback to `uv pip install -e ".[dev]"` if no lock present in branch).

## Verification: 157 passed, 3 skipped (+40 net new). ruff & mypy clean. Reporting coverage 98%.
