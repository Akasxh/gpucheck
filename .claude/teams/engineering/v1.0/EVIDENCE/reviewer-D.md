# reviewer-D.md

**Branch**: `feat/track-d-bundle` @ `02507da`

## Stage 1: PASS — All four sub-deliverables (D.1-D.5) shipped.

## Stage 2

### Strengths
- HTML dashboard is genuinely zero-dep (no Jinja, no D3, inline SVG). Vendorizes cleanly into any static-hosting setup.
- `assert_deterministic`'s failure message is actionable: it explicitly cites SYNTHESIS §4 + the xfail-block escape hatch.
- 98% reporting coverage exceeds the 90% target by a healthy margin.
- `permissions: contents: read` is the cleanest possible CFG-2 fix — five lines, no lateral risk.

### Concerns
- **CONCERN**: `uv sync --frozen` requires the `astral-sh/setup-uv@v3` action which is mutable-tag-pinned in the new ci.yml. Security finding CFG-1 flagged "actions pinned to mutable tags" as LOW severity. We're consistent with the existing `actions/checkout@v4`, `actions/setup-python@v5` usage; tightening to commit SHAs is a v1.1 improvement.
- **NIT**: Coverage gap on `reporting/__init__.py` is the lazy-import scaffold; covered transitively in normal use. Not load-bearing for the 98% headline.
- **NIT**: HTML escape uses Python's stdlib `html.escape(quote=True)` — sufficient for v1.0 but doesn't guard against attribute-context injection in tests that don't go through `_esc`. Current code paths all do, so no actionable issue.
- **NIT**: `assert_deterministic` for non-torch + non-numpy outputs uses Python `==`, which silently passes for objects with broken `__eq__`. Acceptable since the pattern is "users compare the kind of values their kernels produce".

## Verdict: APPROVED.
