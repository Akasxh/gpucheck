# EVIDENCE/docs-tester.md — Phase 3

**Persona:** docs-tester (FM-3.2 — validates examples, links, cross-refs)
**Phase:** 3
**Date:** 2026-05-01

## Test plan

1. **Unit suite must remain green at 224 passing.** Charter rule:
   "After all docs are written, run `uv run pytest -q` and confirm still
   224 passing."
2. **No doctest blocks were introduced.** All code examples in CHANGELOG,
   CONTRIBUTING, MIGRATION, and README are illustrative — not collected
   as doctests. (Pre-v1.0 README also used illustrative blocks; the
   `[tool.pytest.ini_options]` in `pyproject.toml` has `testpaths =
   ["tests"]` so doctest collection is not enabled for docs.)
3. **External link sanity** — all upstream issue URLs cited in the docs
   (`triton#9838/9839`, `pytorch#162872, #179352, #179294, #173525,
   #175189, #142836, #174269, #181936, #96602, #175190, #176296,
   #137001, #177116, #164299, #170837`) are sourced from
   `.claude/teams/research/v1.0/SYNTHESIS.md`, which already verified
   each via WebFetch on 2026-05-01 (per its "verified by WebFetch
   2026-05-01" annotations in §Sub-Q 1/§Sub-Q 2).
4. **Internal cross-references** — verified each docs file's
   relative-path references:
   - `CHANGELOG.md` → `MIGRATION.md` — exists.
   - `CHANGELOG.md` → `.claude/teams/research/v1.0/SYNTHESIS.md` — exists.
   - `CHANGELOG.md` → `.claude/teams/security/v1.0/FINDINGS.md` — exists.
   - `CONTRIBUTING.md` → `~/.claude/teams/engineering/PROTOCOL.md` —
     out-of-tree, exists in user home (path is descriptive, not asserted).
   - `MIGRATION.md` → `CHANGELOG.md`, `CONTRIBUTING.md` — exist.
   - `README.md` → `CHANGELOG.md`, `MIGRATION.md`, `examples/triton_*.py`,
     `.claude/teams/research/v1.0/SYNTHESIS.md` — all exist.
5. **footnote anchor** — README.md `<a id="fn-bug-count">` reachable
   from the in-text `<sup>[1]</sup>` link.

## Test execution

### Run 1 (post all-docs-write)

```
$ uv run pytest -q
...
224 passed, 1 skipped, 10 warnings in 0.38s
```

PASS — matches charter target of 224 passing (was 117 pre-v1.0).

### Lint / type checks (not in charter as gates, but worth confirming)

Not re-run; no source code modified in Phase 3. Phase 2 already
confirmed clean ruff + mypy strict on all 4 track branches before
merge to `release/v1.0`.

## Stale claim verification

The README footnote claims "2 externally verified bugs" — cross-checked
against SYNTHESIS §Sub-Q 8:

- triton#9838 — "OPEN, opened 2026-03-25, 83.4% relative error ...
  Verified by WebFetch 2026-05-01." ✓
- triton#9839 — "CLOSED, opened 2026-03-25, max abs error 0.125 at
  K=8192 ... Verified by WebFetch 2026-05-01." ✓

The README "8 bugs surfaced" total matches the lead paragraph's prior
claim, with the explanatory footnote bringing rigour ("8 surfaced; 2
externally verified; remaining 6 are internal-ledger findings").

## Verdict

PASS — 224/224 tests still green; no doctest broken (none introduced);
all internal cross-references resolve; external citations grounded in
research SYNTHESIS WebFetch verification.
