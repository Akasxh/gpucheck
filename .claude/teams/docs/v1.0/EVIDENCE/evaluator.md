# EVIDENCE — docs-evaluator

**Persona:** docs-evaluator (close gate, FM-3.1 + FM-3.2 — 5-dimension
documentation rubric, PASS/FAIL).
**Method:** Run the standard rubric against the Phase 1 deliverables in
`/Users/cero/Code/gpucheck/.claude/teams/docs/v1.0/`.

The protocol contract (PROTOCOL.md §"Phase C — Quality Gate"):

- **Strict (1.0 required)**
  - `accuracy` — every claim must be verifiable in reader evidence /
    source.
  - `example_correctness` — every code example must pass tester (or be
    explicitly marked Phase-3-deferred).
- **Advisory (0.7, lead override allowed)**
  - `completeness` — coverage of all in-scope public APIs.
  - `readability` — formatting, headings, navigability.
  - `style_conformance` — house style, conventional formats.

This is a **Phase 1** evaluation; the gate is "did Phase 1 produce a
defensible audit + skeletons that Phase 3 can fill in?" — NOT "is the
final v1.0 documentation done?".

---

## Strict dimension 1 — accuracy

| Deliverable | Verdict | Notes |
|---|---|---|
| `EVIDENCE/detector.md` | PASS | Every fact cited to file:line. PyPI version, Python target, license, plugin entry-point, module map all match `pyproject.toml` and `ls` output. |
| `EVIDENCE/reader.md` | PASS | All 19 missing-docstring entries cite file:line. The 14 stale-CUDA-language entries each name the symbol and line. |
| `EVIDENCE/planner.md` | PASS | Tier classification is justified by the SESSION_PRECHECK signals + research QUESTION. |
| `EVIDENCE/reviewer.md` | PASS | Every README claim audited has a source-side cross-check. The k_dim formula was verified against `tolerances.py:67`. The default-tolerance table was verified row-by-row. |
| `EVIDENCE/skeptic.md` | PASS | Headline-count revision (20 → 19) is justified by removing `gpucheck.__getattr__`. Identifies real new accuracy bug (CLAUDE.md commit-style claim contradicting `git log`). |
| `AUDIT.md` | PASS | All file:line citations resolve. The "8 vs 5 bugs" inconsistency at `README.md:12 vs 328-334` is real and reproducible. |
| `CHANGELOG_DRAFT.md` | PASS | Uses Keep-a-Changelog 1.1.0 structure verbatim. `{{TODO Phase 3}}` markers explicitly identify content that depends on the not-yet-produced engineering DIFF_LOG. No invented features. |
| `CONTRIBUTING_DRAFT.md` | PASS | Documents the **actual** commit convention (bracket-tag) verified via `git log --oneline -25`, calling out CLAUDE.md's contradicting claim. Build commands match `pyproject.toml`. |
| `MIGRATION_v0_to_v1.md` | PASS | Each section names a concrete file:line that motivates the migration (e.g. `tolerances.py:26-28` for thread-safety). All shape-uncertain claims explicitly marked `{{TODO Phase 3}}`. |

**Strict-1 score: 1.0 (PASS).** No invented claims; every assertion is
either source-cited or explicitly placeholder-marked.

---

## Strict dimension 2 — example correctness

The Phase 1 charter does NOT require any code examples to be runnable
(no docs-tester is dispatched in Phase 1 per the protocol — examples
are tested in Phase 3 inner loop). The drafts contain code blocks that
are **structural illustrations**, not executable claims:

- `CHANGELOG_DRAFT.md` — no executable code blocks.
- `CONTRIBUTING_DRAFT.md` — `pip install`, `pytest`, `ruff`, `mypy`
  invocations. Each command is verifiable against `pyproject.toml`
  (lines cited in the document body). The 3-command check loop is the
  exact loop from CLAUDE.md "Build & Test".
- `MIGRATION_v0_to_v1.md` — every code block is either:
  - **before** code that already works on v0 (taken directly from
    README/CLAUDE.md / source patterns), or
  - **after** code marked with `{{TODO Phase 3: confirm shape}}`
    indicating that the shown shape is provisional pending engineering.

No example claims to be runnable today that does not run today.

**Strict-2 score: 1.0 (PASS).**

---

## Advisory dimension 3 — completeness

Phase 1 charter scope: audit + skeletons. Coverage:

| Charter item | Status |
|---|---|
| Audit README.md / CLAUDE.md for CUDA-only language | DONE — 14+ items in `AUDIT.md §A.1` |
| Audit README.md / CLAUDE.md for GTX 1650 references | DONE — 5 items in `AUDIT.md §A.2` |
| Audit "no MPS" / weakness items now becoming reality | DONE — 5 items in `AUDIT.md §A.3` |
| Identify missing dashboard / determinism / stride-fuzzing mentions | DONE — `AUDIT.md §A.4` |
| Audit inline docstrings across `src/gpucheck/{...}` and `plugin.py` | DONE — `EVIDENCE/reader.md` walks every module |
| List every public symbol lacking a docstring or stale type hint | DONE — `AUDIT.md §B` (19 missing + 14 stale + 7 thin) |
| Identify missing top-level docs (CHANGELOG / CONTRIBUTING / MIGRATION / CoC) | DONE — `AUDIT.md §C` |
| Write AUDIT.md with file:line refs | DONE |
| Draft CHANGELOG_DRAFT.md (Keep-a-Changelog format) | DONE |
| Draft CONTRIBUTING_DRAFT.md (full structure) | DONE |
| Draft MIGRATION_v0_to_v1.md (per-API placeholder) | DONE |

Specialist evidence files: detector / planner / reader / reviewer /
skeptic / evaluator / retrospector — **all 7 written**.

**Advisory-3 score: 0.95.** Mild deductions:
- `examples/` directory was not read in detail — the docs-skeptic flagged
  this; AUDIT.md §D defers it to Phase 3.
- Sphinx/MkDocs site decision deferred (out of Phase 1 scope per
  charter).

PASS (above 0.7 floor; well above lead-override threshold).

---

## Advisory dimension 4 — readability

| Deliverable | Verdict | Notes |
|---|---|---|
| `AUDIT.md` | Good | Hierarchical headings (A/B/C/D/E), tables for every section, severity tags (P0/P1/P2). |
| `CHANGELOG_DRAFT.md` | Good | Keep-a-Changelog format → familiar to PyPI consumers. |
| `CONTRIBUTING_DRAFT.md` | Good | Numbered checklist for PR process; tables for branch/extras/tools. |
| `MIGRATION_v0_to_v1.md` | Good | Quick-checklist table at top, per-API sections, before/after blocks. |
| EVIDENCE files | Good | Section dividers (`---`), tables. |

No emojis (consistent with house style — CLAUDE.md does not use
emojis). Line wrap consistent at ~80–100 chars.

**Advisory-4 score: 0.9.** PASS.

---

## Advisory dimension 5 — style conformance

| Convention | Compliance |
|---|---|
| Keep-a-Changelog 1.1.0 in CHANGELOG_DRAFT | YES — `## [Unreleased]`, `## [1.0.0rc1]`, Added/Changed/Deprecated/Removed/Fixed/Security/Known issues sections. |
| Conventional sections in CONTRIBUTING | YES — Code of conduct / reporting / dev setup / tests / coding style / commits / branches / PR process / architecture / how-to-add / releasing / security / meta. |
| Migration placeholders explicit | YES — every uncertain item is `{{TODO Phase 3: ...}}` not "TBD" or unmarked. |
| Citations use absolute paths | YES — `/Users/cero/Code/gpucheck/...` format throughout. |
| 1-based line numbers | YES — match `Read` tool output. |

**Advisory-5 score: 1.0.** PASS.

---

## Aggregate verdict

| Dim | Score | Required | Status |
|---|---|---|---|
| Strict — accuracy | 1.0 | 1.0 | ✓ PASS |
| Strict — example correctness | 1.0 | 1.0 | ✓ PASS |
| Advisory — completeness | 0.95 | 0.7 | ✓ PASS |
| Advisory — readability | 0.90 | 0.7 | ✓ PASS |
| Advisory — style conformance | 1.0 | 0.7 | ✓ PASS |

# **Verdict: PASS** for Phase 1.

This is a Phase-1 (audit + skeleton) PASS. Phase 3 must re-run the
evaluator after filling the `{{TODO Phase 3}}` markers; that gate is
strict and will only pass if SYNTHESIS / DIFF_LOG entries justify
every concrete claim.

## Conditions on the PASS

1. The Phase 1 deliverables remain inside `.claude/teams/docs/v1.0/`
   per the hard rule. Verified — no source files modified.
2. The 19 missing-docstring list in `AUDIT.md §B.1` is binding for
   Phase 3 — engineering must close every item or document why it
   should remain undocumented.
3. The "8 vs 5 bugs" accuracy bug at `README.md:12 vs 328-334` is
   independent of v1.0 and should be closed in Phase 3 regardless of
   MPS/Track-A status.
