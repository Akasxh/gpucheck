# EVIDENCE/docs-writer.md — Phase 3

**Persona:** docs-writer (FM-1.2, FM-2.3 — writes from reader evidence only,
never invents)
**Phase:** 3 (production landing)
**Date:** 2026-05-01

## Inputs consumed

- `/Users/cero/Code/gpucheck/.claude/teams/docs/v1.0/CHANGELOG_DRAFT.md`
- `/Users/cero/Code/gpucheck/.claude/teams/docs/v1.0/CONTRIBUTING_DRAFT.md`
- `/Users/cero/Code/gpucheck/.claude/teams/docs/v1.0/MIGRATION_v0_to_v1.md`
- `/Users/cero/Code/gpucheck/.claude/teams/docs/v1.0/AUDIT.md`
- `/Users/cero/Code/gpucheck/.claude/teams/engineering/v1.0/DIFF_LOG.md`
- `/Users/cero/Code/gpucheck/.claude/teams/engineering/v1.0/CHARTER.md`
- `/Users/cero/Code/gpucheck/.claude/teams/research/v1.0/SYNTHESIS.md`
- `/Users/cero/Code/gpucheck/.claude/teams/security/v1.0/FINDINGS.md`
- `/Users/cero/Code/gpucheck/README.md` (pre-edit)
- `/Users/cero/Code/gpucheck/CLAUDE.md` (pre-edit)
- `/Users/cero/Code/gpucheck/pyproject.toml`
- `/Users/cero/Code/gpucheck/src/gpucheck/__init__.py`

## Files written

| Path | Operation | Diff stat | Source of every claim |
|---|---|---|---|
| `/Users/cero/Code/gpucheck/CHANGELOG.md` | new | +256 | DIFF_LOG (4 tracks), SYNTHESIS §Sub-Q 2/§Sub-Q 7/§Sub-Q 8, FINDINGS (CFG-2/TM-E1/DEP-1/N1-N5) |
| `/Users/cero/Code/gpucheck/CONTRIBUTING.md` | new | +341 | CLAUDE.md, pyproject.toml, CHARTER §"Hard rules" (conv-commits transition), git log (224 tests, MPS tests auto-run) |
| `/Users/cero/Code/gpucheck/MIGRATION.md` | new | +391 | DIFF_LOG, SYNTHESIS, GPUInfo dataclass (`arch/detection.py`), `[tool.gpucheck.mps.xfail]` block in pyproject.toml |
| `/Users/cero/Code/gpucheck/README.md` | edit | +15 / -4 | AUDIT §A.1 (CUDA-only language), AUDIT §A.5 item 32 (8-bug reconciliation), SYNTHESIS §Sub-Q 8 |
| `/Users/cero/Code/gpucheck/CLAUDE.md` | edit | +40 / -22 | DIFF_LOG (v1.0 highlights), AUDIT §A.3 items 21/22/23/25/26 (gaps moved out), CONTRIBUTING.md (commit transition) |

## Reader-before-writer compliance

- The 12-entry MPS xfail list in CHANGELOG / MIGRATION is copied verbatim
  from `pyproject.toml` `[tool.gpucheck.mps.xfail]` (real source) and
  cross-referenced with SYNTHESIS §Sub-Q 2 (research-grounded selection).
- The 2× MPS tolerance multiplier is described as **PROVISIONAL** in
  every doc-section that references it (CHANGELOG, MIGRATION, CLAUDE.md)
  per SYNTHESIS §Sub-Q 7 hard rule and skeptic §2.
- The "8 bugs / 2 externally verified" reconciliation in README + CHANGELOG
  matches AUDIT §A.5 item 32 + SYNTHESIS §Sub-Q 8 ("rigorous count of
  externally-verified is 2; SYNTHESIS recommends sharpening").
- No invented API surfaces. Every `gpucheck.foo` mentioned is grounded in
  `src/gpucheck/__init__.py:_LAZY_MAP` or in DIFF_LOG track-A row 11.
- Every cited pytorch issue number (#162872, #179352, #179294, #173525,
  #175189, #142836, #174269, #181936, #96602, #175190, #176296, #137001,
  #177116, #164299, #170837) is present in SYNTHESIS §Sub-Q 1/§Sub-Q 2.

## Hard rule: deferred files

- `CODE_OF_CONDUCT.md` — deferred per AUDIT §C item 4. Not written.
- `SECURITY.md` — deferred to security team per AUDIT §C item 5.
  Not written. CONTRIBUTING points to FINDINGS.md.

## Stale claims fixed

| AUDIT row | File:line (pre-edit) | Resolution |
|---|---|---|
| §A.1 #1 | README.md:10 | "CUDA kernel" → "CUDA or Metal kernel" |
| §A.1 #14 | README.md:367 | "AMD ROCm and Intel XPU not supported yet" → explicit "Apple Silicon supported via MPS as of v1.0; ROCm/XPU planned" |
| §A.5 #32 | README.md:12 vs 328-334 | "8 real bugs" → "8 bugs, 2 externally filed and verified" with footnote citing SYNTHESIS §Sub-Q 8 |
| §A.3 #21 | CLAUDE.md "No stride/contiguity fuzzing" | moved out of gaps into "v1.0 highlights" / Strengths |
| §A.3 #23 | CLAUDE.md "Thread-safety issue in tolerance override stack" | moved out of gaps into v1.0 highlights |
| §A.4 #25 | CLAUDE.md "No HTML/dashboard reporting" | moved out of gaps |
| §A.4 #26 | CLAUDE.md "No determinism testing support" | moved out of gaps |
| §A.4 #30 | CLAUDE.md "Reporting module zero coverage" | moved out of gaps (now 98%) |
| §A.4 #31 | CLAUDE.md "No changelog, contributing guide, migration docs" | closed by Phase 3 deliverables |
| §A.5 #33 | CLAUDE.md "Git Conventions: conventional commits" | reconciled — Conventional Commits going forward, bracket-style history unchanged |

## Verdict

PASS for Phase 3 writer responsibilities. Every claim traces to a
binding input file. PROVISIONAL items are flagged as such. Deferred
items (CoC, SECURITY) are explicitly not written.
