# EVIDENCE — docs-planner

**Persona:** docs-planner (Phase A, FM-1.1 — coverage gap analysis,
priority matrix, doc plan).
**Run:** Phase 1 audit, gpucheck v1.0.

## Tier classification

**Comprehensive** (full roster, all gates).
Rationale:
- v1.0 is a release tag with breaking-API potential (MPS backend,
  stride-fuzzing, thread-safety fix per `SESSION_PRECHECK.md` impl
  worktrees `track-{a-mps,b-strides,c-thread-safety,d-bundle}`).
- README and CLAUDE.md both contain CUDA-only language that needs
  widening (see `EVIDENCE/reader.md` stale-claim list, 14 items).
- Four top-level docs are missing (CHANGELOG, CONTRIBUTING, MIGRATION,
  CODE_OF_CONDUCT). Two of them (CHANGELOG, MIGRATION) are
  release-blockers for a v1.0 tag.
- 20 public symbols lack docstrings.

## Coverage matrix (now → v1.0)

| Doc artifact | Owner | Phase 1 (this turn) | Phase 3 (after DIFF_LOG) |
|---|---|---|---|
| `README.md` | docs-lead → writer | Audit-only — list stale claims | Rewrite §"Step by step" §6/§8/§"Tested hardware"; add MPS section |
| `CLAUDE.md` | docs-lead → writer | Audit-only | Update Architecture, "Strengths", "Known Weaknesses" |
| `CHANGELOG.md` | docs-lead | **Draft skeleton** (Keep-a-Changelog) | Fill `[1.0.0rc1]` from DIFF_LOG |
| `CONTRIBUTING.md` | docs-lead | **Draft full skeleton** | Polish, link to expert system |
| `MIGRATION_v0_to_v1.md` | docs-lead | **Draft per-API placeholder** | Fill from research SYNTHESIS + engineering DIFF_LOG |
| `CODE_OF_CONDUCT.md` | (deferred) | Flag in AUDIT.md, recommend Contributor Covenant 2.1 | Adopt verbatim |
| `SECURITY.md` | (deferred to security team) | Flag in AUDIT.md | Cross-team handoff |
| Public-symbol docstrings (20 missing) | docs-lead → writer | List in AUDIT.md | Add docstrings in source (engineering will commit) |
| Architecture diagram (Mermaid) | docs-diagrammer | — | Phase 3 |
| Examples re-validation on MPS | docs-tester | — | Phase 3 (needs hardware run) |

## Priority matrix

P0 (blocks v1.0 release):
- README MPS section + remove "no MPS" caveats
- CHANGELOG.md `[1.0.0rc1]` — semver discipline for downstream installers
- MIGRATION_v0_to_v1.md — anyone pinning `gpucheck<1` needs this
- 20 missing docstrings on public API (mypy strict + ruff doesn't enforce, but DocAgent reader-before-writer principle requires them)

P1 (release-quality, not blocking):
- CONTRIBUTING.md — needed for incoming PRs
- Update CLAUDE.md "Known Weaknesses" — currently lists "no MPS / no
  stride-fuzzing / no thread-safety" all of which v1.0 will close
- Update tested-hardware list (add Apple Silicon results)
- Tolerance table: add MPS rows

P2 (nice-to-have):
- CODE_OF_CONDUCT.md (Contributor Covenant 2.1)
- SECURITY.md (links to security team output)
- Architecture Mermaid diagram in README

## Phase 1 deliverable list (this turn)

1. `EVIDENCE/detector.md` — DONE.
2. `EVIDENCE/reader.md` — DONE.
3. `EVIDENCE/planner.md` — this file.
4. `EVIDENCE/reviewer.md` — README + CLAUDE.md vs source.
5. `EVIDENCE/skeptic.md` — adversarial pass.
6. `EVIDENCE/evaluator.md` — 5-dim rubric (becomes top-level `evaluator.md`).
7. `EVIDENCE/retrospector.md` — Phase 1 lessons.
8. `AUDIT.md` — aggregated, with file:line cites.
9. `CHANGELOG_DRAFT.md` — Keep-a-Changelog `[Unreleased]` + `[1.0.0rc1]`.
10. `CONTRIBUTING_DRAFT.md` — full structure.
11. `MIGRATION_v0_to_v1.md` — per-API placeholder, references SYNTHESIS.

## Doc plan — narrative arc for v1.0 announcement

The README's hero paragraph needs to shift from
"GPU kernel testing is painful... CUDA kernel..." to
"GPU kernel testing is painful, on **CUDA and Apple Silicon
Metal Performance Shaders**...". The "no MPS" caveat
(`README.md:367` — "AMD ROCm and Intel XPU are not supported yet";
implicitly excludes MPS) becomes "MPS supported; ROCm/XPU planned".

The "Bugs found" section (`README.md:324`) keeps the 8-bug record but
adds a parallel "MPS bugs found" subsection once research/engineering
produce hard data (research QUESTION sub-question 2).

## Cross-team plumbing

Phase 3 will read:
- `<cwd>/.claude/teams/research/v1.0/SYNTHESIS.md` (does not yet exist —
  research session is still in-flight per `research/v1.0/QUESTION.md`).
- `<cwd>/.claude/teams/engineering/v1.0/DIFF_LOG.md` (does not yet
  exist — `ls` returned only `TURN_LOG.md`).

Phase 1 only references these files via `{{TODO: cite SYNTHESIS}}`
placeholders inside `MIGRATION_v0_to_v1.md`. We do not invent claims.

## Termination plan

Phase 1 is intentionally bounded — no source-file modifications, only
audit + skeletons inside `.claude/teams/docs/v1.0/`. Hard rule from
charter: "Do NOT modify any file outside `.claude/teams/docs/v1.0/` in
Phase 1". Honored.
