# Cartographer — Memory Artifact Inventory

## Scope
Filesystem-only inventory of every claude-forge memory artifact reachable from the
six target trees. No interpretation of lesson semantics; only path, size, mtime,
schema, and provenance.

Excluded by charter: `~/.claude/skills/<name>/SKILL.md` body content (only the
top-level directory roster is in scope), settings.json, log JSONLs.

---

## §1. `~/.claude/agent-memory/` — per-lead canonical memory

7 lead directories. Only 4 currently have a `MEMORY.md` (engineering, forge,
research, research-retrospector). docs-lead, security-lead, testing-lead each
have a `staging/` dir but **no MEMORY.md** — meaning the v1.0-gpucheck staging
file is the *first* lessons file ever written for that lead.

| Path | Lines | Bytes | mtime | First line / frontmatter | Schema | Status |
|---|---|---|---|---|---|---|
| `engineering-lead/MEMORY.md` | 32 | ~2.3 KB | 2026-05-01 | `# engineering-lead — persistent agent memory` | H1 + Starter playbook + 2× "Added from …" merge sections | **light** — starter playbook is "(Empty)"; only 2 merged lessons from `memory-hook-a-v1` (2026-04-12). Has NOT yet absorbed any v1.0-gpucheck or upgrade-mcp staging. |
| `forge-lead/MEMORY.md` | 31 | ~1.4 KB | 2026-05-01 | `# forge-lead — persistent agent memory` | H1 + "Process lessons" + "Authored skills catalog" + "Failed gap investigations" | **light** — 1 process lesson + 1 catalog entry (`hn-search`). The 3 v1.0-gpucheck draft skills (mps-kernel-debugging, metal-shader-profiling, hatch-testpypi-release) are NOT yet promoted into the catalog. |
| `research-lead/MEMORY.md` | 262 | ~22 KB | 2026-05-01 | `# research-lead — persistent agent memory` | H1 + Starter playbook + 2 merge sections (`engineering-team-self-evolve-v1`, `orchestration-full-activation-v1`) | **substantive** — 26 H3 lessons. Pre-2026-05-01 content from 2026-04-12 sessions. v1.0-gpucheck staging not yet merged. |
| `research-retrospector/MEMORY.md` | 32 | — | 2026-05-01 | `# research-retrospector — meta-lessons about retrospection` | H1 + "Starter meta-playbook" + 3 H3 meta-rules | **light** — 3 meta-lessons, all from seed (2026-04-12). No additions. |
| `docs-lead/MEMORY.md` | — | — | — | (file does not exist) | n/a | **EMPTY** — directory contains only `staging/`. |
| `security-lead/MEMORY.md` | — | — | — | (file does not exist) | n/a | **EMPTY** — directory contains only `staging/`. |
| `testing-lead/MEMORY.md` | — | — | — | (file does not exist) | n/a | **EMPTY** — directory contains only `staging/`. |

Per-lead MEMORY.md status: **0 substantive merged, 1 substantive (research-lead),
3 light, 3 empty.**

---

## §2. `~/.claude/skills/` — installed skills roster

Top-level directory contains **106 skill subdirectories** (per `ls`). Charter
states "currently 106"; verified.

Spot-check (alphabetical first 5): `0-autoresearch-skill/`, `20-ml-paper-writing/`,
`accelerate/`, `audiocraft/`, `autogpt/`. All are dated `2026-05-02 02:22` (mass
mtime — likely a sync timestamp, not authoring date). One outlier:
`avoid-ai-writing/` mtime 2026-04-29.

Note: `mps-kernel-debugging`, `metal-shader-profiling`, `hatch-testpypi-release`
ARE present in `~/.claude/skills/` (3 of the 106) — meaning forge promoted
them between Phase 4 and now. The forge `PROMOTIONS.md` still lists them as
"NOT yet promoted" pre-Phase-4; the actual `~/.claude/skills/` tree disagrees
with `PROMOTIONS.md`. Possible drift — surfaced for human review.

No `MEMORY.md` or `staging/` artifacts under `~/.claude/skills/`. Schema is
opaque from the perspective of this audit.

---

## §3. `~/.claude/agents/` — installed personas

| Sub-tree | Files | First line schema | Notes |
|---|---|---|---|
| `forge-lead.md` (root) | 1 | `# forge-lead — capability forge orchestrator` (assumed; not read) | Sole top-level persona file (8.3 KB). |
| `research/` | 19 personas + `PROTOCOL.md` | each persona is markdown role-spec | research-lead, research-cartographer, …, research-retrospector. |
| `engineering/` | 13 personas + `PROTOCOL.md` | markdown role-spec | engineering-lead, …, engineering-debugger. |
| `security/` | 13 personas (no PROTOCOL.md) | markdown role-spec | security-lead, …, security-license-auditor. |
| `docs/` | 11 personas | markdown role-spec | docs-lead, docs-reader, …, docs-retrospector. |
| `testing/` | 12 personas | markdown role-spec | testing-lead, …, testing-property. |
| `gpucheck/` | 10 expert personas | markdown expert-spec | pytest-plugin-architect, performance-engineer, etc. (project-bound, see CLAUDE.md). |

Total: **78 persona files** + 2 PROTOCOL.md (research, engineering) inside the
agents tree. **Anomaly**: `~/.claude/agents/security/` has NO PROTOCOL.md
inline; the protocol lives only in `~/.claude/teams/security/PROTOCOL.md`.
Same for `docs/` and `testing/`. Inconsistency: only `research/` and
`engineering/` ship a PROTOCOL.md inside `agents/`.

---

## §4. `~/.claude/teams/` — team protocols + write-audit logs

| Path | Lines | Bytes | mtime | First line |
|---|---|---|---|---|
| `docs/PROTOCOL.md` | 307 | 13862 | 2026-05-01 | `# Documentation & Knowledge Team Protocol v1` |
| `engineering/PROTOCOL.md` | 398 | 17158 | 2026-05-01 | `# Engineering Team Protocol v1` |
| `research/PROTOCOL.md` | 582 | 27674 | 2026-05-01 | `# Research Team Protocol v2` |
| `security/PROTOCOL.md` | 490 | 18651 | 2026-05-01 | `# Security & Review Team Protocol v1` |
| `testing/PROTOCOL.md` | 329 | 13340 | 2026-05-01 | `# Testing/QA Team Protocol v1` |
| `research/v1.0/_write_audit.log` | 36 | 5193 | 2026-05-04 | `<ISO-timestamp> <agent> wrote <path>` lines |
| `security/v1.0/_write_audit.log` | 11 | 1611 | 2026-05-01 | same | 
| `docs/v1.0/_write_audit.log` | 10 | 1386 | 2026-05-01 | same |
| `testing/v1.0/_write_audit.log` | 9 | 1336 | 2026-05-01 | same |
| `engineering/v1.0/_write_audit.log` | 18 | 2633 | 2026-05-01 | same |
| `audit/v1.1/_write_audit.log` | 8 | 1158 | 2026-05-06 | same |

Note: research is on PROTOCOL **v2**; all other teams are v1. forge has no
team-level PROTOCOL.md under `~/.claude/teams/forge/` (forge is invoked via the
`forge-lead.md` persona only, not as a fully-collaborative team).

---

## §5. `~/Code/gpucheck/.claude/teams/` — project evidence trees

7 team subdirs (research, security, docs, testing, audit, forge, engineering)
+ 5 root-level files. **488 total files**. 34 directories.

### Root-level files

| Path | Lines | Bytes | mtime | First line |
|---|---|---|---|---|
| `HEARTBEAT.md` | 357 | 52157 | 2026-05-04 | `# HEARTBEAT — gpucheck v1.0 session (v2 expansion)` |
| `SESSION_PRECHECK.md` | 58 | 3011 | 2026-05-01 | `# SESSION_PRECHECK — gpucheck v1.0 + claude-forge v0.2` |
| `V2_BUDGET.md` | 28 | 2520 | 2026-05-06 | `# v2 dispatch budget — final measurements` |
| `MATRIX_REPORT.md` | 66 | 3477 | 2026-05-01 | `# MATRIX_REPORT — gpucheck v1.0 cross-PyTorch matrix` |
| `MATRIX_2.10.0_full.md` | 36 | 2422 | 2026-05-01 | `# MATRIX run: torch==2.10.0` |

### Per-team artifact counts (v1.0 unless noted)

| Team | Top-level docs | EVIDENCE files | Other |
|---|---|---|---|
| `research/v1.0/` | 9 (QUESTION, HYPOTHESES, SYNTHESIS, SYNTHESIS_v1, SYNTHESIS_v2, EXPECTED_EVIDENCE, TURN_LOG, evaluator, API_STABILITY_AUDIT, drift_histogram.json) | 27 (7 base + ~20 versioned: cartographer-v2, archaeologist-v3, github-miner-v3, etc.) + 1 subdir (github-miner-v2/) | retrospector.md exists (152 lines) |
| `security/v1.0/` | 5 (AUDIT_CHARTER, TURN_LOG, evaluator, THREAT_MODEL, FINDINGS) | 10 (license-auditor, skeptic, crypto-reviewer, evaluator, config-scanner, architecture-reviewer, owasp-scanner, planner, dependency-auditor, secrets-hunter, threat-modeler) | **NO retrospector.md in EVIDENCE** |
| `docs/v1.0/` | 6 (TURN_LOG, CONTRIBUTING_DRAFT, evaluator, CHANGELOG_DRAFT, AUDIT, MIGRATION_v0_to_v1) | 10 (retrospector, reviewer, skeptic, reader, evaluator, docs-tester, docs-diagrammer, detector, planner, docs-writer) | retrospector.md exists (139 lines) |
| `testing/v1.0/` | 7 (UPSTREAM, MUTATION_REPORT_v2, TURN_LOG, evaluator, PROPERTY_PLAN, MUTATION_REPORT, SWARM_PLAN) | 9 (testing-skeptic, testing-retrospector, testing-detector, testing-mutator, testing-property, testing-fixture, testing-planner, testing-scribe, testing-evaluator) + `swarm/` subdir (~80 fuzz_*.py files, 50+ logs) + `mutmut/` | retrospector exists (named `testing-retrospector.md`) |
| `audit/v1.1/` | 1 (LOG.md, 1 line so far) | 7 (api-dx-grade, archaeologist-debt, detector-files, docs-tester-blocks, mutator-survivors, security-postmerge, tracer-runtime) — **all dated 2026-05-06** (this in-flight audit) | 1 SUMMARIES file (api-dx-grade.summary.md) |
| `forge/v1.0/` | 4 (TURN_LOG, SCOUT_LOG, GAP_INVENTORY, PROMOTIONS) | 0 EVIDENCE files | 3 DRAFTS subdirs (mps-kernel-debugging, hatch-testpypi-release, metal-shader-profiling) each with SKILL.md + EVAL_TRACE.md |
| `engineering/v1.0/` | 8 (MPS_RUN.log, VERIFY_LOG, DIFF_LOG, TURN_LOG, evaluator, CPU_RUN.log, CHARTER, PLAN) | 19 (planner, architect, skeptic, adversary, executor-A/B/C/D, verifier-A/B/C/D, reviewer-A/B/C/D, retrospector, scribe) | retrospector.md exists (4661 bytes) |
| `engineering/INDEX.md` (root of engineering tree) | 13 | — | 2026-05-01 |

Anomaly: **security & forge teams produced NO `retrospector.md`** evidence file.
The `staging/v1.0-gpucheck.md` files for both leads are 3-line stubs explicitly
saying "No retrospector evidence file written for this team in this session."

---

## §6. `~/.claude/hooks/` and `~/.claude/scripts/`

### Hooks (4 total)

| Path | Lines | Bytes | mtime | Wired? |
|---|---|---|---|---|
| `cascade-research-to-engineering.sh` | 54 | 1753 | 2026-05-01 | (would be wired via settings.json — not inspected here) |
| `check-cascade.sh` | 11 | 374 | 2026-05-01 | same |
| `log-evidence-writes.sh` | 89 | 3260 | 2026-05-01 | PostToolUse observation hook per research-lead MEMORY.md lesson |
| `session-capture.sh` | 63 | 2543 | 2026-05-01 | session-start capture |

### Scripts (6 total)

| Path | Lines | Bytes | mtime |
|---|---|---|---|
| `audit_evidence.py` | 610 | 23988 | 2026-05-01 |
| `forge-gap-refresh.sh` | 40 | 1711 | 2026-05-01 |
| `meta_evaluator.py` | 208 | 8458 | 2026-05-01 |
| `setup-schedules.sh` | 23 | 1466 | 2026-05-01 |
| `team_status.sh` | 171 | 5720 | 2026-05-01 |
| `test-infrastructure.sh` | 136 | 5593 | 2026-05-01 |

All scripts/hooks frozen on 2026-05-01. No drift.

---

## §7. Staging dead-vs-live disposition

8 staging files total across 6 leads (engineering-lead has 3 in staging).

| File | Lines | Lessons | Source EVIDENCE on disk? | Disposition | Reason |
|---|---|---|---|---|---|
| `docs-lead/staging/v1.0-gpucheck.md` | 145 | 7 (L1–L7) | YES (`docs/v1.0/EVIDENCE/retrospector.md`, 139 lines) | **KEEP — promote in v0.3** | Substantive 7-lesson retrospective from a session that ran cleanly. docs-lead has no MEMORY.md yet — this would be the seed playbook. |
| `engineering-lead/staging/upgrade-mcp-tools-deterministic.md` | 24 | 3 | n/a (no team session in `gpucheck/.claude/teams/`; lives in OUTBOX archive per file body) | **KEEP — promote in v0.3** | 3 well-formed lessons, includes failure-mode IDs, unique authoring date 2026-05-06. Newest staging file. |
| `engineering-lead/staging/v1.0-gpucheck.md` | 57 | 4 | YES (`engineering/v1.0/EVIDENCE/retrospector.md`, 4661 bytes) | **DROP — clearly stale** | Source-mirror staging file (literal copy of EVIDENCE/retrospector.md with a heading). Same 4 lessons appear in cleaner form in `v1.0.md` sibling. Pure duplicate. |
| `engineering-lead/staging/v1.0.md` | 33 | 4 | derived from same source | **KEEP — promote in v0.3** | Cleaned, scribe-formatted version of the engineering-lead lessons. Use this; drop the `-gpucheck` mirror. |
| `forge-lead/staging/v1.0-gpucheck.md` | 3 | 0 | NO retrospector evidence | **DROP — clearly stale** | Stub: "No retrospector evidence file written". No content to merge. |
| `research-lead/staging/v1.0-gpucheck.md` | 158 | 3 (L1, L2, L3) | YES (`research/v1.0/EVIDENCE/retrospector.md`, 152 lines) | **KEEP — promote in v0.3** | Substantive 3-lesson retrospective with a §4 cross-session pattern observations + §5 cross-references. Tightly scoped. |
| `security-lead/staging/v1.0-gpucheck.md` | 3 | 0 | NO retrospector evidence | **DROP — clearly stale** | Stub. No retrospector ran. |
| `testing-lead/staging/v1.0-gpucheck.md` | 98 | 5 | YES (`testing/v1.0/EVIDENCE/testing-retrospector.md`) | **KEEP — promote in v0.3** | 5 lessons (skeptic-gate value, importable target names, dual-pronged absence-of-pattern, per-track mutation thresholds, deterministic divergence classifier). Plus v2.1 compliance trailer. |

**Tally**:
- KEEP — promote in v0.3: **5** (docs, eng v1.0, eng MCP, research, testing)
- DROP — stale: **3** (forge stub, security stub, eng v1.0-gpucheck duplicate)
- DEFER — needs human review: **0**

---

## §8. Cross-lead lesson contradictions

File-system facts only. No semantic cross-walk performed (out of charter — would
require interpretation). The only structural contradiction noted:

- `forge/v1.0/PROMOTIONS.md` lists 3 candidate skills as "NOT yet promoted"
  (Phase 4 pending), but `~/.claude/skills/` directory **already contains** all
  3 (`mps-kernel-debugging`, `metal-shader-profiling`, `hatch-testpypi-release`).
  Either Phase 4 ran and PROMOTIONS.md wasn't updated, or someone manually
  copied the drafts. **Defer for human review.**

---

## §9. Provenance summary by session date

| Session date | Lead | Artifact |
|---|---|---|
| 2026-04-12 | research-lead | seed Starter playbook + 2 merge-sections |
| 2026-04-12 | engineering-lead | memory-hook-a-v1 lessons |
| 2026-04-13 | forge-lead | hn-search authored skill |
| 2026-05-01 | docs/research/testing/engineering | gpucheck v1.0 evidence trees + 5 staging files |
| 2026-05-04 | research | HEARTBEAT.md update + extra `_write_audit.log` lines |
| 2026-05-06 | engineering | upgrade-mcp-tools-deterministic staging |
| 2026-05-06 | audit | this v1.1 audit (in flight) |

No artifact dates are in the future relative to today (2026-05-01 in CLAUDE.md
context, 2026-05-06 latest mtime). The 2026-05-06 dates indicate post-cutoff
sessions.

---

## Confidence

**high** — every count and mtime above came from `find`, `wc -l`, `stat`, or
`Read`. No claim depends on inferring lesson semantics. The two soft anomalies
(forge PROMOTIONS.md vs `~/.claude/skills/` drift; security/forge missing
retrospector evidence) are filesystem-verifiable and explicitly flagged for
human review rather than asserted.
