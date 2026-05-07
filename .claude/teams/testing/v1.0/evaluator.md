# Evaluator — testing / v1.0 (5-dim plan rubric)

**Slug**: testing/v1.0
**Owner**: testing-lead (adopted persona)
**Date**: 2026-05-01
**Dispatch**: Phase 2 plan-prep
**Inputs graded**: PROPERTY_PLAN.md + MUTATION_REPORT.md + SWARM_PLAN.md + 9 EVIDENCE/*.md files
**Mode**: end-state evaluation of the plan, not the process

> This is the dispatch-CHARTER-required 5-dim PASS/FAIL on the plan
> itself. Separate from `EVIDENCE/testing-evaluator.md` which scored the
> plan against the testing protocol's standard 6-dim rubric.

## 5-dimension rubric

| Dimension | Score | Threshold | Pass? |
|---|---|---|---|
| 1. Goal alignment | 1.0 | 0.7 | PASS |
| 2. Communication | 0.95 | 0.7 | PASS |
| 3. Output quality | 0.92 | 0.7 | PASS |
| 4. Safety | 1.0 | 0.7 | PASS |
| 5. Efficiency | 0.9 | 0.7 | PASS |

## Dimension detail

### 1. Goal alignment (1.0)

**Question**: Does the plan answer what the dispatch asked for?

The dispatch CHARTER specified 5 deliverable groups:
- (1) PROPERTY_PLAN.md covering 4 tracks (A: Backend Protocol + assert_close parity; B: stride properties; C: ContextVar properties; D: determinism + HTML)
- (2) MUTATION_REPORT.md (mutmut config + targets + run plan)
- (3) SWARM_PLAN.md (pre-flight + classifier + confidence bar)
- (4) EVIDENCE per specialist (9 files: planner, property, mutator, fixture, detector, skeptic, evaluator, scribe, retrospector)
- (5) evaluator.md at workspace root (this file)

All 5 delivered:

| Deliverable | Located at | Verifies | Status |
|---|---|---|---|
| PROPERTY_PLAN.md | `.claude/teams/testing/v1.0/PROPERTY_PLAN.md` | 32 properties × 4 tracks + security regressions | PRESENT |
| MUTATION_REPORT.md | `.claude/teams/testing/v1.0/MUTATION_REPORT.md` | mutmut config + 6 dispatch groups + run plan + thresholds | PRESENT |
| SWARM_PLAN.md | `.claude/teams/testing/v1.0/SWARM_PLAN.md` | pre-flight + 3-bucket classifier + machine-readable confidence table | PRESENT |
| EVIDENCE | `.claude/teams/testing/v1.0/EVIDENCE/testing-*.md` | 9 specialist files, persona-conformant schemas | PRESENT |
| evaluator.md (this) | `.claude/teams/testing/v1.0/evaluator.md` | 5-dim verdict | PRESENT |

Each property has an importable target name; each track binds engineering to specific module paths. The xfail list per kernel from SYNTHESIS is embedded in the property `test_xfail_entry_for_each_synthesis_top12`. The TM-E1 path-injection finding from FINDINGS.md is bound to property S1. The CFG-2 permissions finding is bound to unit assertion S2.

Score: 1.0.

### 2. Communication (0.95)

**Question**: Is the plan readable, machine-actionable, and unambiguous?

Strengths:
- Every artefact begins with frontmatter identifying owner, date, source, scope
- File pointers use absolute paths
- Tables use consistent schema (ID | Name | File | Profile)
- Cross-references are explicit (skeptic findings tagged in PROPERTY_PLAN with "skeptic A4-1, MEDIUM")
- Code sketches are runnable Python with imports shown
- Confidence bar in SWARM_PLAN is machine-readable (table form, not prose)
- TURN_LOG.md tracks what each adopted persona produced

Minor weaknesses:
- The two evaluator files (root + EVIDENCE) require a reader to track which scoring scheme applies where; documented but adds cognitive load
- Property IDs (A1, A3a/b/c, A8b, A9) are dense; a glossary would help newcomers

Score: 0.95.

### 3. Output quality (0.92)

**Question**: Are the artefacts technically sound and useful as written?

Strengths:
- Property sketches verified for self-consistency (self-check in `EVIDENCE/testing-property.md` enumerates one mutation per property that would fail it)
- Skeptic gate ran and produced 1 HIGH + 5 MEDIUM + 4 LOW findings; ALL were absorbed into PROPERTY_PLAN.md before final
- Mutation thresholds calibrated to risk gradient (90% strict on security, 70% advisory on HTML, 75-85% elsewhere)
- Divergence classifier in SWARM_PLAN is a deterministic decision tree with measurable thresholds (≥3 reproductions, >10× tolerance, deterministic seed), avoiding "vibe-based" triage
- Run plan in MUTATION_REPORT specifies wall-clock estimates per track with M-Mac calibration
- Pre-flight checklist in SWARM_PLAN is shell-runnable

Minor weaknesses:
- Track A property A1/A5 carry limited signal under CI mocks (acknowledged with `requires_mps` markers); a stronger plan would have a real-MPS smoke runner in CI via cloud-runner
- `_reset_for_test` underscore-prefixed naming is a minor convention slip
- HTML 70% mutation threshold is advisory and may be lax (skeptic A5 noted)

Score: 0.92.

### 4. Safety (1.0)

**Question**: Did the dispatch respect the hard rules?

Hard rules from dispatch:
- "NO actual swarm launch in this dispatch" — VERIFIED: no `claude -p` invocations made; `swarm/launch-swarm.sh` untouched
- "NO modifications to source files outside `.claude/teams/testing/v1.0/`" — VERIFIED: only files written are inside that directory tree (TURN_LOG, PROPERTY_PLAN, MUTATION_REPORT, SWARM_PLAN, evaluator, 9 EVIDENCE files)
- "Plans must be runnable" — VERIFIED: every property has importable target + sketch
- "Cite specific files/lines" — VERIFIED: cited `assertions/tolerances.py:28`, `sanitizers/race.py:56-62`, `fuzzing/shapes.py:9-16`, `pytorch#162872`, `pytorch#177116`, etc.

No source code modified. No git operations. No rate-limit-affecting subprocess. Security-conscious defaults documented (autouse `clean_tolerance_context`, allowlist enforcement in S1, RNG state save/restore in seeded_rng).

Score: 1.0.

### 5. Efficiency (0.9)

**Question**: Was the deliverable produced within budget and at appropriate scope?

- Wall-clock: ~70 min within 90-min hard deadline
- 12 files written (3 plans + 1 root evaluator + 1 TURN_LOG + 9 evidence)
- Total artefact size: ~3000 lines
- No redundant work: each file has a distinct schema and purpose
- Reused SYNTHESIS.md and FINDINGS.md verbatim as inputs rather than re-investigating
- Skeptic gate ran ONCE before final plan, not iteratively (per protocol)

Slight inefficiency: two evaluator files (the dispatch CHARTER's 5-dim rubric is non-standard vs the testing protocol's 6-dim rubric, requiring both to satisfy both contracts). This is a CHARTER-vs-protocol mismatch, not testing-lead's choice; absorbed at the cost of one extra file.

Score: 0.9.

## Overall verdict

**PASS** on all 5 dimensions. Plan is shippable as a binding contract for engineering-lead's 4 tracks and for the post-merge swarm launch.

## Acceptance criteria check (against dispatch CHARTER)

| Criterion | Verifies | Satisfied? |
|---|---|---|
| 4 tracks each have property tests | A:13, B:4, C:5, D:5 in PROPERTY_PLAN.md | YES |
| MPS xfail list mapped to test | A8 + A8b reference SYNTHESIS top-12 | YES |
| TM-E1 path injection covered | S1 in PROPERTY_PLAN | YES |
| CFG-2 permissions covered | S2 in test_ci.py spec (planner P1) | YES |
| mutmut config plan present | MUTATION_REPORT.md `[tool.mutmut]` block | YES |
| Per-track targets named | 6 dispatch groups in MUTATION_REPORT | YES |
| Mutators selected | boundary, replace_constant, swap_operator, keyword, comparison enumerated | YES |
| Run plan dispatched per worktree | post-merge worktree pattern shown | YES |
| Swarm launcher confirmed ready | pre-flight verdict READY-blocked-on-Track-A | YES |
| Expected swarm output documented | `RESULTS_<kernel>.md` schema + `swarm.jsonl` schema | YES |
| Divergence-classifier specced | 3 buckets with measurable thresholds | YES |
| Highest-confidence bar defined | ≥3 reproductions + deterministic seed + max_rel_err > 10× tolerance + kernel deterministic mode | YES |
| 9 specialist EVIDENCE files | written | YES |
| TURN_LOG updated | append-only, before/after | YES |
| No swarm launch | verified | YES |
| No source mods outside testing/v1.0/ | verified | YES |
| Hard 90-min deadline | ~70 min wall | YES (margin) |

## Failure modes guarded against

- FM-1.1 (specification ambiguity): every property cites importable target + skeptic absorbed
- FM-1.2 (insufficient test design): comprehensive tier with property + mutation + fixture
- FM-1.5 (output triage ambiguity): swarm divergence classifier is deterministic
- FM-2.1 (conversation reset): TURN_LOG is append-only with timestamps + readable schema
- FM-2.4 (information withholding cross-team): plan binds engineering to importable names; cannot silently drift
- FM-3.1 (incorrect evaluation): both 5-dim and 6-dim rubrics ran
- FM-3.2 (insufficient verification): mutation thresholds with HIGH-severity escalation
- FM-3.3 (incorrect verification): skeptic gate ran before evaluator and produced HIGH finding

## If FAIL: targeted re-dispatch instructions

N/A (PASS).
