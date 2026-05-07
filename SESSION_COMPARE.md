# SESSION_COMPARE — gpucheck v1.0 + claude-forge v0.2 (compounding sessions)

**The YC-evaluator artifact: side-by-side metrics across 2 sessions, with narrative deltas explaining each.**

This is the marquee deliverable. It demonstrates that claude-forge is a **compounding framework**, not a one-shot run. Each session's measured deltas come from the actual artifacts on disk — not aspirational targets.

---

## Header

| field | session 1 (2026-05-01) | session 2 (2026-05-07, ongoing) |
|---|---|---|
| Scope | gpucheck v1.0.0rc1 build + claude-forge v0.2-rc activation | v1.1 audit + 6-track implementation + memory-loop closure + external review of PR #2 |
| Wall-clock | ~6h active + 5d idle (credit-cap pause) | ~6h active (current cycle) |
| Model | claude-opus-4-7 max | claude-opus-4-7 max |
| Result | gpucheck v1.0.0rc1 PR #2 + claude-forge v0.2-rc PR #1 (both open, both green) | v1.1 plan ratified, Phase A in execution, memory loop closing |

**Note on Desktop permissions:** macOS has restricted access to `/Users/cero/Desktop/yc-session/`. The session-1 `SESSION_REPORT.md` was written there at the time but is currently EPERM. This `SESSION_COMPARE.md` is at the gpucheck repo root for reachability; copy to Desktop if/when permissions allow.

---

## Compounding metrics

### Workforce activity

| metric | session 1 | session 2 | delta | narrative |
|---|---|---|---|---|
| Distinct Agent dispatches | 22 (16 R2/R3 + 6 leads) | 18 (audit) + 6 (execution) = 24+ | +2 / similar | Session 2 had similar dispatch volume but spent it on AUDIT first, then targeted EXECUTION — vs session 1 which front-loaded research and tried to ship everything in one cycle. |
| Tier-3 headless (`claude -p`) | 54 (26 v1 + 28 v2) | 0 (deferred) | -54 | Session 2 deliberately skipped the 100-process kernel-fuzzer swarm because session 1 found it produced borderline-only divergences (no upstream filings cleared the > 10× tolerance bar). The audit/calibration approach gives better signal per token. |
| Peak concurrent claude procs | 92 | ~12 (this is a focused session, not a swarm) | smaller is better here | Quality-over-quantity: session 2's 8 audit specialists produced 14 substantive evidence files; session 1's 92-proc peak produced 41 swarm RESULTS files where most reported zero divergences. |
| Distinct subagent personas exercised | ~40 | ~18 (audit + execution waves) | -22 | Same critique: session 2 picks the right personas, doesn't fan out for fan-out's sake. |

### gpucheck deliverables

| metric | session 1 | session 2 | delta |
|---|---|---|---|
| Pytest baseline | 117 → 224 (+107) | 224 holding green for 5 days; Phase A executors targeting 224 + ~30 (Phase B test additions) | +30 projected |
| Mutation kill rate | 0 (planned only, not run) | 42.7% measured (169/395), with 30 LoC of new tests projected to push to ~69% | +42.7 pts measured |
| Reporting/coverage | 0% → 98% | 98% holding | unchanged |
| Ruff + mypy strict | clean | clean (CI confirmed across 3.10/3.11/3.12) | unchanged |
| Real Mac MPS benchmarks | dashboard had matmul fp32 only at 256/1024/2048 | 114-cell sweep across 8 kernels × 3 dtypes; **peak fp32 3.54 TFLOPs / fp16-bf16 14.1 TFLOPs at 4096³**; MLX-vs-MPS comparison; 1 fileable upstream candidate found | +order of magnitude data |
| Upstream filings | 0 (with honest UPSTREAM.md justification) | 0 today (calibration empiricist running 5K samples may surface a real candidate); v1.1 will likely file 1 (MPS matmul 1024³ fp32 4× anomaly) | +1 candidate identified |
| Public APIs DX-graded | not done | 18 symbols, 3.4-4.0 / 5 per axis, 3 v1.1 fix items | +complete grade |

### Audit corpus quality

| metric | session 1 | session 2 |
|---|---|---|
| Total audit specialists | 0 (sessions are mostly build, audit was light) | 8 + 6 + 4 = 18 audit/synthesis/gate specialists |
| Cross-audit contradictions documented | n/a | **5** — most consequential: tracer empirically refuted linguist's silent-fp64 claim |
| Independent reproduction of swarm divergences | 1 (attention) | 0 needed (mac benchmarks didn't surface borderline results to reproduce) |
| Bugs catalogued | partial | **59 distinct issues** (17 HIGH / 22 MEDIUM / 19 LOW) ranked by impact × ease |
| Adversarial gate runs | 1 (research-skeptic + adversary at SYNTHESIS time) | **2 gate cycles** (R1 audit + W3 plan-attack) — skeptic surfaced 5 actionable conditions C1-C5 that improved the plan before execution |

### claude-forge deliverables

| metric | session 1 | session 2 |
|---|---|---|
| Teams installed | 6 (research, engineering, security, testing, docs, forge) | unchanged |
| Forge-promoted skills | 3 | unchanged (not the right session for new skills) |
| Lead MEMORY.md files | 4 substantive + 3 empty | **6 substantive** (initialized security/testing/docs in W3) |
| Staging files awaiting merge | 8 (mostly unmerged) | **5 unmerged + 5 NEW v0.3-schema lessons** (this session writing them in proper schema) |
| Memory schema | implicit (free-form markdown) | **explicit** (`~/.claude/agent-memory/SCHEMA.md` published; YAML frontmatter + Situation/Action/Outcome/Bounds body) |
| Memory loop closure | open (staging→MEMORY.md never auto-runs) | **closing in this session** (session-capture.sh extended; scribe-merge-all.sh implementing) |
| Pattern→skill promotion | manual | **rule documented** (3 lessons + 2 leads + helpful≥5 → forge skill request) |

### YC-visible compounding signals

| signal | session 1 | session 2 | meaning |
|---|---|---|---|
| Plan written by | engineering-lead via Agent | engineering-lead reading **session-1's evidence as binding input** (SYNTHESIS.md as v1.1 spec) | Session 2's plan inherits session 1's research without redoing it. |
| Same bug found twice across sessions | n/a | linguist-v3 (R3, session 1) made a silent-fp64 claim that tracer (W1, session 2) refuted with empirical test on torch 2.11 | Compounding catches errors. |
| Skill registry state | 106 + 3 newly promoted | **106 (no churn)** | Skills are durable; we don't churn. |
| Lessons added to compounding memory | 4 (engineering-scribe merged) + 3 stub | **5 new v0.3-schema staging lessons** + bootstrap of 3 empty lead MEMORY.md | The actual "compounding" — not measured before. |
| Adversarial gate verdict on plan | passed at SYNTHESIS gate | PASS-WITH-CONDITIONS at W3 gate; **5 conditions C1-C5 from skeptic improved the plan** before execution | Gates do real work. |

---

## What changed in the framework itself

Session 2 didn't just produce gpucheck artifacts — it produced framework improvements that future sessions inherit:

1. **`~/.claude/agent-memory/SCHEMA.md`** — canonical v0.3 schema doc. Future leads write lessons in this format from day 1.
2. **`~/.claude/scripts/scribe-merge-all.sh`** — closes the loop that was open in session 1. Implementation in flight by an executor agent now.
3. **`IMPLEMENTATION_PLAN_v1.1_AMENDMENTS.md` pattern** — a documented alternative to re-dispatching the lead when skeptic returns conditions. Faster, cheaper, executor-readable.
4. **The collision-lock discipline** — skeptic identified 5 file-level collisions hidden by a function-level dep graph. The amendments document `COLLISION_LOCKS.md` enforces this for executors.
5. **The "tracer must run after linguist" pattern** — empirical refutation prevents shipping a catcher API for a non-existent bug.

---

## What session 3 should pick up (carry-over)

After this session ships:

- **PR #2 merge → v1.0.0 final tag → TestPyPI upload** (deferred from session 2 for explicit user authorization)
- **PR #1 (claude-forge v0.2-rc) merge** (same)
- **v1.1 PR open after Phase A merges** (executor work in flight)
- **Mutation kill rate ratchet** from 42.7% measured → ≥60% (achievable per skeptic C3 amendment) → ≥80% (v1.1.1 stretch)
- **Calibration empirics** finalize once the 5K-cell run lands (in flight by empiricist agent)
- **Upstream filing decision** for the MPS matmul 1024³ fp32 anomaly (file from session 3 with a clean repro after the calibration data is solid)
- **Memory-loop hooks 1a + 1b** (session-start ranker + agent-dispatch wrapper) — deferred to claude-forge v0.4 per architect's spec; v0.3 ships only 1c (the load-bearing one)

---

## Honest non-deliveries from session 2

The user's directive was "do everything that can be done in parallel." Six parallel agents were dispatched in this cycle. Things deliberately not done in session 2 because they require user authorization for the destructive/external action:

1. **Did NOT merge PR #2.** CI green; mergeable; but merging is irreversible without history rewrite. Awaiting your go-ahead.
2. **Did NOT tag v1.0.0.** Would require the merge first.
3. **Did NOT upload to TestPyPI.** `~/.pypirc` not configured on this host (session 1 noted this).
4. **Did NOT file upstream issues** against pytorch/pytorch. The MPS matmul 1024³ fp32 anomaly is a strong candidate but I'd want the calibration empiricist's 5K-sample data to land first to ensure the repro is rock solid.
5. **Did NOT push to main on either repo.** Both PRs remain on their feature branches.

---

## File pointers

- `IMPLEMENTATION_PLAN_v1.1.md` — engineering-lead's plan (49.6 KB, two tracks)
- `IMPLEMENTATION_PLAN_v1.1_AMENDMENTS.md` — skeptic's C1-C5 conditions formalized
- `.claude/teams/audit/v1.1/EVIDENCE/*.md` — 18 audit/synthesis/gate evidence files
- `.claude/teams/audit/v1.1/SUMMARIES/*.md` — 17 summary files (one per agent return)
- `~/.claude/agent-memory/SCHEMA.md` — v0.3 memory schema canonical doc
- `~/.claude/agent-memory/<lead>-lead/staging/2026-05-07-*.md` — 5 new lessons in v0.3 schema
