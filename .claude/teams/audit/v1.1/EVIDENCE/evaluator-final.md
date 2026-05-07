# Evaluator — v1.1 implementation plan + audit corpus

**Date:** 2026-05-01
**Evaluator:** research-evaluator
**Verdict (TL;DR):** **FAIL** — primary deliverable absent; no adversarial gate ran.

---

## §0. Critical pre-flight failure

Per my charter, I was asked to grade three artifacts:

1. `/Users/cero/Code/gpucheck/IMPLEMENTATION_PLAN_v1.1.md` — **does not exist** at the cited path.
2. `EVIDENCE/skeptic-plan-attack.md` — **does not exist** in `/Users/cero/Code/gpucheck/.claude/teams/audit/v1.1/EVIDENCE/`.
3. `EVIDENCE/adversary-corpus-attack.md` — **does not exist** in the same directory.

Verified by direct filesystem checks (`ls`, `find`):

- `find /Users/cero/Code/gpucheck -name "IMPLEMENTATION_PLAN*"` → no results.
- `EVIDENCE/` listing has 14 files: `api-dx-grade`, `archaeologist-debt`, `architect-continuous-learning`, `cartographer-memory-map`, `detector-files`, `docs-tester-blocks`, `empiricist-mac-benchmarks`, `forge-memory-schema`, `historian-memory-prior-art`, `mutator-survivors`, `planner-v1.1-tasks`, `security-postmerge`, `synthesist-bugs-inventory`, `tracer-runtime`. No skeptic, no adversary.
- The most recent file by mtime is `synthesist-bugs-inventory.md` (2026-05-07T00:39). Nothing in the workspace was written after that. Charter says the skeptic + adversary are "in-flight — wait for them"; they never landed.

The charter says this is the FINAL gate. Per my hard rules, **"You never grade a session with no skeptic pass. That's an automatic fail."** I therefore cannot pass this dispatch on Goal Alignment, Output Quality, or Safety regardless of how good the planner output is, because:

- I have no implementation plan to grade — only a `planner-v1.1-tasks.md` (a planner specialist's task table, not the engineering-lead's consolidated implementation plan that the charter requires me to grade).
- I have no skeptic attack to incorporate.
- I have no adversary attack to incorporate.

I will, however, give a provisional read of the **planner output as a stand-in** so the lead has actionable signal for re-dispatch. Treat what follows as conditional: "if the planner output were the implementation plan, here is how it would score." It is not a substitute for a real eval of a real plan with real adversarial gates.

---

## §1. Per-dim score table (provisional, treating `planner-v1.1-tasks.md` as the plan)

| # | Dimension | Threshold | Score | Pass? |
|---|---|---|---|---|
| 1 | Goal alignment | 0.90 | 0.78 | **N** |
| 2 | Communication | 0.90 | 0.92 | Y |
| 3 | Output quality | 0.85 | 0.88 | Y |
| 4 | Safety | 0.95 | 0.65 | **N** |
| 5 | Efficiency (advisory) | 0.70 | 0.85 | Y |

**Verdict on the provisional read alone:** FAIL (Goal alignment + Safety below threshold).
**Verdict including missing artifacts:** **FAIL — re-dispatch required.**

---

## §2. Per-dim rationale

### Dim 1 — Goal alignment: 0.78 (FAIL, threshold 0.90)

User's stated request: *"Lots of bugs, errors → audit + improve. Continuous-learning so Claude improves over time. Mac/Metal focus. 6-12 hour cycle."*

Where the planner output (`planner-v1.1-tasks.md`) lands on each pillar:

- **Audit + improve (bugs/errors).** STRONG. 27 atomic tasks, 17 high-severity issues mapped from synthesist's 59-bug inventory, citations to specific `path:line` in every task. Mutation kill rate 42.7%→≥80% is a measurable improvement target. **+0.95 sub-score.**
- **Mac/Metal focus.** PARTIAL. Phase D ships R3 Mac deliverables (T-23 Apple-tile fuzzer, T-24 per-(kernel,dtype) MPS overlay, T-25 xfail expansion 12→41, T-26 deadlock probe), and T-09 fixes the 52-hard-fail MPS auto-skip gap. Synthesist explicitly calls out a "Mac/Metal cluster of 14 issues" and minimum track ISS-05/19/20/21/32/35/36/58 — only T-23..T-26 + T-09 hit a subset. **No task addresses ISS-56 / ISS-57 (the two upstream-fileable PyTorch performance issues on Apple Silicon)** even as "file upstream" actions. **+0.80 sub-score.**
- **Continuous-learning so Claude improves over time.** WEAK. The planner output makes ZERO reference to the continuous-learning charter element. Two of the 14 evidence files address it directly (`architect-continuous-learning.md`, `cartographer-memory-map.md`, `forge-memory-schema.md`, `historian-memory-prior-art.md` — that's 4 of 14, ~29% of the corpus), and **none of their findings appear in any of the 27 tasks**. No memory schema task. No agent-memory write-back wiring. No retrospector hand-off. No mention of MEMORY.md updates. The §8 file list does not touch `~/.claude/agent-memory/` or any memory-system file. This is a **fundamental scope gap**. **+0.40 sub-score.**
- **6-12 hour cycle.** CLEAN. 8.75h estimated, within the 6-12h window with 2-3h headroom for the 25% overrun the planner flags. **+0.95 sub-score.**

Average ≈ 0.78. The continuous-learning gap alone drops this below 0.90 and is non-negotiable: the user named it explicitly as one of four pillars, and 4/14 of the evidence specialists worked on it.

**Failed criterion to fix on re-dispatch:** every continuous-learning evidence file (architect / cartographer / forge / historian) must produce at least one task in the implementation plan, OR the plan must explicitly defer continuous-learning to v1.2 with a written rationale.

### Dim 2 — Communication: 0.92 (PASS, threshold 0.90)

The planner output is unusually readable for an implementer:

- §1 atomic task table has all 9 columns (ID / Tag / Title / Files / Deps / Blast / Min / Rollback / Acceptance) populated for every task. No "TBD" placeholders.
- §2 dependency graph is rendered both as text DAG and as a Gantt sketch in §3 with wall-clock hour markers.
- §4 has a coverage check mapping every charter acceptance criterion to ≥1 task ID.
- §6 calls out 5 high-risk tasks with named mitigations.
- §7 lists 8 specific caveats and open questions, including the explicit refutation chain for the dropped fp64 catcher (cited to `tracer-runtime §4`).
- §8 alphabetical file list with task IDs lets the implementer scan for collisions.

Demerits:
- The continuous-learning silence noted in Dim 1 is also a communication failure — Akash, reading this, would not know the plan deliberately skipped the memory-system charter element.
- "v1.0.0rc2" is referenced 6× without a cross-link to where rc2 is defined. Akash will guess from context, which works, but a one-line "rc2 = Phase A bundle, ships independently of v1.1" near the top would help.

**Verdict:** PASS. The plan is implementable as written for everything it covers. The one structural gap (continuous-learning) is upstream of communication.

### Dim 3 — Output quality: 0.88 (PASS, threshold 0.85)

I sampled 5 tasks for `path:line` precision, citation, rollback:

| Task | path:line precise? | Source cited? | Rollback specified? |
|---|---|---|---|
| T-01 | Yes — `assertions/close.py` lines 13-19 (verified: try/except torch import block IS exactly at lines 13-19) | Yes — `detector-files.summary.md fix #1` | Yes — `git revert <sha>` |
| T-03 | Yes — `plugin.py:86-88` (verified: bare `except Exception: pass` IS at lines 86-88) | Yes — `security-postmerge.summary.md PM-2` | Yes — "revert single hunk" |
| T-19 | Yes — `fixtures/benchmark.py:283-327` plus `backends/mps.py` (file exists, range plausible at 10980 bytes) | Yes — `tracer-runtime.summary.md finding #1 + #2` | Yes — "restore deleted `_run_mps`" |
| T-24 | Yes — `assertions/tolerances.py` (file exists at 8534 bytes) + `pyproject.toml [tool.gpucheck.mps.tolerances]` | Yes — `SYNTHESIS.md §1 + empiricist-v3-extended.md §"Shape B"` | Yes — "restore single-multiplier 2×" |
| T-26 | Yes — three new files specified, all under `src/gpucheck/diagnostics/` and `fixtures/mps_safety.py` | Yes — `tracer-v3-deadlock.md sketches 1+2` | Yes — "delete new package" |

5/5 sampled tasks have file:line precision, source citation, and named rollback. Acceptance criteria are testable (e.g. T-01: `python -c "import gpucheck"` does not import torch verified by `sys.modules` check). Estimates have minute-grained budgets.

Demerits:
- T-15 acceptance criterion says "drop tests" in the rollback column instead of in the rollback column header (cosmetic — the table at line 55 is mis-aligned: minutes and rollback columns are swapped for T-15/16/17). Not a content error but it'd confuse a fast reader.
- Phase D acceptance does not specify how to verify "behavioral parity" for T-23's "bit-for-bit identical to v1.0 on cuda" — a parity test is named but not authored.

**Verdict:** PASS. The granularity and cite-density is well above the threshold. Two cosmetic rough edges, no functional gaps.

### Dim 4 — Safety: 0.65 (FAIL, threshold 0.95)

Three independent safety problems compound here:

1. **No skeptic pass ran.** This is the load-bearing failure. The skeptic's job is to find the v1.0-breaking case the planner missed. Without it, I have no independent check that any of T-01..T-27 preserves the v1.0 contract. The plan SAYS T-23 is "bit-for-bit identical to v1.0 on cuda" but no specialist has stress-tested that claim with an attack. Same for T-21's `@require_arch` deprecation, T-20's three-way detection consolidation, and T-19's `_run_mps` deletion — each of these touches public surface area that v1.0 callers depend on.
2. **No adversary pass ran.** The 14-evidence corpus has zero adversarial review. Synthesist's job is cross-cut, not attack. The planner's job is decomposition, not defense. There is currently no specialist whose deliverable is "I tried to break v1.0 callers using these tasks and here's what survives." This is exactly the role the charter assigns to the adversary file — and it never landed.
3. **The plan does include named risks (§6) but the mitigations are self-graded.** T-19's risk note says "preserve the `_FLUSH_L2_WARNED` gate" — that's the planner asserting a property; an adversary would write the test that fails before the fix and passes after. T-21's mitigation is "revert rename; restore `@require_arch` only" — a fine rollback, but no test guards the deprecation path.

Plus the structural items I can verify:

- No task touches files outside the `src/gpucheck/` and `tests/` and four root .md files. **No unauthorized files.** ✓
- No task pushes to main or to a published branch. The §0 statement is "Phase A is the v1.0.0rc2 patch bundle — every task here is independently mergeable." Mergeable to a release branch, not main. ✓
- No task spams upstream. T-23..T-26 file no PRs to PyTorch. ISS-56/ISS-57 (upstream-fileable, per synthesist) are notably **not in any task** — which is a Goal-alignment gap (§1) but a Safety win (no upstream spam). ✓

Sub-score:
- v1.0-caller breakage: unverified (no skeptic) → **0.50**.
- File scope: **1.00**.
- Branch / push hygiene: **1.00**.
- Upstream restraint: **1.00**.
- Adversarial review: absent → **0.30**.
- Average ≈ 0.65.

**Failed criterion to fix on re-dispatch:** the skeptic-plan-attack.md and adversary-corpus-attack.md MUST run before I (or any evaluator) can pass this dim. The plan as written may be perfect; I just have no instrument to verify it.

### Dim 5 — Efficiency (advisory): 0.85 (PASS, threshold 0.70)

Total budget 525 min / 8.75h, within the 6-12h target. Phase A's 230 min wall-clock is parallelizable to ~90 min wall-clock with 4-way executor pool — the planner identifies the parallel pool sizes (8/4/4/3) per phase. No padding I can identify. The 25% overrun cushion (8.75h × 1.25 = 10.9h) still fits the 12h cap.

Concerns:
- Phase A alone has 10 tasks for 230 min — at 23 min/task average this is well-paced.
- T-24 at 90 min is the largest single task; reasonable for adding a class-bucketed multiplier table with new tests.
- T-19 at 75 min is appropriate given it touches both fixtures and backends with a tracer-flagged subtle difference.

**Verdict:** PASS. The plan is right-sized for the work it specifies. Note this is advisory — efficiency is meaningless if Goal alignment fails (you can be very efficient at the wrong thing).

---

## §3. Conditions for re-dispatch (must-fix list)

Re-dispatch with these specific repairs, in order:

1. **Engineering-lead must produce `IMPLEMENTATION_PLAN_v1.1.md`** — this evaluator was asked to grade a plan that does not exist. The planner output is one specialist's task table; the implementation plan must consolidate that with the architect, the historian, the cartographer, and the empiricist's deliverables into one document the implementer reads.
2. **Skeptic must run `EVIDENCE/skeptic-plan-attack.md`** — specifically attack the 5 high-risk tasks named in planner §6 plus the silently-dropped continuous-learning charter pillar. Without skeptic, Safety stays at 0.65.
3. **Adversary must run `EVIDENCE/adversary-corpus-attack.md`** — specifically test that T-19 / T-20 / T-21 do not break v1.0 callers under realistic usage patterns (existing 117-test baseline + community examples).
4. **Plan must address the continuous-learning charter pillar** — at least one task wiring the architect / forge / historian / cartographer findings into v1.1, OR an explicit deferral with rationale referencing the user's stated "continuous-learning so Claude improves over time" goal. Currently, 4 of 14 evidence files (29% of the corpus) contributed zero tasks. That is a Goal-alignment failure.
5. **Plan must surface ISS-56 / ISS-57 (upstream-fileable PyTorch issues)** — either as a task to file the bugs upstream OR as an explicit "out of v1.1 scope, file in v1.2" line. Synthesist explicitly flagged these as Mac/Metal Goal-alignment items; the planner output is silent on them.

After those five repairs, re-dispatch for evaluation. With them, I expect the plan to clear all 5 dimensions.

---

## §4. What can ship at MEDIUM confidence today (provisional)

If the user wants to extract value from the corpus immediately while waiting on the missing artifacts:

- **Phase A bundle (T-01..T-10) at MEDIUM confidence** — these are 10 single-file low-blast fixes citing specific lines I verified exist. Even without skeptic, the blast radius is bounded (≤2) and the rollback is one-liner per task. Akash could merge these as v1.0.0rc2 with reviewer eyes and a `git revert` plan.
- **Phase B test additions (T-11..T-14, ex T-15/16/17) at MEDIUM confidence** — these only ADD tests. Cannot break v1.0 callers. The mutation-kill targets are achievable and testable.
- **Phase C refactors (T-18..T-22) MUST WAIT for skeptic** — these change public surface area. Do not ship at any confidence without an adversarial pass.
- **Phase D features (T-23..T-26) MUST WAIT for skeptic** — these add new public APIs and modify tolerance semantics. The blast radius is wider than Phase C.

---

## §5. Confidence in my own verdict

**HIGH** on the FAIL verdict.

The two load-bearing reasons are factual and not interpretation-dependent:

1. The implementation plan named in my charter does not exist on disk. I checked.
2. Two adversarial-gate files named in my charter do not exist on disk. I checked.

My hard rules say I never grade a session without a skeptic pass and I never pass a session where any strict dim is below threshold. Both rules fire here independently. There is no judgment call to make.

**MEDIUM-HIGH** on the provisional sub-scores (0.78 / 0.92 / 0.88 / 0.65 / 0.85). I treated `planner-v1.1-tasks.md` as a stand-in for the absent plan; if the real plan, when produced, includes more than just the planner's task table (e.g. integrates architect's continuous-learning), the Goal-alignment score will rise. If it does not, the score will not change.

**LOW** confidence only on which exact tasks should be added for continuous-learning — that's the architect's job, not mine. I'm flagging the gap, not prescribing the fix.
