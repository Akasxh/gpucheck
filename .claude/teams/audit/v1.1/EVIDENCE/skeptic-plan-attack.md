---
specialist: research-skeptic
slug: v1.1-plan-attack
charter: adversarial review of engineering-planner v1.1 plan + architect continuous-learning + forge memory schema
inputs:
  - .claude/teams/audit/v1.1/EVIDENCE/planner-v1.1-tasks.md
  - .claude/teams/audit/v1.1/EVIDENCE/synthesist-bugs-inventory.md
  - .claude/teams/audit/v1.1/EVIDENCE/architect-continuous-learning.md
  - .claude/teams/audit/v1.1/EVIDENCE/forge-memory-schema.md
  - /Users/cero/.claude/agent-memory/research-lead/MEMORY.md (lessons L236-L248)
  - /tmp/yc-recon/claude-forge/BENCHMARKS_v0.2.md
  - /Users/cero/.claude/agents/engineering/engineering-scribe.md (canonical merge pattern)
  - source: src/gpucheck/{assertions/close.py, backends/mps.py, fixtures/benchmark.py, sanitizers/memory.py}
date: 2026-05-01
status: red-team
plan_status: AT-RISK (PASS-with-conditions)
---

# Skeptic — red-team of the v1.1 implementation plan

## Leading hypothesis (as I understand it)

The engineering-planner has decomposed the v1.1 charter into 27 atomic tasks across 4 phases (A: rc2 bug bundle ~3.8h serial / ~1.5h parallel-of-4; B: mutation-kill tests; C: refactors; D: features), with a stated total of 525 minutes (~8.75h). Phase A is independently mergeable as v1.0.0rc2. The architect's continuous-learning loop and the forge's MEMORY schema are *out of band* — they're not in the 27 tasks but they bracket every future v1.1 session. The plan rejects linguist-v3's silent-fp64-downcast catcher because tracer-runtime refuted it on torch 2.11. Confidence is HIGH on decomposition, MEDIUM on timing.

The implicit promise is: "execute these 27 tasks in dep-order with 4-way parallelism and v1.1 ships in ~8.75h wall-clock."

---

## Attack vectors

### Attack 1 — "8.75h is achievable" rests on a parallelism guarantee that isn't in evidence

**The claim under attack.** §5 says total minutes = 525 (~8.75h). §3 explicitly notes Phase A's 3.8h "is wall-clock parallel-safe in ~90min with a 4-way pool". The Gantt sketch in §3 shows W1-A/B/C/D tracks running in parallel — implying 4 concurrent executors land Phase A in ~90 min and rc2 ships in <2h.

Three load-bearing assumptions hide here:

(a) **The executor harness can sustain 4-way parallelism on a Mac for the engineering team.** Research-lead/MEMORY.md L243-248 documents the empirical ceiling: *"4 concurrent background subagents is the parallel-team empirical ceiling — running more than 4 parallel causes silent deaths via 529 Overloaded."* The plan's Wave 1 dispatches **8 tasks** in a single pool ("Pool: T-01, T-02, T-03, T-04, T-05, T-06, T-07, T-09 — run concurrently; each ≤30min"). That's 2× the documented ceiling. The plan does not cite the L243-248 lesson, does not show how it gets around it, and the Gantt explicitly draws four parallel lanes (W1-A through W1-D) — but in `engineering` mode (not `research`), parallel pools have not been load-tested at this size.

(b) **The 8 tasks in Wave-1 don't share state.** They do — see Attack 2.

(c) **Each task hits its budget.** Most tasks are 25-30 min; mutator-survivors §3 "kill rate 42.7% → ≥80%" assumes the THREE highest-leverage tests deliver the leverage cited (~30+17+12 = ~59 mutants), but these are point-estimates from a 85-of-221 sample. If actual leverage is 30-40% lower (sample noise on 38% of survivors clustered into one file), the kill-rate target slips and Phase B grows from 140 min to ~210 min.

The compound effect: if 4-way parallelism degrades to 2-way (one of the eight tasks blocks two more on a shared file), Phase A becomes ~3h instead of ~90 min. If Phase B's leverage is 30% lower than estimated, that's another +70 min. Realistic worst case: 12-13h wall-clock, comfortably outside the 6-12h CHARTER target. This is **load-bearing optimism, not documented**.

**Competing strategy.** Mark Wave 1 as a 4-way pool explicitly (T-01, T-04, T-07, T-09 in batch 1; T-02, T-03, T-05, T-06 in batch 2) per the L243-248 ceiling, and serialize within each batch's file conflicts (Attack 2). Add a "Phase A budget gate" that auto-falls-back to single-stream if 529 errors fire within 5 min. This raises Phase A's projected wall-clock to ~2h (still inside the 1-2h CHARTER if the harness is healthy, gracefully degrading to 4h if not) and is *implementable today* — it's a one-line dispatch-policy change in the planner's §3 Wave-1 description.

---

### Attack 2 — Tasks marked "independent" share state; the dep graph hides 5 collisions

**The claim under attack.** §1 marks T-01..T-07 and T-09 as "Phase A — parallel-friendly bug fixes ... independent". §2's dep graph shows them as parallel-safe. §3 says "run concurrently".

Walking the dep graph against the §8 "Files this plan touches" table reveals **at least 5 file-level collisions** that the dep-graph notation doesn't capture:

1. **`src/gpucheck/__init__.py`** is touched by T-05, T-22, T-26. T-05 (Phase A, Wave 1), T-22 (Phase C), T-26 (Phase D). Different waves, but T-05 declares no deps; if T-22 lands first in some accidentally-promoted ordering, it re-touches `__init__.py` and may break T-05's `_MutableReport → MemoryReport` rename. The plan calls this "should land after T-20 to not re-touch __init__" for T-22 specifically — but the same hazard applies between T-05 and T-22. Not flagged.

2. **`src/gpucheck/assertions/close.py`** — T-01, T-02, T-10. T-10 declares dep on T-01. T-02 declares **no** dep on T-01, but `_to_numpy` (line 22, T-02 site) is in the same file as the import-block T-01 modifies (lines 13-19). Two parallel agents editing the same file = merge conflict, not "independent". Acceptance: each is a "single hunk" but pytest cannot tell that; if executor-A applies T-01 and executor-B applies T-02 on a stale snapshot, B's diff applies cleanly only if line numbers match.

3. **`src/gpucheck/backends/mps.py`** — T-04 (bare-except cleanup at lines 99,137,141,148,191) and T-19 (`_run_mps` refactor that calls into MPSBackend.event_timer). Plan says T-19 depends on T-04. But T-04 also touches lines 99,137,141,148,191 — and `event_timer` lives at line 111, immediately between two of T-04's edit sites. If T-04's narrowed-exception change happens to swallow a different exception class than T-19 expects, the regression test for T-19 may pass on the new T-04 baseline but fail on revert. The plan never verifies this composition.

4. **`src/gpucheck/arch/detection.py`** — T-04 (bare-except cleanup), T-20 (deduplicate GPU-detection), T-21 (rename `@require_arch`). Plan dep graph: T-20 ← T-01, T-21 ← independent. Three tasks touching the same file with three different "independent" labels.

5. **`pyproject.toml`** — T-24 (per-(kernel,dtype) MPS overlay) AND T-25 (xfail registry expansion). Both add new `[tool.gpucheck.mps.*]` sections to the SAME file. Plan dep graph shows them as independent siblings, both dependent on Phase C precursors. Concurrent merges will conflict.

The dep graph as drawn is correct **at the function level**; it is **incomplete at the file level**. The planner's notion of "independent" assumes a reverter sees only the diff, not file-level merge conflicts. With 4-way parallelism this assumption breaks.

**Competing strategy.** Add a "files-touched serialization" pass to the executor harness: build a {file → tasks-touching-it} map at dispatch time, and for any file with ≥2 tasks queued, serialize them within their wave even if their dep graph says they're independent. For pyproject.toml specifically (used by T-24 + T-25), commit-mode serialization is the safer bet (T-24 lands first, T-25 rebases on top). This is a 30-LOC change to whatever harness already exists; it does not require new infrastructure.

---

### Attack 3 — "All tests pass" is mutation-blind; Phase B's own success criteria don't apply to Phase A & C

**The claim under attack.** §4 acceptance criteria coverage table shows "Mutation kill rate 42.7% → ≥80%" mapped to T-11, T-12, T-13, T-14 (Phase B). §1's per-task acceptance for Phase A/C tasks reads as "existing tests stay green / `mypy strict` green / `pytest -q` green" — none of them require mutation re-run.

Mutator-survivors §"Per-category counts" identifies that **30 mutants are TEST_BUGs** (substring matches like `pytest.raises(match="NaN")` accepting `XXNaNXX`) and **165 are REAL_GAPs**. The Phase A tasks LANDED CHANGES INTO files that mutator-survivors classified — `assertions/close.py` (T-01, T-02, T-10), `backends/mps.py` (T-04, T-19), `arch/detection.py` (T-04, T-20, T-21) — but the only mutation-test acceptance is in T-11/12/13/14, and those test SUITES, not the post-refactor source. After T-18 (unify `compute_tolerance`), the surviving mutants in `assertions/tolerances.py` change identity. T-12+T-13's tests were designed against the pre-refactor file; nothing in the plan re-runs mutmut against the *post-T-18* baseline.

Concretely: T-18's acceptance says `pytest -q green; mypy strict green`. But T-18 unifies two different `compute_tolerance` signatures into one. The renamed `compute_tolerance_arch_aware` may have **different mutmut survivors** than the old `tensor_cores.compute_tolerance`, because the two functions diverged at line 96 (tensor_cores) vs line 70 (assertions). The kill rate post-T-18 is unknown; it could *drop below 42.7%* if the consolidation introduces equivalent mutants on the merged signature.

This is **load-bearing optimism**: the plan treats "kill rate ≥80%" as a Phase-B-only goal, but Phase C structural refactors invalidate the substrate Phase B was measured against.

**Competing strategy.** Add a single explicit gate after each Phase C task that touches a Phase B target file: re-run `mutmut run --paths-to-mutate=<changed-file>` and require kill rate ≥ pre-Phase-C rate (whatever it was, no slippage). This adds ~10 min per Phase C task (5 tasks × 10 min = +50 min, fits inside the 25% overrun envelope §9 already concedes). Implementation: a shell snippet `scripts/mutmut-ratchet.sh <file>` that diffs against `mutmut-baseline.txt` and blocks merge on regression. Mutator-survivors already gives the per-file numbers needed to seed the baseline.

---

### Attack 4 — The architect's auto-merge hook is bootstrap-blocked, and "92-concurrent on a Mac" was measured under conditions that don't match the SessionEnd hook

**The claim under attack.** Architect §1c says SessionEnd will *automatically run scribe-merge* under flock+atomic-rename. Architect cites engineering-scribe's "10 concurrent at 0.07s" validation. The user's prompt asserts "92-concurrent on a single Mac" as the real workload. Forge §8 inherits the same pattern verbatim.

Three load-bearing assumptions, none of them tested at the actual workload scale:

(a) **The 10-concurrent / 0.07s benchmark was measured on `engineering-lead` only.** Engineering-scribe.md L60: *"empirically validated (10 concurrent scribes, 0.07s total, zero lost writes, zero duplicates)."* This was for ONE lead's MEMORY.md. The architect plans to deploy this to **7 leads** (research, engineering, security, testing, docs, forge, research-retrospector), each with their own lock. That changes the workload character: 7 separate locks, not 7 contenders for one lock. If the SessionEnd hook is "always run scribe-merge, branching on team-vs-adhoc-vs-trivial", then a *single concurrent session* may emit ≥6 staging files (one per dispatched team) and the hook now runs 6 merges back-to-back, each one taking 0.07s ⇒ 0.42s, which the architect's ≤500ms latency budget for SessionStart (NOT SessionEnd, but referenced) was never sized for.

(b) **The "92 concurrent" claim from BENCHMARKS_v0.2.md is conflated.** That number measured `1 main + 6 leads + 67 specialists + ~20 swarm + monitor + mutmut` — i.e., 92 *Claude processes* at peak. It was NOT 92 concurrent *flock contenders on a single MEMORY.md*. The 67 specialists each write to their **own EVIDENCE** files in **their team's slug subdirectory**, not to ANY MEMORY.md. The bottleneck the architect's hook actually creates is at the 6-7 lead-level locks, peak ~6 contenders on the same lock when 6 sibling teams close in the same minute. That's the workload to test, and **it has not been tested**. The "10 concurrent" benchmark validated a *different load pattern* (10 fake scribes hammering one lock in a controlled test) than the production load (6 sibling-team scribes closing within a 60-second window of each other).

(c) **The flock+atomic-rename protocol's failure mode in the test was "deferred merge"** (architect §FM-6), not "lost write". The `|| { ... exit 0 }` outer handler turns lock contention into "staging file stays put for next session". That's safe only if **next session actually runs scribe-merge**. The architect ALSO says (§1c, "Critical change vs today") that the current Stop hook never runs scribe-merge; the v0.3 hook will. But the deferred-merge fallback assumes a future session of the same lead will *also* run the hook, fire scribe-merge, and pick up the staging file. If a user runs `claude -p "quick question"` in a chat-only mode (Stop hook fires but the §1c "trivial session: skip" branch triggers), the deferred merge sits indefinitely. There's no GC for stale staging files.

The race the user's prompt asks about ("what if flock+atomic-rename has a race the test missed?") — the most plausible one is *not* a write-loss race, it's a **deadlock-by-staleness**: a staging file written under harmful_count semantics never gets merged because every session it's there is a "trivial session" that skips the merge. Lesson rot then becomes worse, because the lesson was *already* harmful enough to need archiving but the harm signal can't propagate.

**Competing strategy.** Add a **standalone, idempotent `scribe-merge-all` script** that scans every `~/.claude/agent-memory/*-lead/staging/` directory and runs the canonical merge for any directory with non-`_merged` files. Schedule it via launchd every 10 minutes regardless of session activity. This converts "merge happens at session close" into "merge is reconciled continuously"; the SessionEnd hook becomes a latency-optimization (merge while it's hot), not a correctness mechanism. Implementation: ~30 LOC of shell wrapping the existing canonical pattern; `launchctl bootstrap` + plist. This addresses (c) directly and (a)/(b) by decoupling the merge throughput from the session-close burst.

---

### Attack 5 — REFUTED-but-might-recur findings have no recurrence monitor; "rejection" is not "deletion"

**The claim under attack.** §0.4: *"the silent-fp64 downcast catcher is dropped — tracer-runtime §4 verified torch 2.11 raises TypeError ... so the catcher solves a problem that no longer exists."* §7.3 hedges: *"Possible reopen path in v1.2: if a user reports the bug on torch <2.11, re-add the catcher behind a torch-version gate."*

Three concerns:

(a) **PyTorch could regress the fix.** Tracer's evidence is one version (2.11.0). PyTorch has previously regressed user-visible MPS behavior (the `torch.mps.synchronize()` calling-conventions changed in 2.7→2.8 IIRC). The plan stores the refutation in T-26's caveats but creates no automated probe that fires on torch upgrades. If torch 2.12 or 2.13 reintroduces silent fp64 conversion (regression, accidental fast-path, or compile-mode optimization), gpucheck users have **no signal** — the catcher was deleted, the test was never written, and no CI matrix tests fp64-on-MPS for fail-loud-vs-silent.

(b) **The refutation is a single observation.** Tracer-runtime §4 exhibits four cases that all raise TypeError. None of them tests `torch.compile(fn)` where the autograd graph might lower a fp64 op silently. None tests `torch.jit.trace`. None tests the `to_metal` private path. The refutation is "the obvious paths fail loud"; the original linguist-v3 finding may have been about a non-obvious path the refutation never probed.

(c) **"REFUTED" with hedged language is not the same as "no risk".** Tracer's own summary: *"May be different on torch <2.11."* The plan's gpucheck Python-pin is `>=3.10` and torch is unpinned. A user installing gpucheck on torch 2.10 would inherit linguist-v3's hypothesized failure mode with **zero help from gpucheck** — the catcher that would have caught it was rejected.

The unstated assumption is: **"PyTorch only changes one direction (toward stricter)"**. Empirically false in the MPS ecosystem; the plan has zero recurrence guard.

**Competing strategy.** Add **T-26b** (5-min task, blast=1): a single test in `tests/test_mps_fp64_loud.py` that constructs `torch.tensor(0.5, device='mps', dtype=torch.float64)` and asserts a `TypeError` is raised. **This test is the recurrence monitor.** If a future torch version makes the path silent, the test fails on the next CI run and the team rebuilds the catcher behind a torch-version gate (per §7.3 reopen path). Cost: 5 lines of test code + 1 xfail-on-torch-version-out-of-range marker. Benefit: regression detection moves from "user reports a bug 6 months later" to "CI fails the day torch ships the change". This is implementable today, fits in Phase B (next to T-15/16/17 for v1.1 features), and has zero cost when the refutation holds.

---

## Unstated assumptions in the current synthesis

1. **The 4-way executor parallelism in Phase A is supported by the harness.** Consequence if false: Phase A wall-clock = 3.8h, not 90 min; rc2 ships in 4h not 2h; Phase B starts 2h late; total v1.1 wall-clock 11-13h.

2. **Mutmut leverage estimates from a 85-of-221 sample generalize to the full 221-mutant pool.** Consequence if false: ~30 LoC of new tests yields kill rate of 60-70% instead of 80%; Phase B exceeds budget by 50% to chase the long tail.

3. **Phase C refactors don't invalidate the Phase B mutation baseline.** Consequence if false (and it almost certainly is, see Attack 3): the v1.1 release ships with `tolerances.py` post-T-18 having a different and possibly worse mutation profile than the pre-refactor target; the "≥80% kill rate" claim in CHANGELOG/marketing is technically untrue.

4. **`pyproject.toml` writes are append-safe.** Consequence if false (T-24 + T-25 both write new `[tool.gpucheck.mps.*]` sections): merge conflict in CI; one PR has to rebase; possibly invalidates the cross-task acceptance criterion.

5. **The architect's "auto-merge at SessionEnd" hook will actually run for every lead.** Consequence if false (and the SessionEnd hook has a "trivial session: skip" branch that defines "trivial" loosely): scribe-merge runs only for "team sessions"; adhoc and chat-only sessions never trigger merges; staging files for the 6 silent leads (security, testing, docs) accumulate without being merged. Architect §FM-6 mitigates write-races but says nothing about merge-skip-races.

6. **The user's "smartest-guy active" mode is the operational mode being optimized for.** Architect §3 routes adhoc-session memory to a (not-yet-existing) `general-lead/MEMORY.md` (§Open question 2) — but right now there's no such directory. Consequence if false: every chat-only session emits a staging file the system has no place to put.

7. **`engineering-scribe`'s validated 10-concurrent benchmark generalizes to the actual production case.** See Attack 4 for the full unpacking. The validation conditions (all 10 contenders hammering one lock in a controlled timing test) don't match the SessionEnd reality (6 sibling-team scribes closing within an indeterminate window, each holding the lock for 0.07s + scribe-internal work that may take longer).

8. **REFUTED findings stay refuted across torch versions.** See Attack 5. The plan has no recurrence guard.

---

## Evidence quality audit

Selected high-leverage claims, audited:

- **"525 minutes total, 8.75h"** — backed by sum of per-task minutes in §1. **Weak because**: per-task minutes are planner estimates with no cited historical hit rate. §9 concedes "MEDIUM on timing estimates. Most tasks have ≤30min budgets; a 25% overrun across the board would push total to ~11h". This is the planner's own acknowledged risk, but it's stated in a closing footnote, not in the headline §0 scope. **Stronger evidence would look like**: a 5-task historical sample from prior gpucheck v1.0 commits with measured `git diff --stat → time-to-first-green-CI` deltas, and a regression model extrapolating to the v1.1 task pool.

- **"4-way executor parallelism collapses Phase A's 3.8h to ~90 min"** — backed by Gantt sketch in §3. **Weak because**: Gantt is drawn assuming linear tasks at 25-30 min each. Real parallel work has Amdahl's-law floor (file-conflict serialization, Attack 2) plus harness overhead (529 backoff, Attack 1). **Stronger evidence**: a dry-run of 4 sibling Claude sessions hitting the gpucheck repo simultaneously with synthetic 25-min tasks, reporting wall-clock vs serial. The 4-team sibling test pattern from research-lead/MEMORY.md L250-256 ("Dogfood the design session against its own running sibling sessions") is exactly this experiment, and it has been run before. The plan does not cite it.

- **"42.7% → ≥80% mutation kill rate from ~30 LoC of new tests"** — backed by mutator-survivors §"Top-3 highest-leverage new tests" estimating 30+17+12 = ~59 mutants. **Weak because**: 59/(221-94) = 46.5% of CURRENT survivors. Adding 46.5% × (1-0.427) = ~26.6 percentage points to the kill rate gets us to 0.427 + 0.266 = ~69%, not 80%. The plan's arithmetic is off unless additional mutants are killed by the bug fixes themselves (T-01..T-10) — which is plausible but **uncited**. **Stronger evidence**: a written derivation showing each of T-11/12/13/14's mutmut output and a projected kill rate based on those numbers. This is one mutmut run away.

- **"Engineering-scribe's flock pattern is empirically validated 10-concurrent at 0.07s zero-loss"** — backed by `engineering-scribe.md` L60. **Weak because**: single-source claim with no public test artifact. The validation note says "validated in the engineering-team-self-evolve-v1 session" but no script or output is preserved. **Stronger evidence**: a checked-in `tests/test_scribe_concurrency.sh` that re-runs the 10-concurrent test in CI, with a 92-concurrent stress test added per the user's actual workload. (See Attack 4.)

- **"REFUTED on torch 2.11"** — backed by tracer-runtime §4. **Weak because**: single torch version, single test mode (eager), no torch.compile / jit.trace probe. **Stronger evidence**: a small CI matrix that runs the four tracer §4 cases on torch {2.7, 2.8, 2.9, 2.10, 2.11, nightly}, asserting fail-loud everywhere. Cost: ~30 min of CI time per release. (See Attack 5's competing strategy.)

- **"PyTorch CPU has no half-precision GEMM on Apple Silicon"** (ISS-57, used by empiricist-mac-bench to justify "MPS-MUST-NOT-fall-back-to-CPU on fp16/bf16") — backed by empiricist-mac-bench §"Three signal items" #1. **Weak because**: empiricist's own §"Process notes" caught a 25× allocation-vs-kernel timing error mid-run; the fp16/bf16 measurement was done after the lambda-factory fix, but **the same fix wasn't re-verified for the fp16 conv2d "throws on CPU" claim**. If that throw was a side effect of the original mismeasurement, the entire ISS-57 documentation note is wrong. **Stronger evidence**: a 5-line repro of `torch.nn.functional.conv2d(fp16_tensor, fp16_weight)` on CPU, run in an isolated subprocess outside the benchmark harness.

---

## Verdict

- **Prematurely converged?** YES on three axes (Phase A parallelism feasibility, mutation kill-rate arithmetic, recurrence-monitoring for refuted findings). NO on the bug-finding work itself — the synthesist's 59 issues and the per-task acceptance criteria are well-grounded.

- **Safe to raise plan confidence to "high"?** NO. The plan is internally consistent and the source-line citations are verifiable, but at least 5 unstated assumptions are load-bearing for the 6-12h CHARTER target. Specifically:
  1. The "8 tasks in one Wave-1 pool" plan exceeds the documented 4-concurrent ceiling and silently risks rc2's 1-2h aspiration.
  2. The dep graph is correct at function level but incomplete at file level; pyproject.toml and 4 other files have unflagged collisions.
  3. The 80% kill-rate arithmetic doesn't pencil out from the cited mutator-survivors numbers (it computes to ~69%).
  4. The auto-merge hook conflates a 10-concurrent-on-one-lock benchmark with a 6-leads-closing-near-simultaneously workload that has not been tested.
  5. The REFUTED silent-fp64 finding has no recurrence guard.

- **Required next probes before "high":**
  1. **Probe 1 (research-empiricist)**: dispatch a 4-sibling-Claude-session dry-run on a Mac, executing 8 fake 25-min tasks, measuring wall-clock parallelism degradation. Verifies Attack 1's harness assumption.
  2. **Probe 2 (research-empiricist)**: run mutmut against the post-T-18 hypothetical baseline (apply T-18 in a scratch branch, re-run mutmut on tolerances.py) to confirm the 80% target is reachable from 30 LoC of tests, OR adjust the headline claim. Verifies Attack 3.
  3. **Probe 3 (engineering-scribe-team)**: run a 6-sibling-scribe stress test (one per lead) hitting their respective MEMORY.md locks within a 60-second window, measuring deferred-merge rate and end-state staging-file accumulation. Verifies Attack 4.
  4. **Probe 4 (research-historian)**: verify whether torch versions 2.7-2.10 (i.e., gpucheck's likely user range) actually fail-loud on fp64-MPS, or whether linguist-v3's original hypothesis was correct on those versions. Verifies Attack 5.
  5. **Probe 5 (engineering-planner-revision)**: re-derive the kill-rate arithmetic explicitly. If the answer is ~69%, either set the public goal to 70% or expand Phase B to add 2-3 more leverage tests.

**Gate verdict: PASS-with-conditions.** The plan ships v1.0.0rc2 (Phase A) safely *if and only if* the harness sustains 4-way parallelism without 529 throttling. The plan ships v1.1 (Phase C+D) safely *if and only if* file-level collisions are serialized and Phase C's mutation-baseline regression is gated. The architect's continuous-learning system is **AT-RISK** until the standalone scribe-merge-all reconciler (Attack 4 competing strategy) is in place, because the SessionEnd-only merge has a deadlock-by-staleness mode that the test never explored.

**Conditions to flip to PASS:**
- (C1) Reduce Wave-1 dispatch from 8 to 4 concurrent tasks AND add 529-backoff fallback.
- (C2) Add file-level serialization to the executor harness for the 5 collision sites identified in Attack 2.
- (C3) Add mutation-ratchet gate after each Phase C task (Attack 3 competing strategy, ~50 min budget).
- (C4) Add the standalone `scribe-merge-all` reconciler to architect's plan, scheduled every 10 min, OR explicitly accept "deferred-merge until next same-lead session" as the documented behavior with a stale-staging GC.
- (C5) Add T-26b (`tests/test_mps_fp64_loud.py`) as a 5-min recurrence monitor for the rejected linguist-v3 finding.

If C1-C5 land, the plan is shippable at HIGH confidence. Without them, the plan is shippable at MEDIUM with a real risk of the 8.75h estimate being wrong by 30-50%.

---

## Notes for research-adversary (corpus-quality, not reasoning)

(Per skeptic-vs-adversary scope split, these are not in my attacks above; flagged for the adversary in OPEN_QUESTIONS.md.)

1. The "92 concurrent claude processes" claim from `/tmp/yc-recon/claude-forge/BENCHMARKS_v0.2.md` is a single-source, in-house measurement on Akash's machine. The adversary should question whether that benchmark's measurement methodology (cache-read accounting, monitor-spawn cost) is reproducible.

2. The `engineering-scribe.md` "10 concurrent at 0.07s" validation cites `engineering-team-self-evolve-v1` as the source session, but the actual test artifact is not preserved — only the prose claim. The adversary should question whether this is citation-laundering (a claim referenced enough to feel verified, never re-tested).

3. The mutator-survivors classification is "85 of 221 classified; rest extrapolated by clustering on same source lines". The 38% / 14% / 7% / 4% / 1% percentages may be an extrapolation artifact rather than an empirical distribution. The adversary should question whether the kill-rate arithmetic depends on that extrapolation.

## Verdict

PASS-with-conditions (C1-C5 above). Confidence: HIGH on the attack vectors, MEDIUM on the magnitude of each gap (Probe 1-5 will calibrate).
