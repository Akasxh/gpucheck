# IMPLEMENTATION_PLAN_v1.1 — AMENDMENTS

**Companion to:** `IMPLEMENTATION_PLAN_v1.1.md` (read both)
**Source:** `EVIDENCE/skeptic-plan-attack.md` C1-C5 + `EVIDENCE/adversary-corpus-attack.md` 3 weakest-citation re-source items
**Status:** binding amendments — these merge into the main plan before execution

The skeptic ran `PASS-with-conditions` on the main plan. These 5 conditions (C1-C5) flip the gate to clean PASS. The adversary surfaced 3 corpus-quality items that need re-sourcing before the relevant tasks execute. Both sets are ratified here as task amendments.

---

## C1 — Wave-1 dispatch ceiling

**Source:** `skeptic-plan-attack.md` Attack 1; reinforced by `research-lead/MEMORY.md L243-248`.

**Defect:** Phase A pool dispatches 8 tasks in one wave; documented empirical ceiling is 4 concurrent (529 Overloaded + silent deaths beyond).

**Amendment:**
- Phase A's executor harness MUST cap concurrency at 4 simultaneous Agent dispatches.
- Add explicit fallback: if any dispatch returns a 529 status, drop to serial of 2 with 30s spacing.
- Update Phase A's wall-clock estimate from 90 min → 90-180 min depending on whether 4-way holds.

**New task:** `T-A-CAP` — pre-flight check: orchestrator confirms ≤4 in-flight Phase A Agents at all times. 5 min, blocks Phase A start.

---

## C2 — File-level collisions under "independent" tasks

**Source:** `skeptic-plan-attack.md` Attack 2.

**Defect:** Five files are touched by tasks marked independent in the dep graph — function-level deps are correct but file-level merges will conflict under 4-way parallelism.

| collision file | colliding tasks |
|---|---|
| `src/gpucheck/__init__.py` | T-05, T-22, T-26 |
| `src/gpucheck/assertions/close.py` | T-01, T-02, T-10 |
| `src/gpucheck/backends/mps.py` | T-04, T-19 |
| `src/gpucheck/arch/detection.py` | T-04, T-20, T-21 |
| `pyproject.toml` | T-24, T-25 |

**Amendment:**
- Executor harness MUST serialize within each collision group.
- Order within group is the planner's existing dep-graph order (no further changes).
- Phase A wall: tasks in the collision groups serialize, others 4-way. Empirically expect ~2.5h wall (vs 90 min ideal).

**New task:** `T-A-LOCK` — write `~/.claude/teams/audit/v1.1/COLLISION_LOCKS.md` listing the 5 file-level locks the executor harness honors. 5 min, blocks Phase A start.

---

## C3 — 80% mutation kill-rate math

**Source:** `skeptic-plan-attack.md` Attack 3; flagged also by `adversary-corpus-attack.md` weakness #2.

**Defect:** `mutator-survivors.summary.md` projects 80% kill rate from 30 LoC of new tests. Math doesn't pencil: 30+17+12 = 59 mutants killed, total tested 386 (excluding suspicious + timeout) → kill rate climbs from 169/386=43.8% to 228/386=59.1%, not 80%.

**Amendment:**
- Phase B target revised: ≥60% kill rate as "shipping bar" (achievable with current 30 LoC).
- ≥80% kill rate moved to v1.1.1 stretch goal.
- New task: `T-13a` — write 2-3 additional leverage tests targeting backends/mps.py and assertions/close.py (which together hold ~80 of the 221 surviving mutants beyond reporting/tolerances). Add 30 min to Phase B.
- New task: `T-13b` — mutmut-ratchet CI gate: a regression test that fails if kill rate drops below current measured baseline (whatever Phase B achieves). Blocks v1.1.1 from regressing.

---

## C4 — Scribe-merge bootstrap concurrency model

**Source:** `skeptic-plan-attack.md` Attack 4.

**Defect:** Engineering-scribe's 10-concurrent stress test was 10 workers contending one lock, validated 0-loss at 0.07s. Real production load is 6 sibling-team scribes that close within a 60s window — different distribution, different failure mode (deferred-merge-by-staleness, not write-loss).

**Amendment to T2-04 (`session-capture.sh` extension):**
- Add a separate cron-style reconciler: `~/.claude/scripts/scribe-merge-all.sh` runs every 10 minutes via launchd / `/loop` and walks every `staging/` file older than 5 minutes that hasn't been merged.
- The reconciler is the source of truth for "no lesson is left unmerged." The session-end hook is best-effort.
- This is an addition to T2-04, not a replacement. Add 30 min to Track 2.

---

## C5 — Refuted-finding recurrence guard

**Source:** `skeptic-plan-attack.md` Attack 5 (the most strategic of the 5).

**Defect:** Tracer-runtime refuted linguist-v3's silent-fp64-MPS-downcast claim on torch 2.11. Plan correctly drops the catcher API. But: the refutation is a torch-version observation, not a contract. PyTorch can regress.

**Amendment:**

**New task: `T-26b` — `tests/test_mps_fp64_loud.py` recurrence monitor.**

Roughly 30 lines. Verifies that on torch >=2.10, every fp64-on-MPS path (tensor construction, `.to('mps')`, `.double()`, `set_default_dtype(float64)+tensor()`, kernel-inside-`gpu_benchmark`) raises `TypeError` (not silently returns fp32). If any path fails to raise, the test fails — and gpucheck users get a loud warning that they need to enable a downcast catcher manually. 5 min wall, blocks no other task. Categorized as Phase B test addition.

```python
# tests/test_mps_fp64_loud.py — sketch
import pytest
import torch

requires_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS not available",
)

@requires_mps
@pytest.mark.parametrize("ctor", [
    lambda: torch.tensor(0.5, dtype=torch.float64, device="mps"),
    lambda: torch.tensor(0.5, dtype=torch.float64).to("mps"),
    lambda: torch.tensor(0.5).double().to("mps"),
])
def test_fp64_on_mps_raises_typeerror(ctor):
    """Recurrence guard for linguist-v3 hypothesis (refuted on torch 2.11).

    If this test passes (TypeError raised), gpucheck's choice to NOT ship a
    silent-downcast catcher is justified. If it fails (silent downcast
    returned), users must enable a manual catcher — and gpucheck itself
    needs to revisit linguist-v3's design from R3.
    """
    with pytest.raises(TypeError, match=r"float64|fp64"):
        ctor()
```

---

## Adversary corpus re-source items (3)

### A1 — Tracer probe scripts must be preserved

**Defect:** `/tmp/trace_runtime.py` and `/tmp/trace_silent_downcast.py` referenced by tracer-runtime evidence file do not exist on disk; cited line ranges unreproducible.

**Amendment to T-19:** before refactoring `_run_mps`, the executor MUST re-derive the tracer-2 baseline by re-running the trace probes. Add the regenerated probe scripts to `~/Code/gpucheck/.claude/teams/audit/v1.1/probes/` so the next session can reproduce. 15 min add to T-19.

### A2 — Mutation kill-rate is projection until verified

**Already addressed in C3 above.** Phase B reports the *measured* kill rate, not the projected 80%.

### A3 — security PM-4 version-boundary claim is uncited

**Defect:** "torch <2.1 raises RuntimeError on non-contiguous + .numpy()" lacks a PR/changelog citation.

**Amendment to T-02 (`assertions/close.py:_to_numpy` contiguous fix):** acceptance criteria adds a smoke test against torch 2.10 venv (`~/.gpucheck-pyt-2.10.0/`) confirming the fix holds across both versions. 5 min add. The "<2.1" claim is downgraded in commit message to "preventive — known to fire on older PyTorch".

---

## Citation-laundering structural risk

**Source:** `adversary-corpus-attack.md` final note.

**Defect:** Planner cites `SUMMARIES/*.md`, not `EVIDENCE/*.md`. Summaries are derivative; evidence is primary.

**Amendment for executor harness:** every executor task MUST verify its work against the EVIDENCE file (not the summary) before marking complete. Add 1 line to every task's acceptance criteria: "Cite the load-bearing finding from EVIDENCE/<file>.md, not from SUMMARIES/."

This is a process amendment, not a task amendment. No new task; ~zero added wall-clock; closes a real risk surface.

---

## Revised wall-clock estimate (post-amendment)

| track | original (lead) | post-amendment | delta |
|---|---|---|---|
| Track 1 (gpucheck) | ~5.5h | ~6.5h | +1h (C1+C2 serialization, C3 extra tests, C5 recurrence test) |
| Track 2 (claude-forge) | ~5h | ~5.5h | +0.5h (C4 reconciler) |
| **combined** (parallel) | ~5.5-7h | **~6.5-7h** | within 6-12h budget |

---

## Adoption protocol

When the executor opens this cycle:

1. Read `IMPLEMENTATION_PLAN_v1.1.md` in full
2. Read this file (`_AMENDMENTS.md`) — these supersede the main plan where they conflict
3. Read `EVIDENCE/skeptic-plan-attack.md` for the underlying reasoning
4. Read `EVIDENCE/adversary-corpus-attack.md` for the corpus-quality context
5. Honor C1-C5 + A1-A3 + the citation-laundering structural rule
6. Phase A starts only after `T-A-CAP` and `T-A-LOCK` are confirmed

If any of C1-C5 cannot be satisfied at execution time (e.g., no way to enforce 4-concurrency cap in the harness), document the deviation in the executor's `DIFF_LOG.md` before proceeding.
