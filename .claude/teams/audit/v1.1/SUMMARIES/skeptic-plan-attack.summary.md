# skeptic-plan-attack summary (W3)

**Verdict:** PASS-with-conditions (C1-C5 required to flip to clean PASS)

## 5 attacks ranked by danger

1. **Wave-1 pool exceeds documented 4-concurrent ceiling.** Plan dispatches 8 tasks in Phase A pool; research-lead/MEMORY.md L243-248 documents 4 as empirical ceiling with 529 Overloaded + silent deaths beyond. Phase A jumps from 90min → 3-4h if Wave-1 collapses to serial-of-2.

2. **Dep-graph hides 5 file-level collisions.** "Independent" tasks share files:
   - `__init__.py`: T-05/T-22/T-26
   - `assertions/close.py`: T-01/T-02/T-10
   - `backends/mps.py`: T-04/T-19
   - `arch/detection.py`: T-04/T-20/T-21
   - `pyproject.toml`: T-24/T-25
   Function-level dep graph correct; file-level merge conflicts WILL fire under 4-way parallelism.

3. **80% kill-rate arithmetic doesn't pencil out.** From mutator-survivors: ~30 LoC of new tests = 30+17+12 = 59 mutants = ~69% kill rate, not 80%. Plan needs revised public goal OR 2-3 more leverage tests.

4. **Scribe-merge bootstrap wrong scale.** 10-concurrent-on-1-lock validation ≠ 6 sibling-team scribes closing within a 60s window. 92-concurrent in BENCHMARKS_v0.2 was 92 *processes total*, not 92 lock contenders. Real failure mode: deferred-merge-by-staleness, not write-loss.

5. **No recurrence guard for refuted silent-fp64 finding.** Tracer refuted on torch 2.11; plan rejects catcher entirely + adds ZERO CI test. PyTorch has regressed MPS behavior before; users on torch 2.10 inherit original failure mode with zero gpucheck help.

## Strongest unstated assumption

**"PyTorch only changes one direction (toward stricter)."** Plan treats tracer's torch 2.11 fail-loud observation as forever-true. No version matrix, no recurrence test, no torch-pin discipline, no fp64-MPS CI cell. Other unstated assumptions are recoverable mid-flight; this one only fires when a user 6-12 months from now silently gets wrong answers on torch 2.13+.

## Conditions C1-C5

- **C1**: Wave-1 dispatch reduced 8→4 + 529 fallback
- **C2**: File-level serialization in executor harness for 5 collision sites
- **C3**: Mutmut-ratchet gate after each Phase C task (~50 min budget)
- **C4**: Standalone `scribe-merge-all` reconciler scheduled every 10 min OR documented stale-staging GC
- **C5**: **T-26b — `tests/test_mps_fp64_loud.py`** as 5-min recurrence monitor

Without C1-C5: shippable at MEDIUM with 30-50% risk. With them: HIGH.
