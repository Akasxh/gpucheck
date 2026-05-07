# skeptic.md

**Specialist**: engineering-skeptic (Phase A gate)
**Date**: 2026-05-01

## Attacks on PLAN.md

### Attack 1: "The Backend Protocol is over-engineered for v1.0"

Counter-strategy: **Drop the Protocol, just add `torch.mps.*` calls inline at every CUDA-only site.**

**Mitigation**: The Protocol is additive (existing CUDA call sites unchanged) and adds ~150 LOC. The trade-off is that v1.1's refactoring story is much cleaner because there's a target API. The Protocol also formalizes "what backend X must do for gpucheck to work", which is useful for ROCm in v1.2. **ABSORBED** — keep the Protocol, but mark it `# TODO(v1.1): migrate fixtures/benchmark.py to use Backend.event_timer` so we don't pretend it's done.

### Attack 2: "The Event API deadlock risk is theoretical; CI has no MPS"

Counter-strategy: **Just use `torch.mps.event.Event` like CUDA and trust pytorch#162872 will be fixed.**

**Mitigation**: The hang is reproducible on PyTorch 2.10+ Apple Silicon per SYNTHESIS §3 (it's a research finding). gpucheck cannot ship a benchmark fixture that hangs. Even if pytorch#162872 closes, our deadlock-safe pattern (`torch.mps.synchronize()` + wall clock) is correct under the docs' contract. The 1ms overhead from wall-clock vs CUDA event is acceptable for the precision claim. **ABSORBED — Plan is correct.**

### Attack 3: "Provisional 2x tolerance is hand-wavy and may mislead users"

Counter-strategy: **Don't ship MPS tolerance overlay; require users to set their own.**

**Mitigation**: SYNTHESIS §7 calls out exactly this risk. The plan addresses it by:
1. Marking `_MPS_TOLERANCE_MULTIPLIERS` PROVISIONAL in code comment.
2. README MPS section (Phase 3 docs) cites the calibration plan.
3. Users can override via `[tool.gpucheck.tolerances]` (existing config mechanism — no new override surface needed).

**ABSORBED — Plan is correct.** The risk is communicated, not hidden.

### Attack 4: "ContextVar fix changes API semantics — what about generators / async?"

Counter-strategy: **The plan glosses over async semantics.**

Detailed: `ContextVar.get()` follows the active execution context. In async code, `asyncio.create_task()` copies the current context, so a child task observes the override. In a generator, the override is captured at `yield` time but only if the generator is iterated within the active context. This is actually MORE correct than the global list, but the docs need to mention it.

**Mitigation**: Add a docstring note in `tolerance_context` explicitly: "Uses contextvars.ContextVar for thread/task isolation. Each thread or asyncio task observes its own override stack independently." The behavior is correct; the documentation just needs to surface it. **ABSORBED.**

### Attack 5: "Stride fuzzing's 7 categories are arbitrary; some don't compose"

Counter-strategy: **Many of these categories overlap (transpose IS a stride pattern)**. List is unprincipled.

**Mitigation**: The categories ARE adversarial — chosen because each independently exercises a different code path in PyTorch's stride handling:
- row_major: baseline (vectorized contiguous path)
- column_major: tests the ATen contiguous→strided fast path
- broadcast: stride-0 dim, exercises broadcast-aware kernels
- transpose: 2D stride permutation (different from broadcast)
- slice: regular non-unit stride (different from transpose's permutation)
- non_contig + gather: irregular stride / scatter-gather paths

These are documented in the test plan and intentionally chosen to maximize bug-finding coverage. **ABSORBED.**

### Attack 6: "What if Akash's machine has neither MPS nor CUDA?"

Counter-strategy: **The implementation skips MPS tests; we never validate the integration.**

**Mitigation**: I confirmed above (via `.venv/bin/python` `torch.backends.mps.is_available()` returns True) that this machine HAS MPS. Track-A's verifier will run pytest WITH MPS available and the Track-A integration tests should run, not skip. **ABSORBED.**

## Competing strategies considered and rejected

- **Strategy X**: Implement MPS as a "translation layer" that calls `tensor.cpu()` for every kernel and re-checks on CPU. **Rejected**: defeats the purpose of MPS testing; we want to find MPS bugs, not work around them.
- **Strategy Y**: Skip the Backend Protocol; ship 4 separate PRs (mps_detection, mps_benchmark, mps_close, mps_tolerance). **Rejected**: the 4 PRs would each touch overlapping files (`pyproject.toml`, README) and create merge churn. One coordinated track is cleaner.

## Unstated assumptions in the plan

1. **Assumption**: `torch.mps.synchronize()` is always non-blocking when no work is pending. Verified by SYNTHESIS §3 (PyTorch 2.11 docs).
2. **Assumption**: `[tool.gpucheck.mps.xfail]` config block can be parsed by gpucheck (no `tomllib` import yet). gpucheck doesn't currently parse pyproject; SYNTHESIS implies a NEW parser. **Risk**: that parser must be added in Track A. PLAN.md §A.7 calls this out — `assertions/tolerances.py` already has `tolerances_from_config(config: dict)`; we need a sister `mps_xfail_from_config`. Acceptable.
3. **Assumption**: Engineers can identify "thread-safety bug" via a fail-without-fix test (Track C). Risk: the bug is non-deterministic. **Mitigation in plan**: use a `threading.Barrier` to synchronize all threads at the moment of contention.
4. **Assumption**: `uv.lock` already exists in repo and just needs to be `git add`-ed. Verified: `git status` shows it untracked.

## Verdict

**PASS** — proceed to Phase B execution. All 6 attacks absorbed. No load-bearing flaw lacks a mitigation path.
