# skeptic-matmul-attack summary

**Verdict: REQUIRES_MORE_REPRO. DON'T FILE AS-IS. Confidence demoted from HIGH → MEDIUM.**

## 5 strongest attacks (ranked by danger)

1. **Off-shape unblock is post hoc (HIGH, LANDED).** Ran 60 sequential 1024³ fp32 calls, *no other shape touched*: spontaneous transition at iter 5. Replicated 3× — transitions at iter 10+, 5, 7. The "different shape invalidates the cache" story is contradicted — the off-shape simply adds wall-time that crosses the spontaneous-transition threshold.

2. **Two fast regimes, headline cherry-picked (HIGH, LANDED).** Warm path is bimodal: ~1.7 ms steady-state OR ~0.85 ms. The "3.7× / 0.85 ms / 2521 GFLOPs" headline is from `warmup=10, n=50` (post-second-transition). Real event-timed cold/warm is 2.64×, not 3.7×.

3. **Reproduction is flaky (MED-HIGH, LANDED).** Ran `02-repro-isolate.py` 3× back-to-back: Run 1 transitions at iter 1, Run 2 stuck full 14 iters, Run 3 transitions at iter 1. The investigation's "deterministic" claim is wrong; even investigation's own raw data shows mid-window transitions.

4. **Async JIT compile finishing mid-stream (MED, untested).** Step transition is consistent with async compile catching up at iter ~5, not a sticky cache. Resolvable only via `MTLCaptureManager` capture (still in flight).

5. **Versioning fragility (LOW-MED).** 26.4.1 may be fixed in 26.4.2; cannot test from this machine.

## Refuted attacks (cleared)

- Sync-overhead artifact (event timing also shows 2.64×; not a `synchronize()` cost story)
- Thermal throttling (`pmset -g therm` clean; fast path follows MORE GPU work, opposite of throttling)

## Required before filing

- MTLCaptureManager trace of slow-iter vs fast-iter pipeline-state IDs (proves kernel-pick vs async-compile) — **agent in flight**
- 10 fresh-process repro runs with transition-iter histogram
- Test "sleep N ms" vs "off-shape dispatch" — does pure idle also unblock?
- Reconcile two fast regimes (1.7 ms vs 0.85 ms)

## Implication for the issue body

If MTLCaptureManager shows the SAME pipeline-state on slow-iter and fast-iter, the bug claim collapses to "WARMUP=3 isn't enough for MPSGraph cold-start" — a much weaker finding, probably not worth filing as a bug. Worth a docs PR or a gpucheck README note instead.

If MTLCaptureManager shows DIFFERENT pipeline-state IDs, the kernel-pick story is vindicated AND we have a much harder filing.

**The whole filing decision now hinges on the MTL capture.**
