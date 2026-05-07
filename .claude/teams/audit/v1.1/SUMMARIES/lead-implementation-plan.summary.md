# engineering-lead IMPLEMENTATION_PLAN_v1.1 summary (W3)

**Output: `/Users/cero/Code/gpucheck/IMPLEMENTATION_PLAN_v1.1.md`** (736 lines, 49.6 KB)

## Two parallel tracks

### Track 1 — gpucheck v1.0.0rc2 + v1.1
- **27 atomic tasks across 5 phases** (A bug-fixes / B test-additions / C refactors / D R3 features / E docs)
- **Sequential 920 min, ~6h wall-clock with 4-way parallelism in Phase A + 3-way in B/C/D**
- **Phase A independently mergeable as v1.0.0rc2 within ~90 min**

### Track 2 — claude-forge v0.3 continuous-learning
- **10 ordered steps for one agent, ~5h wall-clock**
- **Anthropic memory-tool contract over Cline 6-file schema** (per historian's recommendation)
- **forge's YAML + Situation/Action/Outcome/Bounds schema**
- **Architect's 4 hooks** — only 1c (SessionEnd auto-merge) wired in v0.3 (the load-bearing one that closes the loop today). 1a/1b deferred to v0.4.
- **Migrates 22 lessons** from 5 KEEP staging files into v0.3 schema. Drops 3 stubs.

## Top-5 critical tasks

1. **T-04** — bare-except cleanup (unblocks 3 downstream)
2. **T-23** — Apple-tile fuzz patch (R3 deliverable)
3. **T-24** — MPS tolerance overlay (R3 headline; highest-blast)
4. **T-19** — eliminate `_run_mps` duplicate (deletes ~45 LOC; tracer's load-bearing finding)
5. **T2-04** — extend `session-capture.sh` to actually run scribe-merge (CLOSES THE LOOP)

## Acceptance

- v1.0.0rc2 + v1.1.0 tagged
- mutation kill ≥80% on reporting/tolerances
- lazy-import contract holds
- 4 R3 deliverables in (T-24 may slip to v1.1.1)
- linguist-v3 silent-fp64-catcher NOT shipped (refuted)
- 22 lessons migrated
- hook 1c wired
- ranker + index + pattern-extract scripts work
- 10-concurrent merge stress test passes 0-loss

## Most likely to slip

**T-24 (MPS tolerance overlay)** — 2 upstream deps (T-12 + T-18), highest blast-radius (4) of any feature. Documented fallback: ship v1.1 without it, tag T-24 as v1.1.1, keep 2× scale safe in the meantime.
