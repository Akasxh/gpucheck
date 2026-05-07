# planner-v1.1-tasks summary (W2)

**Output:** `EVIDENCE/planner-v1.1-tasks.md`
**Total:** 27 atomic tasks · 525 min · ~8.75h (within 6-12h budget)

## Hours per phase

- **Phase A** parallel bug fixes (v1.0.0rc2): 230 min wall, ~90 min with 4-way parallelism
- **Phase B** test additions → kill rate ≥80%: 140 min
- **Phase C** architectural refactors: 265 min
- **Phase D** R3 features (minus refuted catcher): 270 min
- **Phase E** final docs (T-27): 15 min

## Top-5 highest-leverage tasks

1. **T-11** — pin numeric fields in `format_mismatch_report` → kills ~30 reporting.py mutants in 30 min. Biggest single mutation-coverage move.
2. **T-23** — Apple-tile fuzzer patch (cartographer-v3 +91/-8 unified diff). Ships R3 binding deliverable; no breaking changes.
3. **T-24** — per-(kernel,dtype) MPS tolerance overlay (Shape B class-bucketed). Replaces global 2×; headline R3 finding.
4. **T-19** — eliminate `_run_mps` duplicate of `MPSBackend.event_timer` (deletes ~45 LOC duplication; tracer's load-bearing finding).
5. **T-04** — bare-except cleanup across arch/detection + backends/mps (7 sites). Unblocks T-18/19/26; closes CLAUDE.md contract violation.

## Rejections

- **linguist-v3 silent-fp64-downcast catcher dropped** — tracer-runtime §4 verified torch 2.11 already raises `TypeError` loudly on fp64-MPS. Catcher solves non-existent problem. Documented in T-26 acceptance + §7 caveat #3.

## Caveats

- Phase A's 1-2h wall-clock requires 4-way executor parallelism (single-stream ~3.5h)
- **PM-5 HTML reporter raw-attribute interpolation deferred to v1.2**
- **GPU CI gate re-enable deferred to a future T-28** pending MPS CI runners
