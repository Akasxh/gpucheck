# reviewer-B.md

**Branch**: `feat/track-b-strides` @ `4ede763`

## Stage 1 — Spec compliance: PASS

All 7 categories present, each with documented contract. `parametrize_gpu` integration added without breaking the no-strides path.

## Stage 2 — Quality

### Strengths
- The 7 categories were chosen for kernel-dispatch coverage, not arbitrary lookalikes (skeptic §5 absorbed).
- Hypothesis `StrideStrategy` shrinks toward `row_major` because it's first in the sampled_from list — a reasonable shrink target that mimics `ShapeStrategy`'s pattern.
- 4-arg vs 3-arg `skip` callbacks tolerated via try/except — graceful for users upgrading from pre-v1.0.

### Concerns
- **NIT**: `_gather` returns a *contiguous* tensor that was *built* via gather. The docstring documents this; users wanting an actual non-contiguous gather view can call `src[idx]` themselves. Acceptable for v1.0.
- **NIT**: `non_contig` for ndim ≥ 3 falls back to a transpose-like permutation. This duplicates the transpose category somewhat. Future v1.1 can introduce a richer "permute(2, 0, 1)" path; not load-bearing for v1.0.

## Verdict: APPROVED.
