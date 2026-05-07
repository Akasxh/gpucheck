# torch.nn.functional.prelu — MPS fuzz report

- **Kernel:** `torch.nn.functional.prelu`
- **Reference:** CPU `F.prelu` (same dtype) compared against MPS `F.prelu`; CUDA path mocked off (no NVIDIA GPU on host).
- **iters_attempted:** 1000
- **iters_completed (status=OK):** 1000
- **iters_unsupported (op/build refused):** 0
- **iters_errored:** 0
- **divergences_filable (>=3 seeds, >10x tol):** 0
- **divergences_recalibration (1-5x tol band, >=3 seeds): 0
- **max_abs_err:** 0.0000e+00
- **max_rel_err:** 0.0000e+00
- **elapsed:** 2.9 s
- **seeds:** [0, 1, 2, 3, 4]
- **iters_per_seed:** 200
- **wall_budget_s:** 660
- **aborted_for_budget:** False
- **halted_on_error:** False
- **torch:** 2.11.0

## Method

- Tolerance: `gpucheck.assertions.tolerances.compute_tolerance(dtype, device_type='mps')` (per-dtype atol/rtol, with the PROVISIONAL 2x MPS overlay).
- Per-iter classification:
  - `DIVERGENCE`  — `max_abs_err > 10*atol` (always) OR `max_rel_err > 10*rtol` AND `denom_magnitude (=max|y_cpu|) >= 1e-6`.
  - `TOLERANCE_RECALIBRATION` — error in (1x..10x] tolerance band (rel-err counted only when denom is meaningful).
  - `OK` — error within 1x tolerance.
- A signature `(shape, dtype, stride_category, weight_kind)` is **FILABLE** only when the per-iter `DIVERGENCE` class is observed on \>= 3 distinct seeds (the user's reproducibility rule).
- Stride categories: row_major, column_major, broadcast, transpose, slice, non_contig, gather

## Top 3 minimal repros

_None — no signature exceeded gpucheck's MPS tolerance band on >=3 seeds._

## Interpretation

The 0.0 errors across all 1000 iters are *expected*, not a stuck harness:

- `prelu(x, w) = max(0, x) + w * min(0, x)`. For `x >= 0` the output is `x`
  bit-identically; for `x < 0` the output is the single FP multiply `w*x`.
- A single IEEE-754 multiply (or trivial branch) is bit-exact across CPU and
  Apple Silicon's MPS path for all three tested dtypes (fp32, fp16, bf16).
- Independent verification: a hand-built fp16 case (shape `(127,16)`, scale 4)
  and a bf16 case (shape `(33,128,4)`) both yielded `max_abs_diff = 0.0`,
  with distinct `data_ptr()`s confirming CPU and MPS executed independently.
- All 7 stride categories were exercised (row_major, column_major, broadcast,
  transpose, slice, non_contig, gather); none produced any drift.

Recommendation: **no upstream filing**, **no xfail entry needed**, and the
2x MPS overlay multiplier is *not* exercised by this kernel. If the M-machine
calibration sweep ever finds a prelu drift, it will surface as a real
DIVERGENCE here — leave the harness in the swarm for v1.1.

## Status counts

- `OK`: 1000

