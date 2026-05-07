# RoPE Fuzz Campaign — rope

- **Status:** OK
- **Backend:** MPS (Apple Silicon, real)  +  CUDA (mocked — no NVIDIA GPU)
- **Iterations attempted:** 250 / target 250
- **Iterations completed:** 250
- **Unsupported (MPS NotImplemented / RuntimeError):** 0
- **Hard errors:** 0
- **Divergences (MPS vs CPU):** 0
- **MPS-vs-CPU max relative error (overall):** 1.600e+01
- **MPS-vs-CUDA-mock max relative error:** N/A (CUDA detection mocked, no kernel run)
- **Elapsed:** 5.7s
- **Recommended upstream filing target:** none

## Method

RoPE applied to (B, S, H, D) inputs. Reference path runs on CPU in the
same dtype as the MPS path (cos/sin built from fp32 then cast). Both
paths share an RNG seed per iteration so the underlying values match.
Tolerance comes from `gpucheck.assertions.tolerances.compute_tolerance`
with `device_type="mps"` (the MPS overlay multiplier is applied).
RoPE is element-wise — no `sqrt(k/128)` scaling is applied.

Stride categories sampled per iteration: row_major, column_major, broadcast, transpose, slice, non_contig.
Dtypes sampled: float32, float16, bfloat16.
Shapes drawn from `gpucheck.fuzzing.fuzz_shapes` for `head_dim`
(snapped to the nearest even integer ≥ 2; RoPE precondition).

## Top divergences (minimal repros)

_None — every (shape, dtype, stride) combination stayed within tolerance._

## Notes

- CUDA backend was not exercised — no NVIDIA GPU is present on this
  Mac, and the swarm task explicitly said to mock CUDA detection. The
  CUDA-vs-MPS comparison is therefore reported as N/A rather than
  fabricated.
- Tolerances follow gpucheck's MPS overlay (PROVISIONAL — see
  `assertions/tolerances.py:_MPS_TOLERANCE_MULTIPLIERS`).
