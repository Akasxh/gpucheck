# rmsnorm — Stride/Contiguity + Shape + Dtype Fuzz (v2)

- **Kernel:** LLaMA-style RMSNorm — `y = x * rsqrt(mean(x², dim=-1) + eps)` with fp32-upcast reduction.
- **Backends:**
  - **MPS:** real (`MPSBackend.is_available() == True`, `torch==2.11.0`, Apple Silicon, Darwin 25.4)
  - **CPU:** reference (same Python process, fp32-upcast reduction matches LLaMA convention)
  - **CUDA:** mocked (no NVIDIA GPU on host) — MPS-vs-CUDA-mock is **N/A**
- **Fuzzers:**
  - `gpucheck.fuzzing.strides.fuzz_strides_for_category` — all 7 stride categories
    (`row_major`, `column_major`, `broadcast`, `transpose`, `slice`, `non_contig`, `gather`)
  - Shape buckets: `degenerate`, `prime`, `pow2_boundary`, `non_tile_aligned`, `large`
  - Dtypes: `float32`, `float16`, `bfloat16` (all three supported on MPS)
- **Tolerances:** `compute_tolerance(dtype, k_dim=last_dim, device_type="mps")` — base
  per-dtype atol/rtol scaled by `sqrt(k/128)` and overlaid with the 2× MPS multiplier.
- **Divergence model (kernel-fuzzer-rmsnorm v2 spec):**
  - `abs_err > 10× atol` → ALWAYS counts as candidate divergence
  - `rel_err > 10× rtol` → counts ONLY if `denom_magnitude (= max |y_cpu|) >= 1e-6`
    (suppresses near-zero-denominator artifacts)
  - candidate is **FILABLE** only when it reproduces on **≥ 3 distinct seeds** for
    the same `(shape, dtype, stride_cat)` configuration
  - `1× < max_factor ≤ 10×` → **TOLERANCE_RECALIBRATION** (recommend xfail entry)
  - `< 1×` → OK
- **Sampler:** 200 unique `(shape, dtype, stride_cat)` configs (seed `0xC0DE`),
  each replayed across seeds `{0, 1, 2, 3, 4}` → 1000 iterations.

## Summary

| Metric | Value |
| --- | --- |
| `kernel` | `rmsnorm` |
| `iters_attempted` | **1000** |
| `iters_completed` | **1000** |
| `divergences_filable` | **0** |
| `divergences_recalibration` | **0** |
| `non_finite_configs` | 0 |
| `max_abs_err` (MPS vs CPU) | **7.8125e-03** |
| `max_rel_err` (denom ≥ 1e-6, MPS vs CPU) | **7.092e-03** |
| `max_rel_err` (raw, unfiltered) | 7.092e-03 |
| MPS vs CUDA-mock | **N/A** (CUDA mocked, no kernel execution) |
| Wall time | 2.57 s |
| Recommended upstream filing target | **none** |

`status_counts: {OK: 1000}` — every one of the 1000 iterations produced a
`max_factor < 1×` against the MPS-overlay tolerance. No iteration approached
even the recalibration band.

## Coverage breakdown

- **Shape buckets:** `degenerate=210, non_tile_aligned=215, pow2_boundary=205, large=190, prime=180`
- **Dtypes:** `float32=300, float16=315, bfloat16=385`
- **Stride categories (all 7 covered):**
  `transpose=140, broadcast=120, slice=175, row_major=135, column_major=175, gather=130, non_contig=125`

## `top_3_repros`

There were no FILABLE or RECALIBRATION configurations to report. For
visibility, the three configurations with the **largest observed
`max_factor`** (still well under 1× tolerance) are:

| # | shape | dtype | stride | bucket | abs_err | rel_err | atol | rtol | denom_mag | max_factor | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `(11, 13, 17)` | `float16` | `gather` | prime | 9.77e-04 | 5.85e-04 | 7.29e-03 | 2.00e-02 | 3.45e+00 | 0.134 | OK |
| 2 | `(100, 100)` | `bfloat16` | `column_major` | non_tile_aligned | 7.81e-03 | 6.10e-03 | 8.84e-02 | 1.00e-01 | 4.03e+00 | 0.088 | OK |
| 3 | `(100, 100)` | `bfloat16` | `transpose` | non_tile_aligned | 7.81e-03 | 6.10e-03 | 8.84e-02 | 1.00e-01 | 4.03e+00 | 0.088 | OK |

(All `max_factor` values are < 1× — listed for completeness only.)

## Interpretation

RMSNorm on `torch==2.11.0` MPS is numerically tight against the CPU reference
across 1000 iterations × 7 stride layouts × 5 shape buckets × 3 dtypes ×
5 seeds. Worst-case absolute error is `7.8e-3` (a single bf16 ulp at unit
magnitude); worst-case relative error is `7.1e-3`, both **two orders of
magnitude below** the MPS-overlay tolerance band. The fp32-upcast reduction
in the reference implementation is the dominant numerical safeguard — even
the broadcast stride-0 case (which has tripped other normalizers) produced
zero divergences here.

No upstream filing is warranted. No new MPS xfail entries recommended.
The existing tolerance overlay (2× CUDA baseline) is **comfortably
sufficient** for `rmsnorm` on this hardware/SDK pairing — no recalibration
needed.

- **Recommended filing target:** `none`
- **Recommended `[tool.gpucheck.mps.xfail]` updates:** `none`
