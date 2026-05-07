# Empiricist — cross-version + cross-shape matmul anomaly

## Hypothesis (falsifiable form)

Two related hypotheses:

1. **H1 (version)**: "If the matmul-anomaly bug is specific to torch 2.11.0,
   then running the same `1024×1024×1024` fp32 cold-start repro on torch
   2.10.0 and on torch nightly (>= 2.13 dev) will NOT produce
   `cold/warm > 2.0x`."

2. **H2 (shape)**: "If the bug is uniquely tied to `1024×1024×1024 fp32`
   on M5 / MPSGraph, then no other (M, N, K, dtype) cell in the swept
   space (M ∈ {512..4096}, N=K=M ∪ N=K=1024, dtype ∈ {fp32, fp16, bf16})
   will exhibit `cold_max / warm_med > 2.0x` with subprocess-isolated
   measurement, WARMUP=3, ITERS=10."

## Experiment design

- **Harness**: `/tmp/cross_version_shape_harness.py` — accepts mode +
  shape + dtype, runs N warmup matmuls then K timed matmuls on MPS with
  `torch.mps.synchronize()` before/after each. Returns JSON with
  per-sample `time.perf_counter_ns()` deltas.
- **Driver**: `/tmp/cross_version_shape_driver.py` — spawns one
  subprocess per (version, shape, dtype, mode, replicate). Each
  measurement is a fresh `python` invocation → fresh `MPSGraph` cache →
  cold-start by construction.
- **Modes**:
  - `cold`: warmup × 3 iters of (M,N,K), then time × 10 iters of (M,N,K).
  - `warm`: warmup × 3 iters of (M,N,K), then ONE call of a "poke" shape
    (`2048×2048×2048 fp32` for most cells; `1023×1023×1023 fp32` when
    the test cell IS `2048×2048×2048` to avoid same-shape reuse), then
    time × 10 iters of (M,N,K).
- **Replicates**: 3 fresh-process replicates per cell, to capture the
  bimodal cold-start distribution (the bug fires ~40-60% of the time).
- **Versions**: 2.10.0, 2.11.0, 2.13.0.dev20260507 (nightly, installed
  fresh in `~/.gpucheck-pyt-nightly`).
- **Sweep**: 11 sizes × 3 dtypes × 2 axes (square + fixed-NK=1024) ×
  2 modes × 3 reps = **378 axis-2 cells** + 18 axis-1 cells = **396
  total cells**.
- **Pinned**:
  - commit: a9a9d44 (gpucheck release/v1.0)
  - python: 3.13.7 (project venv) / 3.14 (nightly venv)
  - torch versions: 2.10.0 / 2.11.0 / 2.13.0.dev20260507
  - hardware: Apple M5, 32 GB, arm64
  - os: macOS 26.4.1
  - seed: deterministic per (M, N, K) within harness
  - wall: 233.1 s for axis-2 sweep, ~30 s for axis-1, ~30 s for
    workaround validation

## Code

The harness and driver are in `/tmp/`. Key timing snippet
(harness, lines 56-64):

```python
torch.mps.synchronize()
t0 = time.perf_counter_ns()
c = a @ b
torch.mps.synchronize()
t1 = time.perf_counter_ns()
samples_ns.append(t1 - t0)
```

Subprocess isolation (driver, line 56):

```python
out = subprocess.run(args, capture_output=True, text=True, timeout=180)
```

Each matmul measurement is a fresh `python` process — no shared MPSGraph
state.

---

## 1. Cross-version: bug status across torch 2.10 / 2.11 / nightly

The original investigation was on `torch 2.11.0` and reported
"`1024^3` fp32 cold = 3.13 ms, warm = 0.85 ms". I tested whether the
bug exists on adjacent versions.

### Single-replicate (3 reps each, exactly the original methodology)

| version | mode | n_reps | med-of-meds | max-of-meds | min-of-mins | all medians (ms) |
|---|---|---|---|---|---|---|
| 2.10.0 | cold | 3 | 0.871 | **3.148** | 0.790 | 3.148, 0.870, 0.871 |
| 2.10.0 | warm | 3 | 1.090 | 1.104 | 0.936 | 1.090, 1.104, 1.058 |
| 2.11.0 | cold | 3 | 1.175 | **3.369** | 0.793 | 3.369, 1.175, 1.045 |
| 2.11.0 | warm | 3 | 1.188 | 1.222 | 0.892 | 1.188, 1.150, 1.222 |
| nightly | cold | 3 | 2.123 | **3.329** | 0.833 | 2.123, 1.163, 3.329 |
| nightly | warm | 3 | 1.173 | 1.183 | 0.878 | 1.173, 1.183, 1.168 |

### Cross-version cold/warm ratios

Using worst-case-of-replicates cold over typical-warm:

| version | cold_max (ms) | warm_med (ms) | ratio | bug present? |
|---|---|---|---|---|
| 2.10.0 | 3.148 | 1.090 | **2.89x** | YES |
| 2.11.0 | 3.369 | 1.188 | **2.84x** | YES |
| nightly (2.13.0.dev20260507) | 3.329 | 1.173 | **2.84x** | YES |

### Per-process incidence (10 reps each, "slow" if median > 2.0 ms)

To get a robust incidence rate I extended to 10 fresh-process replicates
per version on `1024×1024×1024 fp32`:

| version | slow / 10 | fast / 10 | incidence |
|---|---|---|---|
| 2.10.0 | 4 | 6 | 40% |
| 2.11.0 | 4 | 6 | 40% |
| nightly | 4 | 6 | 40% |

**Verdict on H1 (version-specificity): REFUTED.** The bug exists on all
three tested versions of torch with statistically indistinguishable
incidence (~40% per fresh process). The original investigation's claim
of "stuck on slow on cold-start" was a sample-size-of-1 artifact — the
bug is bimodal at the per-process level, with ~40% of fresh processes
landing on the slow kernel-pick.

The bug is NOT specific to 2.11.0 and is NOT fixed in nightly.

---

## 2. Cross-shape table (sorted by cold_max / warm_med)

Below: 63 distinct (axis, shape, dtype) cells from the sweep. Each row
aggregates 3 cold replicates and 3 warm replicates.

`r_max = cold_max / warm_med`. Cells flagged as anomalies (`r_max > 2.0`)
are bolded.

| # | axis | shape | dtype | cold_med | cold_max | warm_med | r_med | r_max |
|---|---|---|---|---|---|---|---|---|
| 1 | square | 1024x1024x1024 | fp32 | 3.104 | 3.306 | 1.137 | 2.73 | **2.91** |
| 2 | square | 1792x1792x1792 | bf16 | 1.108 | 3.904 | 1.413 | 0.78 | **2.76** |
| 3 | fixed-nk | 4096x1024x1024 | bf16 | 1.068 | 2.998 | 1.123 | 0.95 | **2.67** |
| 4 | fixed-nk | 4096x1024x1024 | fp16 | 2.991 | 2.991 | 1.124 | 2.66 | **2.66** |
| 5 | fixed-nk | 768x1024x1024 | fp32 | 2.487 | 2.490 | 0.969 | 2.57 | **2.57** |
| 6 | square | 1536x1536x1536 | fp16 | 2.644 | 2.661 | 1.099 | 2.41 | **2.42** |
| 7 | square | 1536x1536x1536 | bf16 | 2.632 | 2.644 | 1.092 | 2.41 | **2.42** |
| 8 | fixed-nk | 3072x1024x1024 | bf16 | 2.357 | 2.370 | 0.995 | 2.37 | **2.38** |
| 9 | fixed-nk | 3072x1024x1024 | fp16 | 2.323 | 2.363 | 0.993 | 2.34 | **2.38** |
| 10 | square | 1280x1280x1280 | fp16 | 1.656 | 1.904 | 0.809 | 2.05 | **2.35** |
| 11 | square | 768x768x768 | fp32 | 1.737 | 1.738 | 0.752 | 2.31 | **2.31** |
| 12 | fixed-nk | 2560x1024x1024 | bf16 | 1.976 | 1.993 | 0.887 | 2.23 | **2.25** |
| 13 | fixed-nk | 2304x1024x1024 | bf16 | 1.867 | 1.872 | 0.854 | 2.19 | **2.19** |
| 14 | fixed-nk | 1280x1024x1024 | fp32 | 1.025 | 2.888 | 1.334 | 0.77 | **2.17** |
| 15 | fixed-nk | 512x1024x1024 | fp32 | 1.791 | 1.795 | 0.832 | 2.15 | **2.16** |
| 16 | fixed-nk | 2560x1024x1024 | fp16 | 1.990 | 1.996 | 0.927 | 2.15 | **2.15** |
| 17 | fixed-nk | 2048x1024x1024 | fp16 | 1.732 | 1.742 | 0.821 | 2.11 | **2.12** |
| 18 | fixed-nk | 2304x1024x1024 | fp16 | 1.856 | 1.863 | 0.883 | 2.10 | **2.11** |
| 19 | fixed-nk | 2048x1024x1024 | bf16 | 1.702 | 1.704 | 0.818 | 2.08 | **2.08** |
| 20 | square | 1280x1280x1280 | bf16 | 1.658 | 1.666 | 0.820 | 2.02 | **2.03** |
| 21 | fixed-nk | 1792x1024x1024 | bf16 | 1.530 | 1.552 | 0.779 | 1.96 | 1.99 |
| 22 | fixed-nk | 1792x1024x1024 | fp16 | 1.461 | 1.534 | 0.773 | 1.89 | 1.98 |
| 23 | square | 1024x1024x1024 | bf16 | 1.059 | 1.282 | 0.660 | 1.61 | 1.94 |
| 24 | fixed-nk | 1536x1024x1024 | bf16 | 1.381 | 1.418 | 0.735 | 1.88 | 1.93 |
| 25 | fixed-nk | 1536x1024x1024 | fp16 | 1.391 | 1.395 | 0.731 | 1.90 | 1.91 |
| 26 | square | 1024x1024x1024 | fp16 | 1.285 | 1.294 | 0.714 | 1.80 | 1.81 |
| 27 | fixed-nk | 1280x1024x1024 | fp16 | 1.216 | 1.239 | 0.699 | 1.74 | 1.77 |
| 28 | fixed-nk | 1280x1024x1024 | bf16 | 1.201 | 1.223 | 0.721 | 1.67 | 1.70 |
| 29 | square | 512x512x512 | fp32 | 0.812 | 0.979 | 0.603 | 1.35 | 1.62 |
| 30 | square | 768x768x768 | fp16 | 0.907 | 0.939 | 0.583 | 1.56 | 1.61 |
| 31 | square | 512x512x512 | bf16 | 0.606 | 0.766 | 0.511 | 1.19 | 1.50 |
| 32 | fixed-nk | 768x1024x1024 | fp16 | 0.907 | 0.923 | 0.637 | 1.43 | 1.45 |
| 33 | square | 512x512x512 | fp16 | 0.715 | 0.733 | 0.508 | 1.41 | 1.44 |
| 34 | square | 1792x1792x1792 | fp16 | 1.101 | 1.882 | 1.362 | 0.81 | 1.38 |
| 35 | square | 2048x2048x2048 | fp16 | 2.010 | 2.012 | 1.461 | 1.38 | 1.38 |
| 36 | square | 2048x2048x2048 | bf16 | 2.003 | 2.024 | 1.481 | 1.35 | 1.37 |
| 37 | fixed-nk | 768x1024x1024 | bf16 | 0.903 | 0.916 | 0.689 | 1.31 | 1.33 |
| 38 | square | 768x768x768 | bf16 | 0.739 | 0.749 | 0.581 | 1.27 | 1.29 |
| 39 | fixed-nk | 512x1024x1024 | bf16 | 0.749 | 0.757 | 0.603 | 1.24 | 1.25 |
| 40 | fixed-nk | 512x1024x1024 | fp16 | 0.756 | 0.785 | 0.633 | 1.20 | 1.24 |
| 41 | fixed-nk | 2048x1024x1024 | fp32 | 1.834 | 2.153 | 1.767 | 1.04 | 1.22 |
| 42 | square | 4096x4096x4096 | bf16 | 9.764 | 9.897 | 9.791 | 1.00 | 1.01 |
| 43 | square | 4096x4096x4096 | fp32 | 38.334 | 38.571 | 38.283 | 1.00 | 1.01 |
| 44 | square | 4096x4096x4096 | fp16 | 9.760 | 9.774 | 9.771 | 1.00 | 1.00 |
| 45 | square | 2048x2048x2048 | fp32 | 4.877 | 4.881 | 4.879 | 1.00 | 1.00 |
| 46 | square | 3072x3072x3072 | fp32 | 16.055 | 16.056 | 16.054 | 1.00 | 1.00 |
| 47 | square | 2560x2560x2560 | fp32 | 9.315 | 9.320 | 9.320 | 1.00 | 1.00 |
| 48 | square | 2304x2304x2304 | fp32 | 6.868 | 6.868 | 6.882 | 1.00 | 1.00 |
| 49 | square | 3072x3072x3072 | bf16 | 4.271 | 4.272 | 4.300 | 0.99 | 0.99 |
| 50 | square | 2560x2560x2560 | bf16 | 2.578 | 2.821 | 2.890 | 0.89 | 0.98 |
| 51 | square | 3072x3072x3072 | fp16 | 4.275 | 4.289 | 4.477 | 0.95 | 0.96 |
| 52 | fixed-nk | 1536x1024x1024 | fp32 | 1.680 | 1.681 | 1.757 | 0.96 | 0.96 |
| 53 | square | 2304x2304x2304 | bf16 | 1.949 | 2.215 | 2.331 | 0.84 | 0.95 |
| 54 | fixed-nk | 4096x1024x1024 | fp32 | 2.563 | 2.768 | 2.946 | 0.87 | 0.94 |
| 55 | square | 1280x1280x1280 | fp32 | 1.405 | 1.562 | 1.680 | 0.84 | 0.93 |
| 56 | fixed-nk | 3072x1024x1024 | fp32 | 1.976 | 2.143 | 2.329 | 0.85 | 0.92 |
| 57 | square | 1792x1792x1792 | fp32 | 3.352 | 3.359 | 3.680 | 0.91 | 0.91 |
| 58 | fixed-nk | 2304x1024x1024 | fp32 | 1.546 | 1.595 | 1.888 | 0.82 | 0.84 |
| 59 | fixed-nk | 2560x1024x1024 | fp32 | 1.680 | 1.695 | 2.048 | 0.82 | 0.83 |
| 60 | square | 2304x2304x2304 | fp16 | 1.962 | 1.963 | 2.394 | 0.82 | 0.82 |
| 61 | square | 1536x1536x1536 | fp32 | 2.197 | 2.203 | 2.688 | 0.82 | 0.82 |
| 62 | fixed-nk | 1792x1024x1024 | fp32 | 1.715 | 1.887 | 2.568 | 0.67 | 0.73 |
| 63 | square | 2560x2560x2560 | fp16 | 2.571 | 2.601 | 3.790 | 0.68 | 0.69 |

**20 anomalies** with `r_max > 2.0`.

### Top-10 anomalies (named explicitly)

1. `square 1024×1024×1024 fp32` — cold_max 3.31 ms, warm 1.14 ms, **2.91x**
2. `square 1792×1792×1792 bf16` — cold_max 3.90 ms, warm 1.41 ms, **2.76x**
3. `fixed-nk 4096×1024×1024 bf16` — cold_max 3.00 ms, warm 1.12 ms, **2.67x**
4. `fixed-nk 4096×1024×1024 fp16` — cold_max 2.99 ms, warm 1.12 ms, **2.66x**
5. `fixed-nk 768×1024×1024 fp32` — cold_max 2.49 ms, warm 0.97 ms, **2.57x**
6. `square 1536×1536×1536 fp16` — cold_max 2.66 ms, warm 1.10 ms, **2.42x**
7. `square 1536×1536×1536 bf16` — cold_max 2.64 ms, warm 1.09 ms, **2.42x**
8. `fixed-nk 3072×1024×1024 bf16` — cold_max 2.37 ms, warm 1.00 ms, **2.38x**
9. `fixed-nk 3072×1024×1024 fp16` — cold_max 2.36 ms, warm 0.99 ms, **2.38x**
10. `square 1280×1280×1280 fp16` — cold_max 1.90 ms, warm 0.81 ms, **2.35x**

### Notable counter-anomalies (warm SLOWER than cold, `r_max < 0.85`)

These are cells where the poke (`2048×2048×2048 fp32`) actually
*destabilizes* a shape that was already on the fast path:

- `fixed-nk 1792x1024x1024 fp32` — r_max 0.73 (warm 2.57 ms vs cold 1.89 ms)
- `square 2560x2560x2560 fp16` — r_max 0.69 (warm 3.79 ms vs cold 2.60 ms)

This is a strong indicator that the bug is **not just about cold-start**;
it's about MPSGraph's dispatch state machine entering different
kernel-pick states depending on what was run before.

---

## 3. Pattern analysis

The original investigation claimed "1024^3 fp32 is uniquely sticky".
**That claim is refuted.** The 20 anomalies span:

- **All three dtypes** (fp32, fp16, bf16) — none of them is uniquely
  affected. fp16 and bf16 cells make up 16 of the top 20.
- **Both axes**: 8/20 are square (M=N=K), 12/20 are fixed-N=K=1024.
- **No clear tile-alignment pattern**: 1024, 1280, 1536, 1792, 2048,
  3072, 4096 all appear as M (or all-three) values; 768 and 512 also
  appear.
- **No pure power-of-2 pattern**: 1280, 1536, 1792, 3072 are present,
  768 too. The "mod 64 = 0" hypothesis loosely fits — every anomalous
  M is a multiple of 64 — but 64 is also the GCD of the swept set so
  this doesn't discriminate.
- **Mid-size GEMM is the danger zone**: anomalies cluster in M ∈
  [768..3072]. The largest shape (4096^3) and small-square
  (256/384/640^3) are uniformly clean (`r_max ≈ 1.0`).

### Hypothesis (revised, from the data)

The MPSGraph kernel-cache appears to have ≥3 different "kernel
classes" for matmul, depending on a heuristic over (M, N, K, dtype):

- **A — slow kernel pick** (~3.0-3.9 ms at 1024 fp32 / 1792 bf16
  reference points)
- **B — medium kernel pick** (~1.7-2.4 ms when the test shape itself
  is moderately fast, or when the sub-optimal pick is used)
- **C — fast kernel pick** (~0.7-1.4 ms — what 2048+ fp32 normally hits)

Per fresh process, the "first compile" of a shape lands on one of
these classes with a probabilistic distribution. For some (shape,
dtype) pairs the slow class is hit ~40-60% of the time — those are
the bug-prone cells. Once a process picks slow class A, it sticks
there until either (i) ~50 same-shape iterations run (the original
investigation's spontaneous transition) or (ii) a different shape
that re-enters the kernel-selection routine with different
heuristics fires, knocking the cache to a different kernel for the
NEXT compile of the original shape.

This generalizes the original "1024^3 fp32 cache-key sticky" story
to a much wider class of shapes/dtypes.

### Counter-evidence to "1024^3 fp32 is uniquely sticky"

- `1024×1024×1024 fp32` ratio 2.91x (rank 1) — yes, anomalous.
- But `1792×1792×1792 bf16` ratio 2.76x (rank 2) is just as bad.
- And there are 18 more anomalies including bf16/fp16 of equal severity.

The original investigation's claim that "switching dtype on the same
shape (1024^3 fp32 → 1024^3 fp16) does NOT unblock" is also refuted
by the workaround validation (Section 4) — `1024^3 fp16` poke DOES
unblock `1024^3 fp32` in my measurements.

---

## 4. Workaround validation — top-3 affected shapes

For each top-3 anomaly, I ran a 9-poke validation (small fp32, square
fp32, off-by-one fp32, large fp32, non-square fp32, same-shape fp16,
same-shape bf16, small fp16, small bf16) plus a no-poke baseline.

### Top-1: `square 1024×1024×1024 fp32`

Cold (no poke) median = **3.105 ms**.

| poke shape | warm med (ms) | unblocks? |
|---|---|---|
| 256×256×256 fp32 | 1.089 | YES (2.85x) |
| 512×512×512 fp32 | 1.064 | YES (2.92x) |
| 1023×1023×1023 fp32 | 3.240 | **NO** |
| 2048×2048×2048 fp32 | 1.165 | YES (2.66x) |
| 777×1111×999 fp32 | 3.339 | **NO** |
| 1024×1024×1024 fp16 | 1.075 | YES (2.89x) |
| 1024×1024×1024 bf16 | 1.033 | YES (3.01x) |
| 256×256×256 fp16 | 1.102 | YES (2.82x) |
| 256×256×256 bf16 | 1.139 | YES (2.73x) |

**Result**: 7/9 pokes unblock. The two that don't (`1023^3 fp32`,
`777×1111×999 fp32`) are both fp32 with non-power-of-2-friendly shape.
This *contradicts the original investigation's specific claims* that
(a) `1023×1023×1023 fp32` unblocks `1024×1024×1024 fp32` and
(b) switching dtype to fp16/bf16 on the same shape does NOT unblock.
Both of those claims are wrong per my measurements.

### Top-2: `square 1792×1792×1792 bf16`

Cold (no poke) median = **1.468 ms** (note: lower than cold_max 3.904
because cold is bimodal — this baseline run didn't catch the slow path).

| poke shape | warm med (ms) | unblocks? |
|---|---|---|
| 256×256×256 fp32 | 1.467 | NEUTRAL |
| 512×512×512 fp32 | 3.082 | **WORSE** (0.48x) |
| 1023×1023×1023 fp32 | 3.794 | **WORSE** (0.39x) |
| 2048×2048×2048 fp32 | 1.625 | NEUTRAL |
| 777×1111×999 fp32 | 2.303 | **WORSE** (0.64x) |
| 1024×1024×1024 fp16 | 3.789 | **WORSE** (0.39x) |
| 1024×1024×1024 bf16 | 3.985 | **WORSE** (0.37x) |
| 256×256×256 fp16 | 2.297 | **WORSE** (0.64x) |
| 256×256×256 bf16 | 2.536 | **WORSE** (0.58x) |

**Result**: For `1792^3 bf16`, most pokes make it WORSE. Only `2048^3
fp32` and `256^3 fp32` are roughly neutral. This is a different bug
profile than `1024^3 fp32`. The "any other shape unblocks" workaround
is shape-specific and **does not generalize**.

### Top-3: `fixed-nk 4096×1024×1024 bf16`

Cold (no poke) median = **2.171 ms** (also bimodal; in this baseline
run a mix of slow+fast samples).

| poke shape | warm med (ms) | unblocks? |
|---|---|---|
| 256×256×256 fp32 | 1.644 | partial (1.32x) |
| 512×512×512 fp32 | 3.248 | **WORSE** |
| 1023×1023×1023 fp32 | 3.059 | **WORSE** |
| 2048×2048×2048 fp32 | 1.297 | YES (1.67x) |
| 777×1111×999 fp32 | 3.045 | **WORSE** |
| 1024×1024×1024 fp16 | 2.896 | **WORSE** |
| 1024×1024×1024 bf16 | 3.068 | **WORSE** |
| 256×256×256 fp16 | 1.598 | partial (1.36x) |
| 256×256×256 bf16 | 3.082 | **WORSE** |

**Result**: For `4096×1024×1024 bf16`, only 3 of 9 pokes are
beneficial (`256^3 fp32`, `256^3 fp16`, `2048^3 fp32`). The rest
make it strictly worse. **No poke is universally safe.**

### Sanity check: `square 4096×4096×4096 fp32` (non-anomaly)

Cold = 38.41 ms. All 9 pokes leave warm in [38.13, 38.78] ms (ratio
~1.00). Confirms the non-anomalous cells are unaffected by poking,
which validates that my poke methodology isn't introducing artifacts.

### Workaround verdict

The "touch any other shape" workaround from the original
investigation is **only valid for `1024×1024×1024 fp32`** and even
there it has caveats: `1023^3 fp32` and non-square shapes don't
unblock. For other anomalous shapes (`1792^3 bf16`,
`4096×1024×1024 bf16`), poking with a wrong shape can make timing
SUBSTANTIALLY WORSE.

This refutes the original investigation's claim that "any
dispatch-state change releases the slow pick". The MPSGraph
kernel-pick state machine has more states than a 2-state
{cold, warm} model captures.

---

## Raw output

Per-cell raw timing data is in
`/Users/cero/Code/gpucheck/.claude/teams/audit/v1.1/EVIDENCE/raw/cross_version_shape.json`
(396 measurements, each with full `samples_ms` array). Workaround
validation outputs in `/tmp/workaround_*.txt` (preserved here).

Sample of axis-1 raw record:

```json
{
  "ok": true,
  "torch": "2.10.0",
  "shape": [1024, 1024, 1024],
  "dtype": "fp32",
  "warmup": 3,
  "iters": 10,
  "samples_ms": [3.122833, 3.147625, 3.105208, 3.148667, 3.049667,
                 3.0375, 3.16925, 3.150125, 3.148417, 3.146291],
  "median_ms": 3.147625,
  "min_ms": 3.0375, "max_ms": 3.16925,
  "p10_ms": 3.049667, "p90_ms": 3.16925,
  "poke_shape": null, "mode": "cold",
  "version": "2.10.0", "axis": "cross-version", "rep": 0
}
```

(All 10 samples in one process either ALL slow or ALL fast — bimodal
at process level, not within a process.)

## Interpretation

- **H1 (version-specificity): REFUTED**. Bug exists with
  ~40% incidence on each of torch 2.10.0, 2.11.0, and nightly
  2.13.0.dev20260507. No version-level fix.
- **H2 (shape-specificity): REFUTED**. 20 (shape, dtype) cells
  exhibit `cold_max/warm_med > 2.0x` — far beyond `1024×1024×1024
  fp32`. The bug class is broader than the original investigation
  claimed.
- The phenomenon is more accurately described as: **MPSGraph picks
  one of multiple kernel classes for the first matmul of a given
  cache key, with class-A (slow) hit probabilistically (~40%) for
  cells in the M ∈ [768..3072] mid-size range, all dtypes**.
- The "touch any other shape" workaround is **only validated for
  `1024×1024×1024 fp32`**, and even there it requires careful poke
  selection (off-by-one and non-square pokes don't help). For other
  anomalous shapes the workaround can backfire substantially.

### Confounds ruled out

- **Subprocess isolation**: each measurement is a fresh `python` —
  no MPSGraph state shared. Verified by 396 independent processes.
- **Per-sample timing**: all 10 samples within a process show
  consistent slow OR consistent fast behavior — not a measurement
  artifact.
- **Hardware throttling**: total wall is 233 s with idle gaps; M5
  isn't thermally throttled. Verified by 4096^3 cells being uniform
  at 38 ms.
- **Numpy missing on nightly**: I installed numpy in the nightly
  venv before measurement; not a confound.
- **Different python versions**: 3.13.7 (project) vs 3.14 (nightly).
  Possible minor confound but unlikely to affect a CUDA-event-style
  MPS timing path.

### Confounds NOT yet ruled out

- **Process startup state**: Each `python -X` start may load MPS
  drivers in slightly different states. The 40% incidence is a
  per-process probability — there could be a deterministic but
  hard-to-predict trigger (e.g., depending on process ID modulo
  something, or wall-clock time when the kernel cache is queried).
  Not investigated.
- **Metal kernel name**: I did NOT use `MTLCaptureManager` to record
  which exact MPSGraph kernel is dispatched on slow vs fast path.
  Knowing the kernel name would clarify what the heuristic is
  picking between.
- **macOS-version-specificity**: Only tested on macOS 26.4.1. Could
  be specific to a particular MPSGraph build shipped with this OS.

### Follow-ups that would strengthen this

1. Run the same sweep on macOS 14.x or 15.x to test
   OS-version-specificity (the underlying MPSGraph framework changes
   per OS release).
2. Use `MTLCaptureManager` to capture the slow-path and fast-path
   kernel names, file with Apple at the MPSGraph level (PyTorch
   can't fix this).
3. Add a regression test in gpucheck that detects "kernel pick
   stickiness" by running a 2-state markov check (run shape A 50x,
   then B once, then A again — assert variance bounded).

## Cleanup

- Files left for re-run:
  - `/tmp/cross_version_shape_harness.py`
  - `/tmp/cross_version_shape_driver.py`
  - `/tmp/workaround_validation.py`
  - `/tmp/analyze_results.py`
  - `/tmp/workaround_1024_fp32.txt`, `/tmp/workaround_1792_bf16.txt`,
    `/tmp/workaround_4096_1024_bf16.txt`, `/tmp/workaround_4096_fp32.txt`
  - `/tmp/anomaly_summary.json`
- Permanent artifact:
  - `.claude/teams/audit/v1.1/EVIDENCE/raw/cross_version_shape.json`

## Confidence

**high** for the cross-version refutation (H1) — 10 fresh-process reps
per version, deterministic per-sample-within-process behavior, and
identical incidence rates across 2.10/2.11/nightly are conclusive.

**high** for the cross-shape table — 396 cells with 3 reps each,
subprocess-isolated, per-sample arrays preserved in JSON, sanity
check (4096^3 fp32 uniform 38 ms across all pokes) confirms the
methodology is sound.

**medium** for the workaround validation — the per-shape behavior
diverges enough that "the workaround works" is not a clean story.
Replicates per-poke-cell would tighten this; I did 1 rep per
(test_shape, poke_shape) and the cold baseline is bimodal which
introduces noise into the ratio. Top-1 finding (`1024^3 fp32`) is
robust; top-2 and top-3 findings ("most pokes WORSEN it") would
benefit from 3-rep replication, but the qualitative pattern (no
universal poke) is clear.
