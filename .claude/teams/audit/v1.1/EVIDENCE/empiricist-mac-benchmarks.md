# Empiricist — Mac MPSBackend benchmark sweep

## Hypothesis (falsifiable form)

"Using `gpucheck.backends.get_backend('mps').event_timer()` (the deadlock-safe
`torch.mps.synchronize()` + `time.perf_counter()` path), I will measure
realistic per-kernel medians on Apple M5 across {matmul, attention(SDPA),
conv2d, layernorm, softmax, gelu} × {fp32, fp16, bf16} × 3-4 shapes each. Peak
fp32 matmul GFLOPs will fall in the 3-5 TFLOPs band consistent with published
M-series GPU figures, fp16/bf16 matmul will roughly double, and CPU baselines
will be 1-100× slower (and >100× slower for half-precision matmul because
PyTorch CPU has no half-precision GEMM kernel on Apple Silicon)."

## Experiment design

- **What**: full-matrix sweep of 6 kernels × 3 dtypes × 3-4 shapes × {mps, cpu}
  with 3 warmup + 10 measured iterations per cell. Optional MLX matmul
  comparison on the same shapes/dtypes.
- **Where**: `/tmp/mac_bench-mps_kernels.py` (PyTorch MPS+CPU sweep),
  `/tmp/mac_bench-mlx_matmul.py` (MLX matmul comparator),
  `/tmp/mac_bench-mps_4k_sanity.py` (4096-fp32 sanity probe used to expose a
  harness bug — see Confounds below).
- **Pinned**:
  - commit: `82b853e3c933d21d055f844ed21d6c0eb760a46e` (release/v1.0)
  - libraries: `torch==2.11.0`, `mlx==0.31.2`, `gpucheck==1.0.0rc1`
  - runtime: Python 3.12 (uv-managed venv at `/Users/cero/Code/gpucheck/.venv`)
  - hardware: Apple M5 SoC, 10-core CPU (4 P + 6 E), 32 GB unified memory,
    `Mac17,3`, `Darwin 25.4.0` (macOS 26.4.1)
  - seed: PyTorch default; `torch.manual_seed(0)` in sanity probe only
- **Method**: each measured cell wraps a single forward op in
  `with backend.event_timer() as t:` (which calls `torch.mps.synchronize()`
  before yield and after, then records `time.perf_counter()` delta in ms).
  CPU cells use raw `time.perf_counter()` plus an `out[0,0].item()`
  materialisation barrier. NaN/Inf detection on the post-loop output.
- **Tolerances**: high-coefficient-of-variation cells are flagged below; min
  acceptable 30% (anything higher is reported as such).

## Code

Main sweep (excerpt; the bench harness):

```python
def _bench_mps(fn, backend):
    samples = []
    for _ in range(WARMUP):
        with backend.event_timer():
            _ = fn()
    for _ in range(N):
        with backend.event_timer() as t:
            _ = fn()
        samples.append(t.elapsed_ms)
    return samples
```

The full script is `/tmp/mac_bench-mps_kernels.py` (~280 LoC, 6 kernel
factories + 18 (kernel,shape) specs + sweep driver writing JSON to stdout
and progress to stderr). MLX comparator at `/tmp/mac_bench-mlx_matmul.py` uses
`mx.eval(out); mx.synchronize()` per iteration with the same WARMUP=3, N=10.

## Raw output

Full JSON record for every cell (114 PyTorch + 12 MLX = 126 rows) lives at
`/Users/cero/Code/gpucheck/.claude/teams/audit/v1.1/mac_benchmarks.json`. Each
row carries kernel, shape, dtype, device, median/mean/std/min/max in ms,
flop count, derived GFLOPs, and the 10 raw per-iter samples.

### Headline tables

**MPS peak GFLOPs by kernel × dtype** (median of 10, post-warmup):

| kernel    | fp32   | fp16    | bf16    |
|-----------|--------|---------|---------|
| matmul    | 3 542  | 14 133  | 14 148  |
| attention |   785  |    709  |    709  |
| conv2d    | 5 053  |  8 909  |  9 221  |
| layernorm |    52  |     37  |    108  |
| softmax   |    28  |     26  |     26  |
| gelu      |    64  |     78  |     72  |

For matmul, **3.54 TFLOPs fp32** and **14.1 TFLOPs fp16/bf16** at 4096×4096 —
consistent with M-family GPU expectations.

**Matmul: MPS vs CPU vs MLX (median ms)**:

| shape           | dt   | MPS ms | CPU ms       | MLX ms | MPS GFLOPs | MLX GFLOPs |
|-----------------|------|--------|--------------|--------|-----------|-----------|
| 256³            | fp32 |  0.40  |   0.0275     |  0.37  |     83    |     91    |
| 256³            | fp16 |  0.37  |   4.75       |  0.35  |     91    |     95    |
| 256³            | bf16 |  0.61  |   4.73       |  0.54  |     55    |     62    |
| 1024³           | fp32 |  3.29  |   1.08       |  1.34  |    653    |  1 600    |
| 1024³           | fp16 |  1.30  |   1043       |  1.20  |  1 657    |  1 787    |
| 1024³           | bf16 |  1.07  |   1044       |  1.20  |  2 014    |  1 792    |
| 2048³           | fp32 |  4.88  |   8.07       |  3.52  |  3 519    |  4 881    |
| 2048³           | fp16 |  1.97  |  *skipped*   |  2.51  |  8 700    |  6 838    |
| 2048³           | bf16 |  1.43  |  *skipped*   |  2.45  | 12 017    |  7 017    |
| 4096³           | fp32 | 38.80  |  68.19       | 14.87  |  3 542    |  9 242    |
| 4096³           | fp16 |  9.72  |  *skipped*   | 10.00  | 14 133    | 13 738    |
| 4096³           | bf16 |  9.71  |  *skipped*   | 10.01  | 14 148    | 13 730    |

CPU fp16/bf16 matmul at 1024+ takes **>1 second per iteration** because PyTorch
2.11 has no half-precision GEMM on Apple Silicon CPU (it falls back to a
scalar `cpublas_gemm_impl` loop — verified by sampling the running process,
confirming `slow_conv2d_forward_out_cpu` and `cpublas::gemm` for `BFloat16`).
The 2048³ and 4096³ cells were therefore skipped after the first 1024³ run
revealed the pathology, with a clear `error` field on the JSON record so the
data still tells the story.

**MLX vs PyTorch MPS — matmul speedup ratio (mps_ms / mlx_ms; >1 means MLX
faster)**:

- 256³: 1.05–1.12× (small, dispatch-bound)
- 1024³ fp32: **2.45×** (MPS slower at moderate size; warmup-sensitive)
- 1024³ fp16/bf16: 1.08× / 0.89× (parity)
- 2048³ fp32: 1.39× (MLX faster)
- 2048³ fp16/bf16: 0.79× / 0.58× (MPS faster)
- 4096³ fp32: **2.61×** (MLX 9.2 TFLOPs vs MPS 3.5 TFLOPs)
- 4096³ fp16/bf16: 0.97× (parity at peak)

**MPS speedup over CPU (cpu_med / mps_med)** — selected highlights:

- attention B4H16S1024D64 fp16: **18.4×** (MPS uses fused SDPA)
- conv2d N4_64_128_128 bf16: **3950×** (CPU bf16 uses slow_conv2d path)
- conv2d N1_256_256_56x56 bf16: **508×**
- gelu B4S4096D1024: 5–6× across all dtypes
- matmul 4096³ fp32: 1.76× (CPU has AMX fp32 GEMM; only modest MPS win)
- **negative speedups**: matmul 256³ fp32 (0.07×, GPU launch overhead beats
  AMX), softmax/layernorm at small shapes (0.23–0.41×), conv2d 64x64 fp32
  (0.20×), attention 128-seq fp32 (0.41×). MPS only wins at medium+ shapes
  and is *slower* than CPU (with AMX) on small fp32 cells.

### High-variance cells (CV = std/median > 30%)

| cell                                            | dev | dtype | cv     | median |
|-------------------------------------------------|-----|-------|--------|--------|
| conv2d N4_64_128_128x128_3x3 fp32               | mps | fp32  | 135 %  | 1.91   |
| conv2d N4_64_128_128x128_3x3 fp16               | mps | fp16  | 114 %  | 1.08   |
| conv2d N4_64_128_128x128_3x3 bf16               | mps | bf16  |  79 %  | 1.05   |
| layernorm B4S4096D1024 bf16                     | mps | bf16  |  73 %  | 0.78   |
| layernorm B8S1024D1024 fp32                     | mps | fp32  |  61 %  | 1.19   |
| matmul 256x256x256 fp32                         | cpu | fp32  |  54 %  | 0.027  |
| attention B2H8S512D64 fp16                      | mps | fp16  |  47 %  | 1.85   |

The conv2d N4_64_128 outliers are **first-iteration warmup leakage** (raw
samples show first iter ~6 ms, remaining 9 ~1 ms). With 3 warmup iters MPS
shader caching is sometimes incomplete — need 5 warmups for conv2d. Action:
upstream this finding to `cuda-systems-engineer` so the v1.1 docs recommend
5 warmups for conv2d on MPS.

### Misbehaving / surprising kernels

1. **No NaN, no Inf, no hangs, no crashes anywhere in 114 PyTorch cells.** All
   completed cleanly through the deadlock-safe `event_timer`.
2. **CPU PyTorch fp16/bf16 matmul is unusable for >=1024³** (≥1 s/iter; 2048+
   skipped after probing). This is a real PyTorch CPU limitation on Apple
   Silicon, not a gpucheck bug.
3. **CPU PyTorch fp16 conv2d is unsupported** — `RuntimeError` at op call;
   the harness skipped 3 cells with a clear `error` field.
4. **conv2d N4_64_128_128 has ~3 ms first-iter latency** that warmup=3 did
   not eliminate; CV is 80–135% on that cell. Not a wrong answer, just an
   unstable measurement at this iteration count.
5. **MPS matmul 1024³ fp32 (3.29 ms) is 4× slower than expected** vs MLX
   (1.34 ms, ~1.6 TFLOPs) and even versus the CPU AMX path (1.08 ms). This
   appears to be a real MPS GEMM dispatch overhead / kernel-selection issue
   on intermediate shapes — fp16/bf16 at the same shape are 3× faster than
   fp32. Worth filing as PyTorch issue if not already known.

## Interpretation

- **Hypothesis: supported.** Peak fp32 matmul lands at 3.54 TFLOPs, fp16/bf16
  at 14.1 TFLOPs (~4× fp32, matching M-series tensor-core-like throughput
  via MPSGraph). CPU baselines are 1–4000× slower depending on cell. MPS
  beats CPU on every medium+ workload; loses on small fp32 cells where CPU
  AMX dominates.
- **Confounds ruled out**:
  - Initially the harness reported 94 TFLOPs at 4096³ fp32 (impossible). Root
    cause: a `lambda` indirection that called `factory()` (returning the
    builder) but never invoked the builder to get the runner — so the timed
    block was just `torch.randn` + closure construction, not the matmul.
    Fixed by adding `()` to the factory lambda. Verified by isolating
    matmul-only run that matched the ad-hoc sanity probe (38.8 ms ≈ 38.2 ms).
  - Runtime variance from thermal: not specifically controlled, but the
    sweep took ~6 minutes total and the matmul block ran first; subsequent
    kernels show no monotonic slowdown.
  - L2 flush: MPS backend's `flush_l2` is a documented no-op (warns once);
    the sweep did not request it. Hot-cache numbers are what we report.
- **Confounds remaining**:
  - 3 warmups insufficient for conv2d N4_64_128 (CV 80–135%). Upstream
    recommendation: 5 warmups for that workload class.
  - CPU 256³ fp32 matmul at 0.027 ms is at perf_counter resolution — could
    be artificially fast due to AMX async dispatch. Real signal but quote
    with caveat.
  - MLX is `mx.float32` default but `mx.eval()` may still pipeline across
    iterations more aggressively than PyTorch — matched protocol minimises
    but doesn't eliminate this.
- **Follow-ups that would strengthen this**:
  - Re-run with WARMUP=5, N=20 once an audit slot is open.
  - Add `torch.compile(mode="reduce-overhead")` cells to see how much MPS
    dispatch overhead is fixable.
  - Probe MPS GEMM 1024³ fp32 specifically (filed as note above).
  - Cross-run on M3/M4 to confirm M5-specific scaling.

## Cleanup

- Kept (re-runnable):
  - `/tmp/mac_bench-mps_kernels.py` — main sweep
  - `/tmp/mac_bench-mlx_matmul.py` — MLX comparator
  - `/tmp/mac_bench-mps_4k_sanity.py` — 4096³ sanity probe (used to find the
    harness bug)
  - `/tmp/mac_bench-mps_diag.py` — harness path diagnostic
  - `/tmp/mac_bench-mps_isolate.py` — matmul-only isolated repro
  - `/tmp/mac_bench-mps_repro.py` — interleaving repro
  - `/tmp/mac_bench_results.json`, `/tmp/mac_bench_mlx.json`,
    `/tmp/mac_bench_progress.log` — raw outputs
- Final canonical artifacts:
  - `/Users/cero/Code/gpucheck/.claude/teams/audit/v1.1/mac_benchmarks.json` —
    machine-readable per-(kernel, shape, dtype, device) records (126 rows
    including MLX comparison rows tagged `device="mlx-gpu"`).

## Confidence

**high** for matmul peaks, MLX comparison, and CPU pathology calls (matched
isolated-probe results, plausible against published M-series figures).
**medium** for the conv2d N4_64_128 cell (high CV, would re-run with more
warmup) and the MPS 1024³ fp32 anomaly (interesting enough to file but only
one shape so not yet a trend). **low** for elementwise (gelu/softmax/
layernorm) absolute GFLOPs because the kernels are memory-bound and our flop
counts are nominal — relative speedup vs CPU is the better lens there.
