# Archaeologist v2 — PyTorch MPS backend git history

Charter: dig into `pytorch/pytorch` to understand WHY the current MPS
implementation looks the way it does, and what commits/PRs encode the deepest
correctness pitfalls. Inform what gpucheck must test.

Note: archaeology done via GitHub Search API (`gh api search/issues` and
`search/commits`). Local PyTorch checkout was not used because the gpucheck
repo does not vendor it; queries against `pytorch/pytorch` are authoritative.
Where I cite a commit SHA on `main`, I have verified it via
`gh api search/commits` (the `pytorchbot`/ghstack quirk makes the `pulls` API
report `merged: false` for landed PRs — I cross-checked all five against
`search/commits`).

---

## Timeline (since 2023, focus on 2025-2026)

| Date       | SHA / PR       | Author        | What changed                                                                |
|------------|----------------|---------------|-----------------------------------------------------------------------------|
| 2023-08-08 | cdfd0ea16282 (#102121) | razarmehr | Introduces `torch.mps.Event()` API and `MPSEventPool::elapsedTime` — bakes in `waitForCpuSync()` blocking call that becomes the deadlock vector |
| 2024-11-21 | b417006e (#141296) | malfet | Adds **regression test** for sync deadlock — but only for `torch.mps.synchronize()`, not for `Event.synchronize()` |
| 2025-09-13 | (#162872 issue) | oraluben | Files `[MPS] dead lock when calling Event.synchronize() before Event.elapsed_time()` — issue still **OPEN** |
| 2025-09-13 | PR #162874     | oraluben      | Proposes one-line fix removing `end_event->waitForCpuSync()` — closed **WITHOUT MERGE**, fix is orphaned |
| 2026-01-25 | a914fee44471 (#173326) | malfet | "[BE][MPS] Adjust cdist tolerances" — admits CI passes but local M4 fails at `1.5e-5` vs `1e-5` allowed |
| 2026-02-05 | 75742d2eb01f (#174411) | malfet | "[BE][MPS] Move tolerance overrides to OpInfo" — institutionalizes per-op MPS tolerance fudge factors at the framework level |
| 2026-02-20 | c68a1d2c01df (#174945) | hvaara | "[MPS] Fix 2-pass SDPA memory corruption by forcing float accumulators" — fixes silent OOB / nondeterministic outputs in `sdpa_vector_2pass_mps` |
| 2026-04-28 | 49e7d4dadbbe (#181466) | malfet | "[MPS] Workaround MetalPerformancePrimitives bug for F.linear on M5+" — Apple's own MPP returns nondeterministic results for `>2D` fp16/bf16 matmul on M5 |

---

## Pivotal commits

### 1. cdfd0ea16282 — `[MPS] Introduce torch.mps.Event() APIs (#102121)` — razarmehr, 2023-08-08
File: `aten/src/ATen/mps/MPSEvent.mm` (the **only** commit ever to touch this
file, per `gh api repos/pytorch/pytorch/commits?path=...`). This is the
original sin: the "elapsed time" path takes a recursive lock and then calls
`end_event->waitForCpuSync()`, which waits for a notify that arrives on a
*different thread*. If the user has already called `Event.synchronize()` on
the end event, no further notify will fire — deadlock.

> "Implement `MPSEventPool` to recycle events. Implement python bindings with
> `torch.mps.Event` class using the MPSEventPool backend. The current member
> functions of the Event class are `record()`, `wait()`, `synchronize()`,
> `query()`, and `elapsed_time()`."

Evidence: `aten/src/ATen/mps/MPSEvent.mm:228` (HEAD as of 2026-05-01) still
contains `end_event->waitForCpuSync();` — verified by reading the raw file
from `pytorch/main`. The 2023 design has not been touched in 2.7 years.

### 2. PR #162874 — `[MPS] Do not explicit wait for sync` — oraluben, 2025-09-13 (CLOSED, NOT MERGED)
URL: https://github.com/pytorch/pytorch/pull/162874

> "The deleted line was waiting for this notify [...] But when we explicitly
> sync stream as in #162872, this will trigger a dead lock as no other
> threads will notify. I think users are responsible for syncing, not here.
> Or we can add a full sync here, not just a wait."

The proposed one-line patch removes line 228 of `MPSEvent.mm`. It also adds a
parametric test that exercises both `sync_start` and `sync_end` permutations:

```python
@parametrize("sync_start", [True, False])
@parametrize("sync_end", [True, False])
def test_mps_event_module(self, sync_start, sync_end):
    ...
    if sync_start: startEvent.synchronize()
    if sync_end:   endEvent.synchronize()
    elapsedTime = startEvent.elapsed_time(endEvent)
```

`gh api repos/pytorch/pytorch/pulls/162874` reports
`merged: false, merge_commit_sha: null, state: closed`. The issue (#162872)
is still **open**. Cross-checked `gh api search/commits` for
`"Do not explicit wait for sync"` — zero hits in `main`. **The deadlock is
unfixed in HEAD.** Any benchmark fixture that calls `Event.synchronize()`
before `elapsed_time()` will hang on Apple Silicon.

### 3. 75742d2eb01f — `[BE][MPS] Move tolerance overrides to OpInfo (#174411)` — malfet, 2026-02-05
This is load-bearing because it formalizes that **MPS systematically needs
looser tolerances than CUDA**. Files touched:
- `test/test_mps.py` (-3 lines)
- `torch/testing/_internal/common_methods_invocations.py` (+9 lines)

By moving overrides into `OpInfo` (PyTorch's central op-test metadata), the
PR encodes Apple-specific tolerance fudge factors as first-class test
metadata, no longer special-cased in `test_mps.py`. The companion PR #173326
landed the same week with the message:

> "It passes in CI, but on local runs on M4 it fails with [...] Greatest
> absolute difference: 1.5139579772949219e-05 at index (1, 3, 3) (up to
> 1e-05 allowed)"

That is a 1.5x tolerance breach on `cdist` between CI hardware and M4 silicon
— in the same backend, on the same code path.

### 4. c68a1d2c01df — `[MPS] Fix 2-pass SDPA memory corruption by forcing float accumulators (#174945)` — hvaara, 2026-02-20
File: `aten/src/ATen/native/mps/operations/Attention.mm` (+2/-2 lines).
Fixes #174861 — out-of-bounds memory access plus nondeterministic / corrupt
outputs in `sdpa_vector_2pass_mps` for bf16/fp16 with GQA, `seq_len > 1023`.
The fix: force `sums` and `maxs` accumulator buffers to `kFloat` instead of
inheriting the input dtype.

> "Fixes out-of-bounds memory access and nondeterministic/corrupt results,
> as reported in #174861 (reproducible with bf16/fp16 and GQA, seq_len > 1023)."

The lesson: a two-character dtype mismatch in an accumulator caused
*silent* memory corruption, not a crash. CUDA testing alone would never
catch it; this is exactly the class of bug a cross-backend fuzzer should hunt.

### 5. 49e7d4dadbbe — `[MPS] Workaround MetalPerformancePrimitives bug for F.linear on M5+ (#181466)` — malfet, 2026-04-28
File: `aten/src/ATen/native/mps/operations/Linear.mm` (+36/-13).

> "`MPSNDArrayMatrixMultiplication` and `MPSGraph matrixMultiplication:`
> produce non-deterministic results for >2D fp16/bf16 inputs on Apple10 GPUs
> (M5). Work around this in `F.linear` by flattening the input to 2D before
> the matmul on affected hardware/dtype combinations [...] At some point
> attempted to fix this by migrating Linear to MPP on MacOS-26, which works
> great on M5, but performance is 2X lower than MPS on M4. Fixes #180776"

This is **Apple's own framework returning nondeterministic results** for
batched fp16/bf16 matmul on M5. PyTorch ships a hardware-version-conditional
workaround. The fix is per-chip-generation (`Apple10`/M5) and per-dtype
(fp16/bf16), so any test matrix that does not parametrize on chip generation
will miss this regression.

---

## Incidents / workarounds

- **`MPSEvent.mm:228 waitForCpuSync()`** — known deadlock since 2025-09-13,
  fix proposed and abandoned, lives in HEAD as of 2026-05-01. Risk: any
  benchmarking fixture using the `torch.mps.Event` API the way Python users
  expect will hang the test runner.
- **OpInfo tolerance overrides for MPS** (#174411 + cdist #173326) — the
  framework now ships per-op tolerance fudge factors specifically for MPS,
  acknowledged divergence between Apple Silicon generations (CI vs M4).
- **`sdpa_vector_2pass_mps` accumulator dtype** (#174945) — silent OOB +
  nondeterminism, fixed by hardcoding `kFloat`. The pattern (accumulator
  inherits input dtype) is exactly the kind of bug that recurs across kernels.
- **F.linear on M5+** (#181466) — workaround flattens >2D inputs to 2D when
  GPU is `Apple10` and dtype is fp16/bf16. Hardware-conditional code path is
  inherently fragile.
- **Recurring pattern: non-contiguous tensor handling on MPS.** Issue search
  surfaced #175188 (tanh/sigmoid backward on permuted tensors), #175187
  (sort/digamma on transposed), #176159 (in-place index_add transposed),
  #181133 (SDPA on permute-produced q/k/v), #180984 (conv2d on
  channels-slice views), #178497 (count_nonzero/mean/nansum/sum/trace
  correctness). All filed 2026-Q1 alone. This is the largest single class of
  MPS correctness issues by issue count.

## Recurring authors (handles)

From `gh api search/issues` filtered to `is:pr "[MPS]" created:>=2025-09-01`:

- **malfet** (9 PRs) — Apple/PyTorch MPS lead. Authors of #173326 cdist
  tolerance, #174411 OpInfo refactor, #181466 M5+ workaround, #141296
  deadlock regression test. Also files MPS issues himself (#176159).
- **Isalia20** (10 PRs in same window, 16 in `is:pr "fix"` window) — most
  prolific external MPS contributor; #171619 grid_sampler non-contig fix,
  #167727 mm/addmm large-tensor issue.
- **hvaara** (4 PRs, plus filed #178497 / #178079 / #174861) — files **and**
  fixes silent correctness bugs; authored the 2-pass SDPA dtype fix #174945.
- **BenjaminDEMAILLE** (13 PRs in fix window) — implemented `batch_norm_update_stats`
  (#173048), `fractional_max_pool2d/3d` (#173150).
- **jhavukainen** (5 PRs) — ULP-guided tolerance work (#168323).
- **anagnorisis2peripeteia** (5 PRs) — ongoing Metal-shader migration of
  softmax (#181503, still open).
- **razarmehr** — original Event API author (#102121, #106938) but no
  recent MPS activity in the 2025-2026 window.
- **kulinseth, DenisVieriu97** — cc'd on the deadlock issue, indicating
  Apple-internal review responsibility, but no merged PRs in the window.

## 5-PR Deep Dive — implications for gpucheck

| # | PR / URL | Lesson for gpucheck |
|---|----------|---------------------|
| 1 | #102121 (razarmehr, 2023-08-08, https://github.com/pytorch/pytorch/pull/102121) — original Event API | gpucheck's `gpu_benchmark` fixture must NOT call `Event.synchronize()` on MPS before `elapsed_time()`. Either route MPS timing through `torch.mps.synchronize()` + wall clock, or special-case the MPS path. The CUDA-Event idiom is unsafe on MPS. |
| 2 | #162874 (oraluben, 2025-09-13, https://github.com/pytorch/pytorch/pull/162874) — abandoned deadlock fix | gpucheck should ship a **deadlock-detection probe** for MPS Event timing: spawn the timing call in a worker thread with a timeout, fall back to host clock if it hangs. The upstream fix is orphaned; users will hit this. |
| 3 | #174411 + #173326 (malfet, 2026-01..02, https://github.com/pytorch/pytorch/pull/174411, https://github.com/pytorch/pytorch/pull/173326) — MPS tolerance overrides moved to OpInfo, cdist tolerance bump | gpucheck's per-dtype tolerance table should expose an **MPS column distinct from CUDA** — even at fp32, M4-class hardware breaches CUDA tolerances (`1.5e-5` vs `1e-5` for cdist). Tolerances should also be **chip-generation aware** (M1 vs M4 vs M5), because PyTorch upstream now is. |
| 4 | #174945 (hvaara, 2026-02-20, https://github.com/pytorch/pytorch/pull/174945) — 2-pass SDPA accumulator dtype | gpucheck should add a **silent-corruption fuzzer** that runs reduction-style ops at small dtypes (fp16/bf16) with `seq_len > 1023` and checks both (a) determinism across reruns and (b) absence of OOB-pattern outputs (NaN clusters, denormals at known stride boundaries). The regression test in #174945 is a template. |
| 5 | #181466 (malfet, 2026-04-28, https://github.com/pytorch/pytorch/pull/181466) — F.linear MPP nondeterminism on M5+ | gpucheck's `arch/` module should detect **Apple chip generation** (M1/M2/M3/M4/M5) the same way it does SM60-SM120 for CUDA, and gate matmul determinism tests by chip family. A test that runs only on M4 will silently pass while M5 users see nondeterminism. The chip-conditional workaround pattern means a single "MPS supported" boolean is insufficient. |

## Unanswered

- Why was PR #162874 abandoned? The author (oraluben, external) closed it
  with no public review thread visible via the API. Worth pinging
  `@malfet` / `@kulinseth` directly or checking the internal Apple/PyTorch
  Slack referenced in some PR descriptions.
- The non-contiguous tensor cluster (#175187/8/9, #176159, #181133, #180984,
  #178497) — is this a *single* underlying bug in MPS's stride handling, or
  multiple independent kernel-level issues? An interview with `hvaara` (the
  most active filer of this class) would clarify whether gpucheck should add
  one stride-fuzzer or per-kernel stride checks. The CLAUDE.md already flags
  "No stride/contiguity fuzzing (only shapes and values)" as a known gap —
  this evidence quantifies the cost of that gap.
- M5-specific behavior in #181466 was discovered ~6 months after Apple
  shipped the silicon. What is PyTorch's release-test cadence for new Apple
  chips? Likely zero pre-release access; this is a structural reason to
  build chip-generation-aware test matrices.

## Confidence

**high** for the deadlock claim (verified via reading raw `MPSEvent.mm` from
`pytorch/main` HEAD; PR #162874 confirmed not-merged via API and
`search/commits`). **high** for the five PR citations (all SHAs cross-checked
in `search/commits` and file-touch lists pulled from `pulls/<n>/files`).
**medium** for the recurring-author leaderboard — the sample is restricted to
PR titles containing `[MPS]` since 2025-09-01 and may underweight reviewers
who don't author. **medium** for the "single underlying stride bug vs many"
hypothesis — this is genuinely unanswered without code-level tracing.
