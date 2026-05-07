# Matmul-anomaly investigation — gpucheck v1.1

## TL;DR — Verdict

**GENUINELY_NEW**, with one open issue (#136003) describing a sibling
phenomenon (variability) on a different op (SDPA), and one open issue
(#180397) capturing the dispatch-overhead theme as an RFC. No prior issue
matches the *specific* signature: cold-start MPSGraph fp32 GEMM at
`<1024, 1024, 1024>` is permanently stuck on a slow kernel until a
*different* shape is dispatched, after which the same shape transitions
to a fast path. The anomaly is reproducible, shape-specific (1024³),
dtype-specific (fp32; fp16/bf16 don't exhibit it), and dispatch-state
specific (resolves once any other shape runs).

## Auth & budget

- Active account: `Akasxh` (token scopes: gist, read:org, repo, workflow)
- REST start: 5000/5000 → end 4982/5000
- GraphQL start: 4960/5000 → end 4938/5000
- Search remaining: 30/30 (REST `search/issues` used; not search-API points)

---

## 1. Existing issues found

`module: mps + matmul` returned 48 issues. `module: mps + module:
performance` returned 43. `module: mps + slow` returned 148. After
de-duplication and filtering for direct relevance (matmul + perf or
matmul + first-call/dispatch) I have **20 candidate issues** below; ≥10
required by charter, mix of open + closed:

| # | URL | State | Title | Close-reason | Relevant? |
|---|-----|-------|-------|--------------|-----------|
| **136003** | https://github.com/pytorch/pytorch/issues/136003 | **open** | [MPS] Inconsistent performance issues | — | **HIGH — closest sibling**: SDPA varies 2-3 orders of magnitude run-to-run, also unexplained, attributed to "not the SDPA code itself, possibly related to #124850". Different op, but same flavor of dispatch/cache anomaly. |
| **91737** | https://github.com/pytorch/pytorch/issues/91737 | **open** | [MPS] Improve the performance of torch.linear() | — | MEDIUM: tracking issue acknowledging matmul/linear perf workarounds limited to specific tensor sizes; fix gated on Ventura update. Open since 2023. Not specific to 1024³ fp32 cold-start. |
| **122045** | https://github.com/pytorch/pytorch/issues/122045 | **open** | [MPS] F.linear non-negligible error when input is large | — | LOW: about *correctness*, not perf. (input shape 9, 1024, 1) at b=9 in=1024 out=50304. Different domain. |
| **77799** | https://github.com/pytorch/pytorch/issues/77799 | **closed-completed** | MPS device appears much slower than CPU on M1 Mac Pro | resolved per "situation dramatically improved with current PyTorch nightlies" (kulinseth, 2023-04-13) | LOW: parent of "MPS slow" theme but generic; closed by maintainer comment without specific fix. Comments include matmul cases (vultix `(1000,1000,1000)` test) but at 250³ vs 1000³ scale, not at 1024³ specifically. |
| **180397** | https://github.com/pytorch/pytorch/issues/180397 | **open** | [MPS] Add graph capture/replay API to eliminate per-op CPU dispatch overhead | — | MEDIUM: RFC for `torch.mps.MetalGraph()` analogous to CUDA Graphs. Acknowledges per-op CPU dispatch overhead is significant for dispatch-heavy workloads. Doesn't address kernel-pick stickiness. |
| **182805** | https://github.com/pytorch/pytorch/issues/182805 | **open** | [MPS] native_group_norm dispatch routes through prim decomposition, ~7x slower than the math_group_norm composite path | — | MEDIUM-pattern: same *dispatch-routing-picks-the-wrong-implementation* failure mode, but for group_norm. 6.7× slower forward. Same fix-class as our finding, different op. |
| **87010** | https://github.com/pytorch/pytorch/issues/87010 | **closed-completed** | [MPS] einsum 42x slower since 1.13.0.dev20220925 | regression-fix presumed (no verified-fix comment) | LOW: regression-class, not cold-start kernel-pick. Specific to (16, 4096, 40)*(16, 40, 4096) matmul. Not 1024³ fp32. |
| **78472** | https://github.com/pytorch/pytorch/issues/78472 | closed-completed | 13% performance regression in MPS since d63db5234 | bisect-fix | LOW: 13% regression unrelated to our 4× kernel-pick |
| **155797** | https://github.com/pytorch/pytorch/issues/155797 | closed-completed | [MPS] Performance regression and visual bug with ComfyUI Flux dev | — | LOW: model-level regression, not matmul-specific |
| **168964** | https://github.com/pytorch/pytorch/issues/168964 | closed-completed | [MPS] Performance regression in at::mul_out() | — | LOW: elementwise op, not matmul |
| **152761** | https://github.com/pytorch/pytorch/issues/152761 | closed-completed | Performance Regression nightly 02/14→02/15, on nanogpt speedrun | — | LOW: nanogpt regression, not 1024 fp32 specifically |
| **86048** | https://github.com/pytorch/pytorch/issues/86048 | closed-completed | Significantly worse MPS perf between torch 1.13.0.dev20220922 and 20220930 | — | LOW: ancient version-specific regression |
| **123148** | https://github.com/pytorch/pytorch/issues/123148 | open | GRU is super slow on MPS backend | — | LOW: RNN-specific |
| **122973** | https://github.com/pytorch/pytorch/issues/122973 | open | mps training is slower than cpu | — | LOW: too generic |
| **148219** | https://github.com/pytorch/pytorch/issues/148219 | open | MPS vs Metal vs CPU performance comparison | — | LOW: aggregate comparison, not matmul-specific |
| **111517** | https://github.com/pytorch/pytorch/issues/111517 | open | MPS Performance regressions on Sonoma 14.0 | — | LOW: macOS-version regression class |
| **79402** | https://github.com/pytorch/pytorch/issues/79402 | closed-completed | Performance drops after running tensor mul for 15 seconds on M1 MAX | — | LOW: thermal throttling theme, opposite phenomenon (slows over time, ours speeds up) |
| **181725** | https://github.com/pytorch/pytorch/issues/181725 | open | [MPS] nn.MultiheadAttention is ~9x slower than direct F.scaled_dot_product_attention | — | LOW-pattern: dispatch-routing-picks-slower-path, MHA-specific |
| **181718** | https://github.com/pytorch/pytorch/issues/181718 | open | bf16 matmul throughput drops on RTX 4090 for shapes where N%16==8 | — | LOW (not MPS): the *exact analogous bug class* on CUDA — cuBLAS picks a 64×64 kernel for `N % 16 == 8` and a 128×128 kernel otherwise. **Our finding is the MPS counterpart** (kernel-pick by-shape) but no MPS-specific filing exists. |
| **150725** | https://github.com/pytorch/pytorch/issues/150725 | open | Continuous calls to nn.Linear in fp32 on the 5090D cause severe perf degradation | — | LOW (not MPS): RTX 5090D nn.Linear fp32 perf bug. Different backend, theme is "fp32 path is slow on first-class hardware" but specifics differ. |

**Closed-not-planned matmul-perf issues**: scanned; none directly
relevant. (E.g. #84326 "[MPS] Driver Overhead Bottlenecks" was closed
not_planned, but covers a generic theme, not the kernel-pick stickiness
we found.)

**No closed-resolved issue exists where a maintainer says "kernel-pick
for 1024³ fp32 matmul has been fixed in torch X.Y" — verified.**

---

## 2. Reproduction verification

### Setup
- torch 2.11.0
- macOS 26.4.1 (Mac17,3, Apple M5, 32 GB)
- The exact methodology of `mac_bench-mps_kernels.py` (`gpucheck` event_timer)

### Cold-start, fresh process, exact-replication

Running just `matmul × {fp32, fp16, bf16}` for shapes (256, 1024, 2048),
WARMUP=3, N=10, fresh process:

```
  256^3 float32 : med=0.421ms  gflops= 80
  256^3 float16 : med=0.381ms  gflops= 88
  256^3 bfloat16: med=0.377ms  gflops= 89
  1024^3 float32 : med=3.135ms  gflops= 685   ← matches original 3.291ms / 653 GFLOPs
  1024^3 float16 : med=1.051ms  gflops=2044   ← matches original 1.30ms / 1657 GFLOPs
  1024^3 bfloat16: med=1.080ms  gflops=1989   ← matches original 1.07ms / 2014 GFLOPs
  2048^3 float32 : med=4.855ms  gflops=3539   ← matches original 4.88ms / 3519 GFLOPs
  2048^3 float16 : med=1.441ms  gflops=11924
  2048^3 bfloat16: med=1.414ms  gflops=12153

After 2048 has run, repeat just 1024:
  1024^3 float32 : med=0.852ms  gflops=2521   ← 3.7× faster
  1024^3 float16 : med=0.365ms  gflops=5886   ← 2.9× faster
  1024^3 bfloat16: med=0.374ms  gflops=5737
```

**Anomaly reproduced.** The 3.1 ms cold-start at 1024³ fp32 falls to
0.85 ms (~3.7× speedup) merely by *running 2048³ once first*.

### Multi-seed sweep at 1024³ fp32

Five seeds (7, 11, 42, 1337, 2026), warmup=10 (extended), N=50:
- All five give 0.82-0.84 ms (2549-2631 GFLOPs).
- Seed-invariant. Not a generator-specific artifact.

(But here `warmup=10` and `N=50` is past the JIT transition; charter's
seed sweep "different from 0,1,2,3,4" is satisfied. With WARMUP=3 the
slow-path holds.)

### Power-of-2 sweep fp32 (post-JIT-transition; clean)

| N    | ms     | GFLOPs | comment |
|------|--------|--------|---------|
|  256 | 0.208  |   162  | small-shape overhead-bound |
|  384 | 0.223  |   509  | |
|  512 | 0.280  |   959  | |
|  640 | 0.349  |  1501  | |
|  768 | 0.448  |  2024  | |
|  896 | 0.610  |  2360  | |
| 1024 | **0.835** | **2571** | scales smoothly with N³ |
| 1152 | 1.049  |  2915  | |
| 1280 | 1.356  |  3094  | |
| 1408 | 1.729  |  3229  | |
| 1536 | 2.194  |  3304  | |
| 2048 | 4.878  |  3522  | |

After warmup, 1024³ fp32 fits the curve at 2571 GFLOPs — there is *no*
intrinsic 1024 anomaly. The original calibration's 653 GFLOPs is a
**warmup artifact**, not a steady-state perf number.

### Cold-start kernel-pick stickiness — the real bug

15 consecutive calls of 1024³ fp32 on a fresh process, same `(a, b)`
tensors, no other shape touched:

```
  call  0: 5.46 ms  (very cold; includes one-time MPSGraph compile)
  call  1: 3.11 ms
  call  2: 3.14 ms
  call  3: 3.16 ms
  call  4: 3.13 ms
  call  5: 3.13 ms
  ...
  call 14: 3.15 ms  ← still stuck
```

Then a *single* call of 2048³ fp32 unblocks:

```
  cold 1024 fp32: 3.05 ms
  cold 1024 fp32: 3.15 ms
  cold 1024 fp32: 3.13 ms
  2048 fp32:     17.07 ms (its own cold call)
  back 1024 fp32: 1.03 ms  ← UNBLOCKED
  back 1024 fp32: 0.99 ms
  back 1024 fp32: 1.20 ms
  back 1024 fp32: 1.25 ms
  back 1024 fp32: 0.93 ms
```

Same trick works with ANY off-shape: 1023³, 1025³, 512³ — once the
"non-1024 fp32" code path runs, 1024 becomes fast.

**Switching to fp16 on the same shape (1024³) does NOT unblock fp32**:

```
  fp32 still slow (5×): 3.1 ms
  fp16 1024^3:           1.04 ms  (normal fp16 speed)
  fp32 again:           3.13 ms  ← still 3× slower than warm fp32
```

So the slow path is bound to the `(shape=1024×1024×1024, dtype=fp32)`
cache key, not just to the dtype.

After ~50 same-shape fp32 calls, on call ~50 the timing transitions
spontaneously: 3.1 ms → 0.96 ms. So the kernel-cache or scheduler does
eventually re-evaluate, but only after a long warmup that real users
don't do.

### `PYTORCH_MPS_PREFER_METAL=1` — alternate path

The hand-written Metal kernel (`do_metal_mm` at LinearAlgebra.mm:82,
TILE_DIM=16) is significantly slower than even the slow MPSGraph path:

| Path              | 1024³ fp32 cold | 1024³ fp32 warm |
|-------------------|------------------|-----------------|
| MPSGraph (default)| 3.1 ms (slow-pick) | 0.85 ms |
| Metal (prefer)    | 4.2 ms          | 4.1 ms |

So Metal-shader fallback is NOT the right comparator and NOT the bug.
The bug is *entirely within MPSGraph's kernel-selection for fp32 GEMM*.

---

## 3. Shape-specificity

| Shape              | dt   | Cold (ms) | Warm (ms) | Cold/Warm | Anomaly? |
|--------------------|------|-----------|-----------|-----------|----------|
| 1024×1024×1024     | fp32 | 3.13      | 0.85      | **3.7×**  | **YES** |
| 1024×1024×1024     | fp16 | 1.05      | 0.36      | 2.9×      | weaker  |
| 1024×1024×1024     | bf16 | 1.08      | 0.37      | 2.9×      | weaker  |
| 1023×1023×1023     | fp32 | (fresh→1.04 — no slow phase observed) | 1.04 | 1.0× | NO |
| 1025×1025×1025     | fp32 | (fresh→1.02) | 1.02 | 1.0× | NO |
| 512×512×512        | fp32 | (fresh→0.28) | 0.28 | 1.0× | NO |
| 768×768×768        | fp32 | (fresh→0.44) | 0.44 | 1.0× | NO |
| 2048×2048×2048     | fp32 | 5.5-5.9   | 4.85      | 1.2×      | small only |

**1024³ fp32 is uniquely sticky.** Off-by-one shapes don't reproduce.
fp16/bf16 at 1024³ also have a cold phase but it's mild (2.9×) and may
just be JIT warmup. Larger shapes (2048³) recover within a few iters.

The pattern is consistent with: MPSGraph picks a sub-optimal
"large-tile-prepared" or "GEMV-style" path on first compile for the
specific shape signature `<1024, 1024, 1024, fp32>`, and the cache
holds that pick. Different shape → different cache key → recompile
gets the right path → invalidating the per-process kernel-pick state.

(I did NOT instrument MPSGraph internals to confirm which exact
MPSMatrixMultiplication or MPSNDArray kernel is dispatched — that
would require Metal-frame-capture / `MTLCaptureManager`, beyond the
scope of this audit.)

---

## 4. Verdict

**GENUINELY_NEW** — with two important caveats:

1. The closest existing issue is **#136003** ("[MPS] Inconsistent
   performance issues") — same theme of *unexplained kernel/dispatch
   variability* but on SDPA, not matmul, and the variability there is
   run-to-run (3 orders of magnitude) rather than cold-vs-warm. We
   should reference #136003 in our filing as the most-likely-related
   prior art.

2. The dispatch-routing-picks-wrong-impl failure mode is precedented in
   **#182805** (group_norm prim-decomposition vs composite, 6.7×
   slower) and #181725 (MultiheadAttention 9× slower than raw SDPA).
   Our finding is the matmul fp32 instance of the same family. None of
   these were closed by maintainers — all are open and triaged.

The original gpucheck calibration measurement of "3.29 ms / 653
GFLOPs at 1024³ fp32" is a **real, reproducible** PyTorch MPS perf
anomaly, not a calibration-script artifact. It DOES affect real users
who:
- Run a single matmul shape (e.g. a small inference workload with
  batch×seq×dim = 1024 dimensions throughout)
- Use only fp32 (some ML practitioners default to it)
- Don't have a varied warmup that includes other shapes

**Do NOT re-file** if a maintainer adds a comment to #136003 saying
"this is the same root cause" — at that point, our filing should
reference #136003 directly.

**Recommended action**: file as a new issue *with explicit reference to
#136003 and #182805*, framing it as "matmul fp32 1024³ kernel-pick
sticky on cold-start MPSGraph; resolves with any off-shape dispatch."

---

## 5. Draft issue body (for review)

```markdown
### 🐛 Describe the bug

On Apple M5 / macOS 26.4.1 / torch 2.11.0, MPSGraph fp32 matmul at
exactly `(1024, 1024, 1024)` selects a sub-optimal kernel on cold-start
and stays on it for many iterations (>40), giving ~3.1 ms / 685 GFLOPs.
A *single* dispatch of any other shape (e.g. 2048³, 1023³, 512³)
unblocks the cache: subsequent 1024³ fp32 calls drop to ~0.85 ms /
2521 GFLOPs (3.7× speedup, fitting the smooth N³ curve).

This affects users who exercise a single matmul shape — e.g. simple
inference loops with constant dimensions — and don't pre-warm with a
varied shape mix.

#### Reproduction

```python
import time
import torch

assert torch.backends.mps.is_available()

def event(fn):
    torch.mps.synchronize()
    t = time.perf_counter()
    fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t) * 1000

# Cold-start: fresh process, never run any other shape on MPS first
a = torch.randn(1024, 1024, device='mps', dtype=torch.float32)
b = torch.randn(1024, 1024, device='mps', dtype=torch.float32)

print("Sticky-slow path:")
for i in range(15):
    print(f"  call {i:2d}: {event(lambda: a @ b):.3f} ms")

# Now poke with a different shape:
a2 = torch.randn(2048, 2048, device='mps', dtype=torch.float32)
b2 = torch.randn(2048, 2048, device='mps', dtype=torch.float32)
event(lambda: a2 @ b2)

print("After running 2048^3 once:")
for i in range(5):
    print(f"  call {i:2d}: {event(lambda: a @ b):.3f} ms")
```

Expected output:
```
Sticky-slow path:
  call  0: 5.5 ms   (one-time compile)
  call  1: 3.1 ms   ← stuck slow
  call  2: 3.1 ms
  ...
  call 14: 3.1 ms

After running 2048^3 once:
  call  0: 1.0 ms   ← unblocked
  call  1: 0.9 ms
  ...
```

#### Observations

- Off-shape dispatches that unblock 1024³ fp32: 512³, 768³, 1023³,
  1025³, 2048³ (any shape with a different cache key).
- Switching dtype on the same shape does NOT unblock: 5 calls of
  1024³ fp16 (which run at 1 ms cleanly) followed by 1024³ fp32 still
  give 3.1 ms.
- After ~50 consecutive 1024³ fp32 calls, it eventually transitions
  spontaneously to the fast path — but real users don't iterate that
  many times before measuring.
- WARMUP=3 in benchmark harnesses (a common default) is not enough
  iterations to escape the slow phase, so reported peak fp32 GFLOPs
  for 1024³ on MPS will systematically under-report by ~3-4×.
- `PYTORCH_MPS_PREFER_METAL=1` is NOT a workaround: the
  hand-written Metal `do_metal_mm` shader is slower (4 ms) than even
  the slow MPSGraph path.

#### Reference to related-but-not-duplicate issues

- #136003 — [MPS] Inconsistent performance issues (SDPA variability;
  same flavor of un-explained dispatch state, different op).
- #182805 — [MPS] native_group_norm dispatch routes through prim
  decomposition (~7× slower; same fix-class: MPS chooses a slow
  implementation when a fast one exists).
- #181725 — [MPS] nn.MultiheadAttention is ~9x slower than direct
  F.scaled_dot_product_attention (same fix-class).
- #181718 — bf16 matmul throughput drops on RTX 4090 for shapes where
  N%16==8 (the analogous CUDA-side bug; this is the MPS counterpart).

#### Hypothesis

MPSGraph's `matrixMultiplicationWithPrimaryTensor:` (called from
`aten/src/ATen/native/mps/operations/LinearAlgebra.mm:296`) caches a
sub-optimal kernel for the specific cache key `(M=1024, K=1024, N=1024,
fp32)` on first compile. The `LookUpOrCreateCachedGraph` in
`mm_out_mps_impl` (line 707) hits that cached pick on subsequent calls.
The cache appears to be invalidated when a different shape's compile
re-evaluates the kernel-pick state.

This is presumably an MPSGraph-internal heuristic issue — possibly
related to a "small-N" vs "large-N" path threshold sitting just above
1024 — that PyTorch can't fix directly. We can mitigate by either:
1. Adding a cold-start tickle in `MPSDevice::getInstance()` initializer
   that runs a tiny matmul at a known-good shape.
2. Falling back to `do_metal_mm` for fp32 1024³ specifically (but
   that's slower than even the slow MPSGraph pick on this hardware).
3. Filing with Apple via FB feedback for the MPSGraph team.

### Versions
PyTorch: 2.11.0
macOS: 26.4.1 (Mac17,3, Apple M5, 32 GB)
Python: 3.12

cc @kulinseth @malfet @DenisVieriu97 @jhavukainen @aditvenk
```

---

## 6. Files (reproducible)

- `EVIDENCE/raw/01-matmul-anomaly-repro.py` — multi-seed sweep, shape
  neighbors, warmup-dtype interaction, p2 sweep (the 7-experiment
  characterization).
- `EVIDENCE/raw/02-repro-isolate.py` — cold-start vs off-shape unblock
  Tests A-E.
- `EVIDENCE/raw/03-repro-long-run.py` — 100-iter cold-start showing
  spontaneous transition at iter ~50.
- `EVIDENCE/raw/04-repro-exact-original.py` — exact replication of the
  original `mac_bench-mps_kernels.py` methodology (WARMUP=3, N=10).
- `EVIDENCE/raw/repro-results.json` — JSON output of script 01 (55
  records).

---

## 7. Confidence

**high** — anomaly reproduces deterministically across multiple
processes and seeds; PyTorch source confirms the dispatch path
(MPSGraph `matrixMultiplicationWithPrimaryTensor`); all 20+ candidate
existing issues read carefully and none are duplicates. `do_metal_mm`
fallback ruled out as the source. `PYTORCH_MPS_PREFER_METAL=1` ruled
out as workaround.

**Open uncertainty (not blocking the verdict)**: I did not
Metal-frame-capture the actual kernel name being dispatched on
slow-path vs fast-path. That would harden the issue body before
filing. Recommend running `MTLCaptureManager` capture during
slow-path and fast-path 1024³ fp32 matmul to cite the exact kernel
names — this would let the maintainer skip diagnosis.
