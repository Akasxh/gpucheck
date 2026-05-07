# SYNTHESIS v2 — gpucheck v1.0 MPS Backend (Round 2 consolidation)

**Slug:** v1.0
**Date:** 2026-05-01
**Owner:** research-synthesist-v2 (under research-lead)
**Supersedes:** `SYNTHESIS_v1.md` where claims conflict; cites it as the prior
synthesis whenever Round 2 updates a Round 1 answer.
**Downstream:** engineering-lead (CHARTER.md must adopt SYNTHESIS_v2.md instead
of SYNTHESIS_v1.md).

---

## Headline (revised)

**Ship gpucheck v1.0 with an MPS backend** — direction unchanged from
SYNTHESIS_v1.md L10–17. Two structural changes from Round 2:

1. **The "2× tolerance multiplier" is dead.** Round 2 empiricist measurement
   on Apple M5 (`EVIDENCE/empiricist-v2.md` §"Compact result table") refuted
   the 2× multiplier for 5 of 12 (kernel × dtype) pairs. Replace with a
   per-(kernel, dtype) overlay; matmul/bf16 needs **32×**, matmul/fp16 needs
   **20×**, matmul/fp32 needs **16×**, conv2d/bf16 needs **8×**, conv2d/fp16
   needs **5×**. Attention and layernorm 2× holds (often 100× conservative).
2. **The xfail list expands from 12 → 43.** GitHub-miner v2 added 31 new
   xfail-eligible MPS bugs (`EVIDENCE/github-miner-v2.md` §"Recommended xfail
   expansion"), most of them silent-correctness — including 5 OOB-indexing-
   silently-returns-zero issues, 4 unsigned-dtype garbage cases, and 5
   non-contiguous failure-class issues.

Two sharpened-but-not-flipped findings:

3. **Determinism skepticism strengthened at source level.** Round 2 librarian
   verified Apple's MSL spec §6.15.4 locks `atomic_float` add/sub to
   `memory_order_relaxed`, and SIMD-group matrix element-to-thread mapping is
   "unspecified" (`EVIDENCE/librarian-v2.md` §1). PyTorch's source contains
   exactly 3 MPS-aware deterministic gates (`index_put_`, `index_add_`,
   `kthvalue`) — none of the gpucheck-targeted ops. SYNTHESIS_v1.md §"Sub-Q 4"
   was correct in conclusion; v2 hardens the *source*.
4. **Deadlock #162872 is STILL UNFIXED in HEAD.** Archaeologist v2 verified
   `MPSEvent.mm:228 waitForCpuSync()` survives in `pytorch/main` as of
   2026-05-01 (`EVIDENCE/archaeologist-v2.md` §"Pivotal commit 1") and PR
   #162874's one-line fix was closed without merge. SYNTHESIS_v1.md §"Sub-Q 3"
   API gotcha is upgraded from "PROVISIONAL" to "PERMANENT for v1.0" — the
   gpucheck `gpu_benchmark` MPS path needs an explicit deadlock probe (worker
   thread + timeout fallback), not just an avoidance pattern.

**Confidence:** HIGH on the v1.0 ship recommendation; HIGH on the per-
(kernel, dtype) overlay direction (numbers MEDIUM, M5-only); HIGH on the
expanded xfail list; HIGH on the determinism contract; HIGH on the deadlock
probe requirement.

---

## Sub-Q 1 — MPS correctness baseline (Round 2 update)

### Round 1 answer

`SYNTHESIS_v1.md` §"Sub-Q 1" enumerated kernel-by-kernel correctness vs CUDA
across matmul, softmax, layer_norm, SDPA, BCE, conv2d. All citations primary
(28 PyTorch issues). Status: HIGH confidence.

### Round 2 update

**Strengthened (no flip).** Three new dimensions:

1. **Quantitative drift envelope on M5** — `EVIDENCE/empiricist-v2.md`
   §"Compact result table" measured P99 `|MPS−CPU|` across 4 kernels × 3 dtypes
   (200 iters each). Headline numbers now backed by direct measurement, not
   inference from CUDA precedent:
   - matmul/fp32 P99 = 1.37e-3 (13.7× CUDA atol baseline 1e-4)
   - matmul/fp16 P99 = 1.67e-1 (16.7× baseline 1e-2)
   - matmul/bf16 P99 = 1.31 (26.1× baseline 5e-2)
   - attention/{fp32, fp16, bf16}: P99 well within 0.2× baseline (FlashAttention
     fused-softmax denominator absorbs accumulation error)
   - layernorm/{fp32, fp16, bf16}: P99 within 0.6× baseline
   - conv2d/fp32: 0.61× baseline; conv2d/fp16: 3.84× baseline; conv2d/bf16:
     6.14× baseline
2. **Apple's own framework is non-deterministic on M5** —
   `EVIDENCE/archaeologist-v2.md` §"Pivotal commit 5" surfaces commit
   49e7d4dadbbe (PR #181466, malfet, 2026-04-28): "MPSNDArrayMatrixMultiplication
   and MPSGraph matrixMultiplication produce non-deterministic results for >2D
   fp16/bf16 inputs on Apple10 GPUs (M5)". This is **upstream Apple** producing
   nondeterminism, not just PyTorch's wrapper. PyTorch ships a chip-conditional
   2D-flatten workaround. Implication: gpucheck must detect chip generation
   (M1/M2/M3/M4/M5).
3. **Long-tail correctness bugs much larger than v1's 12** —
   `EVIDENCE/github-miner-v2.md` §"Recommended xfail expansion" identifies 31
   NEW xfail-eligible bugs beyond v1's list, with patterns:
   - **5 OOB-index-silently-returns-zero**: #144824, #154235, #170370 (offsets[0]),
     #170507 (one_hot), #160553 (advanced indexing). gpucheck must add an
     *error-raising* expectation contract.
   - **4 unsigned-dtype garbage**: #176296 binary ops on uint16/32/64 return
     `[6.88e-16, 1.08e-25]` instead of `[0, 1]`.
   - **5 non-contiguous failure-class**: argmax (#160740), linear (#161640),
     scatter, BatchNorm2d, grid_sampler. CLAUDE.md "No stride/contiguity
     fuzzing" is **the load-bearing v1.0 gap**.
   - **3 large-tensor 64-bit overflow**: #182052 copy_ at offset >2^32, #154322
     fp16→fp32 zero-out at >43000², #161865 SEGFAULT in libomp on M4 Max.
   - **4 closed-without-completion landmines**: #181867, #175191, #89708,
     #150051 — closed for stale/dup, NOT verified-fix.

### What changes for engineering

- The kernel-correctness section in `MPSBackend` documentation expands from
  ~12 known-broken items to ~43 plus 4 landmines.
- `gpucheck.fuzzing` must add **stride/contiguity fuzzing** as a v1.0 priority
  (was a "Known Weakness" in CLAUDE.md; Round 2 quantifies the cost).
- `gpucheck.assertions` needs an `assert_raises_oob` contract for embedding /
  index_select / one_hot tests on MPS, since silent-zero is the bug.

**Citations:** SYNTHESIS_v1.md §"Sub-Q 1"; EVIDENCE/empiricist-v2.md §"Compact
result table" + §"Cross-shape sanity check"; EVIDENCE/archaeologist-v2.md
§"Pivotal commit 5"; EVIDENCE/github-miner-v2.md §"Recommended xfail expansion"
+ §"Cross-repo patterns"; pytorch#181466, pytorch#176296, pytorch#182052,
pytorch#154322, pytorch#144824, pytorch#170507, pytorch#160553, pytorch#181867
(landmine), pytorch#175191 (landmine).

**Confidence:** HIGH (Round 1's HIGH stands; Round 2 measurement strengthens
the quantitative claims; the 31-bug expansion is fully URL-verified).

---

## Sub-Q 2 — Open MPS bugs (Round 2 update)

### Round 1 answer

`SYNTHESIS_v1.md` §"Sub-Q 2" enumerated 12 highest-impact open bugs (live count
255). HIGH confidence.

### Round 2 update

**Strengthened.** Live count: **252 open** as of 2026-05-01 (down 3 in ~24h —
`EVIDENCE/github-miner-v2.md` §"Anomalies"). Round 2 expanded the actionable
xfail list **12 → 43** by adding 31 new entries (full table in
`EVIDENCE/github-miner-v2.md` §"Recommended xfail expansion"). Top-5 by
recommended urgency (`EVIDENCE/github-miner-v2.md` §"Top-5"):

1. **#182052** — `copy_` strided wrap at offset > 2^32 (silent data loss in
   production-sized tensors; updated 2026-04-30, *yesterday*)
2. **#176296** — binary ops on uint{16,32,64} return total numerical garbage
3. **#179608** — avg_pool1d prefix-sum drift produces negative values from
   non-negative input
4. **#163327** — scatter_add_ silent no-op on slice with offset > 0
5. **#162092** — Voxtral ASR returns gibberish on MPS

Plus **4 landmines** (closed-without-completion, NOT verified fix):
#181867 (boolean indexing), #175191 (scatter on non-contig), #89708 (M1 stale),
#150051 (chebyshev_polynomial_t).

### What changes for engineering

- `pyproject.toml` `[tool.gpucheck.mps.xfail]` table grows to 43 entries
  (full mapping `issue → kernel.dtype.shape_cond → recommended xfail key` in
  EVIDENCE/github-miner-v2.md table).
- A separate `[tool.gpucheck.mps.regression_fence]` section captures the 4
  landmines, where bug recurrence indicates upstream regression rather than
  a static known-issue.
- The OOB-indexing class (5 issues) collapses into one shared test pattern:
  any test that calls embedding / index_select / one_hot / advanced indexing
  on MPS must assert that out-of-range indices RAISE (not silently zero).

**Citations:** SYNTHESIS_v1.md §"Sub-Q 2"; EVIDENCE/github-miner-v2.md
§"Recommended xfail expansion" (full table) + §"Top-5 by recommended-xfail
urgency" + §"Dangerous closed-but-unfixed issues (LANDMINES)".

**Confidence:** HIGH (live API verification; all 31 new issues URL-verified).

---

## Sub-Q 3 — torch.mps API surface (Round 2 update)

### Round 1 answer

`SYNTHESIS_v1.md` §"Sub-Q 3" enumerated the public API at docs.pytorch.org/2.11
(librarian §1-§4) and flagged the deadlock #162872 + memory-accounting #164299.
HIGH confidence on docs surface.

### Round 2 update

**Strengthened on three axes.**

1. **Type-stub bugs documented** (`EVIDENCE/linguist-v2.md` §2):
   - `torch.mps.compile_shader` and `torch.mps.load_metallib` are NOT in the
     stub block at `torch/_C/__init__.pyi.in:2072-2096`. mypy --strict against
     gpucheck code that uses these will see `Returns Any`.
   - `set_per_process_memory_fraction(fraction)` is untyped; runtime requires
     `float` (not even `int`).
   - `load_metallib(source)` is untyped despite branching on
     `bytes | bytearray | str | os.PathLike`.
   - Triple-`l` typo in native binding `_mps_loadMetalllib` (NOT to be fixed).
   - `Event.elapsed_time` returns `float` with no unit hint in the type system
     (it is milliseconds, per `MPSEvent.mm:221-238` `1e-6` factor).
2. **dtype × MPS support matrix sharpened with silent-downcast asymmetry**
   (`EVIDENCE/linguist-v2.md` §3): a 0-d `torch.tensor(1.0, dtype=torch.float64)`
   does NOT raise on MPS — it silently runs as fp32. A 1-d
   `torch.tensor([1.0], dtype=torch.float64)` raises. Source:
   `aten/src/ATen/native/mps/OperationUtils.mm:120-158` (scalar path falls
   through `Double → Float`). Same for complex128 → complex64. **gpucheck's
   `assert_close` could be fooled when the user expects fp64 reference precision
   but is silently being given fp32**.
3. **Deadlock #162872 promoted from "API gotcha" to "permanent v1.0
   constraint"** (`EVIDENCE/archaeologist-v2.md` §"Pivotal commits 1 & 2"):
   - The original sin: commit cdfd0ea16282 (PR #102121, razarmehr, 2023-08-08)
     introduced `MPSEvent.mm:228 waitForCpuSync()`. ONLY commit ever to touch
     this file.
   - PR #162874 (oraluben, 2025-09-13) proposed the one-line fix; closed
     WITHOUT MERGE. Issue #162872 still OPEN.
   - File grep on `pytorch/main` HEAD as of 2026-05-01 confirms line 228
     `end_event->waitForCpuSync()` is still there.
   - PR #141296 (malfet, 2024-11-21) added a regression test, but only for
     `torch.mps.synchronize()`, not for `Event.synchronize()` — the deadlock
     vector is uncovered upstream.

### What changes for engineering

- gpucheck's Backend Protocol gets `unsupported_dtypes: frozenset[torch.dtype]`
  and `silently_downcast_dtypes: frozenset[torch.dtype]` (linguist-v2 §5.1).
  MPSBackend implementation:
  ```python
  unsupported_dtypes = frozenset({torch.float64, torch.complex128, torch.float8_e4m3fn, torch.float8_e5m2})
  silently_downcast_dtypes = frozenset({torch.float64, torch.complex128})  # for 0-d only
  ```
- `gpucheck.assertions.assert_close` must consult `silently_downcast_dtypes`
  before picking the reference precision; otherwise the reference will be
  silently downcast and the assertion becomes vacuous.
- `EventTimer` Protocol grows `units: Literal["ms"]` to lock down the unit
  contract (linguist-v2 §5.2).
- `gpu_benchmark` MPS path **needs an explicit deadlock probe**: spawn the
  `Event.elapsed_time(...)` call in a worker thread with a timeout; on
  timeout, fall back to host-clock measurement. Direct avoidance is
  insufficient because user code passing custom Events can re-trigger the
  bug. (archaeologist-v2 §"5-PR Deep Dive" row 2.)
- `arch/` module gets Apple-chip-generation detection (M1/M2/M3/M4/M5) the
  same way SM60-SM120 detection works for CUDA — required to gate matmul
  determinism tests by chip family per #181466 (archaeologist-v2 §"5-PR Deep
  Dive" row 5).

**Citations:** SYNTHESIS_v1.md §"Sub-Q 3"; EVIDENCE/linguist-v2.md §1, §2, §3,
§5; EVIDENCE/archaeologist-v2.md §"Pivotal commit 1", §"Pivotal commit 2",
§"Timeline" rows for #102121, #141296, #162872, #162874.

**Confidence:** HIGH.

---

## Sub-Q 4 — Determinism guarantees (Round 2 update)

### Round 1 answer

`SYNTHESIS_v1.md` §"Sub-Q 4" concluded: PyTorch MPS docs SILENT on determinism;
empirical record contradicts; verdict "best-effort deterministic". Confidence
HIGH on practical conclusion / MEDIUM on quantitative MSL claims (Apple MSL
spec was REPORTED-NOT-VERIFIED).

### Round 2 update

**Strengthened — Apple MSL spec is now VERIFIED.** `EVIDENCE/librarian-v2.md`
§1 cites Apple primary sources via Context7's mirror of Apple's MSL spec PDF:

- **MSL §6.15.4 Atomic Functions** locks `atomic_float` fetch_add to
  `memory_order_relaxed` — **the only memory order Metal supports for atomic
  fetch-and-modify** (Source B). Apple's own recommended SIMD-group reduction
  pattern (which compiles into MPS's matmul/softmax/layer_norm reduction
  trees) ends in `atomic_fetch_add_explicit(output, val, memory_order_relaxed)`.
  This is **the textbook source of run-to-run non-determinism** for
  non-associative FP arithmetic. Apple's spec **mandates** the silence; it is
  not an oversight.
- **MSL §2 SIMD-group Matrix Data Types** says "the mapping of matrix elements
  to threads in the SIMD-group is **unspecified**" (Source C). Two driver
  versions can give different reduction trees and therefore different
  last-bit rounding for the same input — **by spec**.
- **MSL §1 fp pragma** documents `#pragma METAL fp math_mode([relaxed | safe |
  fast])` and `fp contract([off | on | fast])` (Source D). Default is `fast`
  with `-ffast-math`; reassociation and FMA contraction are both permitted.
  Apple does NOT document MPSGraph's internal mode; users have no API to
  override.
- **WWDC sessions** (24/10160, 24/10218, 25/205, 25/262) all retrieved fresh
  2026-05-01 — **zero mentions of determinism, reproducibility, numerical
  accuracy, atomic ops, or IEEE 754** (Source E). Only "lock in random seeds
  for reproducible results" — RNG-seed level only.

**Strengthened on the PyTorch source side.** Librarian v2 grepped 21 MPS
operation files in PyTorch v2.11.0 and found:
- **Only 3 ops** wire into `globalContext().deterministicAlgorithms()` on MPS:
  `index_put_` (Indexing.mm:205), `index_add_` (Indexing.mm:516-536),
  `kthvalue` (Sort.mm:170 with `alertNotDeterministic`).
- **Zero** of the gpucheck-targeted ops (matmul, softmax, layer_norm, attention,
  conv2d, BCE, BatchNorm, F.linear backward) gate on `deterministicAlgorithms()`
  on MPS. The skeptic's worry that "the contract is inherited from CUDA
  equivalents" is **falsified at the source level**.

### What changes for engineering

- `gpucheck.assertions` must NOT promise "deterministic parity on MPS" in
  any user-facing surface. Apple's spec mandates the silence; PyTorch did not
  inherit the CUDA gates.
- New `@gpucheck.mps_determinism_check` fixture (librarian-v2 §5.2): runs the
  wrapped op N=5 times on MPS and asserts bit-exact identical outputs, fails
  if Apple's spec class (atomic-add, simdgroup-matrix layout) prevents that.
- Maintain a **known-non-deterministic-on-MPS xfail list** (librarian-v2 §5.3)
  starting with: matmul, linear, addmm (FP16+BF16); softmax, log_softmax;
  layer_norm, rms_norm, batch_norm; SDPA; cross_entropy, nll_loss; conv2d,
  conv2d_backward; F.linear backward (BF16/FP16); index_add, index_put;
  kthvalue; scatter_add, scatter.
- gpucheck honors `torch.use_deterministic_algorithms(True)` for the 3 ops
  PyTorch already gates on MPS; for the rest, gpucheck auto-xfails unless
  `@gpucheck.allow_nondeterministic_mps('reason')` decoration.

**Citations:** SYNTHESIS_v1.md §"Sub-Q 4" (prior); EVIDENCE/librarian-v2.md
§1 Sources A-E (Apple MSL primary), §2 Sources F-I (PyTorch docs), §3 Sources
J-K (PyTorch source-level gates), §5 (recommended API surface).

**Confidence:** HIGH on the conclusion; HIGH on MSL spec quantitative claims
(no longer REPORTED-NOT-VERIFIED — librarian v2 cited primary).

---

## Sub-Q 5 — llama.cpp Metal dispatch precedent (Round 2 — no update)

### Round 1 answer

`SYNTHESIS_v1.md` §"Sub-Q 5": `ggml-metal supports_op()` / per-op CPU fallback
pattern; `tests/test-backend-ops.cpp` NMSE per-op tolerance; `commit 62bfef5`
discipline of "disable the kernel for that shape, fall back to CPU/alternate
path". HIGH confidence.

### Round 2 update

No new evidence dispatched on this sub-question in Round 2. Cartographer v2
incidentally reconfirmed the `N_SIMDWIDTH 32` constant and `simdgroup_half8x8`
kernel structure in ggml-metal (`EVIDENCE/cartographer-v2.md` Source S3),
which strengthens the precedent that the SIMD-width / 8x8-fragment combination
is the *production* tile shape, not a theoretical one. No flip.

### What changes for engineering

Unchanged from SYNTHESIS_v1.md §"Sub-Q 5".

**Citations:** SYNTHESIS_v1.md §"Sub-Q 5"; EVIDENCE/cartographer-v2.md
Source S3 for ggml-metal `N_SIMDWIDTH 32` cross-validation.

**Confidence:** HIGH (unchanged).

---

## Sub-Q 6 — FlashAttention / Triton on Metal (Round 2 update)

### Round 1 answer

`SYNTHESIS_v1.md` §"Sub-Q 6": no Triton-Metal upstream; MLX has fused SDPA;
PyTorch MPS dispatches to "math" backend (slow path, not fused) per
pytorch#179294. HIGH confidence.

### Round 2 update

**Strengthened with kernel-internal precision data.** Tracer v2
(`EVIDENCE/tracer-v2.md` §1.1-§1.5) traced MLX's SDPA stack file-by-line:

1. **Kernel-internal upcast confirmed**: `mlx/backend/metal/kernels/sdpa_vector.h:50`
   `typedef float U;` — MLX upcasts to FP32 for the inner accumulators
   regardless of input dtype. This is **why MLX can hold atol=3e-4 on FP16**
   (test_fast_sdpa.py:439). Same FlashAttention numeric trick.
2. **MLX's tolerance schedule is empirical, not contractual**:
   - FP32: `atol = 2e-5`
   - FP16/BF16: `atol = 3e-4`
   - long-mask: `atol = 1e-3`
   These are MLX testing MLX, not a public contract.
3. **gpucheck's MPS-FP16 atol = 2e-2 = 66× looser than MLX's 3e-4**
   (`EVIDENCE/tracer-v2.md` §2.2 + §6). This is **the headline differential**.
   gpucheck would PASS test cases where MLX's own harness FAILS.
4. **MLX has THREE Metal SDPA kernels** with different numeric profiles:
   `sdpa_full_self_attention_metal`, `sdpa_full_self_attention_nax` (M5 NAX
   gated), `sdpa_vector` and `sdpa_vector_2pass`. PyTorch MPS-SDPA path
   currently dispatches to `math` decomposition, not any of these
   (pytorch#179294 unfixed).
5. **The right default oracle is `torch.<op>(*tensors_on_cpu)`, NOT MLX**
   (`EVIDENCE/tracer-v2.md` §6 verdict): different kernel populations make
   MLX uninformative as oracle for `torch.SDPA(device='mps')` tests; MLX is a
   **tie-break oracle** for CPU-vs-MPS disagreements, plus a **calibration
   data source** for the tolerance schedule.

### What changes for engineering

- gpucheck adopts **MLX's tolerance schedule as a calibration anchor**, not as
  the kernel oracle. Specifically: gpucheck's SDPA-on-MPS tolerance for FP16
  should be moved from 2e-2 toward 3e-4 (MLX's empirical number) — but only
  AFTER the per-(kernel, dtype) overlay from empiricist v2 is wired in;
  empiricist measured PyTorch MPS SDPA at `attention/fp16 P99 = 1.15e-3`,
  which is 11.5× MLX's 3e-4 and 0.115× gpucheck's 2e-2. The right number for
  PyTorch MPS SDPA-fp16 is closer to **5e-3** (5× MLX, 0.25× current
  gpucheck).
- gpucheck adds an **optional MLX tie-break oracle** for SDPA tests where
  CPU-PyTorch and MPS-PyTorch disagree by more than the MPS overlay tolerance.
  If MLX agrees with MPS, the CPU path is suspect; if MLX agrees with CPU,
  MPS is suspect.

**Citations:** SYNTHESIS_v1.md §"Sub-Q 6"; EVIDENCE/tracer-v2.md §1.1-§1.5
(MLX dispatch trace), §2.2 (gpucheck tolerance), §3 (differential), §6
(verdict).

**Confidence:** HIGH on the differential (66× tolerance gap); HIGH on the
"don't use MLX as default oracle" verdict; MEDIUM on the recommended new
SDPA-fp16 atol of 5e-3 (single-SKU empirical, M5).

---

## Sub-Q 7 — Tolerance defaults (Round 2 — REFUTED + REPLACED)

### Round 1 answer

`SYNTHESIS_v1.md` §"Sub-Q 7" recommended a **2× FlashAttention multiplier on
top of CUDA atol** as the per-dtype tolerance overlay:
- fp32: 1e-4 → 2e-4 (2×)
- fp16: 1e-2 → 2e-2 (2×)
- bf16: 5e-2 → 1e-1 (2×)

Confidence: MEDIUM (PROVISIONAL until M-machine calibration). Skeptic v1
attack #2 flagged this as "hypothesis dressed as recommendation".

### Round 2 update — **REFUTED FOR 5 OF 12 (kernel × dtype) PAIRS**

`EVIDENCE/empiricist-v2.md` §"Compact result table" measured 200 iterations × 4
kernels × 3 dtypes = 2,400 MPS-vs-CPU comparisons on Apple M5 (commit
`82b853e3c933d21d055f844ed21d6c0eb760a46e`, torch 2.11.0, macOS 26.4.1).
Variance check across seeds CAFE/BABE/DEAD: ±5% at P99.

**Hypothesis: REFUTED for 5 of 12 (kernel × dtype) pairs.**

| kernel    | dtype    | abs_p99   | mult vs CUDA atol | covers FA-2× ? |
|-----------|----------|-----------|-------------------|----------------|
| matmul    | fp32     | 1.37e-3   | 13.7×             | NO (refuted)   |
| matmul    | fp16     | 1.67e-1   | 16.7×             | NO (refuted)   |
| matmul    | bf16     | 1.31      | 26.1×             | NO (refuted)   |
| attention | fp32     | 1.01e-6   | 0.01×             | yes (conserv)  |
| attention | fp16     | 1.15e-3   | 0.11×             | yes (conserv)  |
| attention | bf16     | 9.64e-3   | 0.19×             | yes (conserv)  |
| conv2d    | fp32     | 6.10e-5   | 0.61×             | yes            |
| conv2d    | fp16     | 3.84e-2   | 3.84×             | NO (refuted)   |
| conv2d    | bf16     | 3.07e-1   | 6.14×             | NO (refuted)   |
| layernorm | fp32     | 9.54e-7   | 0.01×             | yes (conserv)  |
| layernorm | fp16     | 3.74e-3   | 0.37×             | yes            |
| layernorm | bf16     | 2.94e-2   | 0.59×             | yes            |

The 2× multiplier is **wrong in both directions**: dangerously under-conservative
for matmul/all dtypes and conv2d/{fp16, bf16}, dangerously over-conservative
for attention/all dtypes and layernorm/fp32 (where the FA fused-softmax
denominator absorbs accumulation error per FlashAttention paper Appendix B —
new citation Round 2). Higham 2002 Theorem 3.5 predicts the matmul/bf16 P99
of 1.31 from the bf16 ε ≈ 7.8e-3 floor accumulating over K (NEW citation,
empiricist-v2 §Citations).

### Replacement: per-(kernel, dtype) overlay

```toml
# Drop-in replacement for the Round-1 "2×-everywhere" recommendation
[tool.gpucheck.mps.tolerances]
default = {fp32 = 2e-4, fp16 = 2e-2, bf16 = 1e-1}      # 2× CUDA, FA-precedent — kernels not in tables below

[tool.gpucheck.mps.tolerances.matmul]
fp32  = 2e-3       # 20× — covers measured P99=1.37e-3 with headroom
fp16  = 2e-1       # 20× — covers measured P99=1.67e-1
bf16  = 2.0        # 40× — covers measured P99=1.31, P99.9=1.41

[tool.gpucheck.mps.tolerances.conv2d]
fp16  = 5e-2       # 5× — covers measured P99=3.84e-2
bf16  = 4e-1       # 8× — covers measured P99=3.07e-1
```

**The right shape for the recommendation is per-(kernel, dtype) atol with
per-SKU override, NOT a single global multiplier** (empiricist-v2 §Confidence).

### What changes for engineering

- `assertions/tolerances.py:35-43` `_MPS_TOLERANCE_MULTIPLIERS = {... 2.0 ...}`
  is removed. Replace with a per-(kernel, dtype) overlay table sourced from
  `pyproject.toml [tool.gpucheck.mps.tolerances]`.
- `assertions/close.py:165-167 compute_tolerance(...)` signature gains an
  `op_name` parameter. The `gpucheck_op` pytest mark already exists for xfail
  registry — same plumbing.
- `pyproject.toml` ships per-SKU sections: `[tool.gpucheck.mps.tolerances.m5.matmul]`,
  `[tool.gpucheck.mps.tolerances.m4.matmul]`, etc. SYNTHESIS_v1's "calibrate
  on M-machine" promise is fulfilled FOR M5 only; M3/M4 are Round 3 candidates
  if the engineering team has access.
- `assertions/close.py` uses MLX-derived calibration anchors (3e-4 fp16, 2e-5
  fp32) for SDPA specifically, since gpucheck's existing 2e-2 is 66× looser
  than MLX's empirically-validated number (tracer-v2 §2.2).

**Citations:** SYNTHESIS_v1.md §"Sub-Q 7" (the now-refuted 2× recommendation);
EVIDENCE/empiricist-v2.md §"Compact result table" + §"Cross-shape sanity check"
+ §"Per-dtype recommended replacement" + §"Combos that BREACH 2×"; Higham 2002
Theorem 3.5; FlashAttention NeurIPS 2022 Appendix B; tracer-v2 §1.5 +§2.2 for
MLX calibration anchor.

**Confidence:** HIGH on the refutation (variance-stable across seeds, 12 pairs
direct-measured); HIGH on the per-(kernel, dtype) shape; MEDIUM on the *exact*
numbers (16/20/32× per dtype — single-SKU M5; M3/M4 may shift by factor 2-3).

---

## Sub-Q 8 — gpucheck's existing CUDA bug-finding record (Round 2 update)

### Round 1 answer

`SYNTHESIS_v1.md` §"Sub-Q 8": triton#9838 (83.4% layer_norm error, OPEN) and
triton#9839 (FP16 matmul drift K=8192, CLOSED) verified externally.
Strategy: `TILE_SIZES = (32, 64, 128)`, `PRIMES = (7, 13, 31, 127, 257)`, etc.
Strategy "transfers verbatim to MPS" modulo skeptic §1 caveat about NVIDIA-tile
calibration. HIGH confidence on the 2 external bugs; MEDIUM on the broader
"8 bugs" claim.

### Round 2 update

**Strengthened on the strategy-transfer claim — the skeptic's caveat was
half-right.** Cartographer v2 (`EVIDENCE/cartographer-v2.md`) mapped Apple's
canonical tile constants from MLX + Apple Feature Set Tables + dougallj G13
GPU reference + ggml-metal:

- **32 is correct** for SIMD width: confirmed across MLX (`SIMD_SIZE = 32`),
  ggml-metal (`N_SIMDWIDTH 32`), dougallj G13 RE, philipturner metal-benchmarks,
  Apple Feature Set Tables. Skeptic's worry about 32 was wrong; gpucheck's
  TILE 32 transfers.
- **8 is the load-bearing Apple tile constant** — `simdgroup_matrix<T, 8, 8>`
  is the Apple MMA fragment, used by both MLX (`mma.h:20-40`
  `BaseMMAFrag<T, 8, 8>`) and ggml-metal (`simdgroup_half8x8` etc.). NOT in
  gpucheck's current `TILE_SIZES = (32, 64, 128)`.
- **16 is also load-bearing** — dominant `BK` in MLX GEMM (4 of 6 instantiations)
  and conv (all 6 instantiations); also Apple's threadgroup memory length
  alignment.
- **80 is an Apple-only FA head_dim** that NVIDIA FA never ships. NOT in any
  POWER_OF_2_BOUNDARIES set.
- **128 is M5-NAX only**. For M1-M4, 128 is rare in MLX. gpucheck's existing
  128 still applies (NAX path on M5+).

### Replacement fuzzer extension

Per `EVIDENCE/cartographer-v2.md` §5:

```python
TILE_SIZES_MPS = (8, 16, 32, 64, 128)  # add 8 and 16
POWER_OF_2_BOUNDARIES_MPS = (
    7, 8, 9,        # 8x8 MMA fragment boundary (NEW for MPS)
    15, 16, 17,     # GEMM BK=16 boundary (NEW for MPS)
    31, 32, 33,     # SIMD width / GEMM BK=32
    63, 64, 65,     # GEMM BM/BN=64 dominant
    79, 80, 81,     # FA head_dim 80 (Apple-only) (NEW)
    127, 128, 129,
    255, 256, 257,
    511, 512, 513,
)
```

PRIMES `(7, 13, 31, 127, 257)` and LARGE_DIMS `(2048, 4096, 8192)` are
device-independent and need no Apple recalibration.

### What changes for engineering

- `src/gpucheck/fuzzing/shapes.py:9-16` adds an MPS-specific tile set and
  boundary set per cartographer-v2 §5. The 8x8 fragment, 16 BK, and 80
  head_dim probes are NEW — they would have missed Apple-tile-aligned bugs in
  v1 fuzz.
- The MPS-fuzz priority order (degenerate > non-tile-aligned > prime >
  power-of-2 > large > mixed) is unchanged — only the tile/boundary constants
  shift on MPS device.
- The README claim "8 bugs found" tightens per skeptic §6 retained from
  SYNTHESIS_v1: "8 bugs found via 511 test configurations, of which
  triton#9838 (open) and triton#9839 (closed) are filed and externally
  verified".

**Citations:** SYNTHESIS_v1.md §"Sub-Q 8"; EVIDENCE/cartographer-v2.md §1-§5
(11 NEW citations: Apple Feature Set Tables PDF, dougallj G13 RE,
philipturner metal-benchmarks, MLX `mma.h`, MLX `steel_gemm_fused.metal`,
MLX `matmul.cpp`, MLX `steel_attention.metal`, MLX `sdpa_vector.h`,
MLX `rms_norm.metal`+`softmax.h`, MLX `conv.cpp`+`steel_conv.metal`).

**Confidence:** HIGH on the strategy-transfers-with-extension claim; HIGH on
the new tile constants (production-shipping in MLX, not theoretical); MEDIUM
on PyTorch-MPS-internal tiles (MPSGraph is closed-source — MLX is the proxy).

---

## Engineering team must respect (top 5 findings, revised from v1's top 3)

1. **The "2× multiplier" recommendation from SYNTHESIS_v1 is REFUTED.**
   Replace with the per-(kernel, dtype) overlay table in
   `EVIDENCE/empiricist-v2.md` §"Compact result table" + §"Per-dtype
   recommended replacement". Specifically: matmul/bf16 needs 32×, matmul/fp16
   needs 20×, matmul/fp32 needs 16×, conv2d/bf16 needs 8×, conv2d/fp16 needs
   5×. Attention and layernorm 2× holds. Without this, gpucheck's matmul/bf16
   tests on MPS will report false negatives — quietly passing kernels that
   are 5-26× outside the bf16 precision floor.

2. **The deadlock #162872 is permanent for v1.0.** PR #162874's one-line fix
   was closed without merge; `MPSEvent.mm:228 waitForCpuSync()` survives in
   `pytorch/main` HEAD. gpucheck's `gpu_benchmark` MPS path **needs an explicit
   deadlock probe** (worker thread + timeout + host-clock fallback), not just
   an avoidance pattern. Direct avoidance is insufficient because user code
   passing custom Events can re-trigger the bug. (Source:
   EVIDENCE/archaeologist-v2.md §"Pivotal commits 1 & 2".)

3. **The xfail list grows from 12 → 43.** Round 2 GitHub mining identified 31
   NEW silent-correctness/crash/drift bugs. Five OOB-indexing-silently-zero
   issues mean gpucheck must add an `assert_raises_oob` contract — silent zero
   IS the bug, not just imprecise. Four landmines (closed-without-completion,
   not verified-fix) go in a separate `[tool.gpucheck.mps.regression_fence]`
   section. (Source: EVIDENCE/github-miner-v2.md §"Recommended xfail expansion".)

4. **Apple's spec mandates non-determinism for the gpucheck-targeted ops.**
   MSL §6.15.4 locks `atomic_float` to `memory_order_relaxed`; SIMD-group
   matrix layout is "unspecified". PyTorch source contains 3 MPS-aware
   determinism gates (`index_put_`, `index_add_`, `kthvalue`) — none of
   matmul/softmax/layer_norm/SDPA/conv2d. gpucheck's API surface MUST NOT
   promise "deterministic parity on MPS"; it can only **detect** and surface
   non-determinism. New `@gpucheck.mps_determinism_check` fixture + 10-op
   known-non-deterministic xfail list (librarian-v2 §5.3). (Source:
   EVIDENCE/librarian-v2.md §1, §3, §5.)

5. **gpucheck's fuzzer needs Apple-tile constants.** Add 8 (MMA fragment), 16
   (GEMM BK), and 80 (FA head_dim Apple-only) to `TILE_SIZES_MPS` and
   `POWER_OF_2_BOUNDARIES_MPS` in `src/gpucheck/fuzzing/shapes.py`. Without
   this, gpucheck would miss the Apple tile-alignment regime that the
   triton#9838 / #9839 strategy was designed to exercise. (Source:
   EVIDENCE/cartographer-v2.md §3, §5.)

Plus three v1.0 type-system constraints from linguist-v2:

6. **Backend Protocol gains** `unsupported_dtypes: frozenset[torch.dtype]` and
   `silently_downcast_dtypes: frozenset[torch.dtype]`. MPS adds float64,
   complex128 (both unsupported), and float64+complex128 (silently downcast for
   0-d). `assert_close` must consult these BEFORE picking reference precision;
   otherwise the assertion is vacuous. (Source: EVIDENCE/linguist-v2.md §3, §5.)

7. **`EventTimer` Protocol gains** `units: Literal["ms"] = "ms"`. PyTorch's
   `Event.elapsed_time` returns ms but the type system has no unit hint
   (linguist-v2 §2.3). Without this, callers consuming `elapsed_ms` generically
   across CUDA and MPS can be off by 1000×.

8. **Apple chip-generation detection in `arch/`**. PR #181466 ships per-chip
   workarounds (M5/Apple10 fp16/bf16 matmul flatten); a "MPS supported"
   boolean is insufficient. gpucheck must detect M1/M2/M3/M4/M5 the same way
   SM60-SM120 detection works for CUDA. (Source: EVIDENCE/archaeologist-v2.md
   §"5-PR Deep Dive" row 5.)

---

## What Round 2 REFUTED in Round 1 (most important)

| Refutation | Round 1 claim | Round 2 evidence | Severity |
|---|---|---|---|
| **2× FlashAttention multiplier as global tolerance overlay** | SYNTHESIS_v1 §"Sub-Q 7" Recommended Starting Overlay (PROVISIONAL) | EVIDENCE/empiricist-v2.md §"Compact result table" — measured 13.7-26.1× breach for matmul/all dtypes; 3.84-6.14× breach for conv2d at low precision | **HIGH — false negatives in matmul tests** |
| **gpucheck atol of 2e-2 for fp16-MPS is "calibrated"** | SYNTHESIS_v1 §"Sub-Q 7" Tolerance Recommendation Table | EVIDENCE/tracer-v2.md §2.2 + §6 — MLX's empirically-validated atol for fp16-SDPA is 3e-4, gpucheck's is 66× looser | **HIGH — sub-budget MPS tolerance** |
| **"MPS tile sizes are unknown"** (skeptic v1 attack #1 inherited) | SYNTHESIS_v1 §"Sub-Q 8" "tile set is a v1.1 enhancement" | EVIDENCE/cartographer-v2.md §3 — Apple-canonical tile constants ARE knowable from MLX (8, 16, 32, 64; 80 head_dim Apple-only); `simdgroup_matrix<T, 8, 8>` cited from production code | **MEDIUM — fuzzer can extend in v1.0** |

## What Round 2 STRENGTHENED in Round 1

| Strengthening | Round 1 claim | Round 2 evidence | Effect |
|---|---|---|---|
| **MPS determinism contract is verifiable from Apple primary** | SYNTHESIS_v1 §"Sub-Q 4" MEDIUM confidence on quantitative MSL claims (REPORTED-NOT-VERIFIED) | EVIDENCE/librarian-v2.md §1 Sources A-E — MSL §6.15.4 atomic_float `memory_order_relaxed` only; §2 SIMD-group matrix layout "unspecified"; #pragma fp math_mode default `fast` | Confidence MEDIUM → HIGH |
| **PyTorch source confirms determinism gates are not inherited from CUDA** | SYNTHESIS_v1 §"Sub-Q 4" "best-effort deterministic" verdict | EVIDENCE/librarian-v2.md §3 Sources J-K — only 3 of 21 MPS op files contain deterministic_algorithms() gating | Verdict sharpens to "Apple's spec mandates the silence" |
| **Deadlock #162872 is permanent in HEAD** | SYNTHESIS_v1 §"Sub-Q 3" "API gotcha" PROVISIONAL | EVIDENCE/archaeologist-v2.md §"Pivotal commit 1 & 2" — file:line verified in `pytorch/main` HEAD 2026-05-01; PR #162874 closed-not-merged | "API gotcha" → "permanent v1.0 constraint requiring deadlock probe" |
| **Open-MPS-bugs xfail list is 12 + 31 = 43** | SYNTHESIS_v1 §"Sub-Q 2" "12 highest-impact" | EVIDENCE/github-miner-v2.md §"Recommended xfail expansion" — 31 verified additional issues + 4 landmines | Coverage grows ~3.5× |
| **Apple's own MPP framework returns nondeterministic results on M5** | SYNTHESIS_v1 §"Sub-Q 1" matmul/bwd ≥32K element corruption | EVIDENCE/archaeologist-v2.md §"Pivotal commit 5" — PR #181466 (malfet, 2026-04-28) ships chip-conditional workaround for `MPSNDArrayMatrixMultiplication` and `MPSGraph matrixMultiplication` on M5 | Determinism class extends from PyTorch wrapper to Apple framework; chip-generation detection in `arch/` becomes load-bearing |
| **gpucheck Backend Protocol type contracts** | SYNTHESIS_v1 §"Sub-Q 3" docs surface enumerated | EVIDENCE/linguist-v2.md §2 (3 stub bugs) + §3 (silent-downcast for fp64/complex128 0-d scalars) + §5 (4 Protocol overrides) | Type-system catches MPS-specific class of bugs at mypy-strict |

---

## Lowest-confidence remaining claims (Round 3 candidates)

| Claim | Why low-confidence | Round 3 dispatch |
|---|---|---|
| **The 16/20/32× tolerance multipliers transfer to M3/M4** | Empiricist-v2 §Confidence: "single Apple SKU (M5)... multipliers may differ on M3/M4 by factors of 2-3" | Empiricist-v3 with M3 + M4 hardware (or community-recruited results) |
| **MLX-as-tie-break-oracle is empirically feasible** | tracer-v2 §4 H1 status OPEN — "I did not run the probe — Apple Silicon required" | Empiricist-v3 probe: round-trip torch ↔ MLX, measure pairwise atol |
| **MPSGraph tile sizes match MLX tile sizes** | cartographer-v2 §6 Confidence "MEDIUM on PyTorch-MPS specifically — PyTorch dispatches through MPSGraph.framework which is binary; we infer from MLX as proxy" | Empiricist-v3 micro-benchmark on PyTorch MPS at non-tile-aligned shapes; does drift jump at the same boundary as MLX? |
| **Stride/contiguity fuzzing scope** | github-miner-v2 §"Cross-repo patterns" — non-contiguous failure family is largest by issue count, but "single underlying bug or many?" is open | Archaeologist-v3 deeper trace on stride-handling in aten/src/ATen/native/mps/ + interview hvaara |
| **Backward-pass numerics on MPS** | tracer-v2 §3.3 + empiricist-v2 §"Follow-ups" — "MLX never uses fused backward on Metal; gpucheck has no gradient testing"; backward-pass measurement out of scope this round | Empiricist-v3 with backward fwd-bwd parity test on the matmul/SDPA xfail set |
| **Sentence-level claim about Apple framework reference docs** | librarian-v2 §1 "Apple's MPSGraph and MPS framework reference docs render JS-heavy and return only their page-title in WebFetch's parsed content" — Playwright pass not done | Web-miner-v3 with rendered DOM scrape of `developer.apple.com/documentation/metalperformanceshaders[graph]` |
| **Whether `torch.mps.empty_cache()` between fuzz iters fully eliminates pytorch#177116 corruption** | SYNTHESIS_v1 §"Open questions" Q-C — empirical question never measured | Empiricist-v3 corruption-vs-empty-cache stress test |
| **PR #162874 abandonment cause** | archaeologist-v2 §"Unanswered" — "no public review thread visible via the API" | Web-miner-v3 inspect Apple/PyTorch internal Slack referenced in PR descriptions, OR gh issue-level comment crawl on adjacent issues |

---

## Confidence summary (revised)

| Sub-Q | Round 1 | Round 2 |
|---|---|---|
| 1 (correctness baseline) | HIGH | HIGH (strengthened with M5 measurement + 31-bug expansion) |
| 2 (open bugs) | HIGH | HIGH (12 → 43 xfail list, 4 landmines) |
| 3 (API surface) | HIGH | HIGH (strengthened with type-stub bugs + deadlock-permanence + dtype asymmetry) |
| 4 (determinism) | HIGH/MEDIUM | HIGH (MSL spec verified primary; source-level gates verified) |
| 5 (llama.cpp pattern) | HIGH | HIGH (no change; cartographer-v2 cross-validated SIMD width) |
| 6 (FA/Triton on Metal) | HIGH | HIGH (strengthened with kernel-internal MLX trace + 66× tolerance differential) |
| 7 (tolerance defaults) | MEDIUM (PROVISIONAL) | **REFUTED + REPLACED** with per-(kernel, dtype) table; HIGH on shape, MEDIUM on numbers |
| 8 (CUDA record + MPS transfer) | HIGH/MEDIUM | HIGH on transfer + tile extension (8, 16, 80 added) |

Overall: **HIGH** on the v1.0 ship recommendation; the per-(kernel, dtype)
overlay direction; the expanded xfail list; the determinism contract; the
deadlock probe requirement; the chip-generation detection requirement.
**MEDIUM** on the *exact* tolerance numbers (single-SKU M5; M3/M4 are Round 3
candidates).

---

## Citations summary (Round 1 + Round 2 distinct primaries)

### From SYNTHESIS_v1.md (44 citations carry forward)
- 28 PyTorch GitHub issues (full list in v1)
- 5 Triton GitHub issues (#9838, #9839, #4824, #3443, #1796)
- 4 PyTorch documentation pages (mps.html, randomness.html,
  numerical_accuracy.html, generated/torch.mps.event.Event.html)
- 2 MLX repo paths (kernels listing, scaled_dot_product_attention.cpp file-level)
- 3 llama.cpp / ggml references (#10845, commit 62bfef5, ggml-metal-device.cpp)
- 1 stable-diffusion.cpp issue (#1040)
- 1 Apple MSL spec PDF (REPORTED-NOT-VERIFIED in v1; promoted in v2)

### NEW in Round 2

**EVIDENCE/cartographer-v2.md** (11 NEW primaries):
- Apple Metal Feature Set Tables PDF (page-2/3-5/6 specific)
- dougallj G13 GPU Architecture Reference
- ggml-metal `N_SIMDWIDTH 32 // assuming SIMD group size is 32`
- MLX `mma.h:20-40` BaseMMAFrag<T, 8, 8>
- MLX `steel_gemm_fused.metal:21-26` 6-tuple block tile
- MLX `matmul.cpp:86-160` GEMM_TPARAM_MACRO dispatcher + NAX path
- MLX `steel_attention.metal:14-17` FA tile shapes
- MLX `sdpa_vector.h:42-43` BN=32, BD=32
- MLX `rms_norm.metal:22` SIMD_SIZE=32; `softmax.h:16` same; `defines.h:12` RMS_LOOPED_LIMIT=4096
- MLX `conv.cpp` + `steel_conv.metal:42-48` implicit-GEMM tile set
- philipturner/metal-benchmarks 32-thread SIMD + MATMUL<8x8>

**EVIDENCE/empiricist-v2.md** (5 NEW primaries):
- Higham N. J. — "Accuracy and Stability of Numerical Algorithms" 2002, Theorem 3.5
- Dao et al. — FlashAttention NeurIPS 2022, Appendix B "Numerical Stability"
- NumPy `numpy.testing.assert_allclose` doc v2.3
- Apple MSL Spec v4 §5.7 "Floating-Point Math Functions" `metal::fast::*`
- PyTorch `aten/src/ATen/native/mps/operations/LinearAlgebra.mm`

**EVIDENCE/github-miner-v2.md** (31 NEW pytorch#xxx + 4 landmines + 4 follow-up tracking):
- pytorch#182052, #180776, #179608, #176296, #169738, #169342, #162092,
  #163327, #154322, #144824, #154235, #170507, #163504, #170370, #169236,
  #170639, #160553, #151667, #132086, #160744, #160740, #130295, #122045,
  #121439, #136623, #147510, #151740, #153957, #119677, #122030, #107214,
  #132605, #94691, #154887, #154881, #154890, #154882, #161865, #144634,
  #144445, #164125, #142048
- Landmines: #181867, #175191, #89708, #150051

**EVIDENCE/tracer-v2.md** (8 NEW primaries):
- MLX `fast.cpp:613-862`
- MLX `scaled_dot_product_attention.cpp:18-786`
- MLX `kernels/sdpa_vector.h:50` `typedef float U;`
- MLX `python/tests/test_fast_sdpa.py:439` atol schedule
- gpucheck `src/gpucheck/assertions/close.py:140-189`
- gpucheck `src/gpucheck/assertions/tolerances.py:35-43, 105-110, 181-228`
- gpucheck `src/gpucheck/backends/_protocol.py:18-32, 36-73`
- gpucheck `src/gpucheck/backends/mps.py`

**EVIDENCE/librarian-v2.md** (11 NEW primaries):
- Apple MSL Spec §6.15.4 Atomic Functions
- Apple MSL Spec §6.15.4 Atomic Fetch and Modify, Table 6.25, Memory Order definition
- Apple MSL Spec §2 SIMD-group Matrix Data Types
- Apple MSL Spec §1 #pragma METAL fp math_mode + fp contract
- Apple WWDC sessions 24/10160, 24/10218, 25/205, 25/262
- docs.pytorch.org/docs/2.11/generated/torch.use_deterministic_algorithms.html
- docs.pytorch.org/docs/2.11/notes/randomness.html (re-verify)
- docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html (re-verify)
- docs.pytorch.org/docs/2.11/notes/mps.html (re-verify)
- pytorch v2.11.0 `aten/src/ATen/native/mps/operations/Indexing.mm:194-215, 502-536`
- pytorch v2.11.0 `aten/src/ATen/native/mps/operations/Sort.mm:165-170`

**EVIDENCE/archaeologist-v2.md** (~10 NEW primaries):
- pytorch commit cdfd0ea16282 (PR #102121, razarmehr 2023-08-08)
- pytorch commit b417006e (PR #141296, malfet 2024-11-21)
- pytorch issue #162872 + PR #162874 (oraluben 2025-09-13, closed-not-merged)
- pytorch commit a914fee44471 (PR #173326, malfet 2026-01-25)
- pytorch commit 75742d2eb01f (PR #174411, malfet 2026-02-05)
- pytorch commit c68a1d2c01df (PR #174945, hvaara 2026-02-20)
- pytorch commit 49e7d4dadbbe (PR #181466, malfet 2026-04-28)
- pytorch `aten/src/ATen/mps/MPSEvent.mm:228 waitForCpuSync()` HEAD
- pytorch `aten/src/ATen/native/mps/operations/Attention.mm` (#174945 fix)
- pytorch `aten/src/ATen/native/mps/operations/Linear.mm` (#181466 fix)

**EVIDENCE/linguist-v2.md** (9 NEW primaries):
- pytorch `torch/mps/__init__.py:13, 21-198` (full module)
- pytorch `torch/mps/event.py:14-45`
- pytorch `torch/mps/profiler.py:9, 14-77`
- pytorch `torch/_C/__init__.pyi.in:2072-2096`
- pytorch `aten/src/ATen/mps/EmptyTensor.cpp:45`
- pytorch `aten/src/ATen/native/mps/OperationUtils.mm:49-88` (tensor path)
- pytorch `aten/src/ATen/native/mps/OperationUtils.mm:120-158` (scalar path)
- pytorch `aten/src/ATen/native/mps/OperationUtils.h:612-625`
- pytorch `aten/src/ATen/mps/MPSEvent.mm:221-238` (elapsedTime)

### Total distinct primary citations (Round 1 + Round 2)

- v1: 44 distinct (per SYNTHESIS_v1.md §"Citations summary")
- v2 NEW: 11 cartographer + 5 empiricist + 31+4+4 = 39 github-miner + 8 tracer
  + 11 librarian + ~10 archaeologist + 9 linguist = **~93 NEW**
- After dedup of overlap (e.g., docs.pytorch.org randomness.html re-verified
  in v2 librarian; Apple MSL spec promoted from REPORTED to VERIFIED): **~85
  net-new**

**Combined total distinct primary citations: ~129** (44 + 85), well exceeding
the charter's ≥30 floor.

---

## File pointers (absolute paths)

- `/Users/cero/Code/gpucheck/.claude/teams/research/v1.0/QUESTION.md`
- `/Users/cero/Code/gpucheck/.claude/teams/research/v1.0/HYPOTHESES.md`
- `/Users/cero/Code/gpucheck/.claude/teams/research/v1.0/SYNTHESIS_v1.md` (Round 1 — superseded where this document conflicts)
- `/Users/cero/Code/gpucheck/.claude/teams/research/v1.0/SYNTHESIS_v2.md` (this file — binding for engineering)
- `/Users/cero/Code/gpucheck/.claude/teams/research/v1.0/EVIDENCE/cartographer-v2.md`
- `/Users/cero/Code/gpucheck/.claude/teams/research/v1.0/EVIDENCE/empiricist-v2.md`
- `/Users/cero/Code/gpucheck/.claude/teams/research/v1.0/EVIDENCE/github-miner-v2.md`
- `/Users/cero/Code/gpucheck/.claude/teams/research/v1.0/EVIDENCE/tracer-v2.md`
- `/Users/cero/Code/gpucheck/.claude/teams/research/v1.0/EVIDENCE/librarian-v2.md`
- `/Users/cero/Code/gpucheck/.claude/teams/research/v1.0/EVIDENCE/archaeologist-v2.md`
- `/Users/cero/Code/gpucheck/.claude/teams/research/v1.0/EVIDENCE/linguist-v2.md`
- `/Users/cero/Code/gpucheck/.claude/teams/research/v1.0/drift_histogram.json` (empiricist-v2 raw output)
- `/Users/cero/Code/gpucheck/.claude/teams/research/v1.0/TURN_LOG.md`
