# SYNTHESIS — gpucheck v1.0 MPS Backend

**Slug**: v1.0
**Date**: 2026-05-01
**Owner**: research-lead
**Downstream**: engineering-lead (CHARTER.md cites this as binding spec)

---

## Headline

**Ship gpucheck v1.0 with an MPS backend, in observation-instrument-then-calibrate
mode.** Adopt H1 (CUDA-shaped tolerance + MPS-shaped exclusions) blended with H3
(MPS as correctness oracle, not parity backend). Posterior probability of this
blend: 0.80 (synthesist §3, moderator REFRAME on T1).

The CUDA fuzz playbook that found triton#9838 (83.4% layer_norm error, OPEN)
and triton#9839 (FP16 matmul drift 0.125 at K=8192, CLOSED) transfers to MPS
verbatim because shape adversariality is device-independent (cartographer §3,
archaeologist §3, empiricist §1, with skeptic §1 caveat about NVIDIA-tile
calibration). The 12 high-impact PyTorch MPS bugs (github-miner §2) are
catchable as a finite, citable xfail list. The API surface (`torch.mps.*`,
`torch.backends.mps.*`, `torch.mps.event.Event`) is documented and
reachable in PyTorch 2.11 (librarian §1-§5).

**Confidence: HIGH** on direction; **MEDIUM** on the specific 2× tolerance
multiplier (must be calibrated on M-silicon before publishing); **HIGH** on
the load-bearing xfail list.

Gate status:
- Mid-flight audit: PASS (0 violations, 11/17 evidence files present at gate)
- Skeptic: PASS — 5 of 6 attacks absorbed by careful labeling; no flip
- Adversary: PASS — corpus healthy, no SEO/astroturf/citation laundering
- Moderator: NO_DEBATE — synthesist flagged no load-bearing contradiction;
  3 soft tensions reframed (T1) or labeled NOT_LOAD_BEARING (T2, T3)
- Evaluator: see `EVIDENCE/evaluator.md`

---

## Sub-Q 1 — MPS correctness baseline

The current state of `torch.mps` correctness vs CUDA, per kernel:

### Matmul (FP32 / FP16 / BF16)

- **Small-medium FP32**: parity within precision floor; CUDA atol of 1e-4
  inflated 2× to 2e-4 (PROVISIONAL per skeptic §2; calibrate on M-machine).
- **Small-medium FP16/BF16**: 2× precision-floor multiplier; Apple Silicon has
  no FP16 tensor cores (web-miner §3) so reductions are full-FP16 / sum-FP32
  internally, yielding drift in the FlashAttention regime.
- **Large element counts (>32K)**: CATASTROPHIC GRAD CORRUPTION on backward.
  Per `EVIDENCE/github-miner.md` §2.1: "[#177116](https://github.com/pytorch/pytorch/issues/177116) MPS: catastrophically wrong gradients in backward pass (>32K elements)" — gradient norms deviate by 1,000×–68,000× from CPU when total elements exceed 2^15. Workaround: `torch.mps.empty_cache()` between operations (tracer §1.3).
- **F.linear backward, BF16/FP16, no-bias, >2D, M5**: NON-DETERMINISTIC.
  Per `EVIDENCE/github-miner.md` §2: "[#181936](https://github.com/pytorch/pytorch/issues/181936) max diff between two runs reached 130.0".

Citations: pytorch#177116, pytorch#181936, pytorch#170837, README L300-313.

### Softmax / log_softmax

- **Standard shapes**: parity within precision floor.
- **Large attention shapes (>10000 in last 2 dims)**: NaN OUTPUT. Per
  `EVIDENCE/github-miner.md` §2.7: "[#96602](https://github.com/pytorch/pytorch/issues/96602) softmax returns NaN attention probabilities for large tensors, in float16 and float32" — verified by WebFetch 2026-05-01: shape `[10, 12416, 12416]`, NaN at `diffs = x - maxes`.

Citations: pytorch#96602.

### layer_norm / rms_norm / batch_norm / group_norm

- **layer_norm fwd**: OK (no recent open issues).
- **layer_norm bwd at shape (1,)**: BROKEN. Per `EVIDENCE/github-miner.md` §2: "[#173525](https://github.com/pytorch/pytorch/issues/173525) MPS layer_norm backward numerical issues" — abs diff 1.22e-4, rel diff infinite.
- **batch_norm fwd**: OK.
- **batch_norm bwd, channels_last**: ~7 ORDERS OF MAGNITUDE WRONG. Per
  `EVIDENCE/github-miner.md` §2: "[#175189](https://github.com/pytorch/pytorch/issues/175189) BatchNorm2d backward produces wildly wrong weight gradients on channels_last inputs".
- **group_norm**, **rms_norm**: OK reported (no recent open issues).

Citations: pytorch#173525, pytorch#175189, pytorch#178492 (BN slow on 3D).

### Scaled dot product attention (SDPA) and FlashAttention

- **SDPA fwd, small B×S**: works.
- **SDPA fwd, large B×S (B>2 ∧ seq_len>5120)**: COSINE SIMILARITY DROPS TO ~0.49.
  Per `EVIDENCE/github-miner.md` §2.2: "[#179352](https://github.com/pytorch/pytorch/issues/179352) MPS: scaled_dot_product_attention produces incorrect results for large batch × sequence length combinations" — max abs diff 0.1822.
- **SDPA bwd**: dispatches to math-decomposition backend (slow + partial). Per
  `EVIDENCE/github-miner.md` §2 + tracer §1.2: "[#179294](https://github.com/pytorch/pytorch/issues/179294) the backward pass relies on the device-agnostic 'math' backend instead of a dedicated MPS implementation".
- **FlashAttention native**: NO via PyTorch MPS. MLX has fused FA-equivalent SDPA;
  community projects (pmetal, ZMLX) port FA to Metal but are not callable from
  `torch.Tensor(device='mps')` (historian §2-§3, web-miner §1).

Citations: pytorch#179352, pytorch#179294, pytorch#173943 (return value optim), pytorch#181725 (MultiheadAttention 9× slower than SDPA), pytorch#175873 (SDPA wasted resources).

### cross_entropy / nll_loss / kl_div

- **cross_entropy / nll_loss**: OK (no recent open issues).
- **BCE loss**: BROKEN since 2024. Per `EVIDENCE/github-miner.md` §3 + `EVIDENCE/web-miner.md`: "[#137001](https://github.com/pytorch/pytorch/issues/137001) BCE loss mps device" — labeled `module: correctness (silent)`.
- **kl_div**: not in recent open list; treat as OK pending fuzz.

Citations: pytorch#137001.

### conv2d

- **conv2d fwd, C_out > 65536**: RETURNS ZEROS. Per `EVIDENCE/github-miner.md` §2.4: "[#142836](https://github.com/pytorch/pytorch/issues/142836) Incorrect output from convolution ops with large dimensions" — silent correctness regression on macOS ≥15.1.
- **conv2d bwd, channels_last + contiguous grad_output**: WRONG MEMORY FORMAT.
  Per `EVIDENCE/github-miner.md` §3: "[#174269](https://github.com/pytorch/pytorch/issues/174269) MPS convolution_backward returns wrong memory format" — breaks `torch.compile` with inductor.

Citations: pytorch#142836, pytorch#174269.

**Sub-Q 1 confidence: HIGH.** Every claim cites a primary issue or a verified-absence in the github-miner crawl.

---

## Sub-Q 2 — Open MPS bugs (top 12 highest-impact)

Live count from `gh api search/issues label:"module: mps" state:open`: **255 open
issues** as of 2026-05-01. The top-12 highest-impact triage from
`EVIDENCE/github-miner.md` §2:

| # | Title | Cat | Affected kernel × dtype | gpucheck action |
|---|---|---|---|---|
| 177116 | Catastrophically wrong gradients (>32K elements) | (a) drift | matmul/embedding/residual FP32 | xfail when shape product>32768 OR call empty_cache between fuzz iters |
| 179352 | SDPA incorrect for large B×S | (a) drift | SDPA fwd FP16/FP32 | xfail when B>2 ∧ seq_len>5120 |
| 178497 | Reductions (count_nonzero/mean/nansum/sum/trace) 50-90% off | (a) drift intermittent | reductions all dtypes | xfail/skip these reductions |
| 142836 | Conv2d C_out>65536 returns zeros | (a) drift | conv2d all dtypes | xfail conv with C_out>65536 |
| 173525 | layer_norm backward shape (1,) gives 0 | (a) drift edge | layer_norm bwd | xfail shape (1,) |
| 175189 | BatchNorm2d backward channels_last 7 OOM wrong | (a) drift | BN2d bwd channels_last | xfail until #181411 lands |
| 96602 | softmax NaN on large tensors | (a) drift | softmax FP16/FP32 large | xfail size>1e8 last 2 dims |
| 162872 | Event.synchronize+elapsed_time deadlock | (b) hang | timing API | gpucheck must NOT call this pattern; use `torch.mps.synchronize()` |
| 175190 | AvgPool2d backward channels_last SIGABRT | (b) crash | avg_pool bwd channels_last | xfail/skip channels_last |
| 160828 | _ctc_loss not implemented | (c) missing op | CTC loss | runtime detect via NotImplementedError |
| 181936 | F.linear backward BF16/FP16 non-deterministic on M5 | (d) determinism | linear bwd >2D no-bias | xfail on M5 OR reshape input to 2D |
| 170837 | BERT/RoBERTa batched inference inconsistent | (a) drift | matmul / embedding | xfail BERT/RoBERTa-shaped batched inference |

Plus tracking issues: [#77764](https://github.com/pytorch/pytorch/issues/77764)
(966 reactions, MPS op coverage umbrella),
[#141287](https://github.com/pytorch/pytorch/issues/141287) (2.6+ tracker),
[#154052](https://github.com/pytorch/pytorch/issues/154052) (most-requested ops:
isin 224 votes, index_copy 57, _upsample_bicubic2d_aa 49, max_pool3d_with_indices 48,
grid_sampler_3d 47, linalg_eig 38, grid_sampler_2d_backward 35, linalg_qr 33,
_linalg_eigh 24, native_dropout 23),
[#150121](https://github.com/pytorch/pytorch/issues/150121) (torch.compile on MPS:
"early prototype phase"; "attempt to use it to accelerate end-to-end network is
likely to fail" — gpucheck v1.0 must test eager-mode only).

**Sub-Q 2 confidence: HIGH.** All 12 issues directly verified by WebFetch this session.

---

## Sub-Q 3 — torch.mps API surface (PyTorch 2.11 stable)

Verified primary at `docs.pytorch.org/docs/2.11/...` on 2026-05-01
(librarian §1-§4):

- `torch.mps.synchronize()` — "Waits for all kernels in all streams on a MPS device to complete."
- `torch.mps.empty_cache()` — "Releases all unoccupied cached memory currently held by the caching allocator..."
- `torch.mps.current_allocated_memory()`, `driver_allocated_memory()`, `recommended_max_memory()`, `set_per_process_memory_fraction()`
- `torch.mps.manual_seed()`, `seed()`, `get_rng_state()`, `set_rng_state()`
- `torch.mps.device_count()`
- `torch.mps.compile_shader()` — custom Metal shader compilation
- `torch.mps.profiler.{start, stop, profile, is_capturing_metal, is_metal_capture_enabled, metal_capture}`
- `torch.mps.event.Event(enable_timing=False)` with methods `record(), wait(), query()→bool, synchronize(), elapsed_time(end_event)→float`
- `torch.backends.mps.is_available() → bool`
- `torch.backends.mps.is_built() → bool`

★ **API gotcha (load-bearing for gpucheck.gpu_benchmark)**: Per
[pytorch#162872](https://github.com/pytorch/pytorch/issues/162872) and
`EVIDENCE/tracer.md` §2: calling `start.record(); end.record(); end.synchronize();
start.elapsed_time(end)` **deadlocks** on PyTorch 2.10 / Apple M4 Pro. gpucheck
must use device-level `torch.mps.synchronize()` instead of per-event
`event.synchronize()` until #162872 closes.

★ **Memory-accounting gotcha**: Per
[pytorch#164299](https://github.com/pytorch/pytorch/issues/164299) and
`EVIDENCE/tracer.md` §1.4: "Memory growth is not recognized by built-in
`torch.mps.current_allocated_memory()` and `torch.mps.driver_allocated_memory()`
methods but visible on Activity Monitor". gpucheck's `memory_tracker` fixture
on MPS must use process-RSS via `psutil` as the leak proxy (matches the
existing pynvml fallback pattern at `sanitizers/memory.py`).

**Sub-Q 3 confidence: HIGH.** Each API element verified at
`docs.pytorch.org/docs/2.11/{mps,generated/torch.mps.event.Event,generated/torch.mps.synchronize,generated/torch.mps.empty_cache}.html`.

---

## Sub-Q 4 — Determinism guarantees

★ **PyTorch's MPS docs are SILENT on determinism** (librarian §5,
`EVIDENCE/librarian.md`). Verified primary:
- [docs.pytorch.org/docs/2.11/notes/randomness.html](https://docs.pytorch.org/docs/2.11/notes/randomness.html), retrieved 2026-05-01: makes NO mention of MPS or Metal.
- [docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html](https://docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html), retrieved 2026-05-01: covers TF32 on Ampere+, AMD MI200 FP16 denormal flushing — NOTHING on MPS.

Apple Metal Shading Language Specification v4 PDF:
[developer.apple.com/metal/Metal-Shading-Language-Specification.pdf](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
— REPORTED-NOT-VERIFIED in this session (10MB PDF exceeds WebFetch cap).
Secondary search results indicate: IEEE 754 conformance with caveats, fast-math
vs precise-math accuracy tables in ULP, atomic_* types subset of C++14. Per
adversary §2.5 and MEMORY.md REPORTED-NOT-VERIFIED protocol, the directional
claim "MSL does not promise bit-exact determinism for fast-math paths" is
load-bearing; specific ULP numbers are NOT claimed.

★ **Empirical record contradicts any assumption of MPS determinism**:
- [pytorch#181936](https://github.com/pytorch/pytorch/issues/181936) F.linear bwd: 130.0 run-to-run drift, M5
- [pytorch#180776](https://github.com/pytorch/pytorch/issues/180776) earlier M5 determinism issue
- [pytorch#170837](https://github.com/pytorch/pytorch/issues/170837) BERT/RoBERTa batched ≠ unbatched
- [pytorch#177116](https://github.com/pytorch/pytorch/issues/177116) buffer-pool corruption corrupting later runs

**Verdict (per `EVIDENCE/synthesist.md` §1, `EVIDENCE/linguist.md` §1)**: MPS is
**best-effort deterministic** — gpucheck-MPS must NOT assume bit-exact
reproducibility. For tests requiring determinism: fix seeds + run twice +
compare. The non-determinism comes in three flavors (linguist's A/B/C scheme):
A) literal run-to-run divergence (xfail), B) MPS-vs-CPU divergence (assert_close
catches), C) precision-floor drift (existing tolerance absorbs OR 2× overlay).

**Sub-Q 4 confidence: HIGH on the practical conclusion (don't assume
determinism); MEDIUM on the MSL-specific quantitative claims (REPORTED-NOT-VERIFIED).**

---

## Sub-Q 5 — llama.cpp Metal dispatch precedent

Per `EVIDENCE/historian.md` §1 and `EVIDENCE/web-miner.md` §1, retrieved
2026-05-01:

**Pattern**: `ggml-metal` implements per-op `supports_op()` checks. When
`supports_op(GGML_OP_X)` returns false, the runtime transparently falls back
to CPU. Cited primaries:
- [llama.cpp#10845](https://github.com/ggml-org/llama.cpp/issues/10845) — IM2COL not implemented, falls back
- [llama.cpp commit 62bfef5](https://github.com/ggml-org/llama.cpp/commit/62bfef5194d5582486d62da3db59bf44981b7912) — disabled FA kernel for HS=256 after correctness regression
- [stable-diffusion.cpp#1040](https://github.com/leejet/stable-diffusion.cpp/issues/1040) — third-party project hits same fallback

**Per-op kernel dispatch**: `ggml_metal_library_get_pipeline_*()` factories
produce Metal kernels for unary, mul_mv, mul_mm, pool, softmax, reductions,
RWKV, SSM. Verified primary file structure at
`github.com/ggml-org/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.cpp`
(retrieved 2026-05-01).

**Validation pattern**: `tests/test-backend-ops.cpp` compares Metal to CPU
output via NMSE per-op tolerances. Specific tolerance constants not extractable
in this session (file 9525 lines, exceeded WebFetch summarization cap), but
pattern confirmed via DeepWiki + multiple issue references (`EVIDENCE/historian.md` §1.3).

**Tradeoff handling**: `commit 62bfef5` shows the discipline — when a Metal
kernel is fast but produces wrong values for a specific shape, the kernel is
DISABLED for that shape, falling back to CPU/alternate path. gpucheck-MPS
mirrors this: xfail/skip is our "disable for this shape".

★ **Difference from PyTorch MPS**: PyTorch MPS does NOT do automatic CPU
fallback for unimplemented ops — it raises `NotImplementedError` (tracer §1.1,
`EVIDENCE/tracer.md`). gpucheck-MPS tests must therefore wrap calls or check
op coverage in preconditions. Citation: [pytorch#160828 _ctc_loss not implemented](https://github.com/pytorch/pytorch/issues/160828).

**Sub-Q 5 confidence: HIGH.** Pattern verified across 3 primary repos; tolerance constants partial.

---

## Sub-Q 6 — FlashAttention / Triton on Metal (status 2026-05)

Per `EVIDENCE/historian.md` §3-§4 and `EVIDENCE/web-miner.md` §1, retrieved
2026-05-01:

- **Triton-Metal backend**: NONE upstream. [triton#4824](https://github.com/triton-lang/triton/issues/4824) (RFC), [triton#3443](https://github.com/triton-lang/triton/issues/3443) (build broken on macOS), [triton#1796 discussion](https://github.com/triton-lang/triton/discussions/1796) (community inquiry, no roadmap).
- **MLX fused SDPA (FA-equivalent)**: YES. [mlx/backend/metal/scaled_dot_product_attention.cpp](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/scaled_dot_product_attention.cpp) confirms `sdpa_vector` (≤8 tokens) + `sdpa_full_self_attention_metal` + Steel attention (head_dims 64/80/96/128/256). MLX kernels: scaled_dot_product_attention, layer_norm, rms_norm, softmax, logsumexp, conv, rope, fft, etc.
- **Community PyTorch-MPS-FA**: pmetal (community FlashAttention port from FLA Triton), ZMLX (Triton-style toolkit on MLX), vllm-metal (vLLM plugin using MLX). All MIXED-quality, none in-tree for PyTorch MPS.
- **PyTorch MPS SDPA path**: dispatches to "math" backend, not "flash". No
  native MPS fused SDPA invoked in current codebase. Per
  [pytorch#179294](https://github.com/pytorch/pytorch/issues/179294): "a Metal kernel implementation exists but is never invoked".

**Gap**: PyTorch MPS lags MLX on fused-kernel coverage by approximately one
generation. **Implication for gpucheck**: testing `torch.Tensor(device='mps')`
SDPA tests PyTorch's slow path, not Apple's optimal path. This is OK for v1.0
because the user we're targeting is the PyTorch user.

**Sub-Q 6 confidence: HIGH.** Three independent repos confirmed.

---

## Sub-Q 7 — Tolerance defaults

Per `EVIDENCE/empiricist.md` §2, mapped from real bug magnitudes (not folk
wisdom). **PROVISIONAL** until calibration on Akash's M-machine (skeptic §2).

### Recommended starting overlay (per dtype × MPS)

| dtype | CUDA atol | CUDA rtol | MPS atol | MPS rtol | Multiplier | Source |
|---|---|---|---|---|---|---|
| float32 | 1e-4 | 1e-4 | 2e-4 | 2e-4 | 2× | precision-floor (FlashAttention precedent, `assertions/close.py:117` baseline_2x flag) |
| float16 | 1e-2 | 1e-2 | 2e-2 | 2e-2 | 2× | no FP16 tensor cores on Apple Silicon (web-miner §3); Apple MSL fast-math defaults |
| bfloat16 | 5e-2 | 5e-2 | 1e-1 | 1e-1 | 2× | reductions intermittent (#178497) — but for the kernels that DO work, 2× absorbs drift |
| float64 | 1e-10 | 1e-7 | unchanged | unchanged | 1× | not load-bearing on MPS (rarely used) |

### Recommended xfail set (initial — DOES NOT cover all 255 open MPS bugs)

```toml
[tool.gpucheck.mps.xfail]
ops = [
  "scaled_dot_product_attention.large",       # B>2 ∧ seq_len>5120 — pytorch#179352
  "scaled_dot_product_attention.backward",    # math decomp — pytorch#179294
  "layer_norm.backward.shape1",               # (1,) input — pytorch#173525
  "batch_norm.backward.channels_last",        # 7-OOM grads — pytorch#175189
  "conv2d.large_channels",                    # C_out>65536 — pytorch#142836
  "conv2d.backward.channels_last_format",     # wrong stride — pytorch#174269
  "F.linear.backward.bf16_3d_nobias_m5",      # M5 only — pytorch#181936
  "softmax.large_attention",                  # >10000 last 2 dims — pytorch#96602
  "avg_pool2d.backward.channels_last",        # SIGABRT — pytorch#175190
  "binary_ops.uint16_uint32_uint64",          # garbage values — pytorch#176296
  "BCE_loss",                                 # silent correctness — pytorch#137001
  "matmul.backward.over_32K_elements",        # gradient corruption — pytorch#177116
]
```

### Calibration plan (release blocker)

1. Implement Phase 1 (cartographer §6): MPS detection + assert_close fast-path
   broadening + tolerance overlay scaffolding.
2. Run `fuzz_shapes()` against the standard kernel suite on Akash's M-machine
   with CPU as reference (FP64 / FP32).
3. Histogram `|MPS - CPU|` per (dtype × kernel). Fit P99 quantile.
4. Set MPS overlay = max(2× CUDA, P99 of measured drift).
5. Publish overlay in pyproject.toml AND in README MPS section, with the
   M-generation specified.

### What if 2× isn't enough?

Per skeptic §2: 2× is a hypothesis. If P99 measured drift on M-silicon exceeds
2×, the overlay must inflate to match — OR the affected op gets added to the
xfail list. The choice depends on whether the drift is "precision-floor"
(linguist Cat C — multiplier OK) or "implementation bug" (linguist Cat A/B —
xfail).

**Sub-Q 7 confidence: MEDIUM.** Direction high; numbers PROVISIONAL.

---

## Sub-Q 8 — gpucheck's existing CUDA bug-finding record

Per `EVIDENCE/archaeologist.md` §2-§3 and `EVIDENCE/empiricist.md` §1, §5:

### Verified primary (externally filed and verified)

- [triton#9838](https://github.com/triton-lang/triton/issues/9838) "Tutorial layer_norm: variance padding bug causes 83% error for non-power-of-2 feature dims" — OPEN, opened 2026-03-25, **83.4% relative error** at `n_cols=17 BLOCK_SIZE=32`, GTX 1650 SM75 FP32. Verified by WebFetch 2026-05-01. Recommended fix (in issue): `xmean = tl.where(mask, x - mean, 0.0)` to mask before squaring.
- [triton#9839](https://github.com/triton-lang/triton/issues/9839) "Tutorial matmul: modular index wrapping causes FP16 error scaling with K (0.125 at K=8192)" — CLOSED, opened 2026-03-25, **max abs error 0.125** at K=8192 FP16, M=128 N=128. Error scales linearly with K. Verified by WebFetch 2026-05-01.

### README-internal (not externally verifiable in this session)

- cuFFT precision N≥4096 (1.26% relative error)
- `torch.baddbmm` FP16 silent overflow (NaN with alpha=1000)
- `torch.bmm` FP32 large-K (2.1e-3 relative error)
- (3 more not enumerated in README ledger but counted in "8 bugs")

### Fuzzing strategy that found those bugs

From `src/gpucheck/fuzzing/shapes.py:9-16`:
- `TILE_SIZES = (32, 64, 128)`
- `PRIMES = (7, 13, 31, 127, 257)`
- `POWER_OF_2_BOUNDARIES = (127, 128, 129, 255, 256, 257, 511, 512, 513)`
- `LARGE_DIMS = (2048, 4096, 8192)`

Priority order: `degenerate > non-tile-aligned > prime > power-of-2 > large > mixed`. triton#9838 reproduced at n_cols=17 (non-tile-aligned via tile-1 / prime); triton#9839 reproduced at K=8192 (LARGE_DIMS exact). Both bugs sit in the active priority cells.

The fuzzing **strategy transfers verbatim to MPS** (cartographer §3, empiricist §1) modulo the skeptic §1 caveat: NVIDIA tile sizes (32/64/128) may not be Apple's tile sizes. v1.0 ships with the existing tile set; an MPS-specific tile probe is a v1.1 enhancement.

**Documentation precision note** (skeptic §6): the README phrases the record as "8 real bugs"; the rigorous count of externally-verified is 2. SYNTHESIS recommends the docs team sharpen to "8 bugs found via 511 test configurations, of which triton#9838 (open) and triton#9839 (closed) are filed and externally verified".

**Sub-Q 8 confidence: HIGH** on strategy verification (file-level read);
**HIGH** on the 2 external bugs (verified WebFetch 2026-05-01); **MEDIUM** on
the broader "8 bugs" claim (internal-only count).

---

## Engineering team must respect (top 3 findings)

1. **The Event API has a deadlock bug**:
   `start.record(); end.record(); end.synchronize(); start.elapsed_time(end)`
   hangs on Apple M4 Pro / PyTorch 2.10+ ([pytorch#162872](https://github.com/pytorch/pytorch/issues/162872)). gpucheck's `gpu_benchmark` fixture on MPS must use `torch.mps.synchronize()` (device-level) instead of per-event synchronize. **This is a v1.0 release blocker** — without it, every MPS benchmark hangs.

2. **The xfail list is the load-bearing artifact, not the tolerance multiplier**:
   12 specific PyTorch MPS bugs are silent-correctness or crash bugs. A
   tolerance multiplier (2×) cannot rescue a kernel that returns NaN or zeros
   or 7-OOM-wrong gradients. The pyproject.toml `[tool.gpucheck.mps.xfail]`
   section IS the correctness story. Treat it as a living document; re-mine
   the issue tracker each minor release.

3. **MPS is best-effort deterministic; do not assume bit-exactness**:
   PyTorch's MPS docs are silent on determinism; the empirical record
   ([pytorch#181936](https://github.com/pytorch/pytorch/issues/181936) F.linear M5, [pytorch#170837](https://github.com/pytorch/pytorch/issues/170837) BERT/RoBERTa, [pytorch#177116](https://github.com/pytorch/pytorch/issues/177116) buffer corruption) shows real run-to-run divergence. Tests requiring determinism: fix seeds + run twice + compare. Do NOT export gpucheck-MPS as a "deterministic
   parity" guarantee.

---

## Confidence summary

| Sub-Q | Confidence | Gate evidence |
|---|---|---|
| 1 (correctness baseline) | HIGH | github-miner §2 + adversary §2.1 |
| 2 (open bugs) | HIGH | live `gh api` + 12 verified WebFetches |
| 3 (API surface) | HIGH | librarian §1-§4 from 2.11 docs |
| 4 (determinism) | HIGH on conclusion / MEDIUM on quantitative MSL claims | librarian §5 + linguist §1 + adversary §2.5 |
| 5 (llama.cpp pattern) | HIGH | historian §1 + 3 primary repos |
| 6 (FA/Triton on Metal) | HIGH | historian §3-§4 + MLX repo verified |
| 7 (tolerance defaults) | MEDIUM (PROVISIONAL on numbers) | empiricist §2 + skeptic §2 |
| 8 (CUDA record) | HIGH on 2 verified, MEDIUM on broader 8 claim | archaeologist §2 + empiricist §5 |

Overall: **HIGH confidence on the v1.0 ship recommendation**, with explicit
**MEDIUM** caveats on the specific tolerance multipliers (require M-machine
calibration before publishing as canonical).

## Open questions / INCOMPLETE sub-questions

None of the 8 sub-questions are INCOMPLETE. Each was answered to at least
MEDIUM confidence with a written caveat where the gate doesn't close to HIGH.

Three open questions for engineering follow-up (not blockers):
- (Q-A) Quantify the Apple MSL ULP table for fast-math paths — fetch the PDF
  locally and document in v1.1
- (Q-B) Calibrate tolerance multipliers on Akash's actual M-generation hardware
  before publishing the v1.0 MPS table as canonical
- (Q-C) Determine whether `torch.mps.empty_cache()` between fuzz iterations
  fully eliminates [pytorch#177116](https://github.com/pytorch/pytorch/issues/177116) corruption or merely reduces frequency

## Citations summary

Primary sources cited in this SYNTHESIS (de-duplicated):
- 28 PyTorch GitHub issues (#77764, #96602, #123148, #137001, #141287, #142836, #150121, #154052, #154329, #160828, #162872, #164299, #170837, #173525, #173640, #173943, #174269, #175189, #175190, #176296, #177116, #178037, #178492, #178497, #179294, #179352, #180776, #181936)
- 5 Triton GitHub issues (#9838, #9839, #4824, #3443, #1796)
- 4 PyTorch documentation pages (mps.html, randomness.html, numerical_accuracy.html, generated/torch.mps.event.Event.html)
- 2 MLX repo paths (kernels listing, scaled_dot_product_attention.cpp)
- 3 llama.cpp / ggml references (#10845 IM2COL, commit 62bfef5, ggml-metal-device.cpp)
- 1 stable-diffusion.cpp issue (#1040)
- 1 Apple MSL spec PDF (REPORTED-NOT-VERIFIED, used directionally only)

**Total distinct primary citations: 44.**

---

## File pointers (absolute paths)

- `/Users/cero/Code/gpucheck/.claude/teams/research/v1.0/QUESTION.md`
- `/Users/cero/Code/gpucheck/.claude/teams/research/v1.0/HYPOTHESES.md`
- `/Users/cero/Code/gpucheck/.claude/teams/research/v1.0/EVIDENCE/{planner,cartographer,archaeologist,librarian,historian,linguist,web-miner,github-miner,tracer,empiricist,synthesist,skeptic,adversary,moderator,evaluator,retrospector,scribe}.md`
- `/Users/cero/Code/gpucheck/.claude/teams/research/v1.0/SYNTHESIS.md` (this file)
- `/Users/cero/Code/gpucheck/.claude/teams/research/v1.0/TURN_LOG.md`
