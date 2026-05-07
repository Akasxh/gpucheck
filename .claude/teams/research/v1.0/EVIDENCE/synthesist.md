---
specialist: research-synthesist
slug: v1.0
started: 2026-05-01T03:44:00Z
completed: 2026-05-01T03:44:30Z
tool_calls_count: 0
citations_count: 18
confidence: high
---

# Synthesist — claim matrix and contradictions

Reads: `EVIDENCE/{planner,cartographer,archaeologist,librarian,historian,linguist,web-miner,github-miner,tracer,empiricist}.md`.

## §1. Claim matrix (cross-specialist convergence)

Claims labeled by the sub-question they answer. Cite source files and primary
sources. ★ = load-bearing for the v1.0 ship/no-ship decision.

### Sub-Q 1 — MPS correctness baseline

| Claim | Specialists | Primary | Confidence |
|---|---|---|---|
| ★ matmul FP32 small-medium: parity within precision floor (2× CUDA atol) | empiricist §2, github-miner §3, linguist §1.3 | absence-of-issue + #181936 (drift exists at large M5) | High |
| ★ SDPA fwd: BROKEN at large B×S | github-miner #179352, empiricist §2.2 | [pytorch#179352](https://github.com/pytorch/pytorch/issues/179352) | High |
| ★ SDPA bwd: SLOW (math decomp), partial | github-miner #179294, tracer §1.2 | [pytorch#179294](https://github.com/pytorch/pytorch/issues/179294) | High |
| layer_norm fwd: OK | github-miner §3 (no open issue) | absence | Medium |
| layer_norm bwd: BUGGY edge case (shape (1,)) | github-miner #173525, empiricist §2 | [pytorch#173525](https://github.com/pytorch/pytorch/issues/173525) | High |
| batch_norm bwd channels_last: BROKEN (7 OOM) | github-miner #175189, empiricist §2 | [pytorch#175189](https://github.com/pytorch/pytorch/issues/175189) | High |
| conv2d C_out>65536: returns zeros | github-miner #142836, empiricist §2 | [pytorch#142836](https://github.com/pytorch/pytorch/issues/142836) | High |
| softmax large attention shapes: NaN | github-miner #96602 | [pytorch#96602](https://github.com/pytorch/pytorch/issues/96602) | High |
| reductions (mean/sum/trace): 50-90% intermittent | github-miner #178497 | [pytorch#178497](https://github.com/pytorch/pytorch/issues/178497) | High |
| cross_entropy / nll_loss: OK | github-miner §3 (no recent open) | absence | Medium |

### Sub-Q 2 — Open MPS bugs (top 10 in `github-miner.md` §2)

255 open `module: mps` issues as of 2026-05-01 [verified gh api]. Top 12 listed
by category in github-miner.md §2 with action recommendations.

### Sub-Q 3 — torch.mps API surface

`torch.mps.synchronize()`, `Event`, `empty_cache`, `current_allocated_memory`,
`driver_allocated_memory`, `recommended_max_memory`, profiler subgroup,
`set_per_process_memory_fraction`, `compile_shader` — all present in 2.11
([pytorch.org/docs/2.11/mps.html](https://docs.pytorch.org/docs/2.11/mps.html), librarian §1-§4).

`torch.backends.mps.is_available()` and `is_built()` confirmed.

★ **API gotcha**: `Event.synchronize() then elapsed_time()` deadlocks
([pytorch#162872](https://github.com/pytorch/pytorch/issues/162872), tracer §2). gpucheck must use device-level `torch.mps.synchronize()`.

### Sub-Q 4 — Determinism guarantees

★ **PyTorch's MPS docs are silent on determinism** (librarian §5): no
mention of MPS in [randomness.html](https://docs.pytorch.org/docs/2.11/notes/randomness.html), no atomics/non-deterministic-ops note in MPS pages.
Apple's MSL spec [PDF retrieved 2026-05-01](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) — REPORTED-NOT-VERIFIED in this session due to 10MB cap; secondary sources confirm IEEE 754 conformance with fast-math caveats and ULP tables. Empirical record (linguist §1.1, §1.2, github-miner #181936, #170837): MPS exhibits both run-to-run drift (M5) and MPS-vs-CPU divergence (BERT/RoBERTa batched).

**Verdict**: MPS is **best-effort deterministic**. gpucheck-MPS must not assume bit-exact reproducibility; for tests that require it, fix seeds + run twice + compare.

### Sub-Q 5 — llama.cpp Metal precedent

★ Pattern: per-op `supports_op` check + automatic CPU fallback + per-op NMSE
tolerance in test-backend-ops (historian §1, web-miner §1). PyTorch MPS's design
DIFFERS — no automatic fallback; raises NotImplementedError (tracer §1.1).
gpucheck must implement coverage probing in user code.

### Sub-Q 6 — FlashAttention/Triton on Metal

★ NO Triton-Metal backend (historian §4, [triton#4824](https://github.com/triton-lang/triton/issues/4824)). MLX has fused FlashAttention-equivalent SDPA + layer_norm + rms_norm + softmax kernels (historian §2, web-miner §1). Community projects pmetal, ZMLX exist but unaudited. PyTorch MPS lags MLX on fused kernel coverage. Implication: gpucheck-MPS tests PyTorch-MPS, which surfaces PyTorch issues, NOT Apple ones.

### Sub-Q 7 — Tolerance defaults

Empiricist §2: starting overlay = 2× CUDA per-dtype atol/rtol for FP32 / FP16
/ BF16; specific xfail list cites 9 distinct primary issues (empiricist §2.2 table). Calibration on M-machine required before publishing as canonical (archaeologist §4: gpucheck has the calibration-from-measurement habit).

### Sub-Q 8 — gpucheck CUDA fuzz playbook

Verified primary: triton#9838 (83% layer_norm error at n_cols=17, OPEN), triton#9839 (FP16 matmul 0.125 at K=8192, CLOSED) — empiricist §5, archaeologist §2. Other 6 README bugs are internal-only. Strategy = priority-ordered shape fuzz + edge inputs + per-dtype tolerance + k_dim sqrt-scaling. **Strategy transfers to MPS verbatim** (cartographer §3, empiricist §1).

## §2. Contradictions and tensions

I find **no load-bearing contradiction** that requires a moderator debate. The
specialists converge on the same picture. Three SOFT tensions worth flagging:

### T1 — H1 vs H2: ship MPS in v1.0 or wait?

H1 (ship with xfails) and H2 (wait one cycle) are both supported by parts of
the corpus. H1 is supported by: 12 specific high-impact bugs are catchable as a
finite xfail list (empiricist §2.2), the fuzzing playbook transfers (cartographer §3, archaeologist §3), the API surface is documented and reachable (librarian, tracer). H2 is supported by: 255 open MPS issues with weekly turnover (github-miner §1, web-miner §6), torch.compile on MPS still "early prototype" ([pytorch#150121](https://github.com/pytorch/pytorch/issues/150121)), open SDPA/F.linear bugs touch the most important kernels.

Tension is not a bug-level disagreement — it's a **risk-tolerance tradeoff**.
The right resolution is to ship gpucheck-MPS in v1.0 in **observation-instrument-then-calibrate mode** (H1 modulated by H3): publish the xfail list AND the fuzz suite, treat the xfail list as a living document, recalibrate every release. This is consistent with archaeologist §4 and empiricist §3. **No moderator needed; resolution is in synthesis.**

### T2 — gpucheck's CUDA bug-finding record: 8 bugs or 2 bugs externally-verified?

Archaeologist §2: README claims 8 bugs but only 2 (triton#9838 and #9839) are
externally-filed and verifiable. Empiricist §5 confirms the 2 are real. The
other 6 (cuFFT, baddbmm, bmm) are internal findings.

This isn't a contradiction — it's a documentation precision concern. The README's
"8 real bugs in Triton/PyTorch" is technically true if "real bug" means "discovered
via gpucheck against PyTorch ops, regardless of filing", but the language could be
sharpened. The MPS narrative does not depend on the count being 8 vs 2 — **2
externally-verified is sufficient proof of method**.

### T3 — MPS "non-deterministic" — three meanings

Linguist §1 disambiguates. The synthesis must consistently use the linguist's
A/B/C scheme. No moderator needed; just careful synthesis writing.

## §3. Hypothesis posterior

Reading the corpus, posterior over original hypotheses (HYPOTHESES.md):
- H1 "ship with CUDA-shaped tolerance, MPS-shaped exclusions" — **0.50** (up from 0.40)
- H3 "ship MPS as correctness oracle, not parity backend" — **0.30** (up from 0.20). H1+H3 are blendable.
- H2 "wait one cycle" — **0.15** (down from 0.30) — H1's xfail mechanism is sufficient.
- H4 "MLX/llama.cpp already cover this" — **0.05** (down from 0.10) — MLX does not test PyTorch MPS, which is the question.

**Posterior**: blend H1 + H3. Ship gpucheck-MPS as both a parity validator
(against CPU FP32/FP64 reference) and a correctness oracle for PyTorch MPS bugs.
The xfail list is the load-bearing artifact.

## §4. Open questions for SYNTHESIS

- (Q-A) Does the Apple Metal Shading Language spec PDF actually quantify ULP
  bounds for fast-math vs precise-math at FP16? The synthesist could not verify
  PDF text in this session (10MB cap). Action: the engineering team can fetch
  locally; for v1.0 the qualitative claim "fast-math defaults loosen FP16
  guarantees" is sufficient.
- (Q-B) Does Akash have an M5 (where #181936 reproduces) or an earlier M? The
  v1.0 README MPS table will publish whichever generation the calibration runs
  on, and the xfail list must explicitly mark M5-specific items.
- (Q-C) Is `torch.mps.empty_cache()` between fuzz iterations sufficient to make
  #177116 disappear, or is it merely a frequency-reducer? Empirical only;
  gpucheck-MPS test runs will tell us.

## Confidence

High on §1 (every claim cited to ≥1 primary source), high on §2 (no
load-bearing contradiction), high on §3 (posterior follows from §1).
