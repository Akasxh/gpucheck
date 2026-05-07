# UPSTREAM — gpucheck v1.0 + v1.1 upstream filing log

**Verdict (final, v1.1 audit cycle):** **1 issue filed.**

- **pytorch/pytorch#182815** — [MPS] MPSGraph GEMM picks 2-3x slow kernel ~40% of cold-starts, M in [768,3072] — filed 2026-05-07 after 5-specialist hardening + skeptic gate + synthesist consolidation
- URL: https://github.com/pytorch/pytorch/issues/182815
- Evidence: `.claude/teams/audit/v1.1/EVIDENCE/synthesist-final-issue.md` (filed body) + 5 hardening summaries
- Status: open, awaiting maintainer response

**Verdict (original v1.0 swarm-discovery, 2026-05-01):** 0 issues filed — no swarm-discovered divergence clears the testing-lead "high-confidence FILABLE-UPSTREAM" bar (≥3 reproductions, deterministic seed, > 10× tolerance, not on existing xfail list).

---

## What ran

Tier-3 kernel-fuzzer swarm — 26 headless `claude -p` processes, one per kernel, each running gpucheck stride/contiguity + shape + dtype fuzzers against the new `MPSBackend` on this Apple Silicon Mac. 250 iterations per kernel = **6 500 iterations total**. CUDA backend via mocked detection (no NVIDIA hardware on host).

Results aggregated: `~/Code/gpucheck/.claude/teams/testing/v1.0/swarm/RESULTS_<kernel>.md` (26 files), `swarm.jsonl` (machine-readable, 9 entries — 17 kernels emitted only markdown).

## Per-kernel divergence verdict

| kernel | divergences | swarm filing rec | bar verdict | rationale |
|---|---|---|---|---|
| relu, gelu, silu, gemm-3d, matmul-fp32, matmul-fp16, rope, topk, scatter, index_select, gather, kl_div, nll_loss, cross_entropy, softmax, flash-attn-v1, flash-attn-v2, cosine_sim, argmax, rmsnorm | 0 | none | NOT FILABLE | swarm itself reported zero divergences |
| matmul-bf16 | borderline | none | NOT FILABLE | sub-tolerance |
| layernorm | 1 | pytorch/pytorch | NOT FILABLE | single repro, magnitude not > 10× |
| batchnorm | 1 | none (swarm explicit) | NOT FILABLE | sub-ULP residual amplified by 1/sqrt(eps); swarm's own filing recommendation: "Borderline sub-ULP residual; recommend gpucheck-internal tolerance carve-out, not pytorch upstream" |
| groupnorm | 11 | pytorch/pytorch | NOT FILABLE | magnitudes within 1-3× threshold; tolerance-recalibration territory |
| conv2d | 14 | pytorch/pytorch (borderline) | NOT FILABLE | swarm's own caveat: "borderline: magnitudes 1.04-1.47x atol, may instead warrant gpucheck MPS xfail registry update". 5.6% divergence rate, all confined to `stride=slice` path at fp16/bf16. Real signal but does NOT clear > 10× bar |
| attention (SDPA) | 58 | pytorch/pytorch | NOT FILABLE | independent reproduction (840 iterations across small primes) confirmed: 0 iterations clear > 10× tolerance. Swarm-reported `max_rel_err=0.3169` is a near-zero-denominator artifact (`abs_err=4.17e-7` vs `atol=1.99e-4` = 0.002× threshold) — high *relative* error at tiny output magnitudes is not a correctness divergence by gpucheck's atol-driven model |

## Independent reproduction of attention (skeptic check)

```
$ uv run python /tmp/repro_attention_mps.py
torch=2.11.0, mps_built=True
total iterations: 840
over 10x tolerance: 0
global max_rel_err: 0.235108 at shape=(2, 4, 127, 64) dtype=float32 seed=2 \
  abs_err=4.172325e-07 thr=1.992172e-04 over_x=0.00
```

Reproduces the swarm's pattern (high relative error at small outputs) but confirms no atol violation > 10×. The swarm's "max_rel_err 0.3169 → recommend filing" inference was wrong because relative-error filtering without atol guarding is unreliable at near-zero references. `gpucheck.compute_tolerance` (and `assert_close`) correctly uses atol as the primary check; rel_err is sanity check. The swarm's standalone rel_err report broke this contract.

## What the swarm DID surface (recommendations, not filings)

1. **conv2d `stride=slice` fp16/bf16 borderline** — 14/250 iterations 1.04-1.47× the MPS-overlay atol. Specific to `stride=slice` (other 6 stride categories: 0 divergences). Recommended action: extend `[tool.gpucheck.mps.xfail]` with `conv2d.stride_slice_low_precision`, OR recalibrate the bf16/fp16 MPS multiplier from 2× to ~2.5× after empirical P99 fit on Akash's M-machine. **Not a PyTorch bug.**

2. **batchnorm broadcast input fp32** — sub-ULP residual amplified by 1/sqrt(eps) at `eps=1e-5`. Single repro. Recommended action: gpucheck internal carve-out for BN-class ops (eps-aware atol). **Not a PyTorch bug.**

3. **PROVISIONAL 2× MPS multiplier needs P99 calibration** — already documented in `CHANGELOG.md`, `MIGRATION.md` §6, and `CLAUDE.md` "Known Weaknesses". The conv2d/groupnorm finds reinforce this — calibration should be a v1.1 milestone.

## What about the prompt's "≥1 upstream issue" gate?

The Phase 3 → Phase 4 gate asks for "at least 1 upstream issue filed (URL captured in `testing/v1.0/UPSTREAM.md`)". The orchestrator's hard rules also say:

> **Do not invent benchmarks. Every number in the dashboard or report comes from a real run captured in a log.**
> **Cap upstream filings at 3. No spam.**

These rules conflict when no finding clears the high-confidence bar. The orchestrator has chosen "no spam" over "file ≥1 to satisfy a counter" — filing a borderline conv2d issue would be tolerance-recalibration noise to the PyTorch maintainers, not a correctness bug.

The 28 known PyTorch MPS issues catalogued in `research/v1.0/SYNTHESIS.md` (e.g. pytorch#162872, #181936, #178497, #142836) already cover the genuine bug surface. Re-filing variants would be duplicative.

**Recommended next step for Akash:** none of the above forces a filing now. If, after P99 calibration on the M-machine, the conv2d `stride=slice` pattern persists at > 10× atol, that becomes a real candidate. Until then, the action is internal: extend the xfail registry and recalibrate multipliers.

## Provenance

- Swarm worktrees: `~/Code/gpucheck-worktrees/fuzz-<kernel>/` (26 directories, all at `release/v1.0` tip `a60e3a5`)
- Per-kernel logs: `~/Code/gpucheck/.claude/teams/testing/v1.0/swarm/logs/<kernel>.log`
- Per-kernel results: `~/Code/gpucheck/.claude/teams/testing/v1.0/swarm/RESULTS_<kernel>.md`
- Aggregate JSONL: `~/Code/gpucheck/.claude/teams/testing/v1.0/swarm/swarm.jsonl` (9 of 26)
- Independent attention repro: `/tmp/repro_attention_mps.py` (840 iters, 0 over 10× threshold)
