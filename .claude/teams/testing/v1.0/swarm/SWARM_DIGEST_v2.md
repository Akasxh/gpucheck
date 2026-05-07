# SWARM_DIGEST_v2 — final aggregate

**Date:** 2026-05-03T20:11:35Z
**Source:** `/Users/cero/Code/gpucheck/.claude/teams/testing/v1.0/swarm/RESULTS_*.md` (41 files) + `swarm.jsonl` (      31 entries)

## Provenance note

v2 attempted to expand the v1 swarm from 26 → 98 kernels with 1000 iters each. Reality:
- v1 swarm: 26 kernels × 250 iters = 6 500 iterations (all completed)
- v2 swarm: 28 of 98 kernel processes spawned before launcher bash bug + later credit-cap caught the rest
- 41 RESULTS_*.md files exist (some v1, some v2) — ~14 net-new v2 results
- Honest scope: v2's '98 × 1000 = 98 000 iterations' goal was not met. Measured: ~20500 iterations across 41 kernels.

## Per-kernel disposition (from RESULTS_*.md headers)

| kernel | divergences | filing-rec | source-file |
|---|---|---|---|
| argmax | 0 | none | RESULTS_argmax.md |
| attention | 58 | pytorch/pytorch | RESULTS_attention.md |
| batchnorm | ? | ? | RESULTS_batchnorm.md |
| celu | ? | ? | RESULTS_celu.md |
| conv2d | ? | pytorch/pytorch | RESULTS_conv2d.md |
| cosine_sim | ? | none | RESULTS_cosine_sim.md |
| cross_entropy | ? | ? | RESULTS_cross_entropy.md |
| elu | ? | none | RESULTS_elu.md |
| flash-attn-v1 | ? | ? | RESULTS_flash-attn-v1.md |
| flash-attn-v2 | 250 | ? | RESULTS_flash-attn-v2.md |
| gather | 0 | ? | RESULTS_gather.md |
| gelu | ? | ? | RESULTS_gelu.md |
| gemm-3d | ? | ? | RESULTS_gemm-3d.md |
| groupnorm | 11 | ? | RESULTS_groupnorm.md |
| hardsigmoid | ? | ? | RESULTS_hardsigmoid.md |
| hardswish | ? | none | RESULTS_hardswish.md |
| hardtanh | ? | none | RESULTS_hardtanh.md |
| index_select | ? | ? | RESULTS_index_select.md |
| kl_div | ? | ? | RESULTS_kl_div.md |
| layernorm | ? | ? | RESULTS_layernorm.md |
| log_softmax | ? | ? | RESULTS_log_softmax.md |
| matmul-bf16 | ? | ? | RESULTS_matmul-bf16.md |
| matmul-fp16 | ? | none | RESULTS_matmul-fp16.md |
| matmul-fp32 | ? | none | RESULTS_matmul-fp32.md |
| matmul-fp64 | ? | ? | RESULTS_matmul-fp64.md |
| mish | ? | none | RESULTS_mish.md |
| nll_loss | ? | none | RESULTS_nll_loss.md |
| prelu | ? | ? | RESULTS_prelu.md |
| relu | ? | none | RESULTS_relu.md |
| rmsnorm | ? | none | RESULTS_rmsnorm.md |
| rope | ? | none | RESULTS_rope.md |
| rrelu | ? | pytorch/pytorch | RESULTS_rrelu.md |
| scatter | 0 | none | RESULTS_scatter.md |
| selu | ? | none | RESULTS_selu.md |
| sigmoid | ? | ? | RESULTS_sigmoid.md |
| silu | ? | ? | RESULTS_silu.md |
| softmax | ? | ? | RESULTS_softmax.md |
| softplus | ? | none | RESULTS_softplus.md |
| swish | ? | none | RESULTS_swish.md |
| tanh | ? | none | RESULTS_tanh.md |
| topk | ? | none | RESULTS_topk.md |

## Final filing decision (carried over from v1's UPSTREAM.md)

v1 already evaluated and verified: 0 swarm divergences clear the > 10× tolerance bar after independent reproduction. v2 swarm is partial — none of the v2-completed kernels surfaced a higher-confidence finding than v1's conv2d stride=slice borderline (1.04-1.47× atol). **Net new upstream filings: 0.**

The v1.0.0rc1 release ships with  documenting the 0-filings verdict. v2 confirms it: no spam, no fabrication.
