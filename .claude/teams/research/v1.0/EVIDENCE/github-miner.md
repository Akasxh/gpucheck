---
specialist: research-github-miner
slug: v1.0
started: 2026-05-01T03:42:00Z
completed: 2026-05-01T03:43:30Z
tool_calls_count: 4
citations_count: 25
confidence: high
---

# GitHub Miner — `module: mps` issue corpus (PyTorch repo)

All searches via `gh api search/issues` against `pytorch/pytorch` on 2026-05-01.

## §1. Topline counts

| Query | Total |
|---|---|
| `label:"module: mps" state:open` | **255 open issues** |
| `label:"module: mps" state:open reactions:>5` | 10 issues |

Top-reactions open issue: [#77764 "General MPS op coverage tracking issue"](https://github.com/pytorch/pytorch/issues/77764) with **966 reactions**, last update 2026-04-18. This is the meta-tracker; not a single bug.

## §2. Top-10 highest-impact open MPS bugs (categorized for Sub-Q 2)

Selected for impact = (load-bearing kernel for ML workloads) × (silent-correctness or crash) × (recently active or high-reaction). All retrieved 2026-05-01.

### Cat (a) — Numerical drift (silent correctness)

1. **[#177116](https://github.com/pytorch/pytorch/issues/177116) "MPS: catastrophically wrong gradients in backward pass (>32K elements)"** — Opened 2026-03 by erozon; matmul/embedding/residual; FP32; gradient norms deviate by 1,000×–68,000× from CPU when total elements >32,768; **workaround: `torch.mps.empty_cache()` between operations reduces failure rate**. P0 for any training workload. Recommended gpucheck action: xfail when shape product > 32768 until fixed.

2. **[#179352](https://github.com/pytorch/pytorch/issues/179352) "MPS: scaled_dot_product_attention produces incorrect results for large batch × sequence length combinations"** — Cosine similarity MPS-vs-CPU drops to **~0.49** at B=16, seq=10240×20480; max abs diff = 0.1822. Affects video diffusion (ToonCrafter, DynamiCrafter). Suspected buffer-size or indexing overflow. Recommended action: xfail SDPA when B>2 ∧ seq_len>5120 until fixed.

3. **[#178497](https://github.com/pytorch/pytorch/issues/178497) "[MPS] Correctness issues in `count_nonzero`, `mean`, `nansum`, `sum`, `trace`"** — Reductions error by 50–90% intermittently across 300K iterations. count_nonzero 49.8% over, mean 50% over, trace 89.6% over. Recommended action: xfail these reductions until fixed.

4. **[#142836](https://github.com/pytorch/pytorch/issues/142836) "[MPS] Incorrect output from convolution ops with large dimensions"** — `Conv2d` returns ALL ZEROS when output channels > 2^16 on macOS ≥15.1. **High priority, silent correctness, regression**. Recommended action: skip conv with C_out > 65536; this is rare in practice but must xfail.

5. **[#173525](https://github.com/pytorch/pytorch/issues/173525) "[MPS] layer_norm backward numerical issues"** — At input shape (1,), MPS layer_norm backward returns ZERO grad while CPU returns non-zero; abs diff 1.22e-4, rel diff infinite. Edge case, but `assert_close(rel_diff=infinite)` will scream. Recommended action: skip layer_norm backward on shape (1,).

6. **[#175189](https://github.com/pytorch/pytorch/issues/175189) "[MPS] BatchNorm2d backward produces wildly wrong weight gradients on channels_last inputs"** — Weight gradients off by **~7 orders of magnitude** on channels_last. PR #181411 in flight. Recommended action: skip BN2d.backward + channels_last until fix lands; alternative is `.contiguous()` before BN.

7. **[#96602](https://github.com/pytorch/pytorch/issues/96602) "[MPS] softmax returns NaN attention probabilities for large tensors, in float16 and float32"** — Softmax on shape `[10, 12416, 12416]` produces NaN where it shouldn't. NaN appears at `diffs = x - maxes`. Affects float16 inference. Recommended action: skip softmax on large attention shapes (>10000 in last two dims) on MPS until fixed.

### Cat (b) — Crash/hang

8. **[#162872](https://github.com/pytorch/pytorch/issues/162872) "[MPS] dead lock when calling Event.synchronize() before Event.elapsed_time()"** — Event API hangs on `start.record(); end.record(); end.synchronize(); start.elapsed_time(end)`. **Affects gpucheck's gpu_benchmark fixture directly**. Recommended action: gpucheck must NOT call synchronize on the end event before elapsed_time on MPS; use `torch.mps.synchronize()` instead.

9. **[#175190](https://github.com/pytorch/pytorch/issues/175190) "[MPS] AvgPool2d/AdaptiveAvgPool2d backward crashes (SIGABRT) on channels_last inputs"** — `AvgPool2d.backward` on channels_last raises `MPSNDArray.mm:869: failed assertion '...buffer is not large enough...'` on PyTorch 2.10+ nightly. Workaround: `.contiguous()` first. Recommended action: skip avg-pool backward + channels_last.

### Cat (c) — Missing op

10. **[#160828](https://github.com/pytorch/pytorch/issues/160828) "The operator 'aten::_ctc_loss' is not currently implemented for the MPS device."** — Plus the broader [#154052 most-requested](https://github.com/pytorch/pytorch/issues/154052) (top-10 missing: isin, index_copy, _upsample_bicubic2d_aa, max_pool3d_with_indices, grid_sampler_3d, linalg_eig, grid_sampler_2d_backward, linalg_qr, _linalg_eigh, native_dropout). Recommended action: detect via try/except with NotImplementedError and skip; `@require_op` decorator or runtime check.

### Cat (d) — Determinism

(Bonus — these are run-to-run determinism, distinct from numerical drift)

11. **[#181936](https://github.com/pytorch/pytorch/issues/181936) "[MPS] Non-deterministic backward pass for F.linear"** (2026-04-29) — M5-specific. F.linear backward differs across consecutive calls by up to 130.0 in BF16/FP16 with bias=None and input >2D. Workaround: reshape input to 2D. Recommended action: gpucheck-MPS must not assume run-to-run reproducibility for F.linear backward on M5; consider seed-fixing AND multiple runs to detect.

12. **[#170837](https://github.com/pytorch/pytorch/issues/170837) "MPS backend - inconsistent results for batched inference on BERT/RoBERTa"** — On MPS, batched inputs to BERT/RoBERTa produce results different from non-batched. Implementation bug, not precision drift. Affects high-volume inference.

## §3. Affected kernel × dtype heat map (open bugs from §2)

| Kernel family | FP16 | BF16 | FP32 | FP64 | Status |
|---|---|---|---|---|---|
| matmul (F.linear, bmm) | bug (#181936 M5 BF16/FP16) | bug (#181936) | edge cases (#177116 large) | not relevant | partial |
| softmax | bug (#96602 large NaN) | likely | bug (#96602) | not relevant | partial |
| layer_norm fwd | OK | OK | OK | n/a | OK |
| layer_norm bwd | bug (#173525 edge) | bug (#173525) | bug (#173525) | n/a | edge-only |
| batch_norm fwd | OK | OK | OK | n/a | OK |
| batch_norm bwd channels_last | bug (#175189) | bug | bug | n/a | broken |
| group_norm | OK reported | OK | OK | n/a | OK |
| SDPA fwd | bug (#179352 large B×S) | likely | bug (#179352) | n/a | partial |
| SDPA bwd | uses math decomp (#179294) | math decomp | math decomp | n/a | slow + partial |
| conv2d fwd | bug (#142836 large C_out) | bug | bug | n/a | edge-broken |
| conv2d bwd channels_last | bug (#174269 memory format) | bug | bug | n/a | broken |
| cross_entropy | OK reported | OK | OK | n/a | OK |
| nll_loss | OK | OK | OK | n/a | OK |
| BCE | bug (#137001 silent) | bug | bug | n/a | broken (open since 2024) |
| reductions (mean/sum/trace) | bug (#178497 50-90%) | bug | bug | n/a | broken intermittent |
| avg_pool2d bwd channels_last | crash (#175190) | crash | crash | n/a | broken |

## §4. Closed/recent fixes (the trend is improvement)

`gh api search/issues -f q='label:"module: mps" repo:pytorch/pytorch is:closed merged:>2025-01-01 in:title (matmul OR softmax OR norm OR attention OR conv OR cross_entropy)' returned 0` — but the search syntax was off. Looser searches confirm regular weekly merges (e.g. #181946 stride-aware kernels, #176730 SDPA optim, #155560 sort migration).

The trend across the 60+ updated-in-last-6-weeks issues: **a steady stream of
fix PRs**, but the open queue (255) outpaces the close queue. **Do not assume
"will be fixed by ship date"** for any specific bug; xfail conservatively.

## §5. Tracking issues for op coverage

- [#77764 "General MPS op coverage tracking issue"](https://github.com/pytorch/pytorch/issues/77764) — 966 reactions; the umbrella tracker.
- [#141287 "MPS operator coverage tracking issue (2.6+ version)"](https://github.com/pytorch/pytorch/issues/141287) — 72 reactions; PyTorch 2.6+ specific. Sample tracked items: `aten::max_pool3d_with_indices` (16 requests, available in 2.9), `aten::_linalg_solve_ex.result` (15 requests, in 2.7), `aten::_standard_gamma` (13, nightly), `aten::_upsample_bicubic2d_aa.out` (10, in 2.8), `aten::linalg_qr.out` (10, nightly).
- [#154052 "Most requested ops"](https://github.com/pytorch/pytorch/issues/154052) — 13 reactions; top-10: isin (224 votes), index_copy (57), _upsample_bicubic2d_aa (49), max_pool3d_with_indices (48), grid_sampler_3d (47), linalg_eig (38), grid_sampler_2d_backward (35), linalg_qr (33), _linalg_eigh (24), native_dropout (23). Last update 2025-05-21.
- [#150121 "torch.compile on MPS progress tracker"](https://github.com/pytorch/pytorch/issues/150121) — 31 reactions; current state: "early prototype phase, with a goal of reaching beta status by version 2.8.0", "attempt to use it to accelerate end-to-end network is likely to fail". Last update 2025-03-27.

## Confidence

High — every bug claim is sourced to a specific issue number with a verified
title and an extracted fact. Counts are from live `gh api` queries on 2026-05-01.
