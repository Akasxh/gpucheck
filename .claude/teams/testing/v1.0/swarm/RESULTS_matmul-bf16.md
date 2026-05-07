# Fuzz results — `matmul-bf16` on MPS (CPU reference)

- Kernel: `matmul-bf16` (`torch.matmul`, bf16 focus)
- Iterations attempted: 250
- Iterations completed: 250
- Elapsed: 12.65s
- Seed: `0xbf16`
- Torch: `2.11.0`
- MPS backend: real (`torch.backends.mps.is_available() == True`)
- CUDA backend: **mocked** — no NVIDIA GPU present on this host

## Status counts

| status | count |
|---|---|
| DEGENERATE_OK | 9 |
| OK | 241 |

## Divergence headline

- Divergences found: **0 / 250**
- MPS-vs-CPU max relative error: `4.0213e-01`
- MPS-vs-CUDA-mock max relative error: N/A (CUDA mocked)

## Top 3 minimal repros

_No divergences observed within budget._

## Recommended upstream filing target

- **none**


## Method notes

- Per-iteration sampling: shape bucket (degenerate / non_tile_aligned / prime / po2_boundary / large / mixed), dtype (bf16 / fp16 / fp32), and an independent stride category for each of the two matmul operands (from ['row_major', 'column_major', 'transpose', 'slice', 'non_contig', 'gather']; `broadcast` excluded because matmul rejects stride-0 inner dims).
- Reference: `torch.matmul(a.float(), b.float())` on CPU. Tolerance is gpucheck's dtype-aware default with k-scaling and the MPS overlay multiplier from `assertions.tolerances`.
- Determinism: single `random.Random(0xBF16)` and `torch.manual_seed(0xBF16)`.
