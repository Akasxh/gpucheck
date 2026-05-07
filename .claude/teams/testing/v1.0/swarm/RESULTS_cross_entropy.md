# Fuzz Results — `cross_entropy`

**Agent:** `kernel-fuzzer-cross_entropy`
**Date:** 2026-05-01
**Runtime budget:** 8 min (used: ~30 s)
**Hardware:** Apple Silicon (real MPS); no NVIDIA GPU present (CUDA path mocked)

## Summary

| Field | Value |
|---|---|
| Kernel | `torch.nn.functional.cross_entropy` (reduction=mean) |
| Iterations attempted | 250 |
| Iterations completed | 250 |
| Unsupported (MPS skipped op) | 0 |
| Harness errors | 0 |
| Divergences (MPS vs CPU > tolerance) | **0** |
| Max rel-err MPS vs CPU (overall) | 1.49e-2 |
| Max rel-err MPS vs CUDA-mock | **N/A** (no NVIDIA hardware; CUDA detection mocked, not executed) |
| Recommended upstream filing | **none** |

### Max rel-err by dtype (MPS vs CPU reference, fp64-cast on CPU)

| dtype | max rel-err | gpucheck MPS tolerance (k=128, base) | margin |
|---|---|---|---|
| float32   | 2.18e-7  | atol 2e-4, rtol 2e-4 | ~3 orders of magnitude under |
| float16   | 3.96e-3  | atol 2e-2, rtol 2e-2 | within 5× of tolerance |
| bfloat16  | 1.49e-2  | atol 1e-1, rtol 1e-1 | within 7× of tolerance |

All measured errors are within the gpucheck per-dtype tolerance (default per
`assertions/tolerances.py` × MPS multiplier ×√(C/128) for the cross_entropy
class-dim reduction).

## Search Space

- **Shape buckets** (drawn uniformly): degenerate `(1,2)`, `(1,1024)`, `(2,2)`;
  prime `(7,13)`, `(13,257)`, `(31,127)`, `(127,251)`; pow-2 boundary
  `(16,32)`…`(256,1024)`; non-tile-aligned `(33,65)`, `(65,129)`, `(129,257)`,
  `(257,513)`; large `(1024,4096)`, `(2048,8192)`, `(4096,1000)`.
- **Dtypes:** float32, float16, bfloat16. (float64 unavailable on MPS;
  fp8 not supported by `nn.functional.cross_entropy` on either backend.)
- **Stride patterns** (7): `contiguous`, `transpose` (build (C,N) contig +
  `.t()`), `slice_rows` (`[::2]`), `slice_cols` (`[:,::2]`),
  `broadcast_row` (`(1,C).expand`), `broadcast_col` (`(N,1).expand`),
  `non_contig` (3-D permute view).
- Targets sampled `randint(0, C, (N,))` with independent seed.

## Top 3 Highest-Error Configurations (all within tolerance)

These are the most error-prone combinations observed; *none* exceeded
gpucheck's per-dtype tolerance. They are recorded as a calibration signal,
not as bugs.

1. `dtype=bfloat16, shape=(N, C∈[513..8192]), stride=broadcast_*` —
   max rel-err **≈1.49e-2**. Driven by bf16's 7-bit mantissa; the broadcast
   patterns produce perfectly-uniform softmax inputs, making the loss
   `−log(1/C)` (a constant), which amplifies relative-error visibility.
2. `dtype=float16, shape large+non-tile-aligned (e.g. (257,513)),
   stride=transpose` — max rel-err **≈3.96e-3**. Within fp16 budget
   (rtol 2e-2 on MPS).
3. `dtype=float32, shape=(4096, 1000), stride=non_contig` — max rel-err
   **≈2.18e-7**. Order of `eps(fp32)` for a reduction over k=1000.

(Full per-iteration dump preserved in `/tmp/ce_fuzz_out.json`.)

## Method

For each iteration: build identical f32 source values; project to the chosen
(dtype, stride pattern) on both CPU and MPS; run
`F.cross_entropy(logits, target, reduction='mean')` on both; compute
absolute and relative error against the CPU result cast to fp64. Tolerance
uses `compute_tolerance(dtype, k_dim=C, device_type='mps')` —
the gpucheck per-dtype default × MPS multiplier × √(C/128).

CUDA path: detection is mocked (no hardware), so cross-CUDA comparison is
not executed. Per task spec, this is recorded as **N/A**, not as a
divergence.

## Recommendation

**No upstream filing.** PyTorch MPS cross_entropy matches the CPU reference
within the published numerical tolerance band across all 250 fuzzed
(shape × dtype × stride) combinations. The bf16 and fp16 errors track the
expected mantissa-precision floor and do not exceed gpucheck's MPS-aware
tolerance. The op is **safe to remove from any MPS xfail list** for these
shape ranges.

## Artifacts

- Fuzzer source: `.fuzz_cross_entropy.py` (in worktree)
- Raw run dump: `/tmp/ce_fuzz_out.json`
- Stderr (empty): `/tmp/ce_fuzz_err.log`
- Seed: `20260501`
