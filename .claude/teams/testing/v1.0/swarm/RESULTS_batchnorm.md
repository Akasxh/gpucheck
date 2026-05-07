# Fuzz Sweep — `torch.nn.functional.batch_norm` (MPS vs CPU)

- **Agent:** kernel-fuzzer-batchnorm
- **Run:** 2026-05-01T10:16:02+0530
- **Kernel:** `torch.nn.functional.batch_norm` (training=True, momentum=0.0, eps=1e-5)
- **Device under test:** `mps` (real, Apple Silicon)
- **Reference:** CPU fp64
- **CUDA backend:** mocked (no NVIDIA GPU on host) → numerical comparison **N/A**
- **PyTorch:** 2.11.0
- **Seed:** `0x6BA7C0DE`
- **Tolerance source:** `gpucheck.assertions.tolerances.compute_tolerance(dtype, k_dim=N*∏spatial, device_type="mps")` — applies the PROVISIONAL 2× MPS multiplier and the `sqrt(k/128)` matmul-class scaling (k = BN reduction = batch × spatial).

## Headline numbers

| Metric                                  | Value          |
|-----------------------------------------|----------------|
| Iterations attempted                    | 250            |
| Iterations completed                    | 217            |
| UNSUPPORTED (PyTorch rejected input)    | 33             |
| Hard errors (process / kernel exception)| 0              |
| **Divergences (max-rel-err over tol)**  | **1**          |
| Max rel-err MPS vs CPU                  | 7.539e+07 *    |
| Max abs-err MPS vs CPU                  | 3.316e-02      |
| Max rel-err MPS vs CUDA-mock            | N/A (mocked)   |
| Wall-clock (250 iters)                  | 1.6 s          |

\* Worst-case rel-err is dominated by a single near-zero-reference case
(reference value is ~1e-12 / exactly 0 in fp64; MPS produces ~7.5e-5).
Max abs-err is the more meaningful headline.

## What was fuzzed

| Axis            | Values                                                                    |
|-----------------|---------------------------------------------------------------------------|
| Shape category  | degenerate, prime, pow2_boundary, non_tile_aligned, large                 |
| Rank            | 70 % NCHW (4-D), 30 % NC (2-D)                                            |
| dtype           | float32, float16, bfloat16                                                |
| Stride pattern  | contiguous, slice (`[..., ::2]`), transpose (last-two-axes), broadcast (`expand`) |

All 33 UNSUPPORTED cases are `ValueError("Expected more than 1 value per
channel when training, got input size …")` from PyTorch on `(1, …)`-batched
or `(N, C, 1, 1)` inputs sliced/transposed down to one element per channel —
a documented training-mode precondition, not a kernel fault. They are
correctly classified UNSUPPORTED, not divergences.

## Top divergence (only 1 found)

| Field         | Value                              |
|---------------|------------------------------------|
| Iter ID       | 172                                |
| Shape         | `(5, 65)`                          |
| dtype         | `float32`                          |
| Stride        | `broadcast` (`expand` over batch)  |
| k_dim (red.)  | 5                                  |
| MPS atol      | 3.95e-05 (2× × sqrt(5/128) × 1e-4) |
| MPS rtol      | 2.0e-04                            |
| max abs-err   | 7.539e-05                          |
| max rel-err   | 7.539e+07 (ref ≡ 0)                |
| Excess over band | +3.587e-05                      |

### Standalone repro

```python
import torch, torch.nn.functional as F
g = torch.Generator(device="cpu").manual_seed(2024)
small = torch.randn((1, 65), generator=g, dtype=torch.float32)

# Inputs are mathematically identical across the batch axis (broadcast).
x_cpu = small.expand(5, 65)
x_mps = small.to("mps").expand(5, 65)

w = torch.ones(65); b = torch.zeros(65)
y_cpu = F.batch_norm(x_cpu.double(), None, None, w.double(), b.double(),
                    training=True, momentum=0.0, eps=1e-5)
y_mps = F.batch_norm(x_mps, None, None, w.to("mps"), b.to("mps"),
                    training=True, momentum=0.0, eps=1e-5)
torch.mps.synchronize()

# CPU: exact zeros. MPS: ~7.5e-5 noise (sub-ULP mean drift × 1/sqrt(eps) ≈ 316).
print("CPU absmax:", y_cpu.abs().max().item())   # 0.0
print("MPS absmax:", y_mps.cpu().abs().max().item())  # ~7.54e-5
```

### Diagnosis

When the input is broadcast across the batch axis, every batch row is
bit-identical, so the per-channel mean equals the input value, `x − mean`
is mathematically `0`, and the BN output is `0`.
CPU returns bit-exact `0`. MPS returns sub-ULP residuals from the
mean-subtraction reduction; those residuals are then divided by
`sqrt(var + eps) = sqrt(eps) ≈ 3.16e-3` (i.e. amplified by ~316×),
producing an O(7e-5) signal where mathematically there should be none.

This is a **reduction-order / mean-cancellation** issue, not a misbehaving
kernel — the absolute magnitude is below the bf16 atol and a hair above
the fp32 atol once `sqrt(k/128)` scales it down (k=5 is small, so atol
shrinks rather than grows). The divergence is borderline.

## Top-3 large-magnitude observations (within tolerance — informational)

These did **not** count as divergences (within MPS-multiplied band) but
are the largest abs-errs the sweep saw, useful for the calibration set:

| Shape                | dtype     | Stride     | abs-err  | rel-err |
|----------------------|-----------|------------|----------|---------|
| (2, 256, 16, 16)     | bfloat16  | broadcast  | 1.87e-02 | 9.34e+01|
| (2, 256, 16, 16)     | bfloat16  | contiguous | 2.77e-02 | 7.41e+01|
| (7, 17, 11, 11)      | bfloat16  | contiguous | 1.85e-02 | 4.04e+01|

bf16 atol band is 5e-2 (× MPS 2× = 1e-1) so all are comfortably inside.
Pattern: bf16 + large reduction (k=8192 for 2×256×16×16 plane) sees the
expected ~k-scaling residual. No reason to widen the band yet.

## Coverage matrix (completed iterations only, 217)

| dtype    | n   |
|----------|-----|
| float32  | 78  |
| float16  | 75  |
| bfloat16 | 64  |

| stride      | n   |
|-------------|-----|
| transpose   | 64  |
| contiguous  | 54  |
| broadcast   | 52  |
| slice       | 47  |

## Recommended upstream filing target

**`none`** (with caveat).

Rationale:

1. The single divergence is a **borderline, sub-ULP residual** in
   `mean − x` on MPS amplified by `1/sqrt(eps)`. Magnitude is ~7.5e-5
   in fp32, which sits at the edge of the configured tolerance after
   the gpucheck PROVISIONAL 2× MPS multiplier. It is not a correctness
   bug in the everyday sense.
2. The repro is unusual: it requires a batch axis that is fully
   broadcast (all rows identical). Real workloads rarely exhibit this
   exact pattern.
3. The cleaner action is **inside gpucheck**, not upstream:
   - Add a dedicated tolerance carve-out for BN with bit-identical
     batch rows (denominator collapses to `sqrt(eps)`, amplifying
     reduction noise ~316×).
   - Or extend `compute_tolerance` to take a `denom_floor` hint for
     normalization-class ops, then bump atol by `1/sqrt(eps_floor)`.
4. If we later find this repeating on M3/M4 with non-degenerate
   inputs, escalate to `pytorch/pytorch` with the standalone repro
   above. Until then, keep it as a calibration data point in the
   `gpucheck` MPS multiplier section (PROVISIONAL → tighten).

Filing target: **none** (gpucheck-internal calibration follow-up).

## Caveats

- 33/250 iterations were UNSUPPORTED by design (training-mode BN
  rejects ≤1 value per channel). Effective coverage is 217 iters.
- CUDA-mock comparison was not numerically meaningful — no NVIDIA GPU
  on the host. Mocked detection only verified the existing CUDA path
  is reachable; we did not produce CUDA tensors.
- MPS multipliers (`assertions/tolerances.py:_MPS_TOLERANCE_MULTIPLIERS`)
  are flagged PROVISIONAL in source. The single divergence reinforces
  that the calibration plan in SYNTHESIS §7 is needed; it does not by
  itself justify changing the multipliers.
- Backward / gradient pass not exercised (forward-only sweep).
- `eval` mode (with running stats) not exercised.
