# Conv2d fuzz results — conv2d

- Kernel: `torch.nn.functional.conv2d`
- Iterations attempted: 250
- Iterations completed (compared): 250
- Iterations UNSUPPORTED on MPS: 0
- Iterations errored: 0
- Divergences (abs>atol AND rel>rtol, gpucheck MPS overlay): 14
- Elapsed: 21.2s
- Backend: MPS real (torch 2.11.0, `torch.backends.mps.is_available()=True`)
- CUDA: not exercised — no NVIDIA GPU on host (mocked detection only)

## Error magnitudes (MPS vs CPU-fp32 reference)

- max abs error (most meaningful): **1.250e-01** (bfloat16, large slice case)
- median relative error: 7.79e-08 (essentially exact for fp32 contiguous cases)
- max relative error printed by harness: 4.77e+05 — **this is a near-zero-denominator artifact**;
  the reference value at that element was ≈ 1e-7, so abs_err / |ref| explodes. The
  meaningful number is `abs_err = 0.125` against atol = 0.106 — only 1.18× over the
  MPS-overlay tolerance, i.e. tolerance-edge, not catastrophic.

## MPS vs CUDA-mock

N/A — no NVIDIA GPU on host. CUDA backend was *not* exercised; the prompt
permitted mocked detection but conv2d numerical comparison requires a real
device. Reference is CPU-fp32 (industry-standard fallback for MPS validation;
see SYNTHESIS §7).

## Top 3 minimal repros (by rel_err, MPS vs CPU-fp32 ref)

1. shape=(N=1, C_in=16, H=192, W=224, C_out=64, k=3x3) dtype=bfloat16 stride=slice
   abs_err=1.250e-01 (atol=1.06e-01, rtol=1.00e-01, k_dim=144)
   → 1.18× atol — bfloat16 mantissa noise on a slice-strided input.
2. shape=(N=1, C_in=32, H=128, W=192, C_out=32, k=3x3) dtype=float16 stride=slice
   abs_err=3.125e-02 (atol=3.00e-02, rtol=2.00e-02, k_dim=288)
   → 1.04× atol — fp16 with k_dim=288, slice-strided input.
3. shape=(N=1, C_in=16, H=128, W=192, C_out=32, k=3x3) dtype=float16 stride=slice
   abs_err=3.125e-02 (atol=2.12e-02, rtol=2.00e-02, k_dim=144)
   → 1.47× atol — fp16, k_dim=144, slice-strided input.

**Pattern observation:** all 14 divergences cluster on `stride=slice`-strided inputs in
the `large` and `non_tile_aligned` shape categories at fp16/bf16. Contiguous, broadcast,
and transpose stride paths produced zero divergences across all dtypes and shape
classes. This points at MPS's conv2d strided-input code path as the locus of extra
numerical drift on Apple Silicon vs the CPU reference.

## Recommended upstream filing target: `pytorch/pytorch`

Rationale: 14/250 (5.6%) divergences exceed gpucheck's 2× MPS tolerance overlay,
all confined to the strided-slice input path on conv2d at reduced precision
(fp16/bf16). Magnitudes are 1.04–1.47× atol — borderline, not catastrophic, but
consistent enough across iterations to warrant filing. Pre-filing actions:

- Reproduce minimal case #2 deterministically with a fixed seed.
- Confirm against a fresh `pip install torch` build (we used 2.11.0).
- Compare against a CUDA fp64 reference if a Linux + NVIDIA host is available
  (rules out CPU-fp32 reference drift).

If on closer inspection the magnitudes are within IEEE-754 expected accumulation
drift for these k_dims, downgrade to `none` and instead update the gpucheck MPS
xfail registry / tolerance multiplier rather than filing upstream.
