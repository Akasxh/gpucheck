# Fuzz Results — `torch.gather` (kernel-fuzzer-gather)

**Run date (UTC):** 2026-05-01
**Driver:** `/tmp/fuzz_gather_driver.py` (gpucheck `fuzz_shapes` + `fuzz_strides_for_category` + `compute_tolerance`)
**Worktree:** `/Users/cero/Code/gpucheck-worktrees/fuzz-gather`
**Seed:** `0xC0FFEE`
**Devices:** MPS (real, Apple Silicon) vs CPU reference. CUDA mocked (no NVIDIA GPU on this host).

## Summary

| metric | value |
|---|---|
| kernel | `torch.gather` |
| iterations attempted | 250 |
| iterations completed | 242 |
| iterations unsupported | 8 (degenerate shape — zero along some dim before gather axis selection) |
| iterations errored | 0 |
| **divergences found** | **0** |
| elapsed (sec) | 3.39 |
| MPS-vs-CPU max relative error | **0.0** (bitwise identical across all completed iterations) |
| MPS-vs-CPU max absolute error | **0.0** |
| MPS-vs-CUDA-mock max rel error | N/A (no NVIDIA GPU; CUDA detection is mock-only on this host) |

## Sweep coverage

- **Shapes:** mix of 80×ndim=2 + 80×ndim=3 from `gpucheck.fuzzing.fuzz_shapes`, covering degenerate, non-tile-aligned (31, 33, 63, 65, 127, 129, …), prime (7, 13, 31, 127, 257), power-of-2 boundaries, large (up to 512 for 2D / 128 for 3D), and mixed asymmetric shapes.
- **Dtypes:** `float32`, `float16`, `bfloat16` (all MPS-supported on Apple Silicon).
- **Stride categories** (from `gpucheck.fuzzing.strides`): `row_major`, `column_major`, `non_contig`, `transpose`, `slice`, `broadcast`. (`gather`-as-source skipped to avoid kernel-meta confusion.)
- **Gather axis:** sampled per iteration from `range(len(shape))`.
- **Index tensor:** `torch.randint(0, input.size(dim), out_shape)` with `out_shape[dim]` randomized in `[1, input.size(dim)]` to exercise both shrinking and full-size index paths.

## Top 3 minimal repros

None — no divergences were observed. `torch.gather` on MPS produced bit-identical output to CPU for every (shape, dtype, stride, dim) combination tested.

## Why zero divergences is the expected (and informative) result

`torch.gather` is a pure index-and-copy primitive: it performs no floating-point arithmetic, so there is no rounding-error path that could differ between MPS and CPU. The value of this fuzz pass is **structural**, not numerical — it confirms that MPS's gather kernel correctly resolves source-tensor strides for every category in `gpucheck.fuzzing.strides.CATEGORIES`, including:

- `broadcast` (stride-0 dim from `expand`) — MPS does not double-count or read past bounds.
- `transpose` / `column_major` (permuted strides) — MPS reads from the correct logical positions.
- `slice` (every-other, stride 2 in every dim) — MPS handles non-unit element strides correctly.
- `non_contig` (transpose-then-slice, neither pure transpose nor pure slice) — MPS handles compound view metadata.

If a gather divergence had appeared *only* under `broadcast` or `non_contig`, that would localize a missing stride-aware branch in MPS's gather dispatch — exactly the failure mode `gpucheck.fuzzing.strides` was designed to surface (see module docstring: "A test that passes `row_major` and fails `broadcast` has likely tripped over a missing broadcast-aware kernel branch."). No such asymmetry was observed.

## Recommended upstream filing target

**none** — no bug to file. `torch.gather` on MPS is consistent with CPU for the surveyed input space.

## Notes & caveats

- The 8 unsupported iterations all hit shapes containing a `0` dimension (from gpucheck's degenerate-shape category). The driver skips them rather than calling `torch.gather` with a zero-sized axis to avoid PyTorch internal-error noise; this is a driver design choice, not a kernel finding.
- Output dtypes were compared after upcasting both sides to `float32` on CPU. Since gather is bitwise pure, the upcast is sufficient — any difference would have appeared as a non-zero `(a - b).abs().max()`.
- CUDA detection on this host is mock-only (no NVIDIA hardware). MPS-vs-CUDA cross-device comparison was therefore skipped; see `gpucheck.arch.detection` for the dual-backend (pynvml/torch) detection contract.
- Per-dtype tolerances came from `gpucheck.assertions.tolerances.compute_tolerance(dtype, k_dim=None, device_type="mps")`. `gather` is not matmul-class, so no `sqrt(k/128)` scaling was applied. The MPS multiplier (currently 2× for fp32/fp16/bf16, PROVISIONAL per SYNTHESIS §7) was applied but is irrelevant given exact agreement.
