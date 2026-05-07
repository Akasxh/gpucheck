# Changelog

All notable changes to **gpucheck** will be documented in this file.

The format is based on
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Phase 1 status (this document):** skeleton. Concrete `[1.0.0rc1]`
> entries are filled from `<cwd>/.claude/teams/engineering/v1.0/DIFF_LOG.md`
> and `<cwd>/.claude/teams/research/v1.0/SYNTHESIS.md` once those exist.
> Items below tagged `{{TODO Phase 3: cite DIFF_LOG}}` are placeholders.

---

## [Unreleased]

### Added

- *(reserved for post-1.0 work)*

### Changed

- *(reserved for post-1.0 work)*

### Fixed

- *(reserved for post-1.0 work)*

---

## [1.0.0rc1] — {{TODO Phase 3: tag date in YYYY-MM-DD}}

First release candidate of the v1.0 line. **Headline feature:** Apple
Silicon Metal Performance Shaders (MPS) backend, plus stride/contiguity
fuzzing, thread-safe tolerance overrides, and a release bundle.

### Added

- **MPS backend.** `assert_close`, `gpu_benchmark`, `memory_tracker`,
  `gpu_device`, `@devices`, and `@parametrize_gpu` now recognize
  `device="mps"` on Apple Silicon hosts. Detection backend extended to
  query `torch.backends.mps.is_available()` and
  `torch.mps.current_allocated_memory()`.
  *Source:* `{{TODO Phase 3: cite engineering/v1.0/DIFF_LOG.md
  feat/track-a-mps commits}}`.
  *Tolerance multipliers:* `{{TODO Phase 3: cite research/v1.0/
  SYNTHESIS.md sub-question 7}}`.

- **Stride / contiguity fuzzing.** New `fuzz_strides()` (or equivalent)
  in `gpucheck.fuzzing` plus `gpu_tensors(strides=...)` Hypothesis
  strategy.
  *Closes:* CLAUDE.md "Known Weaknesses" item "No stride/contiguity
  fuzzing".
  *Source:* `{{TODO Phase 3: cite feat/track-b-strides commits}}`.

- **Thread-safe tolerance override stack.** `tolerance_context()` is
  now safe to use from `pytest-xdist` workers and concurrent threads.
  *Closes:* CLAUDE.md "Known Weaknesses" item "Thread-safety issue in
  tolerance override stack" and the `# NOT thread-safe` comment at
  `src/gpucheck/assertions/tolerances.py:26-28`.
  *Source:* `{{TODO Phase 3: cite feat/track-c-thread-safety commits}}`.

- **Release bundle.** Wheel + sdist + signed checksums published to
  PyPI; release notes auto-generated from this CHANGELOG.
  *Source:* `{{TODO Phase 3: cite feat/track-d-bundle commits}}`.

- `MIGRATION.md` and `CONTRIBUTING.md` published for the first time.

- `CHANGELOG.md` published for the first time.

### Changed

- **`@require_arch` / `@require_capability` semantics on non-CUDA
  hosts.** `{{TODO Phase 3: document final semantics — likely
  `require_arch("Apple-Silicon")` or a new `require_backend("mps")`
  decorator. Reference DIFF_LOG.}}`

- **`GPUInfo` dataclass shape.** `{{TODO Phase 3: document whether
  `compute_capability` becomes optional, whether a `backend` field is
  added, and whether a parallel `MPSDevice`/`AppleSiliconDevice`
  dataclass is introduced. This is the largest single API-shape decision
  documented in `MIGRATION_v0_to_v1.md`.}}`

- **`assert_close` GPU fast-path** at `src/gpucheck/assertions/close.py:
  163-176` now matches both `cuda` and `mps` device types.

- **Tolerance table** (in `src/gpucheck/assertions/tolerances.py:12-24`)
  extended with MPS-calibrated rows. `{{TODO Phase 3: cite research
  SYNTHESIS sub-Q 7 for exact multipliers.}}`

- **Roofline model** `_KNOWN_SPECS` table in
  `src/gpucheck/analysis/roofline.py:40-46` extended with Apple
  Silicon entries (M1, M1 Pro/Max, M2, M2 Pro/Max, M3, M3 Pro/Max,
  M4 family — exact set TBD by engineering).

- **README "Tested hardware" section** now lists Apple Silicon
  validation alongside the existing GTX 1650 reference platform.

- **README "Known limitations" / CLAUDE.md "Known Weaknesses & Gaps"**
  rewritten to remove items closed by v1.0 (MPS, stride fuzzing,
  thread safety) and to add new genuinely-open items.

### Deprecated

- *(none in this RC; deprecations begin in `[1.1.0]` if any)*

### Removed

- *(none — v1.0 is API-additive vs v0.1.0 except for the items
  documented in `MIGRATION_v0_to_v1.md`)*

### Fixed

- 19 public symbols now have docstrings (per
  `.claude/teams/docs/v1.0/AUDIT.md` §B.1).
- The "8 real bugs" lead-paragraph claim in `README.md:12` is now
  reconciled with the `README.md:328-334` "Bugs found" table —
  `{{TODO Phase 3: confirm with engineering whether the table
  expanded to 8 rows or the lead reduced to 5}}`.

### Security

- *(see `SECURITY.md`, owned by the security team)*

### Known issues

- AMD ROCm and Intel XPU still unsupported; planned for a later
  release. CUDA + MPS only.
- Multi-GPU NCCL communication testing, CUDA graph testing, and
  gradient/backward testing remain out of scope.
- HTML / dashboard reporting still TBD `{{TODO Phase 3: confirm
  whether feat/track-d-bundle includes dashboard or only release
  artifacts}}`.

---

## [0.1.0] — 2026-{{TODO: confirm date from git tag if available}}

Initial PyPI release.

### Added

- `assert_close()` — dtype-aware tensor comparison with k_dim scaling,
  baseline_2x mode, mixed-precision auto-resolution, and Rich-formatted
  mismatch reports with error histograms.
- `@dtypes`, `@shapes`, `@devices`, `@parametrize_gpu` parametrize
  decorators with predefined groups (`FLOAT_DTYPES`, `HALF_DTYPES`,
  `ALL_DTYPES`, `FP8_DTYPES`, `SMALL_SHAPES`, `MEDIUM_SHAPES`,
  `LARGE_SHAPES`, `EDGE_SHAPES`).
- `gpu_benchmark` fixture using CUDA events, L2 flushing, and IQR
  outlier removal.
- `gpu_device` fixture and `GPUDevice` dataclass.
- `memory_tracker` fixture and `memory_guard` context manager.
- `fuzz_shapes()` deterministic corpus + `ShapeStrategy` Hypothesis
  factory; `gpu_shapes()` and `gpu_tensors()` strategies; `random_inputs`,
  `edge_inputs`, `mixed_inputs` generators.
- `check_memory_leaks()` and `run_with_sanitizer()` (NVIDIA
  compute-sanitizer wrapper).
- `arch.detection.detect_gpus()` (pynvml-first, torch fallback) plus
  `GPUInfo`, `@require_arch`, `@require_capability`.
- `analysis.regression.detect_regression()` (Mann-Whitney U +
  Cohen's d + simplified E-Divisive); `compute_roofline`,
  `classify_bottleneck`, `auto_classify_bottleneck`,
  `render_roofline_ascii`.
- `reporting.console.ConsoleReporter`, `reporting.json.JSONReporter`,
  GitHub Actions annotations, JUnit XML, PR comment generation.
- pytest hooks: `--gpu-device`, `--gpu-benchmark-warmup`,
  `--gpu-benchmark-rounds`; markers `gpu`, `slow`, `multi_gpu`.

### Findings (validated on NVIDIA GeForce GTX 1650, Turing SM75)

- 8 real bugs surfaced in upstream Triton / PyTorch CUDA ops via 511
  test configurations, including:
  - `triton#9838` — 83% relative error in Triton layer-norm at
    `n_cols=17`.
  - `triton#9839` — 0.125 abs error / FP16 index-wrapping in Triton
    matmul at K=8192.

---

[Unreleased]: https://github.com/Akasxh/gpucheck/compare/v1.0.0rc1...HEAD
[1.0.0rc1]: https://github.com/Akasxh/gpucheck/releases/tag/v1.0.0rc1
[0.1.0]: https://github.com/Akasxh/gpucheck/releases/tag/v0.1.0
