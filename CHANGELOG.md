# Changelog

All notable changes to **gpucheck** will be documented in this file.

The format is based on
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- *(reserved for post-1.0 work)*

### Changed

- *(reserved for post-1.0 work)*

### Fixed

- *(reserved for post-1.0 work)*

---

## [1.0.0rc1] — 2026-05-01

First release candidate of the v1.0 line. Four parallel engineering tracks
delivered: an Apple Silicon Metal Performance Shaders (MPS) backend, stride
and contiguity fuzzing, a thread-safe tolerance override stack, and a
release-bundle hardening pass.

> **Headline:** gpucheck is no longer CUDA-only. The same fuzz playbook that
> found `triton#9838` (83% layer-norm error, OPEN) and `triton#9839` (FP16
> matmul drift, CLOSED) now runs on Apple Silicon, with a curated 12-entry
> xfail list covering known-broken PyTorch MPS kernels.

### Added

- **MPS backend (Track A — `feat/track-a-mps`).**
  - New `gpucheck.backends` package introduces a `runtime_checkable`
    `Backend` Protocol (`src/gpucheck/backends/_protocol.py`) plus
    `CUDABackend` (`backends/cuda.py`) and `MPSBackend` (`backends/mps.py`).
    Public entry points: `gpucheck.available_backends()` and
    `gpucheck.get_backend(name)`.
  - `assert_close`, `gpu_benchmark`, `memory_tracker`, `gpu_device`,
    `@devices`, and `@parametrize_gpu` now recognize `device="mps"`.
    `@devices()` auto-detects MPS via `torch.backends.mps.is_available()`,
    and `@devices("all")` includes MPS when present.
  - **Deadlock-safe MPS benchmarking.** Per
    [pytorch#162872](https://github.com/pytorch/pytorch/issues/162872),
    the pattern `start.record(); end.record(); end.synchronize();
    start.elapsed_time(end)` deadlocks on MPS. `MPSBackend.event_timer`
    routes to device-level `torch.mps.synchronize()` instead of
    per-event `Event.synchronize()`. A test in `tests/test_backends.py`
    AST-introspects the MPS path to assert no `Event.synchronize` call
    site exists.
  - **Per-kernel xfail registry.** A new `[tool.gpucheck.mps.xfail]`
    section in `pyproject.toml` ships with **12 curated entries** drawn
    from the research SYNTHESIS top-impact open MPS bugs. Loaded at
    `pytest_configure` time; queryable via `gpucheck.is_mps_xfailed(op)`,
    `gpucheck.mps_xfail_list()`, and `gpucheck.register_mps_xfail(op)`.
    The 12 entries cite concrete upstream issues (pytorch#179352,
    #179294, #173525, #175189, #142836, #174269, #181936, #96602,
    #175190, #176296, #137001, #177116) — see
    `.claude/teams/research/v1.0/SYNTHESIS.md` §Sub-Q 2 / §Sub-Q 7.
  - **MPS tolerance overlay.** `_MPS_TOLERANCE_MULTIPLIERS` in
    `src/gpucheck/assertions/tolerances.py` adds a 2× per-dtype multiplier
    on top of the CUDA-calibrated baseline for FP32, FP16, and BF16.
    The 2× multiplier is **PROVISIONAL** per SYNTHESIS §Sub-Q 7 — it is
    a hypothesis grounded in precision-floor + Apple-no-FP16-tensor-cores
    arguments, pending P99 calibration on Akash's M-machine. If observed
    drift exceeds 2×, the affected op moves to the xfail registry rather
    than further inflating tolerances.
  - New install extras: `pip install gpucheck[mps]` and
    `pip install gpucheck[apple]` (alias). Both pin `torch>=2.6` (the
    floor where `torch.mps.synchronize()` is stable).
- **Stride / contiguity fuzzing (Track B — `feat/track-b-strides`).**
  - New `src/gpucheck/fuzzing/strides.py` with a 7-category deterministic
    corpus (`row-major`, `column-major`, `broadcast-induced`, `transpose`,
    `slice`, `contiguous-after-clone`, `gather-induced`).
  - Public surface: `fuzz_strides()`, `fuzz_strides_for_category()`,
    `StrideStrategy` (Hypothesis), and `STRIDE_CATEGORIES`.
  - `parametrize_gpu(stride_categories=...)` wiring threads stride
    fuzzing through the existing dtype/shape/device matrix.
- **Thread-safe tolerance overrides (Track C — `feat/track-c-thread-safety`).**
  - The override stack at `src/gpucheck/assertions/tolerances.py` is now
    a `contextvars.ContextVar`. `tolerance_context(atol, rtol)` is safe
    from `pytest-xdist` workers AND concurrent threads inside a single
    worker. `asyncio` tasks are isolated per the standard `ContextVar`
    semantics. **No user-facing API change.**
- **Release bundle (Track D — `feat/track-d-bundle`).**
  - **HTML dashboard** — `gpucheck.reporting.HTMLReporter` produces a
    self-contained static HTML artifact (`src/gpucheck/reporting/html.py`).
  - **Determinism sanitizer** — `assert_deterministic`,
    `@requires_determinism`, and `DeterminismError`
    (`src/gpucheck/sanitizers/determinism.py`). Implements the
    "fix seeds + run twice + compare" recipe required for MPS, where
    bit-exact reproducibility is not guaranteed (SYNTHESIS §Sub-Q 4).
  - **`uv.lock` committed** (DEP-1 mitigation). Downstream installs that
    consume the lock benefit from supply-chain reproducibility.
  - **CI hardened** — `.github/workflows/ci.yml` gains
    `permissions: contents: read` (CFG-2) and installs from the locked
    environment via `uv sync --frozen`.
- `MIGRATION.md`, `CONTRIBUTING.md`, and `CHANGELOG.md` published for the
  first time.

### Changed

- **`GPUInfo` shape** (`src/gpucheck/arch/detection.py`) gains a
  `backend: str = "cuda"` field. Existing CUDA-shaped fields
  (`compute_capability`, `tensor_core_generation`, `cuda_version`)
  remain in place; on MPS, they take backend-appropriate sentinel values.
  See `MIGRATION.md` §1.
- **`assert_close` GPU fast-path** (`src/gpucheck/assertions/close.py`)
  widened from CUDA-only to recognize both `cuda` and `mps` device types.
  `compute_tolerance(...)` now accepts `device_type=` for MPS overlay routing.
- **Tolerance computation** (`src/gpucheck/assertions/tolerances.py`)
  threads `device_type` through and applies `_MPS_TOLERANCE_MULTIPLIERS`
  when `device_type == "mps"` (see PROVISIONAL note above).
- **`@devices(...)` semantics.** With no arguments, auto-detect now
  includes `mps` on Apple Silicon hosts. The string `"all"` resolves to
  the union of all detected devices (CUDA + MPS).
- **`gpu_benchmark` fixture** (`src/gpucheck/fixtures/benchmark.py`) now
  branches on `device.type`: `_run_cuda` keeps the existing CUDA-event
  path; `_run_mps` uses `torch.mps.event.Event` with device-level sync.
- **`pytest_configure` hook** (`src/gpucheck/plugin.py`) reads
  `[tool.gpucheck.mps.xfail]` and `[tool.gpucheck.tolerances]` from
  `pyproject.toml` via `tomllib` and applies them to the active session.
- **CI** runs `uv sync --frozen` instead of `pip install -e ".[dev]"`.
- **Reporting test coverage**: 0% → 98% (tracks console, JSON, CI
  annotations, JUnit XML, PR comments, HTML dashboard).
- **Test count**: 117 → 224 (107 net new tests across the four tracks).

### Deprecated

- *(none in this release candidate)*

### Removed

- *(none — v1.0 is API-additive vs v0.1.0; see `MIGRATION.md` for shape
  changes that did not require removal)*

### Fixed

- **Reporting module zero coverage** (was a documented gap in
  `CLAUDE.md`'s "Known Weaknesses & Gaps") — closed by Track D.
- **Thread-safety in tolerance overrides** (was the `# NOT thread-safe`
  comment at `assertions/tolerances.py`) — closed by Track C.
- **Stride/contiguity fuzzing gap** (was a documented gap) — closed by
  Track B.
- **HTML/dashboard reporting gap** (was a documented gap) — closed by
  Track D.
- **Determinism testing gap** (was a documented gap) — closed by
  Track D's `sanitizers/determinism.py`.
- README "8 bugs" claim reconciled with the visible "Bugs found" table:
  the README now states **8 bugs found via 511 test configurations, of
  which 2 (`triton#9838` OPEN, `triton#9839` CLOSED) are externally
  filed and verified upstream**. The remaining 5 in the table are
  internal-ledger findings reproducible from `examples/`. See AUDIT.md
  §A.5 item 32 and SYNTHESIS §Sub-Q 8.

### Security

- **CFG-2 (MEDIUM)** — `.github/workflows/ci.yml` now declares
  `permissions: contents: read`, locking down the default
  `GITHUB_TOKEN` write-all surface. (Track D.)
- **TM-E1 (MEDIUM)** — `src/gpucheck/sanitizers/race.py` validates
  `CUDA_HOME` / `CUDA_PATH` against an allowlist of canonical install
  prefixes (`/usr/local/cuda`, `/opt/nvidia/cuda`, `/opt/cuda`) using
  `os.path.realpath`, with a warning on rejection. (Track C.)
- **DEP-1 (MEDIUM)** — `uv.lock` is now committed; CI installs from the
  lockfile (`uv sync --frozen`). (Track D.)
- **N1, N2, N3, N5 (MPS design-stage)** — explicitly waived in
  engineering CHARTER.md §"Explicit waivers": gpucheck v1.0 does not
  shell out to `xcrun metal*`, does not read `task_info` directly, and
  does not ship an MPS dispatch sanitizer. PyTorch's hardening is the
  trust boundary.
- **N4 (MPS design-stage)** — partially mitigated by the `torch>=2.6`
  floor on the `[mps]` extra and the DEP-1 lockfile; no third-party
  Apple-only packages introduced.
- See `.claude/teams/security/v1.0/FINDINGS.md` for the full ledger
  (0 CRITICAL · 0 HIGH · 3 MEDIUM · 13 LOW; **ADVISORY** verdict).

### Migration

- See [`MIGRATION.md`](./MIGRATION.md) for the v0.1.0 → v1.0 guide.
- v1.0 is API-additive: existing CUDA-only code continues to work.
  The new entry points are `gpucheck.backends.get_backend()`,
  `@devices("mps")`, the `[tool.gpucheck.mps.xfail]` config block, and
  the `[mps]` install extra.
- The PROVISIONAL 2× MPS tolerance multiplier may be revised in
  v1.0.0 final after M-machine calibration; users hard-coding overlays
  should track this CHANGELOG.

### Known issues

- AMD ROCm and Intel XPU still unsupported; planned for a later release.
  CUDA + MPS only.
- Multi-GPU NCCL communication testing, CUDA graph testing, and
  gradient/backward testing remain out of scope.
- The 2× MPS tolerance multiplier is PROVISIONAL pending M-machine
  P99 calibration; if measured drift exceeds 2× for a specific kernel,
  that kernel will move to the xfail registry rather than further
  inflating tolerances.

---

## [0.1.0] — 2026-03

Initial PyPI release.

### Added

- `assert_close()` — dtype-aware tensor comparison with `k_dim` scaling,
  `baseline_2x` mode, mixed-precision auto-resolution, and Rich-formatted
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

- 8 bugs surfaced via 511 test configurations against Triton tutorials
  and PyTorch CUDA ops. Externally filed and verified:
  - [`triton#9838`](https://github.com/triton-lang/triton/issues/9838)
    — 83% relative error in Triton tutorial layer-norm at `n_cols=17`
    (OPEN as of 2026-05).
  - [`triton#9839`](https://github.com/triton-lang/triton/issues/9839)
    — FP16 index wrapping in Triton tutorial matmul, 0.125 abs error at
    K=8192 (CLOSED).

[Unreleased]: https://github.com/Akasxh/gpucheck/compare/v1.0.0rc1...HEAD
[1.0.0rc1]: https://github.com/Akasxh/gpucheck/releases/tag/v1.0.0rc1
[0.1.0]: https://github.com/Akasxh/gpucheck/releases/tag/v0.1.0
