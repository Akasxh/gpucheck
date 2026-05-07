# gpucheck Development Guide

## Project Overview
gpucheck is a pytest plugin for GPU kernel testing. It provides dtype-aware assertions, parametric testing across dtypes/shapes/devices, CUDA-event benchmarking, shape fuzzing, and memory leak detection.

**Author:** Akash (drakathakash@gmail.com)
**License:** Apache-2.0
**PyPI:** gpucheck v1.0.0rc1 (initial PyPI publish was v0.1.0)
**Python:** >=3.10
**Backends:** CUDA + Apple MPS (v1.0)

## Architecture

```
src/gpucheck/
  __init__.py          # Lazy public API (import-time zero-cost)
  plugin.py            # pytest hooks: markers, fixtures, terminal summary, pyproject loader
  assertions/          # assert_close(), tolerances (ContextVar overrides + MPS overlay), Rich reports
  backends/            # Backend Protocol + CUDABackend + MPSBackend (deadlock-safe events) [v1.0]
  decorators/          # @dtypes, @shapes, @devices (CUDA+MPS), @parametrize_gpu (stride_categories=)
  fixtures/            # gpu_benchmark (CUDA events / MPS device-sync), memory_tracker, gpu_device
  fuzzing/             # fuzz_shapes(), fuzz_strides() [v1.0], ShapeStrategy, StrideStrategy
  sanitizers/          # memory_guard, check_memory_leaks, compute-sanitizer wrapper, determinism [v1.0]
  arch/                # GPU detection (pynvml/torch), @require_arch, tensor cores
  analysis/            # roofline model, regression detection (Mann-Whitney U), bottleneck
  reporting/           # Rich console, JSON, HTML dashboard [v1.0], CI (JUnit XML, GH annotations, PR comments)
```

## Build & Test

```bash
pip install -e ".[dev]"        # Install with dev deps
pytest --tb=short -q           # Run CPU-only tests
ruff check src/ tests/         # Lint
mypy src/                      # Type check
```

## Key Design Decisions

- **Lazy imports everywhere:** torch/pynvml never imported at collection time
- **Dual backend:** pynvml preferred over torch for detection (lighter)
- **Tolerance model:** Base tolerances per dtype, scaled by sqrt(k/128) for matmul ops
- **GPU fast-path:** assert_close checks torch.allclose on-device first, falls back to numpy for rich reporting
- **Statistical benchmarking:** CUDA events + L2 flush + IQR outlier removal
- **Shape fuzzing priority:** degenerate > non-tile-aligned > prime > power-of-2 boundary > large > mixed

## v1.0 highlights (delivered)

Four parallel engineering tracks shipped in v1.0:

- **Track A — MPS backend** (`feat/track-a-mps`): `gpucheck.backends.{Backend, CUDABackend, MPSBackend}`, `@devices("mps")`, deadlock-safe MPS event timing (avoids pytorch#162872), 2× MPS tolerance overlay (PROVISIONAL — pending M-machine P99 calibration), 12-entry `[tool.gpucheck.mps.xfail]` config block citing concrete pytorch issues, `[mps]` and `[apple]` install extras (pin `torch>=2.6`).
- **Track B — Stride fuzzing** (`feat/track-b-strides`): `fuzz_strides()` deterministic 7-category corpus (row-major / column-major / broadcast-induced / transpose / slice / contiguous-after-clone / gather-induced), `StrideStrategy` for Hypothesis, `parametrize_gpu(stride_categories=...)` wiring.
- **Track C — Thread-safety** (`feat/track-c-thread-safety`): `tolerance_context()` now backed by `contextvars.ContextVar` (safe for `pytest-xdist` workers, threads, asyncio). Also mitigates security finding TM-E1 by adding `realpath`+allowlist validation for `CUDA_HOME`/`CUDA_PATH` in `sanitizers/race.py`.
- **Track D — Release bundle** (`feat/track-d-bundle`): HTML dashboard reporter, `assert_deterministic` / `@requires_determinism` / `DeterminismError`, committed `uv.lock` (DEP-1 mitigation), CI permissions hardening (CFG-2 mitigation), reporting test coverage 0% → 98%.

Test count: **224 passing** (was 117 pre-v1.0).

See `CHANGELOG.md`, `MIGRATION.md`, `.claude/teams/engineering/v1.0/DIFF_LOG.md`, `.claude/teams/research/v1.0/SYNTHESIS.md`, and `.claude/teams/security/v1.0/FINDINGS.md`.

## Strengths

- Found 8 bugs in Triton/PyTorch with 511 test configs (2 externally verified: triton#9838 open, triton#9839 closed; 6 internal-ledger findings reproducible from `examples/`)
- 83% error in Triton layer norm (triton#9838), FP16 drift in tutorial matmul (triton#9839)
- Clean pytest plugin architecture with proper hook registration
- Comprehensive dtype coverage including FP8 (E4M3, E5M2)
- Rich mismatch reports with error histograms
- Hypothesis integration via ShapeStrategy and StrideStrategy
- Architecture detection: Pascal through Blackwell (SM60-SM120)
- Tensor core generation tracking with GTX 16xx exclusion
- **Apple MPS backend with curated xfail registry and deadlock-safe benchmarking** (v1.0)
- **Stride / contiguity fuzzing across 7 categories** (v1.0)
- **Thread-safe tolerance overrides via ContextVar** (v1.0)
- **HTML dashboard, determinism sanitizer, committed uv.lock** (v1.0)
- **Reporting module 98% covered** (v1.0; was 0%)
- **Published CHANGELOG, CONTRIBUTING, MIGRATION** (v1.0)

## Known Weaknesses & Gaps

- No AMD ROCm or Intel XPU support (planned)
- No GPU CI (tests run CPU-only on GitHub Actions; GPU validation runs locally on author hardware)
- No profiling integration (Nsight Compute/Systems)
- No gradient/backward pass testing
- No multi-GPU communication testing (NCCL)
- No CUDA graph testing support
- Memory leak detection uses process-level metrics on MPS (psutil RSS proxy per pytorch#164299) — imprecise but deliberate fallback
- The 2× MPS tolerance multiplier is PROVISIONAL — pending P99 calibration on Akash's M-machine; ops with measured drift > 2× will move to the xfail registry rather than further inflate the multiplier

## Code Standards

- **Strict types:** mypy strict mode, all functions typed
- **Linting:** ruff with E/F/W/I/N/UP/B/A/SIM/TCH rules
- **Line length:** 100 chars
- **Python target:** 3.10+
- **Error handling:** No bare except, specific exceptions only
- **Imports:** Lazy for optional deps (torch, pynvml, hypothesis, cupy)
- **Tests:** pytest, run without GPU, mock GPU interactions

## Git Conventions

- Branch: `feat/`, `fix/`, `refactor/`, `docs/`, `test/`, `perf/`, `chore/`
- Commits: **Conventional Commits 1.0** going forward (`feat(scope): message`); legacy `[ Type ] :` bracket-style commits in pre-v1.0 history are unchanged
- Never commit to `main` directly
- One logical change per commit
- Account: Akasxh / drakathakash@gmail.com
- See `CONTRIBUTING.md` for the full PR + commit-message guide

## Expert System

10 expert personas live in `.claude/experts/`. Each has domain-specific context, responsibilities, and review checklists. When working on a module, consult the relevant expert(s):

| Module | Primary Expert | Secondary |
|--------|---------------|-----------|
| assertions/ | numerical-analysis-specialist | api-design-dx-lead |
| decorators/ | pytest-plugin-architect | api-design-dx-lead |
| fixtures/ | pytest-plugin-architect | performance-engineer |
| fuzzing/ | fuzzing-property-testing-lead | numerical-analysis-specialist |
| sanitizers/ | security-safety-specialist | cuda-systems-engineer |
| backends/ | cuda-systems-engineer | api-design-dx-lead |
| arch/ | cuda-systems-engineer | triton-compiler-specialist |
| analysis/ | performance-engineer | numerical-analysis-specialist |
| reporting/ | docs-developer-advocate | cicd-release-engineer |
| plugin.py | pytest-plugin-architect | api-design-dx-lead |
| CI/CD | cicd-release-engineer | security-safety-specialist |
| examples/ | docs-developer-advocate | triton-compiler-specialist |
| pyproject.toml | cicd-release-engineer | api-design-dx-lead |
