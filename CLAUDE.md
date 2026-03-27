# gpucheck Development Guide

## Project Overview
gpucheck is a pytest plugin for GPU kernel testing. It provides dtype-aware assertions, parametric testing across dtypes/shapes/devices, CUDA-event benchmarking, shape fuzzing, and memory leak detection.

**Author:** Akash (drakathakash@gmail.com)
**License:** Apache-2.0
**PyPI:** gpucheck v0.1.0
**Python:** >=3.10

## Architecture

```
src/gpucheck/
  __init__.py          # Lazy public API (import-time zero-cost)
  plugin.py            # pytest hooks: markers, fixtures, terminal summary
  assertions/          # assert_close(), tolerances, Rich mismatch reports
  decorators/          # @dtypes, @shapes, @devices, @parametrize_gpu
  fixtures/            # gpu_benchmark (CUDA events), memory_tracker, gpu_device
  fuzzing/             # fuzz_shapes(), ShapeStrategy, edge_inputs(), gpu_tensors()
  sanitizers/          # memory_guard, check_memory_leaks, compute-sanitizer wrapper
  arch/                # GPU detection (pynvml/torch), @require_arch, tensor cores
  analysis/            # roofline model, regression detection (Mann-Whitney U), bottleneck
  reporting/           # Rich console, JSON, CI (JUnit XML, GitHub annotations, PR comments)
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

## Strengths

- Found 8 real bugs in Triton/PyTorch with 511 test configs
- 83% error in Triton layer norm (triton#9838), FP16 drift in tutorial matmul (triton#9839)
- Clean pytest plugin architecture with proper hook registration
- Comprehensive dtype coverage including FP8 (E4M3, E5M2)
- Rich mismatch reports with error histograms
- Hypothesis integration via ShapeStrategy
- Architecture detection: Pascal through Blackwell (SM60-SM120)
- Tensor core generation tracking with GTX 16xx exclusion

## Known Weaknesses & Gaps

- No stride/contiguity fuzzing (only shapes and values)
- No AMD ROCm or Intel XPU support
- No GPU CI (tests run CPU-only on GitHub Actions)
- No profiling integration (Nsight Compute/Systems)
- No determinism testing support
- No gradient/backward pass testing
- No multi-GPU communication testing (NCCL)
- No CUDA graph testing support
- No HTML/dashboard reporting
- Reporting module (console, json, ci) has zero test coverage
- Thread-safety issue in tolerance override stack
- Memory leak detection uses process-level metrics (imprecise)
- No changelog, no contributing guide, no migration docs

## Code Standards

- **Strict types:** mypy strict mode, all functions typed
- **Linting:** ruff with E/F/W/I/N/UP/B/A/SIM/TCH rules
- **Line length:** 100 chars
- **Python target:** 3.10+
- **Error handling:** No bare except, specific exceptions only
- **Imports:** Lazy for optional deps (torch, pynvml, hypothesis, cupy)
- **Tests:** pytest, run without GPU, mock GPU interactions

## Git Conventions

- Branch: feature/, fix/, refactor/, docs/
- Commits: conventional commits (type(scope): description)
- Never commit to main directly
- One logical change per commit
- Account: Akasxh / drakathakash@gmail.com

## Expert System

10 expert personas live in `.claude/experts/`. Each has domain-specific context, responsibilities, and review checklists. When working on a module, consult the relevant expert(s):

| Module | Primary Expert | Secondary |
|--------|---------------|-----------|
| assertions/ | numerical-analysis-specialist | api-design-dx-lead |
| decorators/ | pytest-plugin-architect | api-design-dx-lead |
| fixtures/ | pytest-plugin-architect | performance-engineer |
| fuzzing/ | fuzzing-property-testing-lead | numerical-analysis-specialist |
| sanitizers/ | security-safety-specialist | cuda-systems-engineer |
| arch/ | cuda-systems-engineer | triton-compiler-specialist |
| analysis/ | performance-engineer | numerical-analysis-specialist |
| reporting/ | docs-developer-advocate | cicd-release-engineer |
| plugin.py | pytest-plugin-architect | api-design-dx-lead |
| CI/CD | cicd-release-engineer | security-safety-specialist |
| examples/ | docs-developer-advocate | triton-compiler-specialist |
| pyproject.toml | cicd-release-engineer | api-design-dx-lead |
