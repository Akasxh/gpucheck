# gpucheck Expert System

## Overview

This directory contains 10 expert agent personas that collectively cover every dimension of the gpucheck project. Each expert has deep domain knowledge, specific file ownership, review checklists, and improvement priorities.

## How to Use

When working on any module, consult the relevant expert(s) by reading their `AGENT.md` file. The expert system enforces cross-cutting quality checks:

```
You are modifying: src/gpucheck/assertions/close.py
  Primary expert: numerical-analysis-specialist
  Secondary expert: api-design-dx-lead
  Read both AGENT.md files before making changes.
```

## Expert Roster

| # | Expert | Domain | Key Ownership |
|---|--------|--------|--------------|
| 1 | **pytest-plugin-architect** | pytest hooks, fixtures, markers, plugin lifecycle | plugin.py, decorators/, fixtures/, __init__.py |
| 2 | **cuda-systems-engineer** | GPU hardware, SM architectures, compute-sanitizer | arch/, sanitizers/race.py |
| 3 | **numerical-analysis-specialist** | FP precision, tolerances, error bounds, IEEE 754 | assertions/, arch/tensor_cores.py |
| 4 | **fuzzing-property-testing-lead** | Shape/input fuzzing, hypothesis, adversarial testing | fuzzing/ |
| 5 | **performance-engineer** | CUDA event timing, roofline, regression detection | fixtures/benchmark.py, analysis/ |
| 6 | **triton-compiler-specialist** | Triton kernels, autotuning, MLIR, common pitfalls | examples/triton_*, advisory on fuzzing |
| 7 | **cicd-release-engineer** | GitHub Actions, PyPI, GPU CI, Docker, versioning | .github/, pyproject.toml |
| 8 | **api-design-dx-lead** | Public API, DX, error messages, type safety | __init__.py, all __all__ exports |
| 9 | **security-safety-specialist** | Subprocess safety, supply chain, input validation | sanitizers/, CI security |
| 10 | **docs-developer-advocate** | README, docs, examples, community, marketing | README.md, examples/, reporting/ |

## Module-to-Expert Mapping

| Module | Primary | Secondary | Tertiary |
|--------|---------|-----------|----------|
| `assertions/close.py` | numerical-analysis | api-design-dx | - |
| `assertions/tolerances.py` | numerical-analysis | performance-engineer | - |
| `assertions/reporting.py` | docs-developer-advocate | numerical-analysis | - |
| `decorators/dtypes.py` | pytest-plugin-architect | api-design-dx | - |
| `decorators/shapes.py` | pytest-plugin-architect | fuzzing-property-testing | - |
| `decorators/devices.py` | pytest-plugin-architect | cuda-systems-engineer | - |
| `decorators/parametrize.py` | pytest-plugin-architect | api-design-dx | - |
| `fixtures/benchmark.py` | performance-engineer | pytest-plugin-architect | - |
| `fixtures/profiler.py` | performance-engineer | cuda-systems-engineer | - |
| `fixtures/gpu.py` | cuda-systems-engineer | pytest-plugin-architect | - |
| `fuzzing/shapes.py` | fuzzing-property-testing | numerical-analysis | - |
| `fuzzing/inputs.py` | fuzzing-property-testing | numerical-analysis | - |
| `fuzzing/strategies.py` | fuzzing-property-testing | pytest-plugin-architect | - |
| `sanitizers/memory.py` | security-safety | cuda-systems-engineer | performance-engineer |
| `sanitizers/race.py` | security-safety | cuda-systems-engineer | - |
| `arch/detection.py` | cuda-systems-engineer | api-design-dx | - |
| `arch/compatibility.py` | cuda-systems-engineer | triton-compiler | - |
| `arch/tensor_cores.py` | numerical-analysis | cuda-systems-engineer | - |
| `analysis/roofline.py` | performance-engineer | numerical-analysis | - |
| `analysis/regression.py` | performance-engineer | numerical-analysis | - |
| `analysis/bottleneck.py` | performance-engineer | cuda-systems-engineer | - |
| `reporting/console.py` | docs-developer-advocate | api-design-dx | - |
| `reporting/json.py` | cicd-release-engineer | docs-developer-advocate | - |
| `reporting/ci.py` | cicd-release-engineer | security-safety | - |
| `plugin.py` | pytest-plugin-architect | api-design-dx | cicd-release-engineer |
| `__init__.py` | api-design-dx | pytest-plugin-architect | - |
| `pyproject.toml` | cicd-release-engineer | api-design-dx | - |
| `.github/workflows/` | cicd-release-engineer | security-safety | - |
| `examples/` | docs-developer-advocate | triton-compiler | - |
| `tests/` | all experts for their owned modules | - | - |
| `README.md` | docs-developer-advocate | api-design-dx | - |

## Cross-Expert Review Protocol

For changes touching multiple modules, the review order is:
1. Primary expert reviews correctness and domain-specific concerns
2. Secondary expert reviews API consistency and integration
3. Security expert reviews any subprocess, file I/O, or external interaction changes

## Key Findings from Analysis (52 agents deployed)

### Critical Bugs Found
- `close.py:121-122` — dead code in baseline_2x path
- `close.py:70-75` — `_resolve_dtype` picks wrong dtype for mixed-precision
- `regression.py:78` — `unique_counts.tolist()` no-op (dead code)
- `compatibility.py:20-29` — SM_ARCH_MAP uses "Blackwell-DC"/"Blackwell-Consumer" but detection.py uses "Blackwell" (mismatch)
- `plugin.py:86-88` — `_register_fixtures()` is dead code
- `plugin.py:27-30` — `--gpu-device` CLI option registered but never consumed
- `detection.py:248-256` — shared memory defaults wrong for Ada Lovelace
- `reporting/ci.py:37` — GitHub Actions annotation format malformed

### Test Coverage Gaps
- 5/18 source modules have tests (28% coverage)
- Zero tests for: reporting/ (3 modules), sanitizers/ (2 modules), fixtures/ (3 modules), fuzzing/inputs.py, fuzzing/strategies.py, arch/tensor_cores.py, analysis/bottleneck.py
- MockGPUInfo fixtures in conftest.py are dead code (never used by any test)

### Architecture Issues
- Duplicate fixture definitions (gpu_benchmark, memory_tracker in both plugin.py and fixture modules)
- `detect_gpus()` returns mutable list from lru_cache (callers can corrupt cache)
- `apply_config_tolerances` is never called (pyproject.toml tolerance overrides are dead)
- Thread-unsafe globals in multiple modules (not just documented tolerance_overrides)
- No AMD ROCm, Intel XPU, or Apple MPS support

### Strategic Gaps
- No GPU CI (all tests CPU-only)
- No documentation site
- No CHANGELOG, CONTRIBUTING, SECURITY, CODE_OF_CONDUCT
- No benchmark history / regression tracking in CI
- No stride/contiguity fuzzing (biggest fuzzing gap)
- No gradient/backward pass testing
- No determinism testing
- Reporting modules are dead code (not wired into pytest hooks)
