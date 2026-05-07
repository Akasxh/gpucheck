# Contributing to gpucheck

Thanks for your interest in **gpucheck**. This document describes how to
set up a development environment, the project's testing conventions, the
PR process, the commit-message format, and how to file effective bug
reports.

> **Phase 1 status (this document):** structurally complete skeleton.
> Sections marked `{{TODO Phase 3}}` will be expanded once the v1.0
> engineering changes land.

---

## Code of conduct

This project adopts the [Contributor Covenant 2.1](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).
By participating, you agree to abide by its terms. Report unacceptable
behaviour to **<drakathakash@gmail.com>**.

`{{TODO Phase 3: vendor a verbatim copy as CODE_OF_CONDUCT.md.}}`

---

## Reporting bugs

1. Check the [open issues](https://github.com/Akasxh/gpucheck/issues)
   first. If a duplicate exists, add information rather than open a new
   one.
2. Include:
   - **gpucheck version** (`python -c "import gpucheck; print(gpucheck.__version__)"`)
   - **PyTorch version** and **device backend** — CUDA driver/toolkit
     version for NVIDIA, macOS + chip family for Apple Silicon.
   - A **minimal reproducer** — ideally a `pytest` test that uses
     `gpucheck` and fails. The `examples/triton_layernorm_bug.py` and
     `examples/triton_matmul_bug.py` files are the reference style.
   - The full traceback or `assert_close` mismatch report.
3. If the bug is in an upstream kernel (Triton, PyTorch CUDA op, MPS op)
   that gpucheck merely surfaced, file the upstream issue first and
   cross-link.

---

## Development setup

gpucheck targets **Python ≥ 3.10**. Dependency management uses pip with
the project's `pyproject.toml` extras.

```bash
git clone https://github.com/Akasxh/gpucheck.git
cd gpucheck

# Editable install with all dev tools
pip install -e ".[dev]"
```

Optional extras (see `pyproject.toml`):

| Extra | Purpose |
|---|---|
| `gpucheck[torch]` | Full PyTorch-backed assertions, fixtures, fuzzing |
| `gpucheck[hypothesis]` | Property-based shape/tensor strategies |
| `gpucheck[cupy]` | CuPy interop for `__cuda_array_interface__` paths |
| `gpucheck[triton]` | Helpers for Triton-kernel test files |
| `gpucheck[all]` | Equivalent to `[torch, hypothesis]` |
| `gpucheck[dev]` | All of `[all]` plus ruff, mypy, pytest-cov |

To run the full check loop:

```bash
ruff check src/ tests/
mypy src/
pytest --tb=short -q
```

These three commands gate every PR.

---

## Test layout

```
tests/
├── unit/                    # Pure-Python tests, no GPU required
└── gpu_integration/         # GPU-required tests, skipped on CPU-only CI
examples/                    # Runnable demos, also collected as tests
```

- The unit suite (`tests/`, excluding `tests/gpu_integration/`) runs on
  every CI box. CI is currently **CPU-only on GitHub Actions** —
  GPU validation runs locally on author hardware.
- GPU integration tests are auto-skipped via the `pytest.mark.gpu` and
  `pytest.mark.multi_gpu` markers (see `src/gpucheck/plugin.py:46-65`)
  when no GPU is detected.
- `examples/*.py` are runnable as `pytest examples/...` and double as
  upstream-bug reproducers.

To run only GPU tests on a machine with a GPU:

```bash
pytest tests/gpu_integration/ -v
```

To run examples:

```bash
pytest examples/basic_kernel_test.py -v
pytest examples/shape_fuzzing_example.py -v
pytest examples/benchmark_example.py -v
```

---

## Coding style

| Concern | Tool | Config |
|---|---|---|
| Formatter / linter | `ruff` | `pyproject.toml:69-76` |
| Type checking | `mypy --strict` | `pyproject.toml:77-85` |
| Line length | 100 chars | `[tool.ruff].line-length = 100` |
| Python target | `py310` | `[tool.ruff].target-version = "py310"` |
| Imports of optional deps | **lazy** — never at module top-level | See `__init__.py:11-45` for the canonical pattern |

**Lazy-import rule (load-bearing):** `torch`, `pynvml`, `hypothesis`,
`cupy` must never be imported at collection time. New code must follow
the pattern at `src/gpucheck/decorators/dtypes.py:36-49` (resolve at
test execution, not at decoration) or
`src/gpucheck/__init__.py:37-45` (lazy module-level `__getattr__`).

**Error handling:** no bare `except:`; specific exception classes
only (`ImportError`, `RuntimeError`, `pynvml.NVMLError`, etc.).

---

## Commit message format

The repository uses **bracket-tag prefixes**, visible in
`git log --oneline`:

```
[ Fix ]      : <description>
[ Feature ]  : <description>
[ Docs ]     : <description>
[ Test ]     : <description>
[ Perf ]     : <description>
[ Refactor ] : <description>
[ README ]   : <description>     # README-only updates
```

Examples from the repo:

```
[ Fix ]    : resolve 7 bugs, add 23 tests, rewrite docs with GTX 1650 validation
[ Feature ] : memory leak detector and compute-sanitizer integration
[ Perf ]   : GPU fast-path for assert_close, fixed tensor core detection
```

Rules:

- One logical change per commit.
- The first line is < 72 chars.
- Body (optional) wraps at 72 chars and explains *why*, not *what*.
- Reference issues with `Closes #N`, `Fixes #N`, or `triton#NNNN` for
  upstream cross-references.
- **Never commit directly to `main`.** Use a feature branch (see PR
  process below).

> Note: `CLAUDE.md`'s "Git Conventions" section currently says
> "conventional commits (type(scope): description)" — this Phase 1 doc
> documents the **actual** convention used in 100% of recent commits.
> If the project later adopts strict Conventional Commits 1.0, that
> would be a deliberate break and should land in a `[1.x]` release.

---

## Branch naming

```
feat/<short-name>      # new feature
fix/<short-name>       # bug fix
refactor/<short-name>  # internal cleanup, no behaviour change
docs/<short-name>      # docs-only
test/<short-name>      # test-only
perf/<short-name>      # performance work
chore/<short-name>     # tooling, deps, CI
```

For the v1.0 effort, the engineering team uses `feat/track-{a,b,c,d}-…`
to match the four parallel impl streams (MPS / strides /
thread-safety / bundle).

---

## Pull request process

1. Open a draft PR early — describe what you intend to do before doing
   it. Cite the file:line you plan to touch.
2. Keep the PR scoped. If it grows past ~400 changed lines, split.
3. Each PR must pass:
   - `ruff check src/ tests/`
   - `mypy src/`
   - `pytest --tb=short -q`
   - `pytest examples/ -v` *(if examples are touched)*
   On a GPU host, also run `pytest tests/gpu_integration/`.
4. Add a CHANGELOG entry under `## [Unreleased]` in the appropriate
   subsection (Added / Changed / Fixed / etc.). Phase 3 will roll
   `[Unreleased]` into the next release section.
5. Update `MIGRATION.md` if the change is breaking.
6. The PR description must include:
   - **Why** the change is needed.
   - **What** behaviour changes (with `before` / `after` if numeric).
   - **Test plan** — explicit list of commands run and on what hardware.

Reviewers will look for:
- Lazy-import discipline preserved.
- Public symbols documented (per `.claude/teams/docs/v1.0/AUDIT.md`
  §B.1).
- No regression in mypy strict pass.
- Tolerance changes traceable to a measured benchmark.

---

## Architecture overview

```
src/gpucheck/
├── __init__.py     # Lazy public API — zero torch/pynvml at import
├── plugin.py       # pytest hooks: addoption, configure, collection_modify, terminal_summary
├── assertions/     # assert_close + dtype-aware tolerances + Rich mismatch reports
├── decorators/     # @dtypes / @shapes / @devices / @parametrize_gpu
├── fixtures/       # gpu_benchmark / gpu_device / memory_tracker
├── fuzzing/        # fuzz_shapes / Hypothesis strategies / edge_inputs
├── sanitizers/     # memory_guard / check_memory_leaks / compute-sanitizer
├── arch/           # detect_gpus / @require_arch / @require_capability / tensor cores
├── analysis/       # roofline / regression / bottleneck classification
└── reporting/      # ConsoleReporter / JSONReporter / CI annotations
```

Detailed design notes live in `CLAUDE.md` § "Key Design Decisions".

---

## How to add a new public API

1. Land the implementation behind a feature flag or extra if it adds a
   new optional dependency.
2. Export from the relevant submodule's `__all__` and add a lazy entry
   to `src/gpucheck/__init__.py:_LAZY_MAP` if it should be importable
   as `gpucheck.foo`.
3. Add a docstring (Numpy or Google style — match neighbouring code
   in the same file).
4. Add unit tests under `tests/` and, if GPU-only, under
   `tests/gpu_integration/`.
5. Add a CHANGELOG `Added` entry.

---

## How to add a new GPU backend

`{{TODO Phase 3: this section is filled once the MPS Track A lands. The
canonical reference will be `src/gpucheck/arch/detection.py` and
`src/gpucheck/decorators/devices.py` showing how the second backend was
threaded through. Until then, the answer is: read the research
SYNTHESIS sub-questions 1, 3, and 7.}}`

---

## Releasing

`{{TODO Phase 3: TestPyPI flow, GitHub release, signed checksums. The
SESSION_PRECHECK.md notes that `~/.pypirc` is not configured and TestPyPI
upload is a manual step before any release.}}`

---

## Security

Security issues should NOT be reported through public GitHub issues.
Instead, email **<drakathakash@gmail.com>** with subject
`gpucheck security: <short description>`.

`{{TODO Phase 3: link to SECURITY.md once the security team publishes
their v1.0 deliverable to `<cwd>/.claude/teams/security/v1.0/`.}}`

---

## Project meta

- **Author / maintainer:** Akash <drakathakash@gmail.com>
- **License:** Apache-2.0
- **Issues:** <https://github.com/Akasxh/gpucheck/issues>
- **PyPI:** <https://pypi.org/project/gpucheck/>
- **Source:** <https://github.com/Akasxh/gpucheck>
