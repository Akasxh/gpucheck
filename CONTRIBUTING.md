# Contributing to gpucheck

Thanks for your interest in **gpucheck**. This document describes how to set
up a development environment, the project's testing conventions, the PR
process, the commit-message format, and how to file effective bug reports.

---

## Reporting bugs

1. Check the [open issues](https://github.com/Akasxh/gpucheck/issues) first.
   If a duplicate exists, add information rather than open a new one.
2. Include:
   - **gpucheck version**:
     `python -c "import gpucheck; print(gpucheck.__version__)"`
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

gpucheck targets **Python ≥ 3.10**. The project uses
[`uv`](https://docs.astral.sh/uv/) for dependency management and ships a
committed `uv.lock` for reproducible installs.

```bash
git clone https://github.com/Akasxh/gpucheck.git
cd gpucheck

# Reproducible install from the lockfile (matches CI exactly)
uv sync --frozen

# Or, an editable install with all dev tools (matches local dev):
uv pip install -e ".[dev]"
```

If you do not want to use `uv`, the equivalent `pip` flow still works:

```bash
pip install -e ".[dev]"
```

Optional extras (see `pyproject.toml`):

| Extra | Purpose |
|---|---|
| `gpucheck[torch]` | Full PyTorch-backed assertions, fixtures, fuzzing |
| `gpucheck[mps]` | Apple Silicon Metal Performance Shaders backend (`torch>=2.6`) |
| `gpucheck[apple]` | Alias for `[mps]` |
| `gpucheck[hypothesis]` | Property-based shape / tensor / stride strategies |
| `gpucheck[cupy]` | CuPy interop for `__cuda_array_interface__` paths |
| `gpucheck[triton]` | Helpers for Triton-kernel test files |
| `gpucheck[all]` | Equivalent to `[torch, hypothesis]` |
| `gpucheck[dev]` | All of `[all]` plus ruff, mypy, pytest-cov |

To run the full check loop:

```bash
ruff check src/ tests/
mypy src/
uv run pytest --tb=short -q
```

These three commands gate every PR.

---

## Running tests

```bash
# Unit suite (CPU-only — runs everywhere, including CI)
uv run pytest -q

# GPU-required tests (CUDA host)
uv run pytest tests/gpu_integration/ -v

# Examples (also collected as tests)
uv run pytest examples/ -v
```

- The unit suite runs on every CI box. CI is currently CPU-only on GitHub
  Actions; GPU validation runs locally on author hardware.
- GPU integration tests are auto-skipped via `pytest.mark.gpu` and
  `pytest.mark.multi_gpu` when no GPU is detected.
- **MPS tests run automatically on Apple Silicon hosts.** When
  `torch.backends.mps.is_available()` returns `True`, `@devices()`
  expands to include `"mps"`, and `tests/test_devices_mps.py` /
  `tests/test_assert_close_mps.py` / `tests/test_mps_xfail.py` exercise
  the MPS backend without further configuration. On non-Apple hosts the
  same tests collect but skip cleanly.
- The full test count as of v1.0.0rc1 is **224 passing** (CPU-only).

---

## Code style

| Concern | Tool | Config |
|---|---|---|
| Formatter / linter | `ruff` | `pyproject.toml` `[tool.ruff]` |
| Type checking | `mypy --strict` | `pyproject.toml` `[tool.mypy]` |
| Line length | 100 chars | `[tool.ruff].line-length = 100` |
| Python target | `py310` | `[tool.ruff].target-version = "py310"` |
| Imports of optional deps | **lazy** — never at module top-level | See `src/gpucheck/__init__.py` for the canonical pattern |

**Lazy-import rule (load-bearing):** `torch`, `pynvml`, `hypothesis`, and
`cupy` must never be imported at collection time. New code must follow the
existing pattern: resolve at test execution, not at decoration; or use
the lazy module-level `__getattr__` shim in `src/gpucheck/__init__.py`.

**Error handling:** no bare `except:`; specific exception classes only
(`ImportError`, `RuntimeError`, `pynvml.NVMLError`, etc.).

---

## Commit message format

**Going forward, gpucheck uses [Conventional Commits 1.0](https://www.conventionalcommits.org/).**
The four track commits that landed v1.0 already follow this:

```
feat(mps): add Apple Silicon MPS backend with deadlock-safe benchmarking
feat(fuzzing): stride and contiguity fuzzing for GPU kernels
fix(tolerances): thread-safe override stack via contextvars; mitigate TM-E1
feat(reporting+sanitizers): HTML dashboard, determinism, lockfile, CI hardening
chore(lock): regenerate uv.lock post-merge to include Track-A [mps] extra
```

**Legacy `[ Type ] :` bracket-style commits** (visible in pre-v1.0
history, e.g. `[ Fix ] : resolve 7 bugs`) are unchanged — we are not
rewriting history. New commits should use Conventional Commits.

Allowed types: `feat`, `fix`, `docs`, `test`, `perf`, `refactor`, `chore`,
`build`, `ci`, `style`. Scope is optional but encouraged for cross-cutting
changes (e.g. `feat(mps)`, `fix(tolerances)`).

Rules:

- One logical change per commit.
- The first line is < 72 chars.
- Body (optional) wraps at 72 chars and explains *why*, not *what*.
- Reference issues with `Closes #N`, `Fixes #N`, or `triton#NNNN` /
  `pytorch#NNNN` for upstream cross-references.
- **Never commit directly to `main`.** Use a feature branch.

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

For the v1.0 release, four parallel tracks lived on
`feat/track-{a,b,c,d}-…` (MPS / strides / thread-safety / bundle).

---

## Pull request process

1. Open a draft PR early — describe what you intend to do before doing it.
   Cite the file:line you plan to touch.
2. Keep the PR scoped. If it grows past ~400 changed lines, split.
3. Each PR must pass:
   - `ruff check src/ tests/`
   - `mypy src/`
   - `uv run pytest --tb=short -q`
   - `uv run pytest examples/ -v` *(if examples are touched)*
   On a GPU host, also run `uv run pytest tests/gpu_integration/`.
4. Add a CHANGELOG entry under `## [Unreleased]` in the appropriate
   subsection (Added / Changed / Fixed / etc.). Each release rolls
   `[Unreleased]` into the next release section.
5. Update `MIGRATION.md` if the change is breaking.
6. The PR description must include:
   - **Why** the change is needed.
   - **What** behaviour changes (with `before` / `after` if numeric).
   - **Test plan** — explicit list of commands run and on what hardware.

Reviewers look for:

- Lazy-import discipline preserved.
- Public symbols documented.
- No regression in mypy strict pass.
- Tolerance changes traceable to a measured benchmark, not folk wisdom.
- For MPS-affecting changes: no `Event.synchronize()` call sites
  (deadlock — pytorch#162872); `torch.mps.synchronize()` only.

---

## Architecture overview

```
src/gpucheck/
├── __init__.py     # Lazy public API — zero torch/pynvml at import
├── plugin.py       # pytest hooks: addoption, configure, collection_modify, terminal_summary
├── assertions/     # assert_close + dtype-aware tolerances + Rich mismatch reports
├── backends/       # Backend Protocol; CUDA + MPS implementations (v1.0)
├── decorators/     # @dtypes / @shapes / @devices / @parametrize_gpu
├── fixtures/       # gpu_benchmark / gpu_device / memory_tracker
├── fuzzing/        # fuzz_shapes / fuzz_strides / Hypothesis strategies
├── sanitizers/     # memory_guard / determinism / compute-sanitizer wrapper
├── arch/           # detect_gpus / @require_arch / @require_capability / tensor cores
├── analysis/       # roofline / regression / bottleneck classification
└── reporting/      # ConsoleReporter / JSONReporter / HTMLReporter / CI annotations
```

Detailed design notes live in `CLAUDE.md` § "Key Design Decisions".

---

## How to add a new public API

1. Land the implementation behind a feature flag or extra if it adds a
   new optional dependency.
2. Export from the relevant submodule's `__all__` and add a lazy entry
   to `src/gpucheck/__init__.py`'s `_LAZY_MAP` if it should be importable
   as `gpucheck.foo`.
3. Add a docstring (Numpy or Google style — match neighbouring code in
   the same file).
4. Add unit tests under `tests/` and, if GPU-only, under
   `tests/gpu_integration/`.
5. Add a CHANGELOG `Added` entry under `[Unreleased]`.

---

## How to add a new GPU backend

The v1.0 MPS backend is the reference. To add a new backend (e.g. ROCm,
XPU):

1. Implement the `Backend` Protocol from
   `src/gpucheck/backends/_protocol.py`. Methods to provide include
   device enumeration, allocator stats, and an `event_timer` context
   manager.
2. Register in `src/gpucheck/backends/__init__.py` so
   `available_backends()` and `get_backend(name)` discover the new
   implementation.
3. Extend `_is_device_available` and `_detect_devices` in
   `src/gpucheck/decorators/devices.py` to recognize the new device-type
   string.
4. Widen the `assert_close` GPU fast-path (`assertions/close.py`) and the
   `gpu_benchmark` runner (`fixtures/benchmark.py`) to dispatch on the
   new device type.
5. If the backend has known broken kernels, add a
   `[tool.gpucheck.<backend>.xfail]` block in `pyproject.toml` and a
   loader in `src/gpucheck/plugin.py`.
6. Mirror the test pattern: `tests/test_backends.py`,
   `tests/test_devices_<backend>.py`, `tests/test_assert_close_<backend>.py`,
   `tests/test_<backend>_xfail.py`.

---

## Releasing

The release flow uses TestPyPI as a staging gate:

1. Bump `version` in `pyproject.toml` and `__version__` in
   `src/gpucheck/__init__.py`. Keep them in lock-step.
2. Update `CHANGELOG.md`: roll `[Unreleased]` into the new version
   section with a date.
3. Update `MIGRATION.md` if the release introduces user-visible API
   shape changes.
4. Tag the release on `release/v<X.Y>` (e.g. `git tag v1.0.0rc1`).
5. Build and upload to TestPyPI first; verify install + smoke test;
   then promote to PyPI. (Note: `~/.pypirc` is currently a manual
   author-only step.)
6. Create a GitHub Release citing the CHANGELOG entry; attach signed
   wheel + sdist checksums.

---

## Security

Security issues should NOT be reported through public GitHub issues.
Instead, email **<drakathakash@gmail.com>** with subject
`gpucheck security: <short description>`.

The current security audit ledger lives in
`.claude/teams/security/v1.0/FINDINGS.md` (0 CRITICAL · 0 HIGH ·
3 MEDIUM mitigated · 13 LOW advisory).

---

## Expert system

`.claude/experts/` holds 10 domain-specific personas (numerical-analysis,
pytest-plugin, fuzzing, security, CUDA-systems, performance, docs,
CI/CD, API-design, Triton). When working on a module, consult the
relevant expert(s):

| Module | Primary | Secondary |
|---|---|---|
| `assertions/` | numerical-analysis | api-design-dx-lead |
| `decorators/` | pytest-plugin-architect | api-design-dx-lead |
| `fixtures/` | pytest-plugin-architect | performance-engineer |
| `fuzzing/` | fuzzing-property-testing-lead | numerical-analysis |
| `sanitizers/` | security-safety-specialist | cuda-systems-engineer |
| `backends/` | cuda-systems-engineer | api-design-dx-lead |
| `arch/` | cuda-systems-engineer | triton-compiler-specialist |
| `analysis/` | performance-engineer | numerical-analysis |
| `reporting/` | docs-developer-advocate | cicd-release-engineer |
| `plugin.py` | pytest-plugin-architect | api-design-dx-lead |
| CI/CD | cicd-release-engineer | security-safety-specialist |
| `examples/` | docs-developer-advocate | triton-compiler-specialist |
| `pyproject.toml` | cicd-release-engineer | api-design-dx-lead |

---

## Engineering Team protocol (claude-forge contributors)

If you contribute via [claude-forge](https://claude.com/claude-code) and
want to dispatch the Engineering Team for a substantial change, the
binding protocol lives at `~/.claude/teams/engineering/PROTOCOL.md`.
The four v1.0 tracks were produced under this protocol; the audit trail
is in `.claude/teams/engineering/v1.0/`.

The Documentation Team protocol is at `~/.claude/teams/docs/PROTOCOL.md`,
with v1.0 docs evidence at `.claude/teams/docs/v1.0/`.

---

## Project meta

- **Author / maintainer:** Akash <drakathakash@gmail.com>
- **License:** Apache-2.0
- **Issues:** <https://github.com/Akasxh/gpucheck/issues>
- **PyPI:** <https://pypi.org/project/gpucheck/>
- **Source:** <https://github.com/Akasxh/gpucheck>
