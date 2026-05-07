# docs-tester — gpucheck v1.1-prep audit

**Persona:** docs-tester (`/Users/cero/.claude/agents/docs/docs-tester.md`)
**Repo:** `/Users/cero/Code/gpucheck` @ `release/v1.0`
**Environment:** macOS Darwin 25.4.0, Apple Silicon, `torch 2.11.0`, CUDA unavailable, MPS available, `gpucheck 1.0.0rc1` (editable install via `uv`).
**Method:** every triple-backtick block in user-facing docs was extracted, classified, and (for executable blocks) saved to `/tmp/doctest_<id>.py` and run with `uv run --project /Users/cero/Code/gpucheck python /tmp/<id>`. Bash blocks classified as **runnable** were executed in `/tmp` with `uv run --project /Users/cero/Code/gpucheck …`; informational `pip install` blocks were not executed.
**Hard-rule note:** `src/gpucheck/__init__.py` contains **no docstrings with `Examples:` sections** (only a module-level summary, `_LAZY_MAP`, `__getattr__`, and `__all__`). `doctest` extraction yields zero blocks. The Examples sections live in submodules (`decorators/parametrize.py`, `decorators/dtypes.py`, `decorators/shapes.py`, `decorators/devices.py`) and are out of scope per the task statement, but flagged here for v1.1.

---

## Per-block table

| file | block-id | type | command | result | matches-docs? |
|---|---|---|---|---|---|
| README.md | R-B1 (L16-31) | python (fragment / pytest test) | `python /tmp/doctest_R1.py` | exit 0 — module imports + decorators bind; test fn never invoked (needs pytest + CUDA) | partial — doc shows snippet decorated with `@pytest.mark.gpu`; runs as a definition, runtime requires CUDA which is the documented "first-class" backend |
| README.md | R-B2 (L66-76) | python ⚠ (actually shell) | `python /tmp/doctest_R2.py` (extracted body only); also tested as raw shell `python -c "…"` | extracted body: exit 0 with `GPU available: False` and no Device line; **literal block as `python` source: SyntaxError** — block is fenced ` ```python ` but the first token is `python -c "` (a shell command, not Python) | **NO** — wrong fence language. Also doc claims `GPU available: True / Device: NVIDIA GeForce GTX 1650 / Compute capability: (7, 5) / Memory: 3715MB`; on this MPS host we get only `GPU available: False` (MPS is not surfaced via `gpu_available()`) |
| README.md | R-B3 (L80-85) | output (no fence lang) | n/a — quoted expected output | not run | claim is environment-specific to GTX 1650; no reproduction obligation |
| README.md | R-B4 (L91-104) | python (fragment, pytest) | `python /tmp/doctest_R3.py` | exit 0 (definition only) | runtime path requires CUDA; auto-skip on this host. Acceptable as illustrative snippet |
| README.md | R-B5 (L108-110) | bash (runnable) | `pytest test_my_kernel.py -v` (in `/tmp`) | exit 0 — `2 skipped` (no CUDA on host) | claim is "generates two test variants automatically" — variants generated; both skip on CUDA-less host. OK |
| README.md | R-B6 (L120-135) | python (fragment, pytest) | `python /tmp/doctest_R4.py` | exit 0 (definition only) | OK |
| README.md | R-B7 (L141-151) | python (fragment with `...`) | `python /tmp/doctest_R5.py` | exit 0 — decorator binds | OK |
| README.md | R-B8 (L159-168) | python (executable) | `python /tmp/doctest_R6.py` | **exit 1** — `AssertionError: Torch not compiled with CUDA enabled` | docs do not flag CUDA dependency in the snippet itself; would fail on any CUDA-less host |
| README.md | R-B9 (L174-176) | python (fragment) | `assert_close(result, expected, baseline_2x=True)` wrapped with trivial tensors → `python /tmp/doctest_R7.py` | exit 0 | OK |
| README.md | R-B10 (L182-191) | python (fragment, pytest fixture) | `python /tmp/doctest_R8.py` | exit 0 (definition only — `gpu_benchmark` fixture not injected) | OK as illustrative snippet |
| README.md | R-B11 (L197-199) | bash (runnable) | `pytest --gpu-benchmark-warmup=20 --gpu-benchmark-rounds=200 test_my_kernel.py` | exit 0 — flags accepted, tests skipped | OK (CLI options registered) |
| README.md | R-B12 (L205-214) | python (executable) | `python /tmp/doctest_R9.py` | **exit 1** — `AssertionError: Torch not compiled with CUDA enabled` after first iteration | hard-coded `device="cuda"` again |
| README.md | R-B13 (L220-229) | python (executable, hypothesis) | `python /tmp/doctest_R10.py` | **exit 1** — `Falsifying example: shape=(0, 0)` → `Torch not compiled with CUDA enabled` | hard-coded `device="cuda"` |
| README.md | R-B14 (L235-252) | python (fragment, pytest fixture + smoke) | `python /tmp/doctest_R11a.py` | exit 0 (`memory_guard` smoke ran) | OK |
| README.md | R-B15 (L260-270) | python (fragment) | `python /tmp/doctest_R12.py` | exit 0 — decorators bind | OK |
| README.md | R-B16 (L282-291) | python (executable, with claimed output) | `python /tmp/doctest_R13.py` | exit 0 — actual: `REGRESSION DETECTED: +11.7% (p=0.0001, Cohen's d=7.48)` | **NO** — README claims `+12.0%` and `Cohen's d=4.21`. Both numbers are wrong |
| README.md | R-B17 (L295-301) | python (executable) | `python /tmp/doctest_R14.py` (filled placeholders `M=N=K=128`, `bytes=1024`, `timing_results=[1.0]`) | exit 0 — prints `compute_bound` | OK; but block uses `M`, `N`, `K`, `bytes`, `timing_results` as undefined free variables — strictly only runs after the user fills them. Doc does not flag this |
| README.md | R-B18 (L319-323) | toml | `tomllib.loads(...)` | parses to `{'tool': {'gpucheck': {'tolerances': {'float16': {'atol': 0.002, 'rtol': 0.002}, 'bfloat16': {'atol': 0.03, 'rtol': 0.03}}}}}` | OK |
| README.md | R-B19 (L35-37, L41-46, L50-52, L62-64) | bash (informational `pip install`) | not executed (would mutate venv) | n/a | OK as documented |
| README.md | R-B20 (L437-448) | bash (mixed clone + install + ruff + mypy + pytest) | `ruff check src/ tests/` (exit 0), `mypy src/` (exit 0), `pytest` (224 passed, 1 skipped) | partial run | matches CONTRIBUTING test count claim. OK |
| README.md | R-B21 (L452-454) | bash | `pytest tests/gpu_integration/ -v` | exit 0 but **52 failed, 119 passed, 64 skipped** | **NO** — README implies tests "skipped automatically when no GPU is available". With MPS available the GPU-integration suite executes but is hard-coded to CUDA → 52 failures. Doc skip-claim is wrong on MPS hosts |
| README.md | R-B22 (L458-462) | bash | `pytest examples/basic_kernel_test.py -v` etc. | not executed in full, but `pytest examples/ -v` → 17 passed, 8 skipped | OK |
| README.md | R-B23 (L392-433) | text/tree (no language tag) | n/a | not run | OK |
| CHANGELOG.md | C-B1..none | — | only fenced-link references; no python/bash code blocks | n/a | OK |
| CONTRIBUTING.md | T-B1 (L14-15, inline) | bash (one-liner inside prose) | `python -c "import gpucheck; print(gpucheck.__version__)"` | exit 0 — prints `1.0.0rc1` | OK |
| CONTRIBUTING.md | T-B2 (L34-43) | bash (clone + uv sync + uv pip install) | informational (would mutate venv) | n/a | OK |
| CONTRIBUTING.md | T-B3 (L47-49) | bash (`pip install -e ".[dev]"`) | informational | n/a | OK |
| CONTRIBUTING.md | T-B4 (L66-70) | bash (lint/typecheck/pytest) | `ruff check src/ tests/` (exit 0); `mypy src/` (exit 0); `uv run pytest --tb=short -q` (224 passed, 1 skipped) | exit 0 across the board | matches "224 passing" claim |
| CONTRIBUTING.md | T-B5 (L78-87) | bash | `uv run pytest -q` (exit 0), `uv run pytest tests/gpu_integration/ -v` (52 failed), `uv run pytest examples/ -v` (17 passed, 8 skipped) | mixed | **gpu_integration claim is broken on MPS hosts** (same as R-B21) |
| CONTRIBUTING.md | T-B6..T-B12 | bash / text | branch / commit / install snippets — informational | n/a | OK |
| CONTRIBUTING.md | T-B7 (L128-134) | text (commit messages) | n/a | n/a | OK |
| CONTRIBUTING.md | T-B8 (L157-165) | text (branch names) | n/a | n/a | OK |
| CONTRIBUTING.md | T-B9 (L205-218) | text (tree) | n/a | n/a | OK |
| CLAUDE.md | K-B1 (L14-27) | text (tree) | n/a | n/a | OK |
| CLAUDE.md | K-B2 (L31-36) | bash (lint/test) | covered above | OK | OK |
| MIGRATION.md | M-B1 (L45-59) | python (executable) | `python /tmp/doctest_M1.py` | **exit 1** — two doc bugs: (a) `available_backends()` actually returns `list[Backend]` (printed `[<MPSBackend object>]`), but doc claims `("cuda",)` / `("cuda", "mps")` / `("mps",)` (str tuple); (b) `get_backend("cuda")` raises on a CUDA-less host **before** the `mps = get_backend("mps")` line — comment "raises if MPS not available" is on the wrong line | **NO** — return-type mismatch + brittle ordering |
| MIGRATION.md | M-B2 (L75-93) | python (dataclass redefinition) | `python /tmp/doctest_M2.py` | exit 0 — definition compiles, `gpucheck.GPUInfo` fields confirm `backend: str = "cuda"` is the trailing field with default | OK |
| MIGRATION.md | M-B3 (L108-121) | python (fragment) | `python /tmp/doctest_M3.py` | exit 0 (warning: `No GPU detection backend available`) | OK — note: pynvml is not installed in this venv even though docs `[mps]` extra implies MPS support is GPU-detectable. `detect_gpu()` returns `None` on this MPS host |
| MIGRATION.md | M-B4 (L132-153) | python (fragment) | `python /tmp/doctest_M4.py` | exit 0 — all `@devices(...)` shapes accept the documented args | OK |
| MIGRATION.md | M-B5 (L173-178) | python | `python /tmp/doctest_M5.py` | exit 0 | OK |
| MIGRATION.md | M-B6 (L187-203) | toml | `tomllib.loads(...)` → 12 ops parsed | exit 0 | OK |
| MIGRATION.md | M-B7 (L210-215) | python (executable) | `python /tmp/doctest_M6.py` | exit 0 but **doc claims do not match runtime values** — `is_mps_xfailed("softmax.large_attention")` returns `False` (docs say `True`); `mps_xfail_list()` returns `[]` (docs say `("scaled_dot_product_attention.large", ...)`); `register_mps_xfail(...)` returns `None` | **NO** — registry only loads inside `pytest_configure`. Outside pytest, the documented values are unreachable. Either the doc must clarify "after `pytest --collect-only`" or the registry must auto-load on import |
| MIGRATION.md | M-B8 (L264-290) | python (executable) | `python /tmp/doctest_M7.py` | **exit 1** — `TypeError: fuzz_strides() missing 1 required positional argument: 'dtype'`. Real signature is `fuzz_strides(shape, dtype, *, n=None, device="cpu", seed=None, categories=None)` returning `list[tuple[str, Tensor]]`; docs call it as `fuzz_strides(shape=(8,16,32), n=20, seed=42)`. `fuzz_strides_for_category` is also reversed: docs use `(category, shape=...)` but real signature is `(shape, dtype, category, *, device, seed)` | **NO** — broken signature on **two** documented entry points |
| MIGRATION.md | M-B9 (L305-314) | python (executable) | `python /tmp/doctest_M8.py` | **exit 1** — `TypeError: my_kernel() got an unexpected keyword argument 'args'`. Docs call `assert_deterministic(my_kernel, args=(x, y), runs=2, atol=0.0)`; real signature is `assert_deterministic(fn, *args, n=3, seed=0, **kwargs)`. The doc's `args=`, `runs=`, `atol=` are forwarded into `fn` as kwargs and explode | **NO** — signature mismatch on three keyword args |
| MIGRATION.md | M-B10 (L330-337) | bash | `uv sync --frozen` / `uv export --no-dev` | not executed (mutates env) | OK as documented |
| MIGRATION.md | M-B11 (L346-350) | bash (`pip install`) | informational | n/a | OK |
| MIGRATION.md | M-B12 (L360-366) | python | `python /tmp/doctest_M11.py` → prints `0.1.0` then `1.0.0rc1` then `gpucheck.__version__ == 1.0.0rc1` | exit 0 | OK |
| `src/gpucheck/__init__.py` | I-B1 | doctest | `python -m doctest src/gpucheck/__init__.py` | no Examples found | doc does not export Examples here. Note for v1.1: add minimal `>>> import gpucheck; gpucheck.__version__`-style smoke if desired |

---

## Setup-needs that the docs do NOT convey

- Every CUDA-pinned snippet (R-B1, R-B3, R-B6, R-B8, R-B12, R-B13) hard-codes `device="cuda"` with no `if torch.cuda.is_available()` guard. README §6 ("Step by step usage guide") asserts the entire walk-through "has been tested on a real NVIDIA GeForce GTX 1650"; on any CUDA-less host (including the MPS box used here) those snippets fail at runtime. Even though the project advertises MPS as first-class, the user-facing examples never use `device="mps"` or auto-detect.
- M-B1 demonstrates `get_backend("cuda")` on a host that may not have CUDA. Comment on line 8 of the snippet ("raises if MPS not available") is misplaced — actually the **CUDA** call raises first.
- M-B7 calls `gpucheck.is_mps_xfailed(...)` outside a pytest session; the registry is empty until `pytest_configure` runs. The doc never says this.
- R-B2 fences a shell command (`python -c "..."`) as ` ```python `. Copy-paste into a Python file is a syntax error.
- R-B17 leaves `M`, `N`, `K`, `bytes`, `timing_results` as undefined free variables in the body; no setup is shown.

---

## Recommendations for v1.1

### MUST FIX (broken examples — copy-paste failures)

1. **MIGRATION.md §7 (M-B8, L264-290) — `fuzz_strides` signature.**
   - Replace with real signature: `fuzz_strides(shape, dtype, *, n=None, device="cpu", seed=None, categories=None)` returning `list[tuple[str, Tensor]]`.
   - Fix `fuzz_strides_for_category` call site: actual signature is `(shape, dtype, category, *, device="cpu", seed=None)`, not `(category, shape=...)`.
   - Suggested replacement:
     ```python
     import torch
     from gpucheck.fuzzing import (
         fuzz_strides, fuzz_strides_for_category, StrideStrategy, STRIDE_CATEGORIES,
     )
     pairs = fuzz_strides(shape=(8, 16, 32), dtype=torch.float32, n=20, seed=42)
     bcast = fuzz_strides_for_category((8, 16, 32), torch.float32, "broadcast-induced")
     ```

2. **MIGRATION.md §8 (M-B9, L305-314) — `assert_deterministic` kwargs.**
   - Real signature: `assert_deterministic(fn, *args, n=3, seed=0, **kwargs)`. There is no `args=`, `runs=`, or `atol=` parameter.
   - Suggested replacement:
     ```python
     from gpucheck.sanitizers import assert_deterministic
     assert_deterministic(my_kernel, x, y, n=2, seed=0)
     ```

3. **MIGRATION.md §1 (M-B1, L45-59) — `available_backends()` return type.**
   - Real signature: `available_backends() -> list[Backend]`. Docs show `("cuda",) / ("cuda", "mps") / ("mps",)` (a tuple of strings). Either change the doc to show the real return type or expose a `available_backend_names() -> tuple[str, ...]` helper for symmetry with the doc.
   - Also fix the comment placement: `get_backend("cuda")` raises first on a CUDA-less host; the "raises if MPS not available" comment must move up to that line, or wrap each call with `try/except`.

4. **MIGRATION.md §5 (M-B7, L210-215) — xfail registry runtime semantics.**
   - The example `gpucheck.is_mps_xfailed("softmax.large_attention") → True` only holds inside a pytest session that has run `pytest_configure`. Outside pytest the registry is empty. Either: (a) load the `[tool.gpucheck.mps.xfail]` block on `import gpucheck`, or (b) add an explicit `# inside pytest configure` comment + a `gpucheck.register_mps_xfail("softmax.large_attention")` line so the snippet is self-contained.

5. **README.md §1 (R-B2, L66-76) — wrong fence language.**
   - Block fenced as ` ```python ` but contents are a shell command (`python -c "..."`). Re-fence as ` ```bash ` (or strip the `python -c "` wrapper and present the body as Python).

6. **README.md §9 (R-B16, L282-291) — wrong claimed output.**
   - Doc says `+12.0%` and `Cohen's d=4.21`; actual on `gpucheck==1.0.0rc1` is `+11.7%` and `Cohen's d=7.48`. Update the comment, or use `# doctest: +SKIP`.

### SHOULD FIX (silent CUDA assumptions on a project that ships MPS as first-class)

7. **README.md §4–§7 (R-B6, R-B8, R-B12, R-B13).**
   - All hard-code `device="cuda"`. On any CUDA-less host these fail with `Torch not compiled with CUDA enabled`. Recommend either:
     - parameterise via a tiny helper `device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")`, or
     - add a leading "this snippet requires CUDA — for MPS see §X" callout, or
     - add `# doctest: +SKIP` markers and label the §"Step by step usage guide" preamble as CUDA-only.

8. **README.md §"Contributing" (R-B21) and CONTRIBUTING.md §"Running tests" (T-B5).**
   - Both claim `pytest tests/gpu_integration/` auto-skips when no GPU is detected. On an MPS host (where `torch.backends.mps.is_available()` is `True`) the suite **runs** and 52 of 235 fail because the tests hard-code `cuda`. Either: gate `gpu_integration/` collection on `torch.cuda.is_available()` (not on the union), or change the doc to "skipped on hosts without **CUDA**".

9. **README.md §10 (R-B17).**
   - The snippet leaves `M`, `N`, `K`, `bytes`, `timing_results` as free variables. Add a 3-line setup or annotate "pseudocode — substitute your own values".

### NICE-TO-HAVE

10. **`src/gpucheck/__init__.py`** has no `Examples:` doctests (only a module-level summary). v1.1 could add a minimal smoke doctest for the lazy-import surface — e.g. `>>> import gpucheck; gpucheck.__version__` — to match the public-API docstring claim in CLAUDE.md.

---

## Top-3 docs that must be fixed in v1.1

1. **`MIGRATION.md`** — three broken signatures (M-B7 `available_backends` return type, M-B8 `fuzz_strides` / `fuzz_strides_for_category`, M-B9 `assert_deterministic`) plus the M-B7 xfail-registry runtime gotcha. This is the single highest-priority fix because it is the v0 → v1 upgrade guide and every example must work as printed.
2. **`README.md` §"Step by step usage guide"** — R-B2 (wrong fence language), R-B16 (wrong numeric output), and R-B6/R-B12/R-B13 (silent CUDA assumption). README is the PyPI landing page; copy-paste failures are most damaging here.
3. **`README.md` + `CONTRIBUTING.md` §"Running tests"** — gpu_integration auto-skip claim is wrong on MPS hosts (R-B21 / T-B5). The fix is either a doc clarification or a collection-time gate.
