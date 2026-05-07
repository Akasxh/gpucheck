# executor-C.md

**Branch**: `feat/track-c-thread-safety` @ `5ddd26e`

## C.1 ContextVar tolerance stack
- File: `src/gpucheck/assertions/tolerances.py`.
- Replaced `_tolerance_overrides: list[...]` with `ContextVar[tuple[tuple[float, float], ...]]`.
- `tolerance_context` now uses `ContextVar.set` (returns Token) and `ContextVar.reset(token)`.
- `compute_tolerance` reads via `_tolerance_overrides.get()`.
- Public API unchanged.

## C.2 Failing-without-fix regression test
- New: `tests/test_tolerance_thread_safety.py`.
- `test_tolerance_context_is_thread_isolated` uses `threading.Barrier` to coordinate 4 threads; without ContextVar, all 4 see the LAST writer's override (race). With ContextVar, each sees its own value.
- Plus 2 additional tests: exception-safe pop, single-thread nesting LIFO order.

## C.3 TM-E1: CUDA_HOME allowlist
- File: `src/gpucheck/sanitizers/race.py`.
- New `_CUDA_HOME_ALLOWLIST = ("/usr/local/cuda", "/opt/nvidia/cuda", "/opt/cuda")`.
- New `_is_allowed_cuda_home(real_path)` does exact-prefix-with-separator match.
- `_find_compute_sanitizer` now resolves symlinks via `os.path.realpath` and rejects paths outside the allowlist with a `RuntimeWarning`.

## C.4 TM-E1 tests
- New: `tests/test_race_cuda_home_allowlist.py` (6 tests covering canonical paths, lookalikes, missing PATH, allowlisted binary, symlink attack).

## Verification: 126 passed, 3 skipped (+9 net new). ruff & mypy clean.
