# Executor — close.py collision group (T-01, T-02, T-10)

Branch: release/v1.0
Owner: engineering-executor
Mutmut baseline at start: 169 killed mutants
Pre-existing test count at start: 225 collected (224 passed + 1 skipped)

## Task T-01 (Lazy torch import)

### What I did
Replaced the module-level `try: import torch as _torch / _has_torch` block in
`src/gpucheck/assertions/close.py` with a lazy `_torch_mod()` helper that
caches the imported module (or `None`) using a sentinel. Both call sites
inside `assert_close` (device-type detection and the GPU fast-path) now bind
`_torch = _torch_mod()` once at the top of the function and guard with
`_torch is not None`.

### Files modified
- `src/gpucheck/assertions/close.py`: lazy-import refactor; `_has_torch` removed; `_torch_mod()` is the single point of access.

### Files created
- `DIFF_LOG.md` (this iteration logs the T-01 change).
- `.claude/teams/audit/v1.1/EVIDENCE/executor.md` (this file).

### Design decisions made during implementation
- Used a sentinel (`_TORCH_UNRESOLVED = object()`) instead of `None` for the
  cache initial state, so a process where torch is genuinely absent still
  short-circuits after the first probe (caches `None` permanently).
- Kept the inner `import torch` inside the `__cuda_array_interface__`
  fallback as-is rather than routing it through `_torch_mod()`. That branch
  needed torch *and* dlpack to be present and was already lazy by virtue of
  living inside a function body — re-routing through the helper would have
  changed behaviour (it would no longer except `ImportError` locally).

### Potential blast radius
- If `gpucheck.assertions.close` is imported on a torch-less host, the
  module load path no longer raises or warns. Behaviour matches the
  previous code (which set `_has_torch = False` silently).
- The `_torch_cached` global is process-wide. If a test fixture
  monkey-patches `sys.modules['torch']` after `_torch_mod()` has run once,
  the cached value will be stale. Existing tests do not do this, but the
  verifier should confirm.

## Task T-02 (`.contiguous()` on slow path)

### What I did
Added `.contiguous()` to both torch.Tensor branches of `_to_numpy` (the
primary `hasattr(tensor, "detach")` branch and the dlpack fallback in the
`__cuda_array_interface__` block). Wrote a new parametrized test module
`tests/test_assert_close_contiguous.py` covering three stride patterns:
slice (`[:, ::2]`), transpose (`.t()`), and broadcast (`.expand`).

### Files modified
- `src/gpucheck/assertions/close.py`: two `.cpu().contiguous()` insertions; PM-4 citation comment.

### Files created
- `tests/test_assert_close_contiguous.py`: 6 parametrized cases (3 stride patterns × 2 entry points).

### Design decisions made during implementation
- Applied `.contiguous()` to the dlpack fallback as well even though only
  the primary branch is in the strict T-02 scope. The same RuntimeError
  surface exists in both code paths and the cost is negligible. Noted
  here as an opportunistic widen so the reviewer can flag if undesired.
- Used `pytest.importorskip("torch")` rather than the existing
  `_has_torch`-style guard so the module skips cleanly on torch-less
  hosts (matches the lazy-import discipline from T-01).

### Potential blast radius
- `.contiguous()` allocates a new tensor when the input is non-contiguous.
  For very large stride-fuzzed tensors this could double peak memory in
  the slow path. Existing CUDA fast-path (which bypasses `_to_numpy`)
  already handles same-shape tensors without copying, so the regression is
  bounded to mismatched / failing comparisons.
