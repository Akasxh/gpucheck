# DIFF_LOG.md — Phase A executor (close.py collision group)

Owner: engineering-executor
Branch: release/v1.0
Collision group: T-01 → T-02 → T-10 (serialized per skeptic C2)

## Iteration 1 — Task T-01: Top-level torch import defeats lazy-import contract
- **File**: `src/gpucheck/assertions/close.py`
- **Change**: Replaced module-level `try: import torch as _torch / _has_torch` block with a lazy `_torch_mod()` cached helper; rewrote `device_type` detection and the GPU fast-path to bind `_torch = _torch_mod()` at the top of `assert_close`, then guard with `_torch is not None`.
- **Reason**: CLAUDE.md Lazy-imports decision-record requires torch to never be imported at collection time. The previous `try: import torch as _torch` ran at module load, breaking the contract. Caching via a sentinel makes the lookup happen at most once per process.
- **Acceptance criterion addressed**: IMPLEMENTATION_PLAN_v1.1 T-01 — `python -c "import gpucheck.assertions.close; import sys; assert 'torch' not in sys.modules"` now passes; existing `tests/test_assertions.py` still green (32 passed).

## Iteration 2 — Task T-02: Add `.contiguous()` to `_to_numpy` slow path
- **File**: `src/gpucheck/assertions/close.py`
- **Change**: Inserted `.contiguous()` between `.cpu()` and `.numpy()` on both torch.Tensor branches of `_to_numpy` (the primary `hasattr(tensor, "detach")` branch and the dlpack fallback inside the `__cuda_array_interface__` block). Added a comment citing PM-4.
- **Reason**: torch <2.1 raises `RuntimeError: input array is not C-contiguous` when `.numpy()` is called on stride-fuzzed / sliced / transposed tensors. Preventive even on newer torch — known to fire on older PyTorch.
- **Acceptance criterion addressed**: security-postmerge PM-4 / planner T-02 — new `tests/test_assert_close_contiguous.py` exercises 3 stride patterns (slice, transpose, broadcast) × 2 entry points (`_to_numpy` directly and `assert_close` end-to-end); all 6 cases pass.

- **File**: `tests/test_assert_close_contiguous.py` (created)
- **Change**: New parametrized test module with 6 cases pinning the contiguity fix.
- **Reason**: T-02 acceptance demands a regression test that would fail without the `.contiguous()` insertion.
- **Acceptance criterion addressed**: planner T-02.
