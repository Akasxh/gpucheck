# DIFF_LOG.md — Phase A executor (close.py collision group)

Owner: engineering-executor
Branch: release/v1.0
Collision group: T-01 → T-02 → T-10 (serialized per skeptic C2)

## Iteration 1 — Task T-01: Top-level torch import defeats lazy-import contract
- **File**: `src/gpucheck/assertions/close.py`
- **Change**: Replaced module-level `try: import torch as _torch / _has_torch` block with a lazy `_torch_mod()` cached helper; rewrote `device_type` detection and the GPU fast-path to bind `_torch = _torch_mod()` at the top of `assert_close`, then guard with `_torch is not None`.
- **Reason**: CLAUDE.md Lazy-imports decision-record requires torch to never be imported at collection time. The previous `try: import torch as _torch` ran at module load, breaking the contract. Caching via a sentinel makes the lookup happen at most once per process.
- **Acceptance criterion addressed**: IMPLEMENTATION_PLAN_v1.1 T-01 — `python -c "import gpucheck.assertions.close; import sys; assert 'torch' not in sys.modules"` now passes; existing `tests/test_assertions.py` still green (32 passed).
