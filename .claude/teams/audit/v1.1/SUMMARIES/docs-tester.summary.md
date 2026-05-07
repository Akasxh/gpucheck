# docs-tester summary (W1)

**47 blocks tested:** 24 README, 12 MIGRATION, 9 CONTRIBUTING, 2 CLAUDE, 0 CHANGELOG runnable, 0 __init__.py doctest blocks.

**Pass / Fail / Skip-Informational:** ~28 / **9 broken** / ~10 informational

**9 Failures:**
- README R-B2 — wrong fence lang
- R-B8 / R-B12 / R-B13 — hard-coded CUDA on a CUDA-less host
- R-B16 — wrong numeric output `+12.0% / d=4.21` (actual `+11.7% / d=7.48`) in regression-detector example
- R-B21 / T-B5 — `pytest tests/gpu_integration/` claims auto-skip without GPU; on MPS hosts this **FAILS 52 tests**
- MIGRATION M-B1 — return type + comment placement wrong
- M-B7 — xfail registry empty outside pytest (gotcha not documented)
- M-B8 — `fuzz_strides` / `fuzz_strides_for_category` wrong signature in example
- M-B9 — `assert_deterministic` wrong kwargs

**Top-3 docs to fix in v1.1:**
1. **MIGRATION.md** — three broken signatures (available_backends, fuzz_strides, assert_deterministic). Every example must work as printed.
2. **README §"Step by step usage guide"** — wrong fence lang at L66-76, wrong numeric output at L282-291, silent CUDA assumption in §4/§6/§7 despite advertising MPS as first-class.
3. **README/CONTRIBUTING "Running tests"** — `pytest tests/gpu_integration/` auto-skip claim is FALSE on MPS hosts (52 hard failures).

**Note:** `src/gpucheck/__init__.py` has zero `Examples:` docstrings. Decorator modules have them but were out of scope.
