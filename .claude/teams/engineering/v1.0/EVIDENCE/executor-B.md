# executor-B.md

**Specialist**: engineering-executor (Track B)
**Branch**: `feat/track-b-strides` @ `4ede763`

## Tasks (PLAN.md §Track-B)

### B.1 fuzz_strides + fuzz_strides_for_category
- New: `src/gpucheck/fuzzing/strides.py`. 7 categories: row_major, column_major, broadcast, transpose, slice, non_contig, gather. Each has its own builder function with documented purpose and stride contract.

### B.2 StrideStrategy (Hypothesis)
- Same file. `__new__`-as-factory pattern matching ShapeStrategy.

### B.3 parametrize_gpu wired
- File: `src/gpucheck/decorators/parametrize.py`. New `stride_categories=` kwarg expands the cartesian to include `stride_category` parameter. Skip predicate accepts both 3-arg and 4-arg signatures.

### B.4 Tests
- `tests/test_fuzz_strides.py` (17 tests) — each category's stride contract (e.g., broadcast has stride 0 on the last dim).
- `tests/test_fuzz_strides_hypothesis.py` (3 tests).
- `tests/test_parametrize_gpu_strides.py` (3 tests).

## Verification: 139 passed, 4 skipped (+22 net new). ruff & mypy clean.
