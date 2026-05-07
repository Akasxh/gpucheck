# MATRIX run: torch==2.10.0

- date: 2026-05-01T09:35:45Z
- venv: /Users/cero/.gpucheck-pyt-2.10.0
- python: Python 3.14.4
- torch: 2.10.0
- mps: True

## Test output
```
tests/test_arch.py::TestBlackwellNamingConsistency::test_check_compatibility_blackwell_resolves
  /Users/cero/Code/gpucheck/tests/test_arch.py:338: UserWarning: Kernel targets SM100 but running on SM90 (Hopper). Forward compatibility is not guaranteed.
    issues = check_compatibility("Blackwell", mock_gpu)

tests/test_arch.py::TestBlackwellNamingConsistency::test_check_compatibility_blackwell_dc_resolves
  /Users/cero/Code/gpucheck/tests/test_arch.py:351: UserWarning: Blackwell-targeted kernels using SM100 features will not run on Hopper.
    issues = check_compatibility("Blackwell-DC", mock_gpu)

tests/test_arch.py::TestBlackwellNamingConsistency::test_check_compatibility_blackwell_dc_resolves
  /Users/cero/Code/gpucheck/tests/test_arch.py:351: UserWarning: Kernel targets SM100 but running on SM90 (Hopper). Forward compatibility is not guaranteed.
    issues = check_compatibility("Blackwell-DC", mock_gpu)

tests/test_fuzzing.py::TestShapeStrategyShrinks::test_strategy_produces_valid_shapes
  /Users/cero/Code/gpucheck/tests/test_fuzzing.py:124: NonInteractiveExampleWarning: The `.example()` method is good for exploring strategies, but should only be used interactively.  We recommend using `@given` for tests - it performs better, saves and replays failures to avoid flakiness, and reports minimal examples. (strategy: tuples(one_of(sampled_from([0, 1, 7, 13, 31, 33, 63]), integers(min_value=1, max_value=64)), one_of(sampled_from([0, 1, 7, 13, 31, 33, 63]), integers(min_value=1, max_value=64))))
    example = strat.example()

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=================================== GPU Info ===================================
  No GPU detected
=========================== short test summary info ============================
FAILED tests/test_assert_close_mps.py::test_assert_close_mps_passes_with_mps_overlay_for_float16
FAILED tests/test_assertions.py::TestMixedPrecisionDtype::test_fp16_fp32_uses_fp16_tolerance_order1
FAILED tests/test_assertions.py::TestMixedPrecisionDtype::test_fp32_fp16_uses_fp16_tolerance_order2
FAILED tests/test_assertions.py::TestMixedPrecisionDtype::test_both_orders_produce_same_result
4 failed, 220 passed, 1 skipped, 10 warnings in 2.69s
```
