# MATRIX_REPORT — gpucheck v1.0 cross-PyTorch matrix

**Date:** 2026-05-01
**Host:** Apple Silicon (MPS available)

## Versions tested

| version | install path | mps_available | python | wheel source |
|---|---|---|---|---|
| 2.11.0 | project venv (`.venv/`) | ✅ | 3.12 | `torch-2.11.0-cp312-cp312-macosx_11_0_arm64.whl` |
| 2.10.0 | `~/.gpucheck-pyt-2.10.0/` | ✅ | 3.14 | `torch-2.10.0-2-cp312-none-macosx_11_0_arm64.whl` |

## Versions skipped (no macOS arm64 wheel on PyPI)

| version | reason |
|---|---|
| 2.6.0 | wheel is `manylinux_2_28_aarch64` (Linux ARM only) |
| 2.7.0 | same |
| 2.7.1 | same |
| 2.8.0 | same |
| 2.9.0 | same |
| 2.9.1 | same |

PyPI release inventory check at session-time: `python -c "urllib.request.urlopen('https://pypi.org/pypi/torch/json')..."` confirmed only `2.10.0` and `2.11.0` ship `macosx_11_0_arm64` wheels for cp312. Older versions on macOS arm64 require building from source. **Honest scope: matrix is constrained to 2.10 + 2.11 on this host.** v2's "≥6 versions" target is unachievable for macOS arm64 wheels in 2026-05-01.

## Results

### torch==2.11.0 (project baseline)
```
$ uv run pytest -q
224 passed, 1 skipped, 10 warnings in 0.34s
```
ruff: PASS · mypy strict: PASS

### torch==2.10.0
```
$ ~/.gpucheck-pyt-2.10.0/bin/python -m pytest tests/ -q --tb=line
4 failed, 220 passed, 1 skipped, 10 warnings in 2.69s
```

**4 NEW failures on 2.10 that pass on 2.11:**
- `tests/test_assert_close_mps.py::test_assert_close_mps_passes_with_mps_overlay_for_float16`
- `tests/test_assertions.py::TestMixedPrecisionDtype::test_fp16_fp32_uses_fp16_tolerance_order1`
- `tests/test_assertions.py::TestMixedPrecisionDtype::test_fp32_fp16_uses_fp16_tolerance_order2`
- `tests/test_assertions.py::TestMixedPrecisionDtype::test_both_orders_produce_same_result`

All 4 failures are in **mixed-precision tolerance handling** — fp16/fp32 promotion path. This is real cross-version evidence: between torch 2.10 and 2.11, either gpucheck's tolerance overlay changed in a way that 2.10 doesn't accept, OR torch's tensor type promotion changed in a way that affects gpucheck's per-dtype tolerance lookup.

### Implication for gpucheck pyproject

Current `[mps]` extra pins `torch>=2.6` (per Track-A's CHARTER). Reality on macOS arm64: minimum installable is `2.10` (PyPI wheel availability). The pin should be tightened to `torch>=2.10` for the `[mps]` extra on macOS, with a note that Linux arm64 supports 2.6+. Or the pin stays `>=2.6` and the macOS user gets a "no matching wheel" error from pip — which is acceptable but unfriendly.

### Implication for v1.0.0rc1 release

**ADVISORY — not a release blocker, but warrants a CHANGELOG note:**
- gpucheck v1.0.0rc1 fully passes only on torch 2.11
- On torch 2.10, 4 mixed-precision tests fail
- Root cause TBD — likely a torch internal type-promotion change between 2.10 and 2.11
- Recommendation: ship with `python_requires` + a stronger pin, OR investigate the failures and either fix gpucheck to handle both versions or document the constraint

## Provenance

- Project venv (.venv/) — torch installed via `uv pip install -e ".[dev,torch]"`
- 2.10 venv — `~/.gpucheck-pyt-2.10.0/`, created via `python3 -m venv`, installed via `pip install torch==2.10.0 pytest hypothesis -e ~/Code/gpucheck`
- Test runner: `pytest -q --tb=line` for both
- Both runs captured live on this Apple Silicon Mac, MPS detected as available, no GPU detection backend warning is informational only
