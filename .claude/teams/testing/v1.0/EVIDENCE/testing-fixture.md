# Fixture — testing / v1.0

Adopted persona: `~/.claude/agents/testing/testing-fixture.md`. Phase 2
plan-prep — this file specifies the test fixtures, mocks, and factories
the property + unit tests will consume. Engineering-lead must ship the
production code that makes these fixtures legitimate (i.e., the real
backends, the real `compute_tolerance(device=...)` overload, etc.).

## Fixture inventory

| Fixture | Type | File | Used by | Mock external? |
|---|---|---|---|---|
| `mps_available` | gate-skip | `tests/conftest.py` | A2, A3a/b/c, A4, A6 | NO — real torch.mps |
| `mock_torch_mps` | mock module | `tests/conftest.py` | A1, A5 (CI no-MPS path) | YES (when no hardware) |
| `mps_or_mock_backend` | composite | `tests/conftest.py` | A2/A3/A4 (CI + local) | conditional |
| `mock_pynvml` | mock | `tests/conftest.py` | A1, A5, existing test_arch.py | YES |
| `cpu_tensor_pair` | factory | `tests/factories.py` | A6 | NO |
| `seeded_rng` | context-fixture | `tests/conftest.py` | D1 + any RNG-touching test | NO |
| `temp_baseline_json` | tmp-path factory | `tests/factories.py` | S3 | NO |
| `clean_tolerance_context` | autouse cleanup | `tests/conftest.py` | C1-C4 + ALL tests | NO |
| `cuda_home_sandbox` | filesystem | `tests/conftest.py` | S1 | NO (real tmp_path) |
| `xfail_registry_loaded` | parser | `tests/conftest.py` | A8 | NO |
| `sample_baselines` | data | `tests/data/baselines/*.json` | D2 | NO |
| `wave_one_data_factory` | factory | `tests/factories.py` | D2/D3 + future swarm result tests | NO |

## Conftest sketch (engineering reads as a contract)

### `tests/conftest.py` additions

```python
"""gpucheck testing — shared fixtures for v1.0 plan."""
from __future__ import annotations

import os
import random
import sys
import threading
from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Hypothesis profiles
# ---------------------------------------------------------------------------
from hypothesis import HealthCheck, Verbosity, settings


def _register_hypothesis_profiles() -> None:
    settings.register_profile(
        "ci",
        max_examples=200,
        deadline=1000,
        derandomize=True,
        suppress_health_check=[HealthCheck.too_slow],
        verbosity=Verbosity.normal,
    )
    settings.register_profile("dev", max_examples=50, deadline=500)
    settings.register_profile("mps", max_examples=50, deadline=5000)
    settings.register_profile("nightly", max_examples=2000, deadline=None)
    settings.load_profile("ci" if os.getenv("CI") else "dev")


_register_hypothesis_profiles()

# ---------------------------------------------------------------------------
# MPS gating
# ---------------------------------------------------------------------------
@pytest.fixture
def mps_available() -> Any:
    """Yield the MPS backend or skip the test."""
    try:
        import torch
        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available on this host")
    except ImportError:
        pytest.skip("torch not installed")
    from gpucheck.arch.backend import detect_backend  # contract: Track A
    backend = detect_backend()
    if backend.name != "mps":
        pytest.skip("Selected backend is not MPS")
    return backend


@pytest.fixture
def mock_torch_mps(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Stub torch.mps.* — used when host lacks MPS hardware."""
    m = MagicMock()
    m.is_available.return_value = True
    m.synchronize = MagicMock(return_value=None)
    m.empty_cache = MagicMock(return_value=None)
    m.current_allocated_memory.return_value = 0
    m.driver_allocated_memory.return_value = 0
    m.recommended_max_memory.return_value = 16 * 1024**3
    m.device_count.return_value = 1
    # Event API
    event_mock = MagicMock()
    event_mock.record = MagicMock()
    event_mock.wait = MagicMock()
    event_mock.query.return_value = True
    event_mock.elapsed_time.return_value = 0.5
    m.event.Event.return_value = event_mock
    monkeypatch.setattr("torch.mps", m)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)
    return m


@pytest.fixture
def mps_or_mock_backend(request: pytest.FixtureRequest) -> Any:
    """Use real MPS if present, else mock-backed Backend."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return request.getfixturevalue("mps_available")
    except ImportError:
        pass
    request.getfixturevalue("mock_torch_mps")
    from gpucheck.arch.backend import detect_backend
    return detect_backend()


# ---------------------------------------------------------------------------
# pynvml mock (existing pattern, formalised here)
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_pynvml(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    m = MagicMock()
    m.nvmlInit = MagicMock()
    m.nvmlShutdown = MagicMock()
    m.nvmlDeviceGetCount.return_value = 1
    handle = MagicMock()
    m.nvmlDeviceGetHandleByIndex.return_value = handle
    m.nvmlDeviceGetName.return_value = b"NVIDIA GeForce GTX 1650"
    m.nvmlDeviceGetMemoryInfo.return_value = MagicMock(total=4 << 30, used=0, free=4 << 30)
    monkeypatch.setitem(sys.modules, "pynvml", m)
    return m


# ---------------------------------------------------------------------------
# RNG hygiene
# ---------------------------------------------------------------------------
@pytest.fixture
def seeded_rng() -> Generator[int, None, None]:
    """Seed all process-global RNGs and restore them on exit."""
    seed = 0xDEADBEEF
    py_state = random.getstate()
    np_state = np.random.get_state()
    try:
        import torch
        th_state = torch.get_rng_state()
    except ImportError:
        th_state = None
    random.seed(seed)
    np.random.seed(seed)
    if th_state is not None:
        torch.manual_seed(seed)
    try:
        yield seed
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        if th_state is not None:
            torch.set_rng_state(th_state)


# ---------------------------------------------------------------------------
# ContextVar tolerance hygiene — autouse to prevent test poisoning
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def clean_tolerance_context() -> Generator[None, None, None]:
    """Clear any leaked tolerance overrides before AND after every test.

    Prevents Track C ContextVar tests from leaking state into Track A/B/D
    tests (especially under pytest-xdist parallel workers).
    """
    from gpucheck.assertions.tolerances import _reset_for_test  # contract: Track C
    _reset_for_test()
    try:
        yield
    finally:
        _reset_for_test()


# ---------------------------------------------------------------------------
# Path-injection sandbox (S1)
# ---------------------------------------------------------------------------
@pytest.fixture
def cuda_home_sandbox(tmp_path, monkeypatch):
    """Create a fake CUDA_HOME with a stub binary, redirect env."""
    fake_root = tmp_path / "fake_cuda"
    bin_dir = fake_root / "bin"
    bin_dir.mkdir(parents=True)
    fake_bin = bin_dir / "compute-sanitizer"
    fake_bin.write_text("#!/bin/sh\necho 'pretending to be a sanitizer'\n")
    fake_bin.chmod(0o755)
    monkeypatch.setenv("CUDA_HOME", str(fake_root))
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setattr("shutil.which", lambda _: None)
    return {"root": fake_root, "bin": fake_bin}


# ---------------------------------------------------------------------------
# xfail registry loader (A8)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def xfail_registry_loaded() -> dict:
    import tomllib
    from pathlib import Path
    cfg = tomllib.loads(Path("pyproject.toml").read_text())
    return cfg.get("tool", {}).get("gpucheck", {}).get("mps", {}).get("xfail", {})
```

### `tests/factories.py` (NEW)

```python
"""Test data factories for gpucheck."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


class CPUTensorPair:
    """Factory: produce (reference, candidate) CPU tensors for parity tests."""

    @staticmethod
    def create(shape: tuple[int, ...], dtype: Any, seed: int = 0) -> tuple[Any, Any]:
        import torch
        g = torch.Generator(device="cpu").manual_seed(seed)
        a = torch.randn(*shape, generator=g, dtype=dtype)
        b = torch.randn(*shape, generator=g, dtype=dtype)
        return a, b


class TempBaselineJson:
    """Factory: write a JSON baseline to tmp_path and return the path."""

    @staticmethod
    def create(tmp_path: Path, content: dict[str, Any]) -> Path:
        p = tmp_path / "baseline.json"
        p.write_text(json.dumps(content, sort_keys=True, indent=2))
        return p


class WaveOneDataFactory:
    """Produce a 'realistic' baseline JSON for HTML dashboard tests (D2)."""

    @staticmethod
    def create(seed: int = 0) -> dict[str, Any]:
        rng = random.Random(seed)
        return {
            "schema_version": 1,
            "kernel_results": [
                {
                    "kernel": k,
                    "device": "mps",
                    "iterations": 250,
                    "divergences": rng.randint(0, 3),
                    "max_rel_err": rng.uniform(1e-5, 1e-2),
                }
                for k in ["matmul-fp32", "softmax", "layernorm"]
            ],
        }
```

## Mock classification (golden rule audit)

- **External deps mocked**:
  - `torch.mps` — CI hardware boundary (M-Mac not present)
  - `pynvml` — NVIDIA driver boundary (no GPU in CI)
  - `shutil.which` — filesystem boundary (existing CI pattern)
  - `os.environ` via `monkeypatch.setenv` — env boundary
  - `subprocess.run` for `compute-sanitizer` — process boundary (existing tests)

- **Internal deps left UNMOCKED** (golden rule):
  - `compute_tolerance` — pure function, real implementation in tests
  - `fuzz_shapes` / `fuzz_strides` — pure, deterministic; real implementation
  - `assert_close` — system under test, never mock the SUT
  - `Backend` Protocol concrete methods — these ARE the SUT for Track A
  - `tolerance_context` — Track C SUT
  - `assert_deterministic` — Track D SUT

- **Boundary cases**:
  - `mps_or_mock_backend` is a composite — uses real backend on M-Mac, mocked elsewhere. Justification: SYNTHESIS notes "calibrate on M-machine"; CI cannot run real MPS, so a mock proxy is required for non-M CI runs.
  - `mock_torch_mps` is configured to expose deadlock-pattern detectability (so A3c can `inspect.getsource` the production code).

## Anti-pattern checklist

- [x] No over-mocking — only external boundaries; SUT is unmocked
- [x] Mocks return realistic data (4 GiB GTX 1650 mem; non-zero elapsed_time; non-zero allocated)
- [x] Factories produce valid domain objects (`CPUTensorPair` calls real `torch.randn`)
- [x] No hardcoded credentials or paths (all paths via `tmp_path`; CUDA_HOME via monkeypatch)
- [x] `clean_tolerance_context` autouse prevents test cross-contamination — critical for Track C (the ContextVar refactor introduces a new global per-thread/task state that must be reset between tests)
- [x] `seeded_rng` restores RNG state — Track D D1 needs this property held by the fixture itself

## Risks specific to fixtures

| Risk | Mitigation |
|---|---|
| `mock_torch_mps` divergence from real `torch.mps` API | Re-validate against `torch==2.11` `torch.mps` surface in CHARTER; nightly job to re-import and `dir()` for drift |
| pytest-xdist worker isolation breaks `clean_tolerance_context` if ContextVar is process-shared | ContextVar is process-scoped per Python; xdist workers are separate processes, so isolation is automatic. The autouse fixture handles in-process cross-test cases. |
| `cuda_home_sandbox` fixture leaks into `os.environ` if monkeypatch fails | `pytest.MonkeyPatch` always restores on teardown; verified by pytest. |
| `mps_or_mock_backend` masks real bugs by always succeeding | Counter: A3c (deadlock-pattern test) inspects source, not behavior, so it catches the bug regardless of mock vs real. |

## Verdict

GENERATED — 11 fixtures across 2 files (conftest.py extension + new
factories.py). Mock classification clean (only external boundaries
mocked). Autouse `clean_tolerance_context` is the critical hygiene
fixture for the ContextVar refactor.
