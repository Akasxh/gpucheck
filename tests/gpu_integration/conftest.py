"""Shared fixtures and skip-gating for GPU integration tests.

These tests are designed to run on real GPU hardware (CUDA or, opt-in, MPS).
On a host with neither, the entire suite is skipped at collection time so the
README claim ``pytest tests/gpu_integration/`` auto-skips without GPU stays
honest. See docs-tester finding R-B21 / T-B5.
"""

from __future__ import annotations

from typing import Any

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add ``--mps-integration`` to opt into running gpu_integration on MPS."""
    parser.addoption(
        "--mps-integration",
        action="store_true",
        default=False,
        help=(
            "Opt in to running tests/gpu_integration/ on Apple Silicon MPS. "
            "Without this flag the suite is skipped on non-CUDA hosts."
        ),
    )


def _cuda_available() -> bool:
    """Return True iff torch reports a usable CUDA device."""
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except (RuntimeError, AssertionError):
        return False


def _mps_available() -> bool:
    """Return True iff torch reports a usable MPS device."""
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except (RuntimeError, AssertionError):
        return False


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip the gpu_integration suite when no usable GPU mode is detected.

    The README claims ``pytest tests/gpu_integration/`` auto-skips without
    a GPU. Previously, on MPS hosts those tests collected and then failed
    on CUDA-specific calls. We now skip at collection time unless either:

    - CUDA is available (preferred path), or
    - MPS is available *and* the user passed ``--mps-integration``.
    """
    if _cuda_available():
        return
    if config.getoption("--mps-integration") and _mps_available():
        return

    if _mps_available():
        reason = "MPS detected but --mps-integration not set; pass it to opt in"
    else:
        reason = "no CUDA GPU available (and no --mps-integration flag set)"

    skip_marker = pytest.mark.skip(reason=reason)
    for item in items:
        item.add_marker(skip_marker)


@pytest.fixture()
def results() -> dict[str, Any]:
    """Mutable dict for benchmark tests to store their results."""
    return {}
