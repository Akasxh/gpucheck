"""Shared fixtures for GPU integration tests."""

from __future__ import annotations

from typing import Any

import pytest


@pytest.fixture()
def results() -> dict[str, Any]:
    """Mutable dict for benchmark tests to store their results."""
    return {}
