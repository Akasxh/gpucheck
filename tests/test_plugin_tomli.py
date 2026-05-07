"""Plugin TOML loading dependency contract (review BLOCKER S1).

Python 3.10 lacks ``tomllib`` in the stdlib; ``gpucheck.plugin``
falls back to ``tomli``. This test pins the dependency:

- On 3.10, ``tomli`` MUST be importable (i.e. declared in
  ``[project.dependencies]`` with the right marker so a fresh
  ``pip install gpucheck`` includes it).
- On 3.11+, the test skips because ``tomllib`` is the stdlib path.

Without this guard, the catch-all in ``_load_pyproject_config``
silently swallows the ImportError and the user's
``[tool.gpucheck.tolerances]`` and ``[tool.gpucheck.mps.xfail]``
overlays no-op without warning — confidence-in-test failure.
"""

from __future__ import annotations

import importlib.util
import sys

import pytest


@pytest.mark.skipif(
    sys.version_info >= (3, 11),
    reason="tomllib is in stdlib on Python 3.11+; tomli is only the 3.10 fallback",
)
def test_tomli_is_importable_on_python_310() -> None:
    """``tomli`` must be a declared dependency on Python 3.10.

    The plugin's ``_load_pyproject_config`` does
    ``import tomli as tomllib`` when the stdlib ``tomllib`` is absent.
    If ``tomli`` isn't installed, the user's pyproject overlays
    silently no-op (review BLOCKER S1).
    """
    spec = importlib.util.find_spec("tomli")
    assert spec is not None, (
        "tomli is required on Python 3.10 (the plugin's TOML fallback "
        "path); declare it in pyproject.toml's [project.dependencies] "
        "with `python_version < '3.11'` marker."
    )


def test_plugin_loader_reads_pyproject_overlay(tmp_path: object) -> None:
    """End-to-end smoke test: a synthetic pyproject.toml with a
    ``[tool.gpucheck.tolerances]`` block must apply on the current
    Python version. This exercises whichever TOML backend the
    interpreter resolves (stdlib on 3.11+, tomli on 3.10).
    """
    from pathlib import Path

    from gpucheck.assertions.tolerances import (
        compute_tolerance,
        reset_config_tolerances,
    )
    from gpucheck.plugin import _load_pyproject_config

    rootpath = Path(str(tmp_path))
    (rootpath / "pyproject.toml").write_text(
        """
[tool.gpucheck.tolerances]
float16 = {atol = 9.99e-3, rtol = 9.99e-3}
""",
        encoding="utf-8",
    )
    try:
        _load_pyproject_config(rootpath)
        atol, rtol = compute_tolerance("float16")
        assert atol == pytest.approx(9.99e-3)
        assert rtol == pytest.approx(9.99e-3)
    finally:
        reset_config_tolerances()
