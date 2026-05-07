"""TM-E1 mitigation: CUDA_HOME / CUDA_PATH allowlist (Track C)."""

from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING

from gpucheck.sanitizers.race import (
    _CUDA_HOME_ALLOWLIST,
    _find_compute_sanitizer,
    _is_allowed_cuda_home,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_is_allowed_cuda_home_canonical_paths() -> None:
    for prefix in _CUDA_HOME_ALLOWLIST:
        assert _is_allowed_cuda_home(prefix) is True
        assert _is_allowed_cuda_home(prefix + "/bin") is True
        assert _is_allowed_cuda_home(prefix + "/12.2") is True


def test_is_allowed_cuda_home_rejects_lookalike_paths() -> None:
    # Trailing characters must NOT match the prefix.
    assert _is_allowed_cuda_home("/usr/local/cuda-evil") is False
    assert _is_allowed_cuda_home("/opt/nvidia/cudawat") is False
    assert _is_allowed_cuda_home("/opt") is False
    assert _is_allowed_cuda_home("/tmp/attacker") is False


def test_find_compute_sanitizer_returns_none_when_path_lookup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Ensure shutil.which fails (point PATH at an empty dir).
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)

    assert _find_compute_sanitizer() is None


def test_find_compute_sanitizer_rejects_outside_allowlist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CUDA_HOME outside the allowlist must be ignored AND emit a warning."""
    monkeypatch.setenv("PATH", str(tmp_path))  # neutralize shutil.which path

    fake_cuda = tmp_path / "fake_cuda"
    (fake_cuda / "bin").mkdir(parents=True)
    binary = fake_cuda / "bin" / "compute-sanitizer"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)

    monkeypatch.setenv("CUDA_HOME", str(fake_cuda))
    monkeypatch.delenv("CUDA_PATH", raising=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _find_compute_sanitizer()

    assert result is None, (
        "fake CUDA_HOME outside allowlist must NOT yield a sanitizer path"
    )
    assert any("allowlist" in str(w.message) for w in caught), (
        "expected a RuntimeWarning explaining the allowlist rejection"
    )


def test_find_compute_sanitizer_accepts_inside_allowlist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If CUDA_HOME points inside the allowlist AND a binary exists there, return it.

    We can't actually create files at /usr/local/cuda in CI; instead, we
    monkeypatch _CUDA_HOME_ALLOWLIST to include tmp_path and verify the
    code path returns the binary when the rest of the conditions hold.
    """
    monkeypatch.setenv("PATH", str(tmp_path / "no_path_here"))

    real_cuda = tmp_path / "real_cuda"
    (real_cuda / "bin").mkdir(parents=True)
    binary = real_cuda / "bin" / "compute-sanitizer"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)

    monkeypatch.setattr(
        "gpucheck.sanitizers.race._CUDA_HOME_ALLOWLIST",
        (str(real_cuda.resolve()),),
    )
    monkeypatch.setenv("CUDA_HOME", str(real_cuda))
    monkeypatch.delenv("CUDA_PATH", raising=False)

    result = _find_compute_sanitizer()
    assert result == os.path.join(str(real_cuda.resolve()), "bin", "compute-sanitizer")


def test_find_compute_sanitizer_resolves_symlink_before_allowlist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A symlink pointing OUTSIDE the allowlist must be rejected.

    This guards against the obvious attack:
        ln -s /tmp/attacker /usr/local/cuda
    """
    monkeypatch.setenv("PATH", str(tmp_path / "nope"))

    attacker = tmp_path / "attacker"
    (attacker / "bin").mkdir(parents=True)
    (attacker / "bin" / "compute-sanitizer").write_text("#!/bin/sh\nexit 0\n")
    (attacker / "bin" / "compute-sanitizer").chmod(0o755)

    symlink_at_canonical = tmp_path / "symlinked_cuda"
    symlink_at_canonical.symlink_to(attacker)

    monkeypatch.setenv("CUDA_HOME", str(symlink_at_canonical))
    monkeypatch.delenv("CUDA_PATH", raising=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _find_compute_sanitizer()
    assert result is None
    assert any("allowlist" in str(w.message) for w in caught)
