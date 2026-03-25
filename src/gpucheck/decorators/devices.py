"""Parametrize tests across GPU devices."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable


def _detect_cuda_devices() -> list[str]:
    """Return list of available CUDA device strings."""
    try:
        import torch

        if not torch.cuda.is_available():
            return []
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    except ImportError:
        return []


def _is_device_available(device: str) -> bool:
    """Check whether a device string is currently usable."""
    try:
        import torch

        if device == "cpu":
            return True
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                return False
            if ":" in device:
                idx = int(device.split(":")[1])
                return idx < torch.cuda.device_count()
            return True
        # Unknown device type — let torch figure it out
        torch.device(device)
        return True
    except (ImportError, RuntimeError, ValueError):
        return False


def _device_id(d: str) -> str:
    """Clean test ID: 'cuda:0' -> 'cuda0'."""
    return d.replace(":", "")


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------

def devices(*device_args: str) -> Callable[..., Any]:
    """Parametrize a test across GPU devices.

    If no arguments are given, auto-detects all available CUDA devices
    (falls back to ``["cuda:0"]`` if detection finds nothing but CUDA
    appears importable).

    Pass ``"all"`` to expand to every visible CUDA device.

    Devices that are not available at collection time get
    ``pytest.mark.skip`` so the test is reported but not run.

    Examples::

        @devices("cuda:0", "cuda:1")
        def test_copy(device): ...

        @devices()          # auto-detect
        def test_kernel(device): ...

        @devices("all")
        def test_broadcast(device): ...
    """
    resolved: list[str] = []

    if not device_args or device_args == ("all",):
        detected = _detect_cuda_devices()
        resolved = detected if detected else ["cuda:0"]
    else:
        for d in device_args:
            if d == "all":
                resolved.extend(_detect_cuda_devices() or ["cuda:0"])
            else:
                resolved.append(d)

    # Build pytest.param entries, skipping unavailable devices
    params: list[Any] = []
    for dev in resolved:
        if _is_device_available(dev):
            params.append(pytest.param(dev, id=_device_id(dev)))
        else:
            params.append(
                pytest.param(
                    dev,
                    id=_device_id(dev),
                    marks=pytest.mark.skip(reason=f"device {dev} not available"),
                )
            )

    return pytest.mark.parametrize("device", params)


__all__ = ["devices"]
