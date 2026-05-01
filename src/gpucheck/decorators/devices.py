"""Parametrize tests across GPU devices (CUDA and MPS)."""

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


def _detect_mps_devices() -> list[str]:
    """Return ``["mps"]`` if Apple Silicon MPS is available, else ``[]``.

    PyTorch's MPS backend exposes a single logical device, so we never emit
    ``mps:0``/``mps:1`` even on machines with an integrated + discrete GPU.
    """
    try:
        import torch
    except ImportError:
        return []
    mps = getattr(torch.backends, "mps", None)
    if mps is None or not mps.is_available():
        return []
    return ["mps"]


def _detect_devices() -> list[str]:
    """Return all available accelerator device strings (CUDA first, then MPS)."""
    return _detect_cuda_devices() + _detect_mps_devices()


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
        if device == "mps" or device.startswith("mps:"):
            mps = getattr(torch.backends, "mps", None)
            return bool(mps is not None and mps.is_available())
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

    Recognized device strings:

    - ``"cuda:N"`` — specific NVIDIA GPU
    - ``"mps"`` — Apple Silicon GPU (single logical device)
    - ``"all"`` — every available accelerator (CUDA devices + MPS if present)

    If no arguments are given, auto-detects all available accelerators
    (CUDA devices first, then MPS). Falls back to ``["cuda:0"]`` if
    detection finds nothing — the test then skips at collection.

    Devices that are not available at collection time get
    ``pytest.mark.skip`` so the test is reported but not run.

    Examples::

        @devices("cuda:0", "mps")
        def test_copy(device): ...

        @devices()          # auto-detect (CUDA + MPS)
        def test_kernel(device): ...

        @devices("all")
        def test_broadcast(device): ...
    """
    resolved: list[str] = []

    if not device_args or device_args == ("all",):
        detected = _detect_devices()
        resolved = detected if detected else ["cuda:0"]
    else:
        for d in device_args:
            if d == "all":
                resolved.extend(_detect_devices() or ["cuda:0"])
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
