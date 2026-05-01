"""@devices("mps") parametrization (Track A)."""

from __future__ import annotations

from gpucheck.decorators.devices import (
    _detect_devices,
    _detect_mps_devices,
    _is_device_available,
    devices,
)


def _has_mps() -> bool:
    try:
        import torch
    except ImportError:
        return False
    mps = getattr(torch.backends, "mps", None)
    return bool(mps is not None and mps.is_available())


def test_detect_mps_devices_when_available() -> None:
    if _has_mps():
        assert _detect_mps_devices() == ["mps"]
    else:
        assert _detect_mps_devices() == []


def test_detect_devices_includes_mps_when_available() -> None:
    devs = _detect_devices()
    if _has_mps():
        assert "mps" in devs


def test_is_device_available_mps_string() -> None:
    if _has_mps():
        assert _is_device_available("mps") is True
    else:
        assert _is_device_available("mps") is False


# Explicit @devices("mps") usage — parametrizes the test even if MPS is
# unavailable (test gets pytest.mark.skip in that case).
@devices("mps")
def test_devices_decorator_passes_mps_string(device: str) -> None:
    assert device == "mps"
    if _has_mps():
        import torch

        x = torch.zeros(2, 2, device=device)
        assert x.device.type == "mps"


@devices("cuda:0", "mps")
def test_devices_decorator_mixed_cuda_and_mps(device: str) -> None:
    assert device in {"cuda:0", "mps"}


def test_all_keyword_includes_mps_on_apple_silicon() -> None:
    """The 'all' keyword should expand to MPS on Apple Silicon."""
    devs = _detect_devices()
    if _has_mps():
        assert "mps" in devs
