"""assert_close on MPS: fast-path widening + 2x tolerance overlay (Track A)."""

from __future__ import annotations

import pytest

from gpucheck.assertions import assert_close, compute_tolerance


def _has_mps() -> bool:
    try:
        import torch
    except ImportError:
        return False
    mps = getattr(torch.backends, "mps", None)
    return bool(mps is not None and mps.is_available())


def test_compute_tolerance_mps_doubles_float32() -> None:
    base_atol, base_rtol = compute_tolerance("float32")
    mps_atol, mps_rtol = compute_tolerance("float32", device_type="mps")
    assert mps_atol == pytest.approx(base_atol * 2.0)
    assert mps_rtol == pytest.approx(base_rtol * 2.0)


def test_compute_tolerance_mps_doubles_float16() -> None:
    base_atol, _base_rtol = compute_tolerance("float16")
    mps_atol, _mps_rtol = compute_tolerance("float16", device_type="mps")
    assert mps_atol == pytest.approx(base_atol * 2.0)


def test_compute_tolerance_mps_doubles_bfloat16() -> None:
    base_atol, _base_rtol = compute_tolerance("bfloat16")
    mps_atol, _mps_rtol = compute_tolerance("bfloat16", device_type="mps")
    assert mps_atol == pytest.approx(base_atol * 2.0)


def test_compute_tolerance_mps_keeps_float64_unchanged() -> None:
    """float64 is rarely load-bearing on MPS; we don't inflate."""
    base = compute_tolerance("float64")
    mps = compute_tolerance("float64", device_type="mps")
    assert base == mps


def test_compute_tolerance_cuda_unchanged_when_device_type_cuda() -> None:
    base = compute_tolerance("float32")
    cuda = compute_tolerance("float32", device_type="cuda")
    assert base == cuda


def test_compute_tolerance_with_kdim_and_mps_overlay() -> None:
    """MPS overlay applies AFTER k_dim sqrt scaling so the order is documented."""
    cuda = compute_tolerance("float32", k_dim=512)
    mps = compute_tolerance("float32", k_dim=512, device_type="mps")
    assert mps[0] == pytest.approx(cuda[0] * 2.0)


@pytest.mark.skipif(not _has_mps(), reason="MPS not available")
def test_assert_close_mps_fast_path_no_cpu_transfer() -> None:
    """On equal MPS tensors, assert_close returns without going through numpy.

    We patch _to_numpy to raise — if the fast-path is taken, _to_numpy is
    never called and the test passes; if the slow path is taken, it raises.
    """
    import torch

    from gpucheck.assertions import close as close_mod

    original = close_mod._to_numpy

    def trip_wire(*_a, **_kw):
        raise AssertionError("_to_numpy was called — fast path missed!")

    close_mod._to_numpy = trip_wire  # type: ignore[assignment]
    try:
        a = torch.ones(8, 8, device="mps")
        b = torch.ones(8, 8, device="mps")
        assert_close(a, b)
    finally:
        close_mod._to_numpy = original  # type: ignore[assignment]


@pytest.mark.skipif(not _has_mps(), reason="MPS not available")
def test_assert_close_mps_passes_with_mps_overlay_for_float16() -> None:
    """Two MPS fp16 tensors that differ by ~1.5e-2 must pass under MPS 2x
    overlay (base atol=1e-2, MPS atol=2e-2).
    """
    import torch

    a = torch.full((16, 16), 1.0, device="mps", dtype=torch.float16)
    b = torch.full((16, 16), 1.0 + 1.5e-2, device="mps", dtype=torch.float16)
    assert_close(a, b)
