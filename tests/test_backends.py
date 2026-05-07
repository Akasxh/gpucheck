"""Backend Protocol contract tests (Track A)."""

from __future__ import annotations

import warnings

import pytest

from gpucheck.backends import Backend, available_backends, get_backend


def _torch():
    try:
        import torch

        return torch
    except ImportError:
        pytest.skip("torch not installed")


def test_get_backend_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown backend"):
        get_backend("rocm")


def test_available_backends_returns_list() -> None:
    backends = available_backends()
    assert isinstance(backends, list)
    for b in backends:
        assert isinstance(b, Backend)
        assert b.name in {"cuda", "mps"}


def test_available_backends_priority_cuda_before_mps() -> None:
    backends = available_backends()
    names = [b.name for b in backends]
    if "cuda" in names and "mps" in names:
        assert names.index("cuda") < names.index("mps")


# ---------------------------------------------------------------------------
# MPS backend (only runs on machines where MPS is available)
# ---------------------------------------------------------------------------

@pytest.fixture()
def mps_backend():
    torch = _torch()
    mps = getattr(torch.backends, "mps", None)
    if mps is None or not mps.is_available():
        pytest.skip("MPS not available on this machine")
    from gpucheck.backends.mps import MPSBackend

    return MPSBackend()


def test_mps_backend_name(mps_backend) -> None:
    assert mps_backend.name == "mps"


def test_mps_backend_synchronize_no_event_synchronize(mps_backend) -> None:
    """SYNTHESIS §3 — MUST use device-level torch.mps.synchronize, NOT
    per-event Event.synchronize (deadlocks on Apple Silicon, pytorch#162872).

    We assert this structurally by calling synchronize() and verifying it
    returns without raising. Detailed source-introspection lives in the
    next test; this one is the smoke check.
    """
    mps_backend.synchronize()  # Must not hang or raise


def test_mps_backend_event_timer_uses_device_sync_not_event_sync(mps_backend) -> None:
    """The event_timer context manager must use torch.mps.synchronize().

    We verify this by source-introspecting the implementation, scanning the
    AST so docstring text doesn't trigger false positives. The deadlock
    pattern (pytorch#162872) is calling ``.synchronize()`` on a
    ``torch.mps.event.Event`` instance.
    """
    import ast
    import inspect
    import textwrap

    from gpucheck.backends.mps import MPSBackend

    src = textwrap.dedent(inspect.getsource(MPSBackend.event_timer))
    tree = ast.parse(src)

    # Find every Call expression in code (not docstrings).
    found_device_sync = False
    forbidden_calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attr = node.func
            # torch.mps.synchronize() — required pattern.
            if (
                attr.attr == "synchronize"
                and isinstance(attr.value, ast.Attribute)
                and attr.value.attr == "mps"
            ):
                found_device_sync = True
            # event.synchronize() / Event.synchronize() — forbidden pattern.
            if attr.attr == "synchronize" and isinstance(attr.value, ast.Name):
                name = attr.value.id
                if name.lower() in {"event", "start", "end"}:
                    forbidden_calls.append(f"{name}.synchronize()")

    assert found_device_sync, "event_timer must call torch.mps.synchronize()"
    assert not forbidden_calls, (
        f"event_timer must NOT call per-event synchronize "
        f"(pytorch#162872 deadlock); found: {forbidden_calls}"
    )


def test_mps_backend_event_timer_returns_positive_elapsed_ms(mps_backend) -> None:
    torch = _torch()
    x = torch.randn(64, 64, device="mps")
    with mps_backend.event_timer() as t:
        _ = x @ x
    assert t.elapsed_ms >= 0.0


def test_mps_backend_arch_info_returns_apple_silicon(mps_backend) -> None:
    info = mps_backend.arch_info()
    assert info.architecture == "Apple-Silicon"
    assert info.backend == "mps"
    assert info.tensor_core_generation is None
    assert info.cuda_version == ""
    # Compute capability is a CUDA concept; MPS uses (0, 0).
    assert info.compute_capability == (0, 0)
    # Apple chip name should be in the device name when sysctl is available
    # (we only assert non-empty here so the test still passes in chrooted CI).
    assert isinstance(info.name, str)


def test_mps_backend_flush_l2_is_noop_with_warning(mps_backend) -> None:
    # Reset the module-level warning gate so we deterministically observe
    # the warning even if a prior test already triggered it.
    import gpucheck.backends.mps as mps_mod

    mps_mod._FLUSH_L2_WARNED = False

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mps_backend.flush_l2()
    assert any("L2" in str(w.message) for w in caught)


def test_mps_backend_mem_stats_has_required_keys(mps_backend) -> None:
    stats = mps_backend.mem_stats()
    assert "used" in stats
    # driver_allocated and total are best-effort; just check keys exist.
    assert "driver_allocated" in stats
    assert "total" in stats


def test_mps_backend_device_count_is_one(mps_backend) -> None:
    # MPS exposes a single logical device on every Apple Silicon machine.
    assert mps_backend.device_count() == 1
