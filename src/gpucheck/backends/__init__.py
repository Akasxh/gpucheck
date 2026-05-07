"""Backend abstraction for gpucheck — CUDA and MPS implementations.

This package defines a structural :class:`Backend` Protocol that captures the
GPU-specific operations gpucheck needs:

- ``synchronize`` — block until pending work completes on a device
- ``event_timer`` — context-managed timer that uses the cheapest accurate
  primitive available (CUDA events on NVIDIA, wall-clock + device sync on MPS,
  see SYNTHESIS §3 / pytorch#162872 for the deadlock context)
- ``mem_stats`` — per-device memory accounting
- ``flush_l2`` — best-effort L2-cache eviction for stable benchmark timings
- ``arch_info`` — populate a :class:`gpucheck.arch.GPUInfo` for the device

The Protocol is **additive** in v1.0: existing CUDA-only call sites in
``fixtures/benchmark.py``, ``fixtures/profiler.py``, etc. retain their direct
``torch.cuda.*`` calls. New MPS code uses the Protocol so the deadlock-safe
timing path is the **only** path on Apple Silicon.

Public API (re-exported via ``gpucheck.backends``)::

    from gpucheck.backends import Backend, available_backends, get_backend

    backends = available_backends()           # list[Backend], priority order
    cuda = get_backend("cuda")                # raises if unavailable
    mps = get_backend("mps")                  # raises if unavailable
"""

from __future__ import annotations

from gpucheck.backends._protocol import Backend, EventTimer


def available_backends() -> list[Backend]:
    """Return all currently-available backends in priority order.

    Priority is CUDA → MPS, mirroring PyTorch's own dispatch order. CPU is
    intentionally excluded — gpucheck targets accelerators.
    """
    backends: list[Backend] = []

    # CUDA first (typical on Linux/Windows GPU servers)
    try:
        from gpucheck.backends.cuda import CUDABackend

        cuda = CUDABackend()
        if cuda.is_available():
            backends.append(cuda)
    except ImportError:
        pass

    # MPS second (Apple Silicon)
    try:
        from gpucheck.backends.mps import MPSBackend

        mps = MPSBackend()
        if mps.is_available():
            backends.append(mps)
    except ImportError:
        pass

    return backends


def get_backend(name: str) -> Backend:
    """Return the named backend, or raise :class:`RuntimeError` if unavailable.

    Recognized names: ``"cuda"``, ``"mps"``.
    """
    name_lower = name.lower()
    if name_lower == "cuda":
        from gpucheck.backends.cuda import CUDABackend

        b: Backend = CUDABackend()
    elif name_lower == "mps":
        from gpucheck.backends.mps import MPSBackend

        b = MPSBackend()
    else:
        raise ValueError(f"Unknown backend {name!r}; expected 'cuda' or 'mps'")

    if not b.is_available():
        raise RuntimeError(
            f"Backend {name!r} is not available on this system "
            f"(missing torch, missing hardware, or driver issue)"
        )
    return b


__all__ = [
    "Backend",
    "EventTimer",
    "available_backends",
    "get_backend",
]
