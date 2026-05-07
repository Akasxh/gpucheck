"""Backend and EventTimer Protocols.

Kept in a private module so user code imports from ``gpucheck.backends``
(public surface) rather than ``gpucheck.backends._protocol``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from gpucheck.arch.detection import GPUInfo


@runtime_checkable
class EventTimer(Protocol):
    """A single benchmark interval with millisecond elapsed time.

    Concrete instances are produced by :meth:`Backend.event_timer` and used as
    context managers::

        with backend.event_timer() as t:
            kernel(x, y)
        elapsed_ms = t.elapsed_ms
    """

    @property
    def elapsed_ms(self) -> float:
        """Elapsed wall time of the protected block in milliseconds."""
        ...


@runtime_checkable
class Backend(Protocol):
    """Structural interface for a GPU backend supported by gpucheck."""

    name: str  # "cuda" | "mps"

    def is_available(self) -> bool:
        """Return ``True`` if this backend can run kernels on this machine."""
        ...

    def device_count(self) -> int:
        """Number of devices this backend exposes."""
        ...

    def synchronize(self, device_id: int = 0) -> None:
        """Block until pending work on the device has finished."""
        ...

    def event_timer(
        self, device_id: int = 0,
    ) -> AbstractContextManager[EventTimer]:
        """Return a context manager that times the wrapped block."""
        ...

    def mem_stats(self, device_id: int = 0) -> dict[str, int]:
        """Memory accounting in bytes; keys at minimum: ``used``, ``total``."""
        ...

    def flush_l2(self, device_id: int = 0, buf: Any = None) -> None:
        """Best-effort L2-cache flush for stable benchmark timings.

        On backends without L2-flush support (e.g. MPS), this is a no-op and
        emits a one-time :class:`UserWarning`.
        """
        ...

    def arch_info(self, device_id: int = 0) -> GPUInfo:
        """Populate a :class:`GPUInfo` describing the device."""
        ...


__all__ = ["Backend", "EventTimer"]
