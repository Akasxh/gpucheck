"""Thread-safety regression test for the tolerance override stack (Track C).

Without the ContextVar fix, ``tolerance_context`` mutates a shared
module-level list — overrides leak between concurrent threads, and a
thread reading inside its ``with`` block can observe a sibling thread's
override.

This test is designed to fail loudly on the unfixed code AND pass on the
fixed code. It uses a ``threading.Barrier`` so all 4 threads enter their
context manager before any of them reads, maximizing observable contention.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from gpucheck.assertions.tolerances import compute_tolerance, tolerance_context


def _worker(
    thread_id: int,
    barrier: threading.Barrier,
    enter_release: threading.Event,
    leave_release: threading.Event,
) -> tuple[int, tuple[float, float]]:
    """Each thread enters a context with thread-distinct overrides, then waits.

    The barrier ensures all 4 threads have entered their context before
    ANY of them reads. Without ContextVar isolation, the read sees the
    LAST writer's override (4 in 4 threads, race-y).
    """
    my_atol = 1.0 + thread_id  # 1.0, 2.0, 3.0, 4.0
    my_rtol = 0.1 + thread_id

    with tolerance_context(my_atol, my_rtol):
        # All threads are now inside their context. Synchronize so reads
        # happen with maximum overlap.
        barrier.wait(timeout=5.0)
        enter_release.wait(timeout=5.0)

        # Read what compute_tolerance sees from THIS thread's perspective.
        observed = compute_tolerance("float32")

        # Hold the context open until the harness signals release. This
        # increases the window during which a sibling thread's broken
        # override could be observed.
        leave_release.wait(timeout=5.0)
        return thread_id, observed


def test_tolerance_context_is_thread_isolated() -> None:
    """Each of N threads must observe its OWN context's atol/rtol.

    The plain-list implementation in pre-Track-C gpucheck serializes
    appends but the read at compute_tolerance time picks the GLOBAL
    top-of-stack — so all threads see the most recently entered
    context's values. With ContextVar each thread has its own stack and
    sees its own atol/rtol.
    """
    n_threads = 4
    barrier = threading.Barrier(n_threads)
    enter_release = threading.Event()
    leave_release = threading.Event()

    with ThreadPoolExecutor(max_workers=n_threads) as exe:
        futures = [
            exe.submit(_worker, tid, barrier, enter_release, leave_release)
            for tid in range(n_threads)
        ]

        # Allow workers to actually read once they've all entered.
        enter_release.set()

        # Brief delay to let reads happen. ContextVar is correct under
        # arbitrary interleavings; the sleep just makes the unfixed
        # version's bug deterministic.
        import time as _time
        _time.sleep(0.05)

        # Now release everyone to leave their context.
        leave_release.set()

        results = [f.result(timeout=10.0) for f in futures]

    # Verify each thread observed its OWN override.
    for tid, (atol, rtol) in results:
        expected_atol = 1.0 + tid
        expected_rtol = 0.1 + tid
        assert atol == pytest.approx(expected_atol), (
            f"thread {tid} observed atol={atol} expected {expected_atol} — "
            f"tolerance_context is NOT thread-isolated"
        )
        assert rtol == pytest.approx(expected_rtol), (
            f"thread {tid} observed rtol={rtol} expected {expected_rtol}"
        )


def test_tolerance_context_pop_is_correct_after_exception() -> None:
    """Exception inside the with-block must still restore the prior state.

    ContextVar.reset(token) is exception-safe by construction; this is a
    regression guard against accidental refactors that might break it.
    """
    base_atol, _ = compute_tolerance("float32")

    class _BangError(RuntimeError):
        pass

    with pytest.raises(_BangError), tolerance_context(99.0, 0.99):
        raise _BangError("boom")

    after_atol, _ = compute_tolerance("float32")
    assert after_atol == pytest.approx(base_atol), (
        "tolerance_context did not restore prior state after exception"
    )


def test_tolerance_context_nesting_in_single_thread() -> None:
    """Nesting in one thread should observe LIFO order — innermost wins."""
    with tolerance_context(1.0, 0.1):
        atol, _ = compute_tolerance("float32")
        assert atol == pytest.approx(1.0)
        with tolerance_context(2.0, 0.2):
            atol2, _ = compute_tolerance("float32")
            assert atol2 == pytest.approx(2.0)
        # After inner exits, outer is restored.
        atol3, _ = compute_tolerance("float32")
        assert atol3 == pytest.approx(1.0)
