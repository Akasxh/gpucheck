"""gpucheck diagnostics — runtime probes for known PyTorch backend pathologies.

The diagnostics package exposes one-shot, opt-in probes that detect
*runtime* failure modes which cannot be caught at import or collection
time. The motivating example is the MPS Event-API deadlock
([pytorch#162872](https://github.com/pytorch/pytorch/issues/162872)),
which silently hangs the test runner the moment a user calls
``torch.mps.event.Event.synchronize()`` on certain PyTorch builds.

Each probe is a pure-Python helper with three guarantees:

1. **Lazy imports.** ``torch`` (and any other heavy backend) is only
   imported inside the probe body, never at module load time.
2. **Bounded cost.** The probe runs in a daemon thread with a hard
   timeout. On a healthy build the cost is sub-10ms; on a broken build
   the daemon thread is leaked (process exit reaps it) but the caller
   returns within the timeout.
3. **Tri-state result.** Probes return ``"healthy"``, ``"deadlocked"``,
   or ``"skipped"``. The third state lets the caller distinguish
   "everything is fine" from "we couldn't even check on this host".
"""

from __future__ import annotations

from gpucheck.diagnostics.mps_event_deadlock import (
    ProbeResult,
    assert_no_event_deadlock,
    mps_event_deadlock_status_fixture,
    probe_mps_event_deadlock,
)

__all__ = [
    "ProbeResult",
    "assert_no_event_deadlock",
    "mps_event_deadlock_status_fixture",
    "probe_mps_event_deadlock",
]
