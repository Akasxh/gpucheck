# tracer-runtime summary (W1) — IMPORTANT: refutes one R3 claim

**Hottest path (Trace 1 fast-path):** `torch.allclose(...)` at assertions/close.py:185 — **1.44 ms median** for 1024×1024 fp32. ~99% of total fast-path latency; every other step (dtype resolve, device detect, tolerance compute, predicate) sums to <6 us. Hot cost is kernel launch + reduce + host-side bool drain on Apple Silicon.

**Slowest step overall:** `format_mismatch_report` at assertions/reporting.py — ~16 ms for rich panel + histogram on 1M-element mismatch. Fast-path torch.allclose is sunk cost (~1.77 ms) before slow path starts.

**Slowest in benchmark fixture (Trace 2):** post-iteration `torch.mps.synchronize()` at fixtures/benchmark.py:324 — ~530 us for 256×256 fp32 matmul. End-to-end fixture is 25.4 ms for 10 warmup + 50 rounds.

## 6 substantive findings:

1. **Architectural drift (real):** fixtures/benchmark.py `_run_mps` is a standalone duplicate of `MPSBackend.event_timer`. Backend Protocol is NOT used by the fixture. v1.1 unification candidate.

2. **flush_l2=True warning fires every call** in the fixture (no gate), unlike `MPSBackend.flush_l2` which gates on `_FLUSH_L2_WARNED`. UX bug.

3. **L2 flush on MPS is silently absent** — warning fires, no replacement (e.g., 16MB buffer fill) offered. MPS benchmarks of small kernels are systematically optimistic vs CUDA.

4. **REFUTED linguist-v3 silent fp64 downcast.** On torch 2.11 every fp64-on-MPS path tested raises `TypeError: Cannot convert a MPS Tensor to float64...`. **Fail-loud, not silent.** May be different on torch <2.11. **This is a v1.1 plan reconciliation item — linguist-v3's "silent-downcast catcher API" is not needed if torch 2.11 already fails loud.**

5. **ContextVar overrides bypass the MPS overlay** at tolerances.py:91-93. Confirmed: active `tolerance_context(7.7e-9, 8.8e-9)` returns those values verbatim even with `device_type="mps"`, ignoring 2× multiplier. Documented but worth flagging — user with tight tolerance gets no MPS slack.

6. **Slow-path tax: 1.77 ms of dead work** — when `torch.allclose` returns False on MPS, that 1.77 ms is wasted before numpy fallback. For guaranteed-mismatch tests this is pure overhead.
