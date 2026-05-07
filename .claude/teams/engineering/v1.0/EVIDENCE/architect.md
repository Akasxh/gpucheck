# architect.md

**Specialist**: engineering-architect (Phase A)
**Date**: 2026-05-01

## Module boundaries

### Track A — Backend Protocol

```
src/gpucheck/backends/
  __init__.py     -- Backend Protocol, available_backends(), get_backend(name)
  cuda.py         -- CUDABackend conforming to Protocol; wraps existing CUDA logic
  mps.py          -- MPSBackend conforming to Protocol; uses torch.mps.* exclusively
```

Protocol surface (final):
```python
class Backend(Protocol):
    name: str          # "cuda" | "mps"

    def is_available(self) -> bool: ...
    def device_count(self) -> int: ...
    def synchronize(self, device_id: int = 0) -> None: ...
    def event_timer(self, device_id: int = 0) -> AbstractContextManager[EventTimer]: ...
    def mem_stats(self, device_id: int = 0) -> dict[str, int]: ...
    def flush_l2(self, device_id: int = 0, buf: Any = None) -> None: ...
    def arch_info(self, device_id: int = 0) -> "GPUInfo": ...

class EventTimer(Protocol):
    @property
    def elapsed_ms(self) -> float: ...
```

`event_timer` returns a context manager so the caller writes:
```python
with backend.event_timer() as t:
    fn()
elapsed = t.elapsed_ms
```

For CUDA, the context manager wraps `torch.cuda.Event(enable_timing=True)` start/end + sync.
For MPS, the context manager wraps `torch.mps.synchronize()` + `time.perf_counter()` (deadlock-safe per pytorch#162872).

This Protocol is **additive** in v1.0 — existing call sites in `fixtures/benchmark.py`, `fixtures/profiler.py`, etc. keep their direct CUDA calls. New MPS code uses the Protocol. v1.1 can refactor everything to the Protocol.

### Track B — Stride fuzzing

```
src/gpucheck/fuzzing/
  strides.py      -- fuzz_strides(), StrideStrategy, fuzz_strides_for_category()
```

Public API:
```python
def fuzz_strides(
    shape: tuple[int, ...],
    dtype: Any,
    *,
    n: int = 20,
    device: str = "cpu",
    seed: int | None = None,
) -> list[tuple[str, "torch.Tensor"]]: ...

def fuzz_strides_for_category(
    shape: tuple[int, ...], dtype: Any, category: str, device: str = "cpu", seed: int | None = None
) -> "torch.Tensor": ...

class StrideStrategy:  # Hypothesis SearchStrategy factory like ShapeStrategy
    def __new__(cls, shape, dtype=None, ...): ...
```

Categories: `("row_major", "column_major", "broadcast", "transpose", "slice", "non_contig", "gather")`.

### Track C — ContextVar tolerance stack

`assertions/tolerances.py` change is local. Public API unchanged. New imports: `from contextvars import ContextVar`.

`sanitizers/race.py` change is local: `_find_compute_sanitizer` gains a path allowlist.

### Track D — Reporting + HTML + determinism

```
src/gpucheck/reporting/
  html.py         -- HTMLReporter(json_path).render(out_path)

src/gpucheck/sanitizers/
  determinism.py  -- assert_deterministic, requires_determinism

# CI:
.github/workflows/ci.yml  -- add permissions: contents: read; install via uv lock
uv.lock                    -- committed

# tests:
tests/test_reporting_*.py  -- coverage for console, json, ci, html
tests/test_determinism.py  -- coverage for new sanitizer
```

## Dependency choices

- **No new third-party deps for Track A** beyond `torch>=2.6` (already optional). MPS detection uses `subprocess.check_output(["sysctl", ...])` for chip name; no `xcrun`.
- **No new third-party deps for Track B** beyond optional `hypothesis` (already extras).
- **No new third-party deps for Track C**.
- **For Track D**: pure stdlib + `pytest-cov` (already in `[dev]`). No `jinja2`, no `d3`.

## API surface — public additions in v1.0

| Symbol | Module | Purpose |
|---|---|---|
| `Backend` (Protocol) | `gpucheck.backends` | typing surface |
| `available_backends()` | `gpucheck.backends` | list of detected backends |
| `get_backend(name)` | `gpucheck.backends` | concrete backend |
| `is_mps_xfailed(name)` | `gpucheck.assertions` | xfail registry query |
| `fuzz_strides()` | `gpucheck.fuzzing` | stride corpus |
| `StrideStrategy` | `gpucheck.fuzzing` | hypothesis strategy |
| `assert_deterministic` | `gpucheck.sanitizers` | determinism check |
| `requires_determinism` | `gpucheck.sanitizers` | decorator |
| `HTMLReporter` | `gpucheck.reporting` | dashboard |

## Rejected alternatives

- **Reject**: making Backend a base class (vs Protocol). Rationale: Protocol is structural, doesn't force inheritance, doesn't break existing direct-CUDA call sites. PEP-544.
- **Reject**: per-event MPS timing via `torch.mps.event.Event`. Rationale: pytorch#162872 deadlock. Use device-level sync + wall clock until #162872 closes. SYNTHESIS §3 is binding.
- **Reject**: third-party HTML templating. Rationale: zero-dep is in keeping with the existing project; static HTML + inline SVG is sufficient for v1.0.
- **Reject**: a separate `MPSDevice`/`AppleSiliconDevice` dataclass. Rationale: extending `GPUInfo` with `backend: str` keeps a single dataclass for callers; existing CUDA fields become optional via `tuple[int, int] | None` etc. AUDIT.md §E asks docs which choice to make; we chose extension.

## Verdict

PASS. Module boundaries clean, no circular deps, additive Protocol respects existing call sites.
