---
specialist: research-tracer
slug: v1.0
started: 2026-05-01T03:42:00Z
completed: 2026-05-01T03:43:30Z
tool_calls_count: 2
citations_count: 6
confidence: medium
---

# Tracer — MPS dispatch path and runtime contracts

## §1. PyTorch MPS dispatch architecture (from issue threads)

The MPS backend's dispatch shape, reconstructed from primary issues retrieved 2026-05-01:

```
torch.<op>(args, device='mps')
   ↓
ATen dispatcher → MPS DispatchKey
   ↓
   ├─ Native MPS kernel (Metal/MPSGraph) ────────→ Apple GPU
   ├─ "math" backend (op decomposition into MPS ops) ─→ Apple GPU
   └─ NotImplementedError → user must fall back to CPU manually
```

Key contracts (load-bearing for gpucheck):

### 1.1 No automatic CPU fallback

Unlike llama.cpp's Metal backend (which falls back to CPU transparently when
`supports_op` returns false), **PyTorch's MPS backend raises
`NotImplementedError`** when an op has no MPS impl. From [#160828, retrieved 2026-05-01](https://github.com/pytorch/pytorch/issues/160828): "The operator 'aten::_ctc_loss' is not currently implemented for the MPS device." gpucheck-MPS tests must therefore wrap MPS calls in try/except for missing-op detection, or check op coverage in a precondition (`@require_op('aten::_ctc_loss', device='mps')`).

This is a **design difference from CUDA**: CUDA either has the op or doesn't compile;
MPS has the op or raises at runtime. Test infrastructure must accommodate.

### 1.2 SDPA dispatches to "math" backend

[#179294, retrieved 2026-05-01](https://github.com/pytorch/pytorch/issues/179294) verbatim: "the dedicated MPS implementation is called under the 'math' backend despite being platform-specific. It should move to the 'flash' backend to align with CUDA conventions." Also: "the backward pass relies on the device-agnostic 'math' backend instead of a dedicated MPS implementation". And: "a Metal kernel implementation exists but is never invoked in the current codebase."

**Implication for gpucheck v1.0**: setting `attn_implementation="flash_attention_2"`
on MPS will silently fall through to math decomposition and run slowly. Tests
that assert "FlashAttention path was used" must check by timing or `torch.profiler`,
not by `attn_implementation` flag — the flag is a hint, not a guarantee on MPS.

### 1.3 Run-to-run buffer pool corruption is observed

[#177116, retrieved 2026-05-01](https://github.com/pytorch/pytorch/issues/177116) original poster: "Trial 0 shows correct results (~0.2398); subsequent trials degrade catastrophically". The reporter observed that `torch.mps.empty_cache()` between operations "appeared to reduce the failure rate, suggesting buffer pool corruption when tensor shapes change between backward passes."

**Implication for gpucheck**: between fuzz iterations with shape change,
`torch.mps.empty_cache()` should be inserted. This is the MPS analog of
gpucheck's existing CUDA `torch.cuda.empty_cache()` between benchmarks (see
`fixtures/benchmark.py` — gpucheck's L2-flush pattern). The MPS empty_cache
serves a different purpose (correctness, not just cache flush) but the call
site is the same.

### 1.4 Memory accounting is unreliable

[#164299, retrieved 2026-05-01](https://github.com/pytorch/pytorch/issues/164299): "Memory growth is not recognized by built-in `torch.mps.current_allocated_memory()` and `torch.mps.driver_allocated_memory()` methods but visible on Activity Monitor". The leak culprit: graph cache handling logic.

**Implication for gpucheck v1.0**: the `memory_tracker` fixture cannot rely on
`torch.mps.current_allocated_memory()` to detect leaks. Either:
- a) use `psutil` against the process RSS as a proxy (matches gpucheck's existing
  pynvml fallback pattern at `sanitizers/memory.py`)
- b) explicitly note in v1.0 that MPS memory leak detection is "best-effort" and
  may miss MPS-internal leaks not reflected in the public counters.

Recommendation: ship (a) with a docstring note; the existing CLAUDE.md weakness
"Memory leak detection uses process-level metrics (imprecise)" is even more apt
on MPS.

## §2. The Event API contract (load-bearing)

`torch.mps.event.Event` exposes record / wait / query / synchronize /
elapsed_time. Per librarian §3 verified from PyTorch 2.11 docs:

- `record()` records on the default stream (no stream arg unlike CUDA).
- `synchronize()` blocks the CPU until the event completes.
- `elapsed_time(end_event)` returns ms.

**Bug contract** (load-bearing): per [#162872, retrieved 2026-05-01](https://github.com/pytorch/pytorch/issues/162872), calling `end.synchronize()` before `start.elapsed_time(end)` deadlocks. The recommended pattern (from CUDA-equivalent and the issue thread) is:

```python
torch.mps.synchronize()  # device-wide barrier
ms = start_event.elapsed_time(end_event)
```

gpucheck's `gpu_benchmark` fixture (CUDA-shaped) uses CUDA events with
`event.synchronize()` per round. The MPS path must use device-level
`torch.mps.synchronize()` instead, NOT per-event synchronize, until #162872 closes.

## §3. Stream semantics

MPS has only the **default stream** in 2.11 — no multi-stream API surface.
This is simpler than CUDA (which has streams + graphs). Two consequences:
- gpucheck's `@parametrize_gpu` doesn't need stream parametrization for MPS.
- Concurrency tests are limited; "race condition" testing on MPS is mostly N/A.

## §4. The forked-subprocess hazard

[#178037 "[MPS] Raise clear error when MPS is used in forked subprocess", retrieved 2026-05-01](https://github.com/pytorch/pytorch/issues/178037) — MPS is undefined behavior across `os.fork()`. pytest-xdist uses fork on macOS by default. gpucheck-MPS must:
- detect xdist + MPS combination and warn/error
- recommend `pytest-xdist --start-method=spawn` for MPS test runs

This is a deployment-doc concern, not a code-path concern, but it's load-bearing
for "v1.0 MPS tests work in parallel CI".

## §5. Comparison: CUDA-event vs MPS-event timing accuracy

CUDA events have ~1µs resolution. MPS events' resolution is not documented in
the PyTorch docs reviewed. PyTorch ships a `torch.mps.profiler.metal_capture()`
for Xcode-grade profiling; for gpucheck v1.0's "median-of-N" benchmark
methodology, `Event.elapsed_time` is sufficient unless P99 deltas matter — the
existing IQR outlier removal in gpucheck's benchmark fixture absorbs jitter.

## Confidence

Medium-high. The dispatch architecture is reconstructed from issue threads, not
from a primary source-code trace (gpucheck's research-tracer normally reads
PyTorch source via Grep, but in this 60-min budget the primary issue threads
provided sufficient signal). The 5 contracts (§1.1–§1.4 and §2) are each cited
to a primary GitHub issue.
