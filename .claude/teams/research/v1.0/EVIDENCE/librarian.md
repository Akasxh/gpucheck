---
specialist: research-librarian
slug: v1.0
started: 2026-05-01T03:42:00Z
completed: 2026-05-01T03:43:30Z
tool_calls_count: 8
citations_count: 12
confidence: high
---

# Librarian — torch.mps API surface (PyTorch 2.11 stable)

All quotes verified directly from `docs.pytorch.org/docs/2.11/...` on
2026-05-01. The site's `/stable/` URLs return only a redirect HTML stub
without doc content; `/2.11/` URLs serve full docs.

## §1. torch.mps top-level functions (2.11 stable)

From [docs.pytorch.org/docs/2.11/mps.html, retrieved 2026-05-01]:

| Function | Documented? |
|---|---|
| `torch.mps.device_count()` | yes |
| `torch.mps.synchronize()` | yes — "Waits for all kernels in all streams on a MPS device to complete." |
| `torch.mps.get_rng_state()` | yes |
| `torch.mps.set_rng_state()` | yes |
| `torch.mps.manual_seed()` | yes |
| `torch.mps.seed()` | yes |
| `torch.mps.empty_cache()` | yes — "Releases all unoccupied cached memory currently held by the caching allocator so that those can be used in other GPU applications." |
| `torch.mps.set_per_process_memory_fraction()` | yes |
| `torch.mps.current_allocated_memory()` | yes |
| `torch.mps.driver_allocated_memory()` | yes |
| `torch.mps.recommended_max_memory()` | yes |
| `torch.mps.compile_shader()` | yes (custom Metal shader compilation) |

## §2. torch.mps.profiler

| Function | Documented? |
|---|---|
| `torch.mps.profiler.start()` | yes |
| `torch.mps.profiler.stop()` | yes |
| `torch.mps.profiler.profile()` | yes |
| `torch.mps.profiler.is_capturing_metal()` | yes |
| `torch.mps.profiler.is_metal_capture_enabled()` | yes |
| `torch.mps.profiler.metal_capture()` | yes |

## §3. torch.mps.event.Event class

From [docs.pytorch.org/docs/2.11/generated/torch.mps.event.Event.html, retrieved 2026-05-01]:

```
class torch.mps.event.Event(enable_timing=False)
```

> "Wrapper around an MPS event. MPS events are synchronization markers that can
>  be used to monitor the device's progress, to accurately measure timing, and
>  to synchronize MPS streams."

| Method | Signature | Returns | Documented behavior (verbatim) |
|---|---|---|---|
| `__init__` | `Event(enable_timing=False)` | — | "indicates if the event should measure time (default: False)" |
| `record` | `record()` | None | "Records the event in the default stream." |
| `wait` | `wait()` | None | "Makes all future work submitted to the default stream wait for this event." |
| `query` | `query()` | bool | "Returns True if all work currently captured by event has completed." |
| `synchronize` | `synchronize()` | None | "Waits until the completion of all work currently captured in this event. This prevents the CPU thread from proceeding until the event completes." |
| `elapsed_time` | `elapsed_time(end_event)` | float | "Returns the time elapsed in milliseconds after the event was recorded and before the end_event was recorded." |

**Caveat (verified primary)**: PyTorch issue [#162872, opened 2025-09-13](https://github.com/pytorch/pytorch/issues/162872), confirmed via WebFetch this session: calling `start.record(); end.record(); end.synchronize(); start.elapsed_time(end)` **deadlocks** on PyTorch 2.10 / Apple M4 Pro. gpucheck's `gpu_benchmark` fixture must NOT call `synchronize()` on the end event before `elapsed_time` until this is fixed; it must mirror CUDA's pattern of `torch.mps.synchronize()` instead.

## §4. torch.backends.mps

Standard surface, two functions (verified from secondary docs and matched to
torch source semantics):

```
torch.backends.mps.is_available() -> bool
torch.backends.mps.is_built() -> bool
```

`is_available()` returns True only on macOS 12.3+ on Apple Silicon (or Intel
Mac with eligible AMD GPU). `is_built()` indicates whether the wheel was
compiled with MPS support — typically True on modern PyPI macOS wheels.

## §5. Determinism — what the docs DO NOT say

The 2.11 MPS doc and `torch.mps` page contain **no** explicit text about:
- atomic floating-point operations
- non-deterministic kernels
- the contract of `torch.use_deterministic_algorithms(True)` on MPS

The general randomness doc ([docs.pytorch.org/docs/2.11/notes/randomness.html, retrieved 2026-05-01](https://docs.pytorch.org/docs/2.11/notes/randomness.html)) makes **no** mention of MPS or Metal at all (verified). The only example given for non-deterministic CUDA ops is `index_add_()`. PyTorch's stance on MPS determinism is therefore **silent by omission** — a meaningful negative result for Sub-Q 4.

## §6. PyTorch numerical-accuracy notes (mps NOT covered)

[docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html, retrieved 2026-05-01]
covers TF32 (Ampere+) and AMD MI200 FP16 denormal flushing, but contains **no**
MPS or Metal section. There is no PyTorch-side authoritative tolerance guidance
for MPS — gpucheck's MPS overlay must be derived from observed bug data
(empiricist's job, see `EVIDENCE/empiricist.md`).

## §7. Custom Metal shaders via torch.mps.compile_shader

`torch.mps.compile_shader()` IS exposed and IS documented in 2.11. This is the
escape hatch that let MLX/community projects write Metal kernels callable from
PyTorch tensors. For gpucheck v1.0, **we do not need this surface** — we are
testing existing torch.* ops, not writing kernels — but it is worth knowing it
exists for future "test custom MPS kernels" support.

## Confidence

High for everything except §5 (silent on determinism) — that is a high-confidence
**negative** finding, which the synthesist will surface to the engineering team
explicitly.
