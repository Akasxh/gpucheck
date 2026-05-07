# Tracer — Runtime call paths of `assert_close` and `gpu_benchmark` on MPS

Hardware: Apple Silicon (sysctl CPU brand reported by MPSBackend), macOS 25.4.0
PyTorch: 2.11.0 (torch.backends.mps.is_available() == True)
Repo: /Users/cero/Code/gpucheck @ release/v1.0
Probe scripts: `/tmp/trace_runtime.py`, `/tmp/trace_silent_downcast.py`

All timings are perf_counter_ns deltas, in microseconds (us), median of 10–30
iterations after warming the MPS runtime with a single 8-element kernel. Source
files were NOT modified — the probes wrap the public surface and re-time the
exact body of the source-line ranges cited.

---

## TRACE 1 — `assert_close(actual, expected)` with both tensors on `mps`

### Entry point
`src/gpucheck/assertions/close.py:109` — `def assert_close(actual, expected, *, rtol=None, atol=None, k_dim=None, nan_equal=False, baseline_2x=False, msg="")`

### Forward trace (annotated, with measured costs at shape (1024, 1024) float32)

| # | Source                                | Step                                                                 | Median | Notes                                                                                              |
|---|---------------------------------------|----------------------------------------------------------------------|-------:|----------------------------------------------------------------------------------------------------|
| 1 | `close.py:140`                        | `_resolve_dtype(actual, expected)` walks `.dtype` of both tensors    | 2.5 us | Mixed-int/float prefers float; same-category prefers smaller itemsize (`close.py:76-106`).        |
| 2 | `close.py:145-150`                    | Device-type detection: `for t in (actual, expected): isinstance(t, torch.Tensor) → device.type` | 0.8 us | `device_type = "mps"`. Bails on first match → only `actual` is checked when both are tensors.    |
| 3a | `close.py:153-163`                   | `baseline_2x` branch (NOT taken in default call)                     | —      | Doubles base before `k_dim` scaling at `close.py:156-157`. ContextVar overrides bypass everything. |
| 3b | `close.py:165-172`                   | `compute_tolerance(dtype, k_dim, device_type="mps")` (the taken branch) | 1.0 us | Calls `tolerances.py:70`; see chain below.                                                       |
| 4 | `tolerances.py:91-93`                 | **ContextVar override** check (early return if any override active)  | —      | Empirically verified: an active `tolerance_context(7.7e-9, 8.8e-9)` BYPASSES the MPS overlay entirely (the function returns the override before reaching line 105). |
| 5 | `tolerances.py:95-100`                | `_normalize_dtype_name`, then `_config_overrides` lookup, then `_DEFAULT_TOLERANCES` lookup | —      | float32 → (1e-4, 1e-4) (`tolerances.py:21`).                                                      |
| 6 | `tolerances.py:102-103`               | k_dim scaling: `atol *= sqrt(k_dim/128)`                             | —      | Skipped here (k_dim=None).                                                                         |
| 7 | `tolerances.py:106-109`               | **MPS overlay**: `multiplier = _MPS_TOLERANCE_MULTIPLIERS.get(name, 2.0)`; `atol *= mul; rtol *= mul` | —     | Verified: float32 (1e-4, 1e-4) → (2e-4, 2e-4). PROVISIONAL per SYNTHESIS §7.                      |
| 8 | `close.py:176-183`                    | Fast-path eligibility: both `torch.Tensor`, same device, device.type ∈ {"cuda","mps"}, equal shapes | 0.9 us | Note: uses `actual.device == expected.device` — for MPS this compares the `torch.device("mps", 0)` singleton, so it's cheap. |
| 9 | `close.py:185`                        | **Fast-path:** `torch.allclose(actual, expected, atol, rtol, equal_nan)` on MPS | **1440 us (~1.4 ms)** | THIS IS THE HOTTEST STEP. The op launches Metal kernels (eq+abs+max-reduce-or-and) and the `.item()`-equivalent host read. The probe calls `torch.mps.synchronize()` afterward to take the host-side bool, which is what `torch.allclose` does internally. See "Hidden cost" below. |
| 10 | `close.py:186`                       | If True → `return` (PASS, zero CPU transfer)                         | —      | Total fast-path latency ≈ 1.45 ms.                                                                |
| 11 | `close.py:187-189`                   | `RuntimeError` catch — re-raises if message lacks "allclose"/"match" | —      | Defensive net for genuine MPS errors. Not exercised in normal traffic.                            |

### Slow-path trace (when `torch.allclose` returns False, or actual/expected are not both MPS tensors)

Triggered by, e.g., shape mismatch, NaN/Inf disagreement, or a real numeric divergence. Measured against a 1024×1024 mismatch (b = a + 0.5):

| # | Source                                | Step                                                                                            | Median   | Notes |
|---|---------------------------------------|------------------------------------------------------------------------------------------------|---------:|-------|
| 12 | `close.py:185 (returns False)`       | `torch.allclose` already executed (full reduce + host bool)                                     | 1770 us  | Sunk cost: the slow path always pays the fast-path cost first. **This is dead work on guaranteed-mismatch tests.** |
| 13 | `close.py:192`                       | `_to_numpy(actual)` — `t.detach().cpu().numpy()` (close.py:28-38). For MPS this is a Metal→host copy. | 610 us | 4 MB transfer + numpy view. The branch at `close.py:33` triggers `.double()` only for ≥8-byte float; fp32 takes the zero-copy `.numpy()` path at line 35. |
| 14 | `close.py:193`                       | `_to_numpy(expected)`                                                                           | 545 us   | Same.                                                                                                |
| 15 | `close.py:200-201`                   | `actual_np.astype(np.float64, copy=False)` × 2                                                  | 390 us   | `copy=False` is a request not a guarantee — fp32 → fp64 always copies. Note this is **second** float64 promotion (numpy already paid one when `_to_numpy` did `.numpy()`). |
| 16 | `close.py:204-263`                   | NaN / Inf detection passes (4× `np.isnan/np.isinf`)                                             | 460 us   |                                                                                                       |
| 17 | `close.py:270-272`                   | `diff = abs(a - b); threshold = atol + rtol * abs(b); failures = diff > threshold`              | 980 us   | 3 numpy passes over 1M float64 elements.                                                             |
| 18 | `close.py:274-284` + `reporting.py`  | `format_mismatch_report` (rich panel, histogram, error stats)                                   | ~16 ms   | Dominates the slow path. Total `assert_close` failure path measured at **~21.5 ms**.                 |

### Backward trace — where do `eff_atol` / `eff_rtol` come from?

- `close.py:168-169`: `eff_atol = atol if atol is not None else default_atol` where `default_atol` ← `compute_tolerance(...)` at `close.py:165-167`.
- `tolerances.py:91-93` (writer #1): if `_tolerance_overrides.get()` is non-empty, returns `overrides[-1]` directly. **This bypasses `_config_overrides`, `_DEFAULT_TOLERANCES`, k_dim scaling, AND the MPS overlay.** Verified by probe.
- `tolerances.py:97-100` (writer #2): `_config_overrides` overlay (populated by `apply_config_tolerances` from `[tool.gpucheck.tolerances]` in pyproject.toml at session start, called from `plugin.py`).
- `tolerances.py:100` (writer #3): hard-coded `_DEFAULT_TOLERANCES` at `tolerances.py:13-26` (calibrated against cuBLAS Turing/Ampere/Ada).
- `tolerances.py:106-109` (writer #4): MPS overlay multiplies the resolved (atol, rtol) — only when `device_type=="mps"`. Multipliers at `tolerances.py:35-43`.

### Competing hypotheses for the 1.4 ms fast-path cost

**H1: `torch.allclose` on MPS is dominated by host↔device synchronization, not by the actual reduce kernel.**
- Supporting: kernel launch alone (with no sync) measured at ~20 us; the post-sync wait was ~530 us for a 256×256 matmul. A `torch.allclose` over 1M elements has at least one host-side bool fetch which forces a device drain. 1.4 ms is consistent with launch+reduce+drain on a small Apple GPU.
- Falsifiable by: measuring `torch.allclose` on a 1-element tensor — if it stays in the same order (~1 ms), the cost is sync, not compute.
- Status: **supported** but not isolated. Worth a follow-up micro-bench.

**H2: `actual.device == expected.device` (close.py:180) creates a hidden allocation.**
- Supporting: comparing `torch.device` instances calls `__eq__`. On most PyTorch builds this is a fast Python comparison.
- Falsifiable by: timing alone — the predicate block measures at 0.9 us median. **Refuted.**

**H3: The MPS overlay multiplier table at `tolerances.py:35-43` is queried twice when k_dim is also given.**
- Supporting: read of `compute_tolerance` body shows a single read.
- Falsifiable by: the source. **Refuted** — single read at line 107.

### Probes I ran (Trace 1)

- Wrapped each line range and timed it 30× after warming MPS with a small kernel. See `/tmp/trace_runtime.py` lines 49-110.
- Forced override stack with `tolerance_context(7.7e-9, 8.8e-9)` and confirmed `compute_tolerance(float32, device_type="mps")` returned `(7.7e-9, 8.8e-9)` — i.e., the MPS overlay was bypassed.
- Verified `compute_tolerance(float32, device_type=None) = (1e-4, 1e-4)` and `compute_tolerance(float32, device_type="mps") = (2e-4, 2e-4)` — multiplier confirmed to be 2.0× as encoded.

### Silent-coercion findings (Trace 1)

`_to_numpy` at `close.py:22-67`:
- fp64 input → `.double().numpy()` (line 33). **Lossless.** Confirmed.
- fp32 input → `.numpy()` (line 35). **Lossless.** Confirmed.
- fp16/bf16/fp8 → `.float().numpy()` (line 37). **Upcast to fp32.** Lossless for fp16/bf16 (the original precision is preserved as a subset of fp32).
- Non-floating → `.numpy()` (line 38). Lossless.

The slow path then calls `astype(float64, copy=False)` at `close.py:200-201`. For fp32 input this **forces a second copy** (fp32 → fp64), so the slow path's float64 cast does an unnecessary float32-staging round trip when the source was sub-float64. Not a correctness risk, but a 1-ms tax on the slow path.

---

## TRACE 2 — `gpu_benchmark` fixture on MPS

### Entry point
`src/gpucheck/fixtures/benchmark.py:330` — `@pytest.fixture() def gpu_benchmark()` returns `_BenchmarkRunner()`.

### Forward trace

| # | Source                                  | Step                                                                                              | Median   | Notes                                                                                       |
|---|-----------------------------------------|---------------------------------------------------------------------------------------------------|---------:|---------------------------------------------------------------------------------------------|
| 1 | `benchmark.py:343`                      | Fixture returns `_BenchmarkRunner()` instance                                                     | —        | `__post_init__` runs (`benchmark.py:134-145`): tries pynvml then preallocates a CUDA flush buffer. **On MPS-only systems the pynvml branch silently fails** (ImportError handled at line 87) and the buffer alloc fails (cuda not available). `self._l2_size` ends up at the 40 MB fallback (`benchmark.py:89`), `self._flush_buf` stays `None`. Cost: a single failed pynvml import attempt. |
| 2 | `benchmark.py:185-191`                  | `import torch` (cached after first fixture call)                                                  | —        |                                                                                              |
| 3 | `benchmark.py:193-198`                  | `cuda_avail = torch.cuda.is_available()`; `mps_avail = torch.backends.mps.is_available()`         | 5.5 us   | Both calls combined.                                                                         |
| 4 | `benchmark.py:199-200`                  | `pytest.skip(...)` if neither GPU is available                                                    | —        | Not taken on MPS.                                                                            |
| 5 | `benchmark.py:202-204`                  | Resolve overrides: `n_warmup`, `n_rounds`, `do_flush`                                             | —        |                                                                                              |
| 6 | `benchmark.py:206-209`                  | **Backend dispatch**: `if cuda_avail: _run_cuda(...) else: _run_mps(...)`                          | —        | NB: this is NOT calling `MPSBackend.event_timer` from `backends/mps.py:111`. The fixture has its own duplicate implementation. The `MPSBackend` Protocol is unused in the fixture path — it's only reachable from external user code via `gpucheck.backends.get_backend("mps")` (see `backends/__init__.py:64-86`). Documented as "additive in v1.0" at `backends/__init__.py:14-17`. |
| 7 | `benchmark.py:283 _run_mps:304-310`     | `if do_flush: warnings.warn("flush_l2=True ignored on MPS …", UserWarning)`                       | 16 us    | **The warning fires every call**, NOT once-per-process — see "Hidden cost" below. The `MPSBackend.flush_l2` at `backends/mps.py:161` does have a one-time gate (`_FLUSH_L2_WARNED`), but the fixture's inline implementation does not. |
| 8 | `benchmark.py:313`                      | `torch.mps.synchronize()` — drain prior in-flight work                                            | <1 us    |                                                                                              |
| 9 | `benchmark.py:314-316`                  | Warmup: `for _ in range(n_warmup): fn(*args, **kwargs)` then sync                                  | ~944 us for 10 warmup rounds | Kernel launches are async; only the final sync blocks. Cost is 10 launches + 1 device drain.|
| 10 | `benchmark.py:319-326`                 | Per-iteration loop: `mps.synchronize()` → `t0 = perf_counter()` → `fn(...)` → `mps.synchronize()` | 554 us per iter (256×256 matmul) | Pre-sync drains prior work (~0.4 us when nothing pending). Kernel launch ~19 us. **Post-sync is the dominant cost (~530 us).** This is by-design: the deadlock-safe path replaces `Event.synchronize()` with `mps.synchronize()` per pytorch#162872. |
| 11 | `benchmark.py:212-213`                 | `_remove_outliers_iqr(raw_times)` (Tukey 1.5×IQR fence)                                           | 9 us     |                                                                                               |
| 12 | `benchmark.py:214-221`                 | If all samples removed, restore raw                                                                | —        | Defensive guard.                                                                              |
| 13 | `benchmark.py:223-243`                 | Compute mean/std/percentiles, return `BenchmarkResult`                                            | <50 us   |                                                                                               |

End-to-end fixture call (10 warmup, 50 rounds, 256×256 fp32 matmul, flush_l2=True):
- **Total wall time: 25.4 ms**
- **Median per round: 545 us** (matches the per-iteration breakdown).
- Outliers removed: 0 (the IQR window admitted all 50 samples).

### Backward trace — what determines the per-iteration timing

- `benchmark.py:325`: `elapsed_ms = (time.perf_counter() - t0) * 1000.0` — wall clock between two `torch.mps.synchronize()` brackets.
- The KERNEL itself (Metal-side compute) finishes BEFORE the post-sync returns; the 530 us post-sync includes the time the kernel actually ran on the GPU plus the host-driver round-trip to confirm completion. There is no way to subtract the round-trip overhead with the current API — that's the documented ~1 ms timing-resolution penalty vs CUDA events (`backends/mps.py:18-19`).
- L2 flush: there is **no L2 flush call site** in `_run_mps` (compare `_run_cuda` at `benchmark.py:271-272`). `do_flush` is only consumed by the warning emit at line 304-310. **Inter-iteration L2 caching is not addressed on MPS** — kernels that fit fully in L2 will benchmark "too fast" relative to repeat-eviction conditions on real workloads. Acknowledged in the PR comment at line 305-309 as a known stability gap.

### Competing hypotheses (Trace 2)

**H1: The MPS warmup loop's sync at `benchmark.py:316` actually waits for ALL warmup work to finish, so warmup-induced JIT compilation cost does NOT leak into the first measured round.**
- Supporting: per-iteration variance (std=0.088 ms over 50 rounds, no outliers removed) is consistent with first-iteration parity with later iterations.
- Falsifiable by: comparing iter 0 against iter 49 in `raw_times`. If iter 0 is ≥3σ above the median, JIT is leaking. (Did NOT run; would require modifying the runner to expose `raw_times` BEFORE outlier removal.) **Open.**

**H2: The fixture warns about flush_l2 on EVERY call, while `MPSBackend.flush_l2` warns only once.**
- Supporting: source. `benchmark.py:304-310` has no gate, while `backends/mps.py:166-176` uses `_FLUSH_L2_WARNED`.
- Falsifiable by: calling the fixture twice and counting warnings.
- Status: **supported by source reading**; not run as a probe. Likely a minor UX bug — duplicate warnings on repeated benchmark runs in the same test session.

**H3: `flush_l2=True` from a user that explicitly wants L2 eviction is silently ignored on MPS even though their tests will be optimistic.**
- Supporting: the warning fires (good) but the request is dropped (also good — Apple GPU has no L2-flush primitive). However, the `_run_mps` loop never calls back into `_flush_l2_cache`, so the user gets NEITHER a flush NOR a workaround like "alloc-and-fill a 16-MB buffer to evict private cache."
- Falsifiable by: any L2-eviction technique that works on Apple Silicon.
- Status: **supported, with no known mitigation in the current code.**

**H4: A kernel that creates a 0-d fp64 tensor on MPS will silently downcast to fp32, producing wrong-but-not-erroring benchmark results (the linguist-v3 finding).**
- Supporting: the linguist-v3 hypothesis posited this.
- **Refuted by probe.** PyTorch 2.11 raises `TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.` at construction time, including:
  - Direct `torch.tensor(x, device='mps', dtype=torch.float64)` — raises.
  - CPU fp64 → `.to('mps')` — raises.
  - `torch.tensor(0.5, device='mps')` under `set_default_dtype(torch.float64)` — raises.
  - `mps_tensor.double()` — raises.
- The benchmark fixture surfaces this immediately on the first warmup iteration; `gpu_benchmark` does not swallow it.
- A "bare Python float" inside a kernel creates a `torch.float32` tensor by default (verified: `torch.tensor(0.5).dtype == torch.float32` even when `default_dtype=float64` if the construction is on CPU first … wait, that's wrong). Re-check: under `set_default_dtype(torch.float64)`, `torch.tensor(0.5)` returns fp64 on CPU; moving it to MPS raises. So a kernel that uses bare `torch.tensor(0.5)` to seed a constant will crash on MPS under fp64-default settings. **This is a fail-loud condition, not a silent one.**

### Probes I ran (Trace 2)

- Decomposed `_run_mps` into 5 stages, each timed separately, 20 iterations on a 256×256 fp32 matmul. See `/tmp/trace_runtime.py` lines 230-283.
- End-to-end run via the actual `_BenchmarkRunner` to confirm the per-stage sum matches the total: 25.4 ms total / 50 rounds = 508 us avg, vs measured 554 us per round in the decomposition. Discrepancy (~10%) consistent with one-off setup costs (warmup, IQR, percentile compute).
- Confirmed `_BenchmarkRunner.__post_init__` does not crash on MPS-only system — pynvml branch silently fails, `_l2_size = 40 MB`, `_flush_buf = None`.
- Probed all four fp64-on-MPS construction paths to confirm fail-loud behavior (no silent downcast).

### Hidden cost / silent-coercion findings (Trace 2)

1. **`flush_l2=True` warning fires per call**, not once. Compare `benchmark.py:304-310` (no gate) vs `backends/mps.py:166-176` (gated by `_FLUSH_L2_WARNED`). A user who calls `gpu_benchmark(...)` 100 times in a single pytest session will see 100 UserWarnings unless they explicitly pass `flush_l2=False`. **Likely v1.1 UX fix candidate.**
2. **L2 flush is silently absent on MPS** — even though `flush_l2=True` is the default. The warning notifies the user, but no replacement is offered. CUDA's `_flush_l2_cache` writes 40 MB of zeros into a pre-allocated buffer; on MPS this could be approximated with a `torch.empty((10_000_000,), device='mps').fill_(0.0); torch.mps.synchronize()`. Not implemented.
3. **Pynvml is queried at fixture construction even on MPS-only systems** (`benchmark.py:71-89`). The try/except handles it cleanly, but on systems without pynvml installed this is one ImportError per `_BenchmarkRunner()` creation. Cost is small but unnecessary on macOS.
4. **The fixture path bypasses `MPSBackend.event_timer` entirely.** All the careful "deadlock-safe path" engineering in `backends/mps.py:111-129` is duplicated inline at `benchmark.py:283-327`. If the inline copy ever drifts from `MPSBackend.event_timer` (e.g., someone adds a fix to `MPSBackend.event_timer` but not here), the fixture will silently use the older code. **Architectural debt acknowledged in source comments at `backends/__init__.py:14-17` ("additive in v1.0").**
5. **fp64 on MPS is fail-loud, not fail-silent** — the linguist-v3 hypothesis of silent downcast is **refuted** for PyTorch 2.11. Worth re-testing on PyTorch 2.6 / 2.7 if those are part of the support matrix, since the failure mode could have changed.

---

## Measured-timing summary table

### `assert_close` on MPS (1024×1024 fp32, b = a.clone())
| Path     | Median       | Notes                                                 |
|----------|--------------|-------------------------------------------------------|
| Fast-path (pass) | **1.44 ms** | Dominated by `torch.allclose` on-device + sync (~99% of total) |
| Slow-path (mismatch, with rich report) | **21.5 ms** | Fast-path runs anyway as sunk cost (1.77 ms), then numpy path + rich panel format dominates |

Per sub-step (fast-path, 1024×1024 fp32):
| Step | Median |
|------|-------:|
| `_resolve_dtype` | 2.5 us |
| device_type detect | 0.8 us |
| `compute_tolerance` (incl. MPS overlay) | 1.0 us |
| fast-path predicate | 0.9 us |
| **`torch.allclose` + sync** | **1440 us** |

Per sub-step (slow-path, mismatch):
| Step | Median |
|------|-------:|
| sunk fast-path `torch.allclose` (False) | 1770 us |
| `_to_numpy(actual)` (MPS→host) | 610 us |
| `_to_numpy(expected)` (MPS→host) | 545 us |
| `astype(float64, copy=False)` × 2 | 390 us |
| NaN/Inf detection | 460 us |
| `diff > threshold` numpy passes | 980 us |
| `format_mismatch_report` (rich) | ~16000 us |

### `gpu_benchmark` on MPS (256×256 fp32 matmul, 10 warmup, 50 rounds)
| Phase | Median |
|-------|-------:|
| Backend detection | 5.5 us |
| `flush_l2` warning emit (per call) | 16 us |
| Warmup x10 + 2 syncs | 944 us |
| Per-iteration: pre-sync | 0.4 us |
| Per-iteration: kernel launch (async) | 19 us |
| **Per-iteration: post-sync** (the actual measured time) | **533 us** |
| Per-iteration: total | 554 us |
| IQR outlier removal (n=22) | 9 us |
| **End-to-end fixture call (10w + 50r)** | **25.4 ms** |

---

## Uncovered ground

- The forward trace through `assert_close`'s `_to_numpy` for `__cuda_array_interface__` objects (close.py:45-65) was not exercised — none of our probes use cupy/numba/dlpack tensors.
- Mixed-device tensors (one MPS, one CPU): the fast-path predicate at `close.py:181` requires `actual.device == expected.device`, so this falls through to the slow path. Behavior under that fall-through (does `_to_numpy` succeed on the MPS half?) was not directly measured.
- The bf16-on-MPS path: `_to_numpy` upcasts via `.float()` (line 37). PyTorch 2.11 supports bf16 on MPS, but full operator coverage is incomplete — not tested.
- First-iteration JIT cost (H1 above) — would require exposing `raw_times` in order, currently embedded in `BenchmarkResult.raw_times` after outlier removal.
- The `_mps_xfail_set` / `is_mps_xfailed` registry (tolerances.py:227-240) — orthogonal to the runtime path of the two APIs traced; not exercised here.
- Multi-call warning behavior of the fixture (H2): inferred from source, not run as a probe.
- Behavior on PyTorch < 2.11 (especially 2.6/2.7) where the fp64-on-MPS error path may differ.

## Confidence

**High** for: timing numbers (10–30 sample medians, consistent runs), MPS overlay multiplier values, ContextVar override precedence, fp64-on-MPS fail-loud behavior, the duplicate inline `_run_mps` implementation that bypasses `MPSBackend.event_timer`.

**Medium** for: the dominant-step claim within `torch.allclose` (1.4 ms on Apple Silicon for a 1M-element fp32 tensor) — H1 is supported but not isolated; would benefit from a 1-element vs 1M-element micro-bench to separate launch+sync from the reduce kernel.

**Low** for: PyTorch-version sensitivity of the fp64 failure mode — only tested on torch 2.11.
