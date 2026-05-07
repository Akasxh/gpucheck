# PLAN — gpucheck v1.0 Phase 2 (4-track parallel execution)

**Owner**: engineering-lead (adopted-persona mode)
**Charter**: `./CHARTER.md`
**Specialists**: planner + architect + skeptic (Phase A, this file); executor + verifier + reviewer per track (Phase B); adversary + evaluator (close).

---

## Track-A — `feat/track-a-mps` (HEADLINE)

### A.1 — Data model: extend `GPUInfo` for MPS

**Spec**: Add `backend: Literal["cuda", "mps"]` field with default `"cuda"` (back-compat). Add MPS-friendly fields: `mps_chip: str | None` (e.g., "Apple M4 Pro"), `mps_unified_memory_mb: int | None` (mutates `memory_total_mb` for MPS — system-wide unified memory). Keep all existing CUDA fields (`compute_capability`, `tensor_core_generation`, `cuda_version`) as `tuple[int, int] | None` / `int | None` / `str` so MPS tests can populate `(0, 0)` / `None` / `""`.

**Blast radius**: `arch/detection.py`, `arch/__init__.py`, `arch/compatibility.py` (require_arch needs to accept "Apple-Silicon"), `arch/tensor_cores.py` (Apple Silicon has no tensor cores; supports_tensor_cores returns False).

**Rollback**: revert by deleting `backend` field.

**Acceptance**: `GPUInfo` is back-compat (existing CUDA tests still pass), and `GPUInfo(backend="mps", ...)` constructs cleanly.

### A.2 — Backend Protocol + CUDA implementation extracted

**Spec**: New module `src/gpucheck/backends/__init__.py` defines `Backend(Protocol)` with methods: `name() -> str`, `is_available() -> bool`, `device_count() -> int`, `synchronize(device_id: int) -> None`, `event_timer() -> EventTimer` (a context-manager-like protocol returning elapsed_ms), `mem_stats(device_id: int) -> dict`, `flush_l2(device_id: int, buf: Any | None = None) -> None`, `arch_info(device_id: int) -> GPUInfo`. Top-level `available_backends()` returns the priority list (cuda before mps).

**`backends/cuda.py`** moves the CUDA-only logic from `fixtures/benchmark.py:_flush_l2_cache`, `fixtures/profiler.py:_snapshot_pynvml`/`_snapshot_torch`, `arch/detection.py:_detect_via_pynvml`/`_detect_via_torch` into a single module that conforms to the Protocol. Existing call sites stay (they still use `torch.cuda.*` directly); the Protocol is **additive** for v1.0 — call sites switch to it incrementally.

**`backends/mps.py`** implements the same Protocol against `torch.mps.*`:
- `synchronize`: `torch.mps.synchronize()` (device-level, NOT per-event — pytorch#162872 deadlock).
- `event_timer`: yields a context manager that internally calls `torch.mps.synchronize()` before/after, computes wall-clock delta. We avoid `torch.mps.event.Event` until #162872 closes.
- `mem_stats`: `torch.mps.current_allocated_memory()` + `torch.mps.driver_allocated_memory()`. Returns `{"used": ..., "driver_allocated": ..., "rss": <psutil if avail>}`. SYNTHESIS §3 notes both queries lag Activity Monitor; psutil RSS is the leak proxy.
- `flush_l2`: no-op on MPS (Apple GPUs don't expose L2 flush). Emit a one-time UserWarning the first time a benchmark requests `flush_l2=True` on MPS.
- `arch_info`: best-effort. Reads `platform.mac_ver`, `subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"])` for chip name (M1/M2/M3/M4/M5). NO `xcrun metal` invocation (N1 waiver). NO mach `task_info` (N3 waiver).

**Blast radius**: net new `backends/`, no existing call site changes in this commit (ADDITIVE).

**Acceptance**: `from gpucheck.backends import available_backends, get_backend; b = get_backend("mps"); b.is_available()` works on MPS-capable machines.

### A.3 — `@devices("mps")` and `@devices("all")` accept MPS

**Spec**: Update `decorators/devices.py:_detect_cuda_devices` → `_detect_devices(types=...)` returns a unified list. `_is_device_available("mps")` checks `torch.backends.mps.is_available()`. `devices("all")` returns `["cuda:0", ..., "mps"]` if both available, just `["mps"]` on Apple Silicon.

**Blast radius**: `decorators/devices.py`, `decorators/parametrize.py` (its `_detect_cuda_devices` import becomes the new `_detect_devices`), tests.

**Rollback**: revert single function.

**Acceptance**: `@devices("mps")` parametrizes a test that runs on MPS-available machine, skips on non-MPS.

### A.4 — `gpu_benchmark` MPS path (deadlock-safe)

**Spec**: In `fixtures/benchmark.py:_BenchmarkRunner.__call__`, branch on tensor device when present, OR a new `device` kwarg. Default detection: if `torch.cuda.is_available()` use CUDA path; else if `torch.backends.mps.is_available()` use MPS path; else skip with "no GPU".

MPS path:
```
torch.mps.synchronize()
for _ in range(warmup): fn(*args, **kwargs)
torch.mps.synchronize()
times = []
for _ in range(rounds):
    if flush_l2: pass  # no-op + one-time warning
    t0 = time.perf_counter()
    fn(*args, **kwargs)
    torch.mps.synchronize()  # device-level — NOT per-event
    times.append((time.perf_counter() - t0) * 1000.0)
```

**The CRITICAL anti-pattern** (DO NOT use): `start.record(); end.record(); end.synchronize(); start.elapsed_time(end)` — pytorch#162872 deadlock.

**Blast radius**: `fixtures/benchmark.py` (substantial new branch).

**Rollback**: feature-flag the MPS branch; revert by removing.

**Acceptance**: a test calling `gpu_benchmark(fn, mps_tensor)` on MPS hardware does NOT hang. (Cannot be tested in CI without MPS hardware, but on this machine we run an integration smoke test.)

### A.5 — `assert_close` GPU fast-path widened

**Spec**: `assertions/close.py:163-176` currently checks `actual.device.type == "cuda"`. Widen to `actual.device.type in ("cuda", "mps")` so MPS tensors don't hit the CPU transfer path unnecessarily. The `torch.allclose` call is device-agnostic.

**Blast radius**: 1-line change in `close.py`.

**Acceptance**: `assert_close(mps_tensor_a, mps_tensor_b)` returns without `.cpu()` transfer when within tolerance.

### A.6 — MPS tolerance overlay (PROVISIONAL 2x)

**Spec**: New module-level dict in `assertions/tolerances.py`: `_MPS_TOLERANCE_MULTIPLIERS: dict[str, float] = {"float32": 2.0, "float16": 2.0, "bfloat16": 2.0, "float64": 1.0}`. Add new function `compute_tolerance(dtype, *, k_dim=None, device_type=None)`. When `device_type == "mps"`, multiply atol/rtol by the dtype-specific multiplier.

**MARK PROVISIONAL** in code comment + README MPS section (Phase 3 docs).

**Blast radius**: `assertions/tolerances.py`, `assertions/close.py` (passes `device_type` from tensor.device.type), tests.

**Acceptance**: `assert_close(a, b)` on MPS uses 2x atol vs CUDA without caller intervention.

### A.7 — `[tool.gpucheck.mps.xfail]` config block

**Spec**: Add a `[tool.gpucheck.mps.xfail]` section to `pyproject.toml` with the 12 entries from SYNTHESIS §7. Plumb a loader into `assertions/tolerances.py` (sister to `tolerances_from_config`) and a registry that test code can query: `gpucheck.is_mps_xfailed("scaled_dot_product_attention.large") -> bool`.

**Blast radius**: `pyproject.toml` (config), `assertions/tolerances.py` (loader), new helper.

**Acceptance**: a test can `if gpucheck.is_mps_xfailed("conv2d.large_channels"): pytest.xfail(...)` and the xfail list is the living document.

### A.8 — `pyproject.toml` extras `[mps]` and `[apple]`

**Spec**: Add `mps = ["torch>=2.6"]` (floor where `torch.mps.synchronize()` is stable). Add `apple = ["gpucheck[mps]"]` as an alias.

### A.9 — Tests

- `tests/test_backends.py` — backend Protocol contract tests (CUDA + MPS): `test_mps_backend_synchronize_does_not_call_per_event_synchronize`, `test_mps_backend_arch_info_returns_apple_silicon`, `test_mps_backend_flush_l2_is_noop_with_warning`.
- `tests/test_devices_mps.py` — `@devices("mps")` parametrizes correctly; skips when MPS unavailable.
- `tests/test_assert_close_mps.py` — fast-path on MPS; tolerance overlay 2x correctly applied.
- `tests/test_mps_xfail.py` — config loader returns the 12 expected entries.
- `tests/test_gpu_benchmark_mps.py` — uses device-level sync (introspection or smoke test on MPS HW).

---

## Track-B — `feat/track-b-strides`

### B.1 — Public API: `fuzz_strides`

**Spec**: New `src/gpucheck/fuzzing/strides.py` with deterministic `fuzz_strides(shape, dtype, *, n=20, seed=None) -> list[tuple[str, Tensor]]`.

7 categories returned in priority order:
1. **row-major (contiguous)** — `torch.empty(shape).contiguous()`. Baseline.
2. **column-major** — `torch.empty(shape[::-1]).contiguous().T` for 2D, transpose-last-two for >2D.
3. **broadcast-induced** — `torch.empty(shape[:-1]).unsqueeze(-1).expand(shape)`. Stride 0 in expanded dim.
4. **transpose** — `torch.empty(transposed_shape).contiguous().transpose(0, 1)`. Non-trivial strides.
5. **slice (every-other)** — `torch.empty(2*shape).contiguous()[::2, ::2]`. Strided view.
6. **contiguous-after-clone vs not** — return both `.clone().contiguous()` AND the non-contig view, so a single test can compare.
7. **gather-induced** — `src.contiguous(); idx = torch.randperm(numel).reshape(shape); src.flatten()[idx]` — irregular access.

Return `(label, tensor)` pairs.

**Blast radius**: net-new file.

### B.2 — Hypothesis: `StrideStrategy`

**Spec**: Sister of `ShapeStrategy`. Returns a `SearchStrategy[Tensor]` that draws shape, dtype, then category, then constructs the tensor.

### B.3 — Wire into `parametrize_gpu`

**Spec**: Optional `stride_categories: Sequence[str] | None = None` kwarg. When set, parametrizes over the cartesian: dtype × shape × device × stride_category. Each test sees an additional `stride_category: str` parameter. Test must call `fuzz_strides_for_category(shape, dtype, category, device)` to get a tensor.

### B.4 — Tests

- `tests/test_fuzz_strides.py` — each category returns ≥1 tensor of correct shape; non-contig tensors are non-contig.
- `tests/test_fuzz_strides_hypothesis.py` — `StrideStrategy` shrinks to row-major.
- `tests/test_parametrize_gpu_strides.py` — stride_categories parametrization wires in.

---

## Track-C — `feat/track-c-thread-safety`

### C.1 — Convert override stack to `ContextVar`

**Spec**: In `assertions/tolerances.py:28`, replace `_tolerance_overrides: list[tuple[float, float]] = []` with `_tolerance_overrides: ContextVar[tuple[tuple[float, float], ...]] = ContextVar("_tolerance_overrides", default=())`.

`tolerance_context(atol, rtol)`:
```
@contextmanager
def tolerance_context(atol, rtol):
    current = _tolerance_overrides.get()
    token = _tolerance_overrides.set(current + ((atol, rtol),))
    try:
        yield
    finally:
        _tolerance_overrides.reset(token)
```

`compute_tolerance` reads `_tolerance_overrides.get()[-1]` if non-empty.

### C.2 — Failing-without-fix regression test

**Spec**: New `tests/test_tolerance_thread_safety.py` (<100 LOC). Spawn 4 threads via `concurrent.futures.ThreadPoolExecutor`. Each thread enters a `tolerance_context(...)` with thread-distinct values, then reads `compute_tolerance(torch.float32)` and asserts it sees its OWN context's overrides. Without the ContextVar fix, threads will observe each other's overrides (race on the mutable list).

### C.3 — Mitigate TM-E1 (`sanitizers/race.py:50-62`)

**Spec**:

```python
_CUDA_HOME_ALLOWLIST: tuple[str, ...] = (
    "/usr/local/cuda",
    "/opt/nvidia/cuda",
    "/opt/cuda",
)


def _find_compute_sanitizer() -> str | None:
    path = shutil.which("compute-sanitizer")
    if path:
        return path
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH", "")
    if not cuda_home:
        return None
    real = os.path.realpath(cuda_home)
    if not any(real == prefix or real.startswith(prefix + os.sep) for prefix in _CUDA_HOME_ALLOWLIST):
        warnings.warn(
            f"CUDA_HOME={cuda_home!r} resolves outside allowlist; ignoring",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    candidate = os.path.join(real, "bin", "compute-sanitizer")
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate
    return None
```

### C.4 — Tests for TM-E1

**Spec**: `tests/test_race_cuda_home_allowlist.py` — temp dir outside allowlist with a fake `compute-sanitizer` binary; assert `_find_compute_sanitizer()` returns None and emits a UserWarning. Allowlisted path returns the binary.

---

## Track-D — `feat/track-d-bundle`

### D.1 — Reporting test coverage 0% → ≥90%

**Spec**: New tests under `tests/test_reporting_*.py` covering:
- `tests/test_reporting_console.py`: `ConsoleReporter` constructs from a `StringIO`; `gpu_info_panel`, `test_summary`, `benchmark_table`, `memory_summary`, `error_detail` produce non-empty output with expected substrings (`PASS`/`FAIL`/`SKIP`).
- `tests/test_reporting_json.py`: `JSONReporter().add_test_result(...).flush()` writes a JSON file with the expected schema; `compare_runs` correctly classifies regression / new / removed / ok.
- `tests/test_reporting_ci.py`: `emit_github_annotations` writes correctly-formatted lines to stdout when `GITHUB_ACTIONS=1`; `write_junit_xml` produces a valid XML with the expected counts; `generate_pr_comment` produces Markdown with regression / ok rows.

Use `pytest-cov` measurement to verify ≥ 90% on `src/gpucheck/reporting/*.py`.

### D.2 — `reporting/html.py` static dashboard

**Spec**: New `src/gpucheck/reporting/html.py` exposes `HTMLReporter(json_path).render(out_path)`. Generates a single self-contained HTML file (no external CSS/JS) with:
- Summary header (counts, pass-rate, GPU info).
- Per-test pass/fail table (sortable via `<details>` per test).
- Per-kernel benchmark table (with simple inline SVG bar chart for median timings).
- Per-shape error histogram if `error_data` keys present in JSON (skipped gracefully if absent).
- Regression highlight band (color rows red if `compare_runs` was performed).

Pure Python, no Jinja, no D3 — just f-strings building HTML and inline SVG. (Time budget excludes Playwright E2E, defer to v1.1; we'll write basic HTML-validity tests instead.)

### D.3 — `sanitizers/determinism.py`

**Spec**: New `src/gpucheck/sanitizers/determinism.py`:
- `assert_deterministic(fn: Callable, *, n: int = 3, seed: int = 0, ..., **kwargs) -> None`: fixes seeds via `torch.manual_seed(seed)` and (if mps) `torch.mps.manual_seed(seed)`, runs `fn(*args, **kwargs)` n times, asserts all outputs are byte-identical. On MPS, **per SYNTHESIS §4**, MPS is best-effort deterministic and may legitimately fail; surface the failure with a structured error.
- `@requires_determinism` decorator that wraps a test with `assert_deterministic` semantics on the test's return value.

### D.4 — Mitigate DEP-1 (commit `uv.lock`)

**Spec**: `uv.lock` is currently untracked. The Track-D branch's commit will `git add uv.lock` and update `.github/workflows/ci.yml` to use `uv sync --frozen` (or `uv pip install --no-deps` after lock parsing). Confirm `uv.lock` reflects current dependency set.

### D.5 — Mitigate CFG-2 (`.github/workflows/ci.yml`)

**Spec**: Add to top-level workflow:
```yaml
permissions:
  contents: read
```

Single-line block. Lives in Track-D.

---

## Dependency graph

- Tracks A, B, C, D are file-disjoint EXCEPT:
  - A and C both touch `assertions/tolerances.py`. Risk: C's ContextVar refactor changes `_tolerance_overrides`; A adds device_type kwarg + `_MPS_TOLERANCE_MULTIPLIERS`. **Mitigation**: A and C touch different lines/symbols. A adds new function signature + helpers; C only changes the override-stack mechanism. Phase 3 merge order C → A avoids conflict.
  - A and D both touch `pyproject.toml` (A adds `[mps]` extra + `[tool.gpucheck.mps.xfail]`; D doesn't add `pyproject.toml` changes — its CI yml change is separate). No conflict expected.
- Phase 3 merge order recommendation: **C → A → B → D**. C is smallest and least surprising, lands first; A is the headline; B is independent; D is last (it adds the biggest test surface so verifies everything else).

## Termination caps

Per LEAD spec: 2.5h wall-clock. Per-track soft cap: 2 × task_count iterations (≈14-18 inner iterations); hard cap 5 × task_count (≈35-45). Token budget 500K. If a track stalls, mark INCOMPLETE in evaluator.md with specific blocker.

## Verdict

PLAN ready. Skeptic gate next.
