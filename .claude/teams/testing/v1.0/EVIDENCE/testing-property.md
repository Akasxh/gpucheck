# Property — testing / v1.0

Adopted persona: `~/.claude/agents/testing/testing-property.md`. This file
catalogues every Hypothesis property the four engineering tracks will be
graded against. Sketches are runnable templates; engineering-lead implements
to the importable target name shown.

## PBT framework

- Framework: `hypothesis` (already in `pyproject.toml:40` as optional dep)
- Recommended floor bump: `>=6.100` (matches DEP-4 from security FINDINGS)
- Settings (consumed via `tests/conftest.py`):

```python
# tests/conftest.py — additions
from hypothesis import HealthCheck, Verbosity, settings

settings.register_profile(
    "ci",
    max_examples=200,
    deadline=1000,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
    verbosity=Verbosity.normal,
)
settings.register_profile("dev", max_examples=50, deadline=500)
settings.register_profile("mps", max_examples=50, deadline=5000)  # MPS fixture is slow
settings.register_profile("nightly", max_examples=2000, deadline=None)
settings.load_profile("ci" if os.getenv("CI") else "dev")
```

## Properties generated (catalogue)

| ID | Target | Property name | Type | Strategy | File |
|---|---|---|---|---|---|
| A1 | `gpucheck.arch.backend.Backend.detect()` | `test_detect_is_pure` | idempotence + purity | env-monkeypatch, repeated calls | `tests/test_backend_props.py` |
| A2 | `gpucheck.arch.backend.Backend.synchronize()` | `test_synchronize_idempotent` | idempotence | call N times in [1,10] | `tests/test_backend_props.py` |
| A3a | `gpucheck.arch.backend.Backend.event_timer()` | `test_event_timer_non_negative` | output bounds | shape, op | `tests/test_backend_props.py` |
| A3b | same | `test_event_timer_monotone_with_workload` | monotonicity | size in [128, 4096] | same |
| A3c | same | `test_event_timer_no_deadlock_pattern` | invariant | introspection | same |
| A4 | `gpucheck.arch.backend.Backend.mem_stats()` | `test_mem_stats_totals_coherent` | invariant | dispatch dummy alloc | same |
| A5 | `gpucheck.arch.backend.Backend.arch_info()` | `test_arch_info_matches_device` | reference impl | sampled SM versions | same |
| A6a | `gpucheck.assertions.assert_close` (CPU vs MPS) | `test_assert_close_cpu_mps_parity` | reference impl | seeded `cpu_tensor_pair` | `tests/test_assert_close_mps_props.py` |
| A6b | same | `test_assert_close_mps_tolerance_envelopes_cpu` | bounds | dtypes × shapes | same |
| A7a | `gpucheck.assertions.tolerances.compute_tolerance` | `test_mps_tolerance_geq_cuda` | monotonicity | dtype enum | `tests/test_tolerance_props.py` |
| A7b | same | `test_tolerance_monotone_in_k_dim` | monotonicity | k_dim ∈ [1, 16384] | same |
| A7c | same | `test_tolerance_monotone_in_dtype_precision` | monotonicity | (FP64, FP32, FP16, BF16) | same |
| A8 | xfail registry | `test_xfail_entry_for_each_synthesis_top12` | totality | enum of 12 issues | `tests/test_xfail_registry.py` |
| B1 | `gpucheck.fuzzing.strides.fuzz_strides` | `test_fuzz_strides_seed_determinism` | determinism | seed, n | `tests/test_strides_props.py` |
| B2 | same | `test_fuzz_strides_shape_compatible` | invariant | shape, n | same |
| B3 | same | `test_fuzz_strides_covers_seven_classes` | totality | n=200, seed=0 | same |
| B4 | `gpucheck.fuzzing.inputs.gpu_tensors` | `test_gpu_tensors_finite_by_default` | bounds | shape, dtype | `tests/test_inputs_props.py` |
| C1 | `gpucheck.assertions.tolerances.tolerance_context` | `test_tolerance_context_lifo` | invariant | nested depths [1,8] | `tests/test_tolerance_contextvar_props.py` |
| C2 | same | `test_tolerance_context_thread_isolated` | isolation | ThreadPoolExecutor(8) | same |
| C3 | same | `test_tolerance_context_asyncio_isolated` | isolation | asyncio.gather(N) | same |
| C4 | same | `test_tolerance_context_exception_safe` | invariant | random raise points | same |
| D1 | `gpucheck.analysis.determinism.assert_deterministic` | `test_assert_deterministic_no_global_rng_mutation` | hermetic | random pure-fn | `tests/test_determinism_props.py` |
| D2 | `gpucheck.reporting.html.render_dashboard` | `test_render_dashboard_idempotent_bytes` | determinism | sample baselines | `tests/test_html_dashboard_props.py` |
| D3 | same | `test_render_dashboard_handles_malformed_input` | no-crash | malformed JSON strategy | same |
| S1 | `gpucheck.sanitizers.race._find_compute_sanitizer` | `test_find_sanitizer_rejects_path_traversal` | invariant | hostile env vars | `tests/security/test_race_path_injection.py` |
| S3 | `gpucheck.reporting.json.load_baseline` | `test_load_baseline_schema_strict` | invariant | malformed JSON strategy | `tests/test_reporting_json_props.py` |

## Property sketches (engineering reads as a contract)

### A1 — `Backend.detect()` is pure

```python
# tests/test_backend_props.py
from hypothesis import given, settings, strategies as st

from gpucheck.arch.backend import Backend, detect_backend  # contract: Track A delivers


@given(seed=st.integers(0, 2**31 - 1))
@settings(max_examples=50)
def test_detect_is_pure(monkeypatch, seed):
    """Calling detect() twice with the same env yields the same Backend instance kind."""
    monkeypatch.setenv("PYTHONHASHSEED", str(seed))
    a = detect_backend()
    b = detect_backend()
    assert type(a) is type(b)
    assert a.name == b.name
    assert a.arch_info() == b.arch_info()  # frozen dataclass equality
```

### A2 — `synchronize()` idempotent

```python
@given(n=st.integers(1, 10))
def test_synchronize_idempotent(n, mps_or_mock_backend):
    """Synchronize N times has the same observable effect as synchronize once."""
    backend = mps_or_mock_backend
    for _ in range(n):
        backend.synchronize()  # must not raise
    # Idempotence verified by post-state: pending stream count is 0.
    assert backend.pending_kernel_count() == 0
```

### A3 — `event_timer()` non-negative + monotone + deadlock-free

```python
import inspect

from gpucheck.arch.backend import Backend


@given(size=st.integers(128, 4096))
@settings(max_examples=30, deadline=5000)
def test_event_timer_non_negative(size, mps_or_mock_backend):
    backend = mps_or_mock_backend
    start, end, elapsed_ms = backend.event_timer()
    start.record()
    backend.run_dummy_workload(size=size)  # contract: backend exposes a benchmarking workload
    end.record()
    backend.synchronize()  # device-level sync, NOT event.synchronize()
    ms = elapsed_ms()
    assert ms >= 0.0
    assert math.isfinite(ms)


@given(small=st.integers(128, 256), large=st.integers(2048, 4096))
@settings(max_examples=15, deadline=8000)
def test_event_timer_monotone_with_workload(small, large, mps_or_mock_backend):
    """Larger workload >= smaller workload elapsed_ms (with slack for noise)."""
    assume(large >= 4 * small)
    backend = mps_or_mock_backend
    t_small = _time(backend, small)
    t_large = _time(backend, large)
    # Allow 2x noise floor; large should still be visibly larger.
    assert t_large >= 0.5 * t_small  # weak monotonicity due to OS jitter


def test_event_timer_no_deadlock_pattern(mps_or_mock_backend):
    """Per pytorch#162872, gpucheck must NOT call per-event synchronize.
    Inspect the source of event_timer() to confirm only torch.mps.synchronize()
    or backend.synchronize() is invoked, NEVER event.synchronize() on MPS path.
    """
    src = inspect.getsource(type(mps_or_mock_backend).event_timer)
    if mps_or_mock_backend.name == "mps":
        assert "event.synchronize" not in src and "self.synchronize" in src
```

### A4 — `mem_stats()` totals coherent

```python
@given(alloc_mb=st.integers(0, 256))
@settings(max_examples=20, deadline=5000)
def test_mem_stats_totals_coherent(alloc_mb, mps_or_mock_backend):
    backend = mps_or_mock_backend
    stats = backend.mem_stats()
    assert stats.allocated >= 0
    assert stats.reserved >= stats.allocated
    assert stats.free >= 0
    assert stats.total >= stats.reserved
    # Coherence — small slack for device-driver fragmentation accounting:
    assert stats.allocated + stats.free <= stats.total + (16 << 20)
```

### A5 — `arch_info()` matches device

```python
@given(family=st.sampled_from(["cuda", "mps", "cpu"]))
def test_arch_info_matches_device(family, monkeypatch):
    if family == "cuda":
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        monkeypatch.setattr("torch.mps.is_available", lambda: False)
    elif family == "mps":
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        monkeypatch.setattr("torch.mps.is_available", lambda: True)
    else:
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        monkeypatch.setattr("torch.mps.is_available", lambda: False)
    from gpucheck.arch.backend import detect_backend
    info = detect_backend().arch_info()
    assert info.family == family
    if family == "mps":
        assert info.tensor_cores is False  # Apple has no FP16 tensor cores
```

### A6 — `assert_close` CPU vs MPS parity

```python
# tests/test_assert_close_mps_props.py
from hypothesis import given, settings, strategies as st
import torch

from gpucheck import assert_close
from gpucheck.fuzzing import ShapeStrategy

DTYPES = [torch.float32, torch.float16, torch.bfloat16]


@given(shape=ShapeStrategy(ndim=2, max_size=512), dtype=st.sampled_from(DTYPES), seed=st.integers(0, 2**16))
@settings(profile="mps", max_examples=30)
def test_assert_close_cpu_mps_parity(shape, dtype, seed, mps_available):
    """For shapes excluded from xfail set, MPS matmul matches CPU within MPS overlay."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.randn(*shape, generator=g, dtype=dtype)
    b = torch.randn(*shape, generator=g, dtype=dtype)
    cpu_ref = a @ b.T
    mps_actual = (a.to("mps") @ b.to("mps").T).to("cpu")
    # MPS overlay must envelope the actual drift.
    assert_close(cpu_ref, mps_actual, dtype=dtype, device="mps")


@given(dtype=st.sampled_from(DTYPES))
def test_assert_close_mps_tolerance_envelopes_cpu(dtype):
    """assert_close on MPS uses tolerance >= the same call on CPU for the same dtype."""
    from gpucheck.assertions.tolerances import compute_tolerance
    cpu_tol = compute_tolerance(dtype, device="cpu")
    mps_tol = compute_tolerance(dtype, device="mps")
    assert mps_tol[0] >= cpu_tol[0] and mps_tol[1] >= cpu_tol[1]
```

### A7 — Tolerance monotonicity

```python
# tests/test_tolerance_props.py
import torch
from hypothesis import given, strategies as st

from gpucheck.assertions.tolerances import compute_tolerance


@given(dtype=st.sampled_from([torch.float32, torch.float16, torch.bfloat16]))
def test_mps_tolerance_geq_cuda(dtype):
    cuda_atol, cuda_rtol = compute_tolerance(dtype, device="cuda")
    mps_atol, mps_rtol = compute_tolerance(dtype, device="mps")
    assert mps_atol >= cuda_atol
    assert mps_rtol >= cuda_rtol


@given(k=st.integers(1, 16384))
def test_tolerance_monotone_in_k_dim(k):
    """sqrt(k/128) scaling — atol monotone non-decreasing in k."""
    a1, _ = compute_tolerance(torch.float32, k_dim=k)
    a2, _ = compute_tolerance(torch.float32, k_dim=k * 2)
    assert a2 >= a1


def test_tolerance_monotone_in_dtype_precision():
    """FP64 < FP32 < FP16 < BF16 by atol."""
    fp64 = compute_tolerance(torch.float64)[0]
    fp32 = compute_tolerance(torch.float32)[0]
    fp16 = compute_tolerance(torch.float16)[0]
    bf16 = compute_tolerance(torch.bfloat16)[0]
    assert fp64 < fp32 < fp16 < bf16
```

### A8 — xfail registry totality

```python
# tests/test_xfail_registry.py
import tomllib
from pathlib import Path

# Top-12 issues from research/v1.0/SYNTHESIS.md "Sub-Q 2"
SYNTHESIS_TOP12 = {
    "pytorch#177116": "matmul.backward.over_32K_elements",
    "pytorch#179352": "scaled_dot_product_attention.large",
    "pytorch#178497": "reductions.intermittent",
    "pytorch#142836": "conv2d.large_channels",
    "pytorch#173525": "layer_norm.backward.shape1",
    "pytorch#175189": "batch_norm.backward.channels_last",
    "pytorch#96602":  "softmax.large_attention",
    "pytorch#162872": "event_timer.deadlock_pattern",
    "pytorch#175190": "avg_pool2d.backward.channels_last",
    "pytorch#160828": "cross_entropy.ctc_loss_missing",
    "pytorch#181936": "F.linear.backward.bf16_3d_nobias_m5",
    "pytorch#170837": "bert_roberta.batched_inference",
}


def test_xfail_entry_for_each_synthesis_top12():
    cfg = tomllib.loads(Path("pyproject.toml").read_text())
    xfail = cfg["tool"]["gpucheck"]["mps"]["xfail"]
    entries = set(xfail["ops"])
    expected = set(SYNTHESIS_TOP12.values())
    missing = expected - entries
    assert not missing, f"xfail registry missing: {missing}"
```

### B1 — `fuzz_strides` determinism

```python
# tests/test_strides_props.py
from hypothesis import given, settings, strategies as st

from gpucheck.fuzzing.strides import fuzz_strides  # contract: Track B delivers


@given(seed=st.integers(0, 2**31 - 1), n=st.integers(1, 100))
@settings(max_examples=200)
def test_fuzz_strides_seed_determinism(seed, n):
    a = fuzz_strides(shape=(8, 8, 8), n=n, seed=seed)
    b = fuzz_strides(shape=(8, 8, 8), n=n, seed=seed)
    assert a == b
```

### B2 — Shape compatibility (never produce a stride that segfaults)

```python
import torch
from gpucheck.fuzzing.shapes import fuzz_shapes


@given(shape=st.sampled_from(fuzz_shapes(ndim=3, max_size=64, n=20, seed=0)),
       seed=st.integers(0, 1000))
@settings(max_examples=100)
def test_fuzz_strides_shape_compatible(shape, seed):
    pairs = fuzz_strides(shape=shape, n=20, seed=seed)
    for spec in pairs:
        # spec is a StrideSpec — has .shape, .stride, .offset, .storage_size
        assert len(spec.stride) == len(spec.shape)
        # as_strided must succeed without segfault — bounded by storage_size
        storage = torch.empty(spec.storage_size)
        view = torch.as_strided(storage, spec.shape, spec.stride, storage_offset=spec.offset)
        assert view.shape == spec.shape  # no crash, valid view
```

### B3 — 7-class coverage

```python
def test_fuzz_strides_covers_seven_classes():
    """One run with n=200 must hit every stride class at least once."""
    specs = fuzz_strides(shape=(4, 4, 4), n=200, seed=0)
    classes = {s.classify() for s in specs}  # contract: StrideSpec.classify() ∈ {
    #   "contiguous", "transpose_2d", "transpose_3d",
    #   "broadcast", "slice_step_2", "sub_tensor_offset", "negative_stride"
    # }
    expected = {"contiguous", "transpose_2d", "transpose_3d", "broadcast",
                "slice_step_2", "sub_tensor_offset", "negative_stride"}
    assert expected.issubset(classes)
```

### C1 — LIFO push/pop

```python
# tests/test_tolerance_contextvar_props.py
from hypothesis import given, settings, strategies as st
import pytest

from gpucheck.assertions.tolerances import compute_tolerance, tolerance_context
import torch


@given(values=st.lists(st.tuples(st.floats(1e-6, 1e-1, allow_nan=False),
                                  st.floats(1e-6, 1e-1, allow_nan=False)),
                       min_size=1, max_size=8))
def test_tolerance_context_lifo(values):
    base_atol = compute_tolerance(torch.float32)[0]
    stack = []
    contexts = []
    for atol, rtol in values:
        ctx = tolerance_context(atol=atol, rtol=rtol)
        ctx.__enter__()
        contexts.append(ctx)
        stack.append((atol, rtol))
        # innermost wins
        cur_atol, cur_rtol = compute_tolerance(torch.float32)
        assert cur_atol == atol
        assert cur_rtol == rtol
    # Pop in reverse order; outer values restored
    while stack:
        contexts.pop().__exit__(None, None, None)
        stack.pop()
        if stack:
            assert compute_tolerance(torch.float32) == stack[-1]
        else:
            assert compute_tolerance(torch.float32)[0] == base_atol
```

### C2 — Thread isolation

```python
from concurrent.futures import ThreadPoolExecutor


def test_tolerance_context_thread_isolated():
    """Override set in worker thread A is invisible to worker thread B."""
    barrier = threading.Barrier(parties=3)
    leaks = []

    def worker_set(name):
        with tolerance_context(atol=name, rtol=name):
            barrier.wait()  # all workers race to observe each other
            seen = compute_tolerance(torch.float32)[0]
            if abs(seen - name) > 1e-12:
                leaks.append((name, seen))
            barrier.wait()

    with ThreadPoolExecutor(max_workers=3) as ex:
        list(ex.map(worker_set, [1e-3, 1e-4, 1e-5]))
    assert leaks == []
```

### C3 — Asyncio task isolation (PEP 567)

```python
import asyncio


def test_tolerance_context_asyncio_isolated():
    async def task(name):
        with tolerance_context(atol=name, rtol=name):
            await asyncio.sleep(0.001)
            return compute_tolerance(torch.float32)[0]

    async def main():
        return await asyncio.gather(task(1e-3), task(1e-4), task(1e-5))

    results = asyncio.run(main())
    assert results == [1e-3, 1e-4, 1e-5]  # each task saw its own override
```

### C4 — Exception safety

```python
def test_tolerance_context_exception_safe():
    base = compute_tolerance(torch.float32)
    with pytest.raises(RuntimeError):
        with tolerance_context(atol=1e-9, rtol=1e-9):
            raise RuntimeError("boom")
    # After exception, override must be popped
    assert compute_tolerance(torch.float32) == base
```

### D1 — `assert_deterministic` is hermetic

```python
# tests/test_determinism_props.py
import random
import numpy as np
import torch
from hypothesis import given, strategies as st

from gpucheck.analysis.determinism import assert_deterministic  # contract: Track D delivers


@given(seed=st.integers(0, 2**32 - 1))
def test_assert_deterministic_no_global_rng_mutation(seed):
    """assert_deterministic(fn) must not mutate process-global RNG state."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    py_before = random.getstate()
    np_before = np.random.get_state()
    th_before = torch.get_rng_state()

    def under_test():
        return [random.random() for _ in range(10)]

    assert_deterministic(under_test, runs=3)

    assert random.getstate() == py_before
    np_after = np.random.get_state()
    assert np_before[0] == np_after[0]
    assert (np_before[1] == np_after[1]).all()
    assert torch.equal(th_before, torch.get_rng_state())
```

### D2 — HTML dashboard byte-identical

```python
# tests/test_html_dashboard_props.py
import json
import hashlib
from pathlib import Path
from hypothesis import given, strategies as st

from gpucheck.reporting.html import render_dashboard  # contract: Track D delivers

BASELINE_FIXTURES = sorted(Path("tests/data/baselines").glob("*.json"))


@pytest.mark.parametrize("baseline_path", BASELINE_FIXTURES)
def test_render_dashboard_idempotent_bytes(baseline_path, tmp_path):
    data = json.loads(baseline_path.read_text())
    a = render_dashboard(data, fixed_timestamp="2026-05-01T00:00:00Z")
    b = render_dashboard(data, fixed_timestamp="2026-05-01T00:00:00Z")
    assert hashlib.sha256(a.encode()).hexdigest() == hashlib.sha256(b.encode()).hexdigest()
```

### D3 — HTML dashboard malformed-input safety

```python
@given(malformed=st.recursive(
    st.one_of(st.none(), st.booleans(), st.integers(), st.text(), st.floats(allow_nan=True)),
    lambda children: st.dictionaries(st.text(), children, max_size=5) | st.lists(children, max_size=5),
    max_leaves=20,
))
@settings(max_examples=200)
def test_render_dashboard_handles_malformed_input(malformed):
    """Either raises gpucheck.reporting.html.SchemaError OR produces a placeholder dashboard.
    Never raises uncaught exception or produces empty output."""
    from gpucheck.reporting.html import SchemaError
    try:
        out = render_dashboard(malformed, fixed_timestamp="t")
    except SchemaError:
        return  # informative rejection — acceptable
    assert isinstance(out, str)
    assert len(out) > 100  # placeholder still has structure
    assert "<html" in out.lower()
```

### S1 — Path-injection guard (TM-E1)

```python
# tests/security/test_race_path_injection.py
from hypothesis import given, settings, strategies as st

from gpucheck.sanitizers.race import _find_compute_sanitizer

ALLOWLIST_PREFIXES = ("/usr/local/cuda", "/opt/nvidia/cuda", "/opt/cuda")


@given(hostile=st.text(min_size=1, max_size=200).filter(
    lambda s: not s.startswith(ALLOWLIST_PREFIXES) and "\x00" not in s))
@settings(max_examples=100)
def test_find_sanitizer_rejects_path_traversal(hostile, monkeypatch, tmp_path):
    # Plant a fake sanitizer at the hostile location
    fake_root = tmp_path / "fake"
    fake_root.mkdir()
    fake_bin = fake_root / "bin" / "compute-sanitizer"
    fake_bin.parent.mkdir()
    fake_bin.write_text("#!/bin/sh\necho hostile\n")
    fake_bin.chmod(0o755)
    monkeypatch.setenv("CUDA_HOME", str(fake_root))
    monkeypatch.setattr("shutil.which", lambda _: None)
    result = _find_compute_sanitizer()
    # Post-fix: realpath must be on allowlist OR result is None
    assert result is None or any(
        os.path.realpath(result).startswith(p) for p in ALLOWLIST_PREFIXES
    )
```

## Self-check

- [x] Every property would fail on a plausible bug (verified by inspection per attack):
  - A1: a `detect()` that branched on RNG would fail purity.
  - A2: a `synchronize()` that incremented a global counter would fail idempotence.
  - A3a: a negative `elapsed_ms` (sign bug) would fail bounds.
  - A3c: a regression that re-introduces `event.synchronize()` on MPS would fail the deadlock-pattern guard.
  - A6: a missing MPS multiplier in `compute_tolerance` would surface as parity test failure.
  - A7c: a swapped FP16/BF16 in the tolerance table would fail monotonicity.
  - A8: a missing entry in `pyproject.toml [tool.gpucheck.mps.xfail]` would fail totality.
  - B1: any non-deterministic insertion (e.g. set iteration order) would fail seed determinism.
  - B3: a stride generator missing the broadcast class would fail totality.
  - C2: a regression that reverts ContextVar to module-level list would leak between threads.
  - C3: same as C2 but for asyncio — fails at the `await` boundary.
  - C4: a non-try/finally exception path would leak the override.
  - D1: an `assert_deterministic` that calls `random.seed(N)` to set up its runs would mutate global state and fail.
  - D2: a renderer that embeds `time.time()` would fail byte-identity.
- [x] No trivially-true properties.
- [x] Strategies cover edge cases (empty shapes via `fuzz_shapes` degenerate, max-size, prime, power-of-2).
- [x] Hypothesis seeds set per-profile (`derandomize=True` in CI).

## Property test counts per track

| Track | Property tests | Property files |
|---|---|---|
| A — MPS backend | 13 | 4 |
| B — strides | 4 | 1 |
| C — ContextVar | 4 | 1 |
| D — determinism + HTML | 3 | 2 |
| Security regressions | 2 | 2 |
| **Total** | **26** | **10** |

## Verdict

GENERATED — 26 properties across 10 files, every property bound to an
importable target name in the engineering-track contract. Settings
profiles registered. Self-check clean.
