# PROPERTY_PLAN — gpucheck v1.0 Phase 2

**Slug**: testing/v1.0
**Owner**: testing-lead
**Date**: 2026-05-01
**Binds**: engineering-lead's 4 parallel tracks (A/B/C/D)
**Source**: `EVIDENCE/testing-property.md` + `EVIDENCE/testing-skeptic.md` (skeptic findings folded in)
**Profile**: Hypothesis ≥6.100, pytest, derandomize=True in CI

This is the contract: every importable target named here is what the
engineering tracks must ship. Property tests that import a missing target
are ENGINEERING bugs (file under FEEDBACK_FROM_TESTING.md).

---

## Hypothesis configuration (added to `tests/conftest.py`)

```python
from hypothesis import HealthCheck, settings

settings.register_profile("ci", max_examples=200, deadline=1000, derandomize=True,
                         suppress_health_check=[HealthCheck.too_slow])
settings.register_profile("dev", max_examples=50, deadline=500)
settings.register_profile("mps", max_examples=50, deadline=5000)        # MPS path
settings.register_profile("nightly", max_examples=2000, deadline=None)  # full M-Mac
settings.load_profile("ci" if os.getenv("CI") else "dev")
```

Switch profile via `HYPOTHESIS_PROFILE=mps pytest tests/test_*_mps_props.py`.

---

## Track A — Backend Protocol invariants + assert_close parity

**New importable contract** (engineering must ship):
- `gpucheck.arch.backend.Backend` — `Protocol` with methods `name: str`, `detect() -> Backend`, `synchronize() -> None`, `event_timer() -> tuple[Event, Event, Callable[[], float]]`, `mem_stats() -> MemStats`, `arch_info() -> ArchInfo`, `pending_kernel_count() -> int`, `run_dummy_workload(size: int) -> None`
- `gpucheck.arch.backend.detect_backend() -> Backend`
- `gpucheck.arch.backend.MemStats` — `dataclass(frozen=True)` `(allocated: int, reserved: int, free: int, total: int)`
- `gpucheck.arch.backend.ArchInfo` — `dataclass(frozen=True)` `(family: Literal["cuda","mps","cpu"], generation: str, sm_major: int|None, sm_minor: int|None, tensor_cores: bool)`
- `gpucheck.arch.xfail.is_xfailed(op: str, *, device: str) -> bool`
- `gpucheck.assertions.tolerances.compute_tolerance(dtype, *, k_dim=None, device=None) -> tuple[float,float]` — **device kwarg is NEW**

### Properties

| ID | Name | File | Profile | Skeptic-tagged |
|---|---|---|---|---|
| A1 | `test_detect_is_pure` | `tests/test_backend_props.py` | ci | requires_mps for full signal |
| A2 | `test_synchronize_idempotent` | `tests/test_backend_props.py` | ci | — |
| A3a | `test_event_timer_non_negative` | `tests/test_backend_props.py` | mps | — |
| A3b | `test_event_timer_finite_and_workload_visible` | `tests/test_backend_props.py` | mps | reframed (skeptic A6-1): median-of-5 with 10× slack, no strict monotonicity claim |
| A3c | `test_event_timer_no_deadlock_pattern_source` | `tests/test_backend_props.py` | ci | source-inspection (cheap) |
| A3c-real | `test_event_timer_no_deadlock_real` | `tests/test_backend_props.py` | nightly | NEW (skeptic A2-1): 5s timeout on M-Mac |
| A4 | `test_mem_stats_totals_coherent` | `tests/test_backend_props.py` | mps | — |
| A5 | `test_arch_info_matches_device` | `tests/test_backend_props.py` | ci | requires_mps for real-hardware signal |
| A6a | `test_assert_close_cpu_mps_parity` | `tests/test_assert_close_mps_props.py` | mps | — |
| A6b | `test_assert_close_mps_tolerance_envelopes_cpu` | `tests/test_tolerance_props.py` | ci | — |
| A7a | `test_mps_tolerance_geq_cuda` | `tests/test_tolerance_props.py` | ci | — |
| A7b | `test_tolerance_monotone_in_k_dim` | `tests/test_tolerance_props.py` | ci | — |
| A7c | `test_tolerance_monotone_in_dtype_precision` | `tests/test_tolerance_props.py` | ci | — |
| A8 | `test_xfail_entry_for_each_synthesis_top12` | `tests/test_xfail_registry.py` | ci | — |
| A8b | `test_xfail_listed_kernel_actually_xfails` | `tests/test_xfail_registry.py` | mps | NEW (skeptic A4-1): registry must affect collection |
| A9 | `test_xfail_hook_applies_marker` (pytester-based) | `tests/test_xfail_plugin_hook.py` | ci | NEW (skeptic A7-1, HIGH): plumbing test |

**Property sketch (A6a, the canonical parity property):**

```python
@given(shape=ShapeStrategy(ndim=2, max_size=512), dtype=st.sampled_from([torch.float32, torch.float16, torch.bfloat16]),
       seed=st.integers(0, 2**16))
@settings(profile="mps")
def test_assert_close_cpu_mps_parity(shape, dtype, seed, mps_available):
    a, b = CPUTensorPair.create(shape, dtype, seed)
    cpu_ref = a @ b.T
    mps_actual = (a.to("mps") @ b.to("mps").T).to("cpu")
    assert_close(cpu_ref, mps_actual, dtype=dtype, device="mps")
```

Full sketches: `EVIDENCE/testing-property.md`.

---

## Track B — Stride / contiguity fuzzing

**New importable contract**:
- `gpucheck.fuzzing.strides.fuzz_strides(*, shape: tuple[int, ...], n: int, seed: int|None=None) -> list[StrideSpec]`
- `gpucheck.fuzzing.strides.StrideSpec` — `dataclass(frozen=True)` `(shape, stride, offset, storage_size)` + `classify() -> Literal[ "contiguous", "transpose_2d", "transpose_3d", "broadcast", "slice_step_2", "sub_tensor_offset", "negative_stride" ]`

### Properties

| ID | Name | File | Profile |
|---|---|---|---|
| B1 | `test_fuzz_strides_seed_determinism` | `tests/test_strides_props.py` | ci |
| B2 | `test_fuzz_strides_shape_compatible` | `tests/test_strides_props.py` | ci (degenerate shapes added per skeptic A4-2) |
| B3 | `test_fuzz_strides_covers_seven_classes` | `tests/test_strides_props.py` | ci |
| B4 | `test_gpu_tensors_finite_by_default` | `tests/test_inputs_props.py` | ci |

The 7 stride classes (per dispatch CHARTER, reproduced for clarity):
1. `contiguous` — default row-major
2. `transpose_2d` — `tensor.T`
3. `transpose_3d` — `permute(2,0,1)`
4. `broadcast` — stride-0 dimension
5. `slice_step_2` — `tensor[::2]`
6. `sub_tensor_offset` — non-zero `storage_offset`
7. `negative_stride` — reversed strides (only via `as_strided`; rare but valid)

Sketches: `EVIDENCE/testing-property.md` §B1-B4.

---

## Track C — `contextvars.ContextVar` tolerance overrides

**New importable contract** (refactor of existing `assertions/tolerances.py`):
- `gpucheck.assertions.tolerances._tolerance_overrides: ContextVar[tuple[float,float]|None]` (replaces the module-level `list`)
- `gpucheck.assertions.tolerances.tolerance_context(atol, rtol) -> ContextManager[None]` — must use `ContextVar.set()` returning a Token, then `reset(token)` in finally.
- `gpucheck.assertions.tolerances._reset_for_test() -> None` — internal hook used by autouse `clean_tolerance_context` fixture

### Properties

| ID | Name | File | Profile |
|---|---|---|---|
| C1 | `test_tolerance_context_lifo` | `tests/test_tolerance_contextvar_props.py` | ci |
| C2 | `test_tolerance_context_thread_isolated` | `tests/test_tolerance_contextvar_props.py` | ci (with `barrier.wait(timeout=10.0)` per skeptic A6-2) |
| C3 | `test_tolerance_context_asyncio_isolated` | `tests/test_tolerance_contextvar_props.py` | ci |
| C4 | `test_tolerance_context_exception_safe` | `tests/test_tolerance_contextvar_props.py` | ci |
| C5 | `test_tolerance_context_inherited_by_child_task` | `tests/test_tolerance_contextvar_props.py` | ci NEW (skeptic A4-3): documents PEP 567 inheritance contract |

Sketches: `EVIDENCE/testing-property.md` §C1-C4.

---

## Track D — Determinism + HTML dashboard

**New importable contract**:
- `gpucheck.analysis.determinism.assert_deterministic(fn, *, runs=3, seed=None) -> None` — must save/restore `random`, `numpy.random`, `torch` RNG state hermetically
- `gpucheck.reporting.html.render_dashboard(data: dict, *, fixed_timestamp: str|None=None) -> str`
- `gpucheck.reporting.html.SchemaError(Exception)`

### Properties

| ID | Name | File | Profile |
|---|---|---|---|
| D1 | `test_assert_deterministic_no_global_rng_mutation` | `tests/test_determinism_props.py` | ci |
| D2 | `test_render_dashboard_idempotent_bytes` | `tests/test_html_dashboard_props.py` | ci |
| D2b | `test_render_dashboard_byte_identical_across_subprocesses` | `tests/test_html_dashboard_props.py` | ci NEW (skeptic A3): subprocess-level guard |
| D3 | `test_render_dashboard_handles_malformed_input` | `tests/test_html_dashboard_props.py` | ci |
| D4 | `test_render_dashboard_valid_html5` | `tests/test_html_dashboard_props.py` | ci NEW (skeptic A5): structural check via html5lib |

Sketches: `EVIDENCE/testing-property.md` §D1-D3.

---

## Security regression properties (P1)

**Bound to** security FINDINGS.md TM-E1 (path injection) + AR-2 (JSON schema).

| ID | Name | File | Profile |
|---|---|---|---|
| S1 | `test_find_sanitizer_rejects_path_traversal` | `tests/security/test_race_path_injection.py` | ci |
| S3 | `test_load_baseline_schema_strict` | `tests/test_reporting_json_props.py` | ci |

S1 sketch in `EVIDENCE/testing-property.md` §S1.

CFG-2 (workflow permissions) is a YAML lint, not a property test —
covered in `tests/test_ci.py` as a unit assertion (S2).

---

## Counts

| Track | Property files | Property tests | Skeptic-additions |
|---|---|---|---|
| A | 4 | 16 | 3 (A3c-real, A8b, A9) |
| B | 2 | 4 | 1 (degenerate shapes in B2) |
| C | 1 | 5 | 1 (C5) + 1 (timeout in C2) |
| D | 2 | 5 | 2 (D2b, D4) |
| Sec | 2 | 2 | 0 |
| **Total** | **11** | **32** | **8** |

---

## Acceptance gates

For Phase 3 merge, the property suite must:
1. Run `pytest tests/test_*_props.py tests/test_*_registry.py tests/test_xfail_plugin_hook.py tests/security/` → 0 failures, 0 errors, 0 flakes (3 consecutive runs).
2. Run with `HYPOTHESIS_PROFILE=mps` on Akash's M-Mac → 0 failures (smoke).
3. Mutation-tested per `MUTATION_REPORT.md` → ≥75% aggregate, ≥90% on `_find_compute_sanitizer`.

If any property test imports a target that engineering didn't ship,
file `FEEDBACK_FROM_TESTING.md` to engineering's workspace immediately —
that's a contract violation, not a test bug.

---

## File pointers

- `EVIDENCE/testing-detector.md` — project profile
- `EVIDENCE/testing-planner.md` — full target table P0/P1/P2
- `EVIDENCE/testing-property.md` — full property sketches with code
- `EVIDENCE/testing-fixture.md` — conftest + factories
- `EVIDENCE/testing-skeptic.md` — adversarial findings (10 total, 1 HIGH absorbed)
