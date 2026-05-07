# Skeptic — testing / v1.0

Adopted persona: `~/.claude/agents/testing/testing-skeptic.md`. Phase 2
plan-prep — this is a pre-flight attack on the plan ITSELF (PROPERTY_PLAN +
MUTATION_REPORT + SWARM_PLAN drafts and the EVIDENCE supporting them),
NOT on generated tests (none exist yet).

## A1 — Over-mocking

**Finding**: The plan mocks `torch.mps` wholesale via `mock_torch_mps`
when CI lacks Apple hardware. This is necessary, but property A1
(`test_detect_is_pure`) and A5 (`test_arch_info_matches_device`) under the
mock will trivially pass because the mock IS pure by construction. These
properties only become meaningful on the real M-Mac.

- **Severity**: MEDIUM
- **Files affected**: `tests/test_backend_props.py` (A1, A5)
- **Fix required**: ADVISORY. Mark A1, A5 with `@pytest.mark.requires_mps`
  and have CI run a single nightly job on the M-Mac (or in the swarm
  worktrees on Akash's machine) — that's the only place these properties
  carry signal. CI no-MPS run is regression coverage only.

## A2 — Implementation-testing

**Finding 1**: A3c (`test_event_timer_no_deadlock_pattern`) uses
`inspect.getsource()` to grep for `event.synchronize` in the production
code. This is implementation-testing — it tests the source string, not
the behavior. **However**, the actual deadlock from pytorch#162872 only
manifests on real M-hardware with PyTorch 2.10+, so a behavioral test
would require the M-Mac in CI, which is the constraint we're working
under.

- **Severity**: MEDIUM
- **Files affected**: `tests/test_backend_props.py:test_event_timer_no_deadlock_pattern`
- **Fix required**: ACCEPT with justification + add a stronger behavioral
  variant that runs ONLY on M-Mac (gated by `mps_available` fixture):

  ```python
  @pytest.mark.requires_mps
  def test_event_timer_no_deadlock_real(mps_available):
      """On real M-Mac, the deadlock would hang. Use a 5s timeout."""
      import threading
      result = {}
      def run():
          backend = mps_available
          start, end, ms = backend.event_timer()
          start.record(); backend.run_dummy_workload(128); end.record()
          backend.synchronize()
          result["ms"] = ms()
      t = threading.Thread(target=run, daemon=True)
      t.start()
      t.join(timeout=5.0)
      assert not t.is_alive(), "event_timer deadlocked (pytorch#162872)"
      assert result.get("ms", -1) >= 0
  ```

  Keep BOTH: source-inspection on CI (cheap regression guard) + behavioral
  on M-Mac (real deadlock guard).

**Finding 2**: A8 (`test_xfail_entry_for_each_synthesis_top12`) tests
that `pyproject.toml` has the right strings. This is testing the
configuration, not the code. Borderline implementation-testing but
necessary because the xfail registry IS load-bearing per SYNTHESIS
"Engineering team must respect §2".

- **Severity**: LOW
- **Fix required**: NO. The registry is a contract; testing the contract
  is appropriate.

## A3 — Tautological tests

**Finding**: D2 (`test_render_dashboard_idempotent_bytes`) compares
SHA-256 of two `render_dashboard` outputs. The property is fine, but if
the implementation passes `time.time()` into the renderer, the test will
fail (correctly), but if engineering builds a workaround that just
buffers the timestamp once per process, the property still passes while
the dashboard remains non-deterministic across process restarts.

- **Severity**: LOW
- **Fix required**: ADVISORY. Augment with:

  ```python
  def test_render_dashboard_byte_identical_across_subprocesses(baseline, tmp_path):
      """Two separate subprocesses render the same input; bytes match."""
      import subprocess, sys
      p1 = subprocess.run([sys.executable, "-c",
          f"from gpucheck.reporting.html import render_dashboard;"
          f"import json; print(render_dashboard(json.load(open({str(baseline)!r})),"
          f" fixed_timestamp='t'))"], capture_output=True, text=True)
      p2 = subprocess.run(..., capture_output=True, text=True)  # same
      assert p1.stdout == p2.stdout
  ```

## A4 — Missing edge cases

**Finding 1**: Property A6 (CPU vs MPS parity) does not currently
parametrize over the SYNTHESIS xfail-listed shapes. It tests "non-xfail"
shapes; we need a complementary test that asserts xfail-listed shapes
DO get marked xfail (i.e., the registry is wired into pytest collection,
not just stored in TOML).

- **Severity**: MEDIUM
- **Files affected**: `tests/test_assert_close_mps_props.py`
- **Fix required**: YES. Add A8b:

  ```python
  @pytest.mark.parametrize("xfail_op", list(SYNTHESIS_TOP12.values()))
  @pytest.mark.requires_mps
  def test_xfail_listed_kernel_actually_xfails(xfail_op, mps_available):
      """For a kernel on the xfail registry, gpucheck must mark the test xfail."""
      from gpucheck.arch.xfail import is_xfailed  # contract: Track A
      assert is_xfailed(xfail_op, device="mps") is True
  ```

**Finding 2**: B2 (stride shape compatibility) does not test what
happens when the shape contains a 0 (degenerate). `as_strided` with a
0-dim tensor has well-defined but easy-to-miss semantics — must include.

- **Severity**: MEDIUM
- **Fix required**: YES. Add to B2's strategy: `st.sampled_from(fuzz_shapes(... include_degenerate=True))`.

**Finding 3**: C2/C3 (thread + asyncio isolation) test that overrides
DON'T leak. They do NOT test that overrides DO propagate when explicitly
opted in (e.g., `contextvars.copy_context()`). PEP 567 supports both
behaviors and engineering must pick one. Skeptic recommends explicit:
new task starts with COPY of current context (the default), so a parent
override IS visible in spawned tasks unless the task overrides.

- **Severity**: LOW
- **Fix required**: ADVISORY. Add C5:

  ```python
  def test_tolerance_context_inherited_by_child_task():
      async def child():
          return compute_tolerance(torch.float32)
      async def parent():
          with tolerance_context(atol=1e-9, rtol=1e-9):
              return await asyncio.create_task(child())
      assert asyncio.run(parent()) == (1e-9, 1e-9)
  ```

  Documents the LIFO-with-inheritance contract.

## A5 — False confidence

**Finding**: The plan's mutation-score thresholds (75% for Track A,
70% for Track D) leave room for real defects to slip. Specifically,
HTML rendering at 70% means up to 30% of structural mutants could
survive. If the renderer emits malformed HTML that browsers tolerate,
D2/D3 won't catch it.

- **Severity**: MEDIUM
- **Fix required**: ADVISORY. Add a property `test_render_dashboard_valid_html5`:
  use `html5lib` parser and assert no parse errors. This raises the
  effective floor without bumping the mutation threshold.

  ```python
  def test_render_dashboard_valid_html5(sample_baseline):
      from html5lib import HTMLParser
      out = render_dashboard(sample_baseline, fixed_timestamp="t")
      parser = HTMLParser(strict=True)
      parser.parse(out)  # raises on invalid HTML5
      assert parser.errors == []
  ```

## A6 — Flakiness risk

**Finding 1**: A3b (`test_event_timer_monotone_with_workload`) uses
elapsed-time inequality `t_large >= 0.5 * t_small` to allow noise.
Under M-Mac thermal throttling or when CI is loaded, even this can fail.
Real benchmark monotonicity tests should use **median of N runs** and
allow much wider slack.

- **Severity**: MEDIUM (real flake risk)
- **Fix required**: YES. Reframe as:

  ```python
  def test_event_timer_monotone_with_workload(mps_or_mock_backend):
      def median(xs): xs = sorted(xs); return xs[len(xs)//2]
      ms_small = median([_time(mps_or_mock_backend, 128) for _ in range(5)])
      ms_large = median([_time(mps_or_mock_backend, 4096) for _ in range(5)])
      # No strict monotonicity claim — just "elapsed is finite, non-negative,
      # and the larger workload doesn't run impossibly fast"
      assert ms_small >= 0 and ms_large >= 0
      assert math.isfinite(ms_small) and math.isfinite(ms_large)
      assert ms_large >= 0.1 * ms_small  # 10x slack vs real measurement noise
  ```

  Or drop the property entirely and replace with a unit test that
  benchmarks a known-large kernel and asserts elapsed_ms > 0.

**Finding 2**: C2 thread-isolation property uses `threading.Barrier(3)`.
On a single-core CI runner under contention, the barrier could timeout,
producing a flake. Set explicit `timeout=10.0` on `barrier.wait()`.

- **Severity**: LOW
- **Fix required**: YES. One-line change.

## A7 — Test plan completeness

**Finding 1**: The plan MISSES coverage for the **xfail registry's
runtime enforcement path** — the pytest hook that reads
`[tool.gpucheck.mps.xfail]` and applies `pytest.mark.xfail` to matching
tests. The plan tests the registry contents (A8) but not the plumbing
that actually reads + applies them.

- **Severity**: HIGH
- **Files affected**: missing test file `tests/test_xfail_plugin_hook.py`
- **Fix required**: YES. Add:

  ```python
  def test_xfail_hook_applies_marker(pytester):  # pytest's pytester fixture
      pytester.makefile(".toml", pyproject="""
      [tool.gpucheck.mps.xfail]
      ops = ["matmul.backward.over_32K_elements"]
      """)
      pytester.makepyfile(test_inner="""
      import pytest
      @pytest.mark.gpucheck_op("matmul.backward.over_32K_elements")
      def test_a(): assert False  # would fail without xfail mark
      """)
      result = pytester.runpytest("-v")
      result.stdout.fnmatch_lines(["*XFAIL*test_a*"])
  ```

**Finding 2**: The SWARM_PLAN lists 26 kernels but the plan's property
catalogue does NOT cover divergence-classifier logic. If the swarm
output triggers a false-positive divergence, there's no guard.

- **Severity**: MEDIUM
- **Files affected**: missing — divergence classifier
- **Fix required**: YES (covered in SWARM_PLAN's classifier spec, but
  add a property test for the classifier post-implementation).

**Finding 3**: The plan does not cover **CFG-2 (GitHub Actions
permissions block)** end-to-end. S2 spec checks YAML, but doesn't check
that subsequent runs of the workflow on a real PR cannot escalate. This
is an integration test that requires PR-level permissions inspection.

- **Severity**: LOW (defense in depth)
- **Fix required**: ADVISORY. Document in OPEN_QUESTIONS — could be a
  GitHub-actions-policy check rather than a unit test.

## Summary

| Severity | Count |
|---|---|
| HIGH | 1 (A7-Finding 1: xfail hook plumbing untested) |
| MEDIUM | 5 (A1, A2-Finding 1, A4-Finding 1, A4-Finding 2, A5, A6-Finding 1) |
| LOW | 4 (A3, A4-Finding 3, A6-Finding 2, A7-Finding 3) |
| **Total** | **10** |

## Verdict

**FAIL on plan completeness** — 1 HIGH-severity gap (the xfail registry
plumbing test) must be added to PROPERTY_PLAN.md before the property
suite is considered shippable. All other findings are absorbed into the
plan as additions/refinements (no plan deletes required).

Required fixes before Phase 3 merge:

1. Add `tests/test_xfail_plugin_hook.py` with `pytester`-based hook test (A7-Finding 1, HIGH).
2. Reframe A3b to median-based with 10x slack (A6-Finding 1, MEDIUM).
3. Add A8b: `test_xfail_listed_kernel_actually_xfails` (A4-Finding 1, MEDIUM).
4. Augment B2 strategy with degenerate shapes (A4-Finding 2, MEDIUM).
5. Add `test_render_dashboard_valid_html5` (A5, MEDIUM advisory).
6. Add subprocess-level idempotence test for D2 (A3, LOW).
7. Add `barrier.wait(timeout=10.0)` to C2 (A6-Finding 2, LOW).

These have been folded into PROPERTY_PLAN.md as the canonical plan — the
plan is a living document and the skeptic gate runs BEFORE the writer,
so these are pre-emptive fixes, not post-hoc patches.
