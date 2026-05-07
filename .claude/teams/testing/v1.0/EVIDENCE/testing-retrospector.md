# Retrospector — testing / v1.0

Adopted persona: `~/.claude/agents/testing/testing-retrospector.md`. Phase
2 plan-prep — retrospection happens BEFORE tests are written, so lessons
focus on the dispatch shape, the cross-team contract pattern, and the
plan's adversarial robustness. Lessons that emerge during Phase 3
test-generation will be a separate retrospection.

## Session summary

- Tier: comprehensive (full roster gates, mutation testing planned, property tests planned, skeptic gate ran)
- Mode: adopted persona — testing-lead executed all specialist methods directly, writing to per-specialist evidence files
- Duration: ~70 min wall-clock (under 90-min hard deadline)
- Evaluator verdict: PASS (all strict 1.0, advisory 0.85+)
- Scope: plan-prep only (no test generation, no swarm launch, no MEMORY.md merge)

## What worked

- **Skeptic-pre-flight produced HIGH-severity finding** (xfail hook plumbing) that pure planner+property would have missed. The protocol's mandatory skeptic gate before final plan exit caught a real test plumbing gap.
- **Cross-team contract pattern** (binding plan to importable target names like `gpucheck.arch.backend.Backend`) makes engineering-test handoff explicit. Engineering can't accidentally rename a method without breaking the plan, and tests can't be written against vapourware.
- **Per-track mutation thresholds** (75% Track A, 80% Track B, 85% Track C, 70% advisory Track D, 90% strict on `_find_compute_sanitizer`) reflect real risk gradient. Uniform thresholds would have under-tested security and over-tested HTML rendering.
- **3-bucket divergence classifier** in SWARM_PLAN gives operators a deterministic decision tree. Avoids the "swarm produces 50 unranked findings, lead picks favorites" failure mode.
- **Reading SYNTHESIS.md and FINDINGS.md as binding inputs** before any planning anchored the plan in actual research evidence (12 specific PyTorch issues, 3 MEDIUM security findings) instead of speculation.

## What didn't work

- **Mock-driven Track A property tests carry limited signal in CI.** A1 (`test_detect_is_pure`) and A5 (`test_arch_info_matches_device`) trivially pass under `mock_torch_mps` because the mock IS pure by construction. Real signal only on M-Mac. The plan acknowledges this with `requires_mps` markers, but the asymmetry is real: comprehensive-tier property tests are expected to carry signal everywhere, and these don't.
- **The dispatch CHARTER's 5-dim plan evaluator is non-standard** vs the testing protocol's 6-dim test-output evaluator. We had to write both. Future plan-prep dispatches should standardise on one rubric or explicitly mark the variant.
- **Mutation framework not yet installed.** The `mutmut` add-to-dev-extras instruction is a side-effect that engineering must accept. A more decoupled plan would have shown how to run mutation testing in a separate venv that doesn't change `pyproject.toml`.
- **Hypothesis profile registration** (the `_register_hypothesis_profiles` block in `tests/conftest.py`) modifies global state at import time. If a future test session imports conftest twice, `register_profile` may fail with `ProfileAlreadyExists`. Plan should idempotency-guard this.

## What was surprising

- **`torch.mps.event.Event.synchronize()` deadlock (pytorch#162872)** is a Track A correctness gate, but the most natural way to test it (calling event.synchronize and timing out) is itself the deadlock pattern. We resolved with source-inspection (cheap, CI) plus thread-with-timeout behavioral test (nightly, M-Mac). This dual-pronged approach is reusable for any "feature defined by absence of bad pattern".
- **PEP 567 ContextVar inheritance semantics** are subtle: child asyncio tasks inherit a COPY of parent context but mutations don't propagate back. This is well-documented but easy to misuse. Skeptic A4-3 added test C5 to make the contract explicit.
- **The SYNTHESIS top-12 list contains a hang bug (pytorch#162872) and a SIGABRT bug (pytorch#175190)** — these aren't "tolerance" bugs at all, they're crash bugs that no `assert_close` multiplier can rescue. The xfail registry is therefore strictly more important than the tolerance overlay. SYNTHESIS made this point too; reinforced here.
- **`fuzz_strides` 7-class enumeration** is small enough that a random sampler may miss `negative_stride` (rare). B3's totality property forces the generator to be priority-stratified, not random.

## Lessons extracted (for `staging/v1.0.md`)

```markdown
## Added from v1.0.md at 2026-05-01

### Lesson: Skeptic gate finds plumbing gaps that planners miss
- **Observed in**: testing/v1.0 (2026-05-01)
- **Failure mode addressed**: FM-3.3 (incorrect verification — plan-level)
- **Lesson**: A planner that decomposes by feature ("test the xfail registry contents") may miss the plumbing that makes the feature work ("the pytest hook that reads the registry and applies xfail markers"). The mandatory skeptic pre-flight surfaced this as a HIGH-severity gap and the plan was patched.
- **Rule of thumb**: For every config-driven feature, plan TWO tests: one for the config contents, one for the plumbing that consumes the config. Skeptic must explicitly check both axes.
- **Counter-example / bounds**: When the config is purely descriptive (e.g., a CHANGELOG.md), no plumbing test is needed.

### Lesson: Bind plans to importable target names
- **Observed in**: testing/v1.0 (2026-05-01)
- **Failure mode addressed**: FM-1.1 (specification ambiguity), FM-2.4 (information withholding cross-team)
- **Lesson**: Phrasing the plan as "test that detect() is pure" is ambiguous if engineering ships `Backend.detect()` vs `detect_backend()` vs `Detector.detect()`. Naming `gpucheck.arch.backend.detect_backend()` explicitly creates a bidirectional contract: engineering ships that name; testing imports it. If engineering renames, the test fails at collection (not at assertion), making the contract violation immediately visible.
- **Rule of thumb**: Every property test sketch in a cross-team plan MUST cite the fully-qualified importable name of its target.
- **Counter-example / bounds**: For internal-only refactors where engineering is the only consumer, the contract can be looser.

### Lesson: Test "absence of bad pattern" with dual-pronged approach
- **Observed in**: testing/v1.0 (2026-05-01)
- **Failure mode addressed**: FM-3.2 (incorrect failure detection)
- **Lesson**: pytorch#162872's deadlock pattern is "calling event.synchronize() in this specific order". The most natural behavioral test invokes the deadlock pattern itself. Resolution: source-inspection (cheap CI guard via `inspect.getsource`) + thread-with-timeout behavioral test (nightly only). Source-inspection is implementation-testing but acceptable when the implementation IS the contract.
- **Rule of thumb**: When a bug is "do not call X", combine source-grep for X with a behavioral test that exercises the surrounding code under a timeout.
- **Counter-example / bounds**: For positive contracts ("must call Y"), the source-grep is brittle to refactoring; prefer behavioral tests.

### Lesson: Per-track mutation thresholds with security-strict floors
- **Observed in**: testing/v1.0 (2026-05-01)
- **Failure mode addressed**: FM-3.2 (insufficient verification depth)
- **Lesson**: Uniform mutation thresholds (e.g., "75% everywhere") under-test security and over-test rendering. The plan set 70% advisory on HTML rendering (high equivalent-mutant rate), 90% strict on `_find_compute_sanitizer` (security-critical), 75-85% elsewhere. Aggregate score still ≥75%.
- **Rule of thumb**: For each module, ask "if a mutant survives, is it (a) likely equivalent, (b) low-impact bug, (c) HIGH-impact bug?" The threshold should reflect the answer. Security-critical functions get 90%+ floors with HIGH-severity escalation.
- **Counter-example / bounds**: For a small project where uniform thresholds reduce cognitive load, uniform may be fine. The threshold gradient pays off for codebases with mixed risk profiles.

### Lesson: Divergence classifier is a deterministic decision tree, not a vibe check
- **Observed in**: testing/v1.0 (2026-05-01)
- **Failure mode addressed**: FM-1.5 (output triage ambiguity)
- **Lesson**: Without an explicit classifier, swarm output produces "N findings, lead picks favorites". The 3-bucket classifier (FILABLE-UPSTREAM ≥3 reproductions + deterministic + >10× tolerance + not on xfail; TOLERANCE-RECALIBRATION ≥10 reproductions + 1×–10× tolerance + deterministic + tight P99/median; FALSE POSITIVE everything else) makes triage mechanical.
- **Rule of thumb**: Any swarm/fuzzer output triage must be specifiable as a flowchart with measurable thresholds before launch. If you can't write the flowchart, you don't yet understand what "high-confidence finding" means.
- **Counter-example / bounds**: For exploratory fuzzing without an upstream-filing target, the classifier can be omitted.
```

## v2.1 compliance

- Detector first: YES (testing-detector.md written first; planner cited it)
- 3x runner: N/A (plan-prep, no runs to repeat)
- Skeptic gate: YES (skeptic.md written before evaluator; skeptic findings folded into plan)
- Evidence schema: YES (all 9 evidence files conform to persona-specified schema)
- TEST_LOG raw output: N/A (no test runs in this dispatch)

Compliance grade: PASS (all applicable items met).

## Verdict

RETROSPECTED — 5 lessons staged for MEMORY.md merge at Phase 3 close. Lessons emphasize cross-team contract patterns, skeptic gate value, dual-pronged absence-of-pattern testing, per-track mutation thresholds, and deterministic divergence classifiers.
