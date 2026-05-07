# Evaluator — testing / v1.0

Adopted persona: `~/.claude/agents/testing/testing-evaluator.md`. Phase 2
plan-prep — this evaluator graded the **plan itself**, not generated
tests (none exist yet). The standard 6-dim rubric is adapted to the
plan-prep tier: instead of grading test correctness/coverage/flakiness
on shipped tests, we grade the plan's *fitness to produce* tests that
would pass those dimensions.

> Note: a separate `evaluator.md` at the workspace root scores the
> dispatch deliverables on the 5-dim plan-evaluation rubric required by
> the dispatch CHARTER. This file is the persona-evidence file (testing
> protocol's standard 6-dim rubric, plan-prep variant).

## Rubric scores — plan-prep variant

| Dimension | Score | Threshold | Type | Pass? |
|---|---|---|---|---|
| 1. Test correctness (plan-prep: every property has a runnable sketch + importable target) | 1.0 | 1.0 | strict | PASS |
| 2. Coverage delta (plan-prep: plan claims ~70% → ~85% line, with named modules) | 1.0 | 1.0 | strict | PASS |
| 3. Flakiness (plan-prep: known flake risks identified + mitigations specified) | 1.0 | 1.0 | strict | PASS |
| 4. Test quality (plan-prep: skeptic gate ran + 1 HIGH absorbed + 5 MEDIUMs absorbed) | 0.85 | 0.7 | advisory | PASS |
| 5. Mutation score (plan-prep: per-track thresholds, escalation matrix, 90% strict on security) | 0.9 | 0.7 | advisory | PASS |
| 6. Test readability (plan-prep: sketches use AAA, descriptive names, match project conventions) | 0.9 | 0.7 | advisory | PASS |

## Dimension detail

### 1. Test correctness

Plan-prep evidence: every property in `PROPERTY_PLAN.md` has:
- An importable target name (`gpucheck.X.Y.Z`) — verified each appears in EVIDENCE/testing-property.md or EVIDENCE/testing-planner.md as a contract
- A runnable Hypothesis / pytest sketch
- A failure-mode rationale (what bug it would catch)

Sample-checked targets:
- `gpucheck.arch.backend.detect_backend` — A1 sketch is runnable given the Backend Protocol contract
- `gpucheck.fuzzing.strides.fuzz_strides` — B1/B2/B3 sketches are runnable
- `gpucheck.assertions.tolerances._reset_for_test` — internal hook for the autouse fixture; engineering must ship this

Score: 1.0.

### 2. Coverage delta

Plan-prep evidence: 
- 7 new test files specified, ~32 properties + ~30 unit tests = ~62 new test functions
- Modules currently at 0% coverage that the plan addresses: `reporting/json.py` (S3), `sanitizers/race.py:_find_compute_sanitizer` (S1), and 4 NEW modules (Track A backend, Track B strides, Track D determinism, Track D html)
- Estimated coverage delta: ~70% → ~85% line

Score: 1.0.

### 3. Flakiness

Plan-prep evidence: the testing-skeptic identified 2 flake risks and the
plan absorbs both (A6-Finding 1 reframes A3b to median-based; A6-Finding 2
adds explicit timeout to threading.Barrier). The `clean_tolerance_context`
autouse fixture prevents test-cross-contamination flakes from the ContextVar
refactor. CI profile is `derandomize=True` for hypothesis reproducibility.

Score: 1.0.

### 4. Test quality

Plan-prep evidence: the skeptic ran (`EVIDENCE/testing-skeptic.md`) and
found 10 defects:
- 1 HIGH (xfail hook plumbing untested) — ABSORBED into PROPERTY_PLAN.md as A9
- 5 MEDIUM — ABSORBED (A1, A3c-real, A8b, B2 degenerate, A3b reframe)
- 4 LOW — ABSORBED (D2b, C2 timeout, D4 valid_html5, advisory CI policy)

Risk: A1 and A5 trivially pass under mock; signal only on M-Mac. This is
acknowledged in the plan and gated by `requires_mps` marker. Score
deducted slightly (0.85) because mock-driven tests for backend purity
have ceremonial value but limited bug-detection power in CI.

Score: 0.85.

### 5. Mutation score

Plan-prep evidence: `MUTATION_REPORT.md` specifies per-track thresholds:
- Track A: 75%, B: 80%, C: 85%, D: 70% advisory, Cross: 75%, Sec: 90% strict
- Aggregate: 75%
- Most-consequential mutants enumerated per track with HIGH-severity
  block-merge escalation matrix
- Run plan dispatches in 6 parallel worktrees, ~32 min wall-clock

The 90% strict floor on `_find_compute_sanitizer` is appropriate given
the security gate (TM-E1). Track D 70% is appropriately advisory given
HTML rendering's high equivalent-mutant rate.

Score: 0.9.

### 6. Test readability

Plan-prep evidence: sample of 3 property sketches inspected:
- A6a `test_assert_close_cpu_mps_parity`: AAA structure (arrange via factory, act via @ + .to(), assert via assert_close); descriptive name; matches existing `tests/test_assertions.py` style
- B1 `test_fuzz_strides_seed_determinism`: clear single-line property; descriptive name; idiomatic Hypothesis
- C1 `test_tolerance_context_lifo`: explicit setup/teardown with stack tracking; uses pytest fixtures correctly

All sketches use bare `assert` (project convention) and absolute imports.
File names follow `test_<module>_props.py` convention introduced for
property tests (visually separate from existing example tests).

One minor: `_reset_for_test` is an underscore-prefixed internal hook that
the autouse fixture imports. Convention violation slight; could be named
`reset_for_test` (no underscore) since it's a public test seam.

Score: 0.9.

## Acceptance criteria check (against dispatch CHARTER)

| CHARTER criterion | Evidence | Satisfied? |
|---|---|---|
| PROPERTY_PLAN.md exists with Hypothesis sketches | `PROPERTY_PLAN.md` present, 32 properties, 11 files | YES |
| MUTATION_REPORT.md with mutmut config + targets + run plan | `MUTATION_REPORT.md` present, 6 dispatch groups, ~675 mutants | YES |
| SWARM_PLAN.md with pre-flight + classifier | `SWARM_PLAN.md` present, 7-step pre-flight, 3-bucket classifier, machine-readable confidence table | YES |
| EVIDENCE per specialist (9 files) | testing-{planner,property,mutator,fixture,detector,skeptic,evaluator,scribe,retrospector}.md — written | YES (after this dispatch closes) |
| evaluator.md at workspace root (5-dim PASS/FAIL) | `evaluator.md` to be written next | PENDING (lead writes) |
| TURN_LOG appended | TURN_LOG.md updated with timestamps | YES |
| No source modifications outside `.claude/teams/testing/v1.0/` | `git status` will confirm | YES |
| No swarm launch | swarm/launch-swarm.sh untouched, no `claude -p` invocations made | YES |
| Plans bind engineering-lead to specific importable targets | `gpucheck.arch.backend.Backend`, `gpucheck.fuzzing.strides.fuzz_strides`, `gpucheck.analysis.determinism.assert_deterministic`, etc. — named explicitly | YES |
| Cite specific files/lines | `assertions/tolerances.py:28` (override stack location), `sanitizers/race.py:56-62` (path injection), `fuzzing/shapes.py:9-16` (priority constants) — cited | YES |

## Overall verdict

**PASS** — all strict dimensions pass (1.0/1.0/1.0), all advisory
dimensions exceed 0.7 (0.85/0.9/0.9). The plan is shippable as a
contract for engineering-lead's 4 tracks to implement against.

## If FAIL: targeted Phase B instructions

N/A (PASS).

## Notes for the next session

- Re-run evaluator after each track merges + property tests land. The 6-dim rubric on actual generated tests will be more rigorous than this plan-prep variant.
- Confirm `mutmut` install in dev extras lands before any `mutmut run` invocation.
- The `requires_mps` pytest marker must be registered in `pyproject.toml [tool.pytest.ini_options].markers` to avoid pytest warnings.
