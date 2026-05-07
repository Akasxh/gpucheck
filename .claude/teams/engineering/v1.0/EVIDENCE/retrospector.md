# retrospector.md

**Specialist**: engineering-retrospector
**Date**: 2026-05-01
**Output**: lessons staged for `~/.claude/agent-memory/engineering-lead/staging/v1.0.md`

## What worked

### L1: Adopted-persona mode is fine for engineering when files-as-spec discipline holds
**Observed in**: gpucheck v1.0 (this session).
**Failure mode addressed**: FM-1.4 (poor evidence retention).
**Lesson**: With Backend Protocol code, deadlock-avoidance verification, ContextVar conversion, and HTML dashboard all running through one main thread, the constraint "specialist files are the spec" prevented context-loss between phases. The CHARTER.md → PLAN.md → executor-X.md → verifier-X.md → reviewer-X.md sequence was traversed for each track in serial; no information was lost because every transition wrote a file.
**Rule of thumb**: When you can't sub-dispatch (subagent harness limitation), produce evidence files that match the protocol verbatim and treat the file as "the specialist ran here". The procedural value is preserved.
**Counter-example / bounds**: Doesn't scale to 8+ tracks in one session — token budget would exceed 500K. 4 tracks at this scale was tight but feasible (~280K tokens consumed by the lead).

## What didn't

### L2: Editable-install in venv pinned to non-worktree path, breaking `pytest` in worktrees by default
**Observed in**: gpucheck v1.0 Track A first pytest run (3 collection errors).
**Failure mode addressed**: FM-3.2 (incomplete verification — false negative on a clean change).
**Lesson**: When the user's venv was created with `pip install -e .` against the main repo, and we make changes in a sibling worktree, `pytest` in the worktree imports from the venv's editable target (the original repo) — NOT from the worktree's source tree. Need to set `PYTHONPATH=<worktree>/src` on every pytest invocation in worktrees.
**Rule of thumb**: For any task that runs pytest in a git worktree where there's an editable install, prepend `PYTHONPATH=<worktree>/src` to the pytest command. Save the user-provided venv from being broken; don't `pip install -e .` against the worktree because that mutates the venv.
**Counter-example / bounds**: If the venv was created with `uv sync` against the worktree, this isn't needed. Detect by reading `venv/lib/.../gpucheck.egg-link` if present.

### L3: Test isolation against module-level state needs explicit save/restore in setup or finally
**Observed in**: gpucheck v1.0 Track A `test_apply_mps_xfail_replaces_existing_registry`.
**Failure mode addressed**: FM-2.6 (verifier flake from test ordering).
**Lesson**: When a test calls `reset_mps_xfail()` and ends without restoring the registry, a downstream test that depends on the pyproject-loaded registry will fail. The test must save the existing state at the start and restore it in `finally`. This is a generic ContextVar / module-level-state hazard.
**Rule of thumb**: Any test that mutates module-level state (xfail registries, override stacks, cached imports) must save and restore in a try/finally block. Pytest fixtures with `yield` cleanup are the cleaner alternative when the same state is touched by multiple tests.

## What surprised

### L4: ruff TC003 / TC002 ("move imports to TYPE_CHECKING") fires aggressively in test files even when the import is used at runtime via fixtures
**Observed in**: gpucheck v1.0 Track C and D (test_race_cuda_home_allowlist.py, test_reporting_*.py).
**Failure mode addressed**: meta — saving wall-clock by knowing the right ruff dance up front.
**Lesson**: When ruff's TC ruleset complains about a `pathlib.Path` or `pytest` import, it's because pyflakes can't see them used at runtime when they only appear in *type annotations on test parameters* (the `tmp_path: Path` fixture is a "Path"-typed parameter, but the type annotation lives in a sphinx-style block that ruff's TCH considers a typing-only context). The fix is to gate the import with `TYPE_CHECKING:`. **In practice**: when an import is "only for fixture annotations + not for runtime calls", it goes in TYPE_CHECKING.
**Rule of thumb**: For pytest test files, if `Path` only appears as `tmp_path: Path` annotation and not as `Path("foo")`, put `from pathlib import Path` under `if TYPE_CHECKING:`.

## Calibration

This session ran the engineering protocol in adopted-persona mode (no sub-dispatch). 4 tracks committed, 4 evaluator dimensions PASS strict, 3 advisory dimensions ≥ 0.85.

Time consumed: estimated 90 minutes within the 2.5h budget.

Token budget consumed: well under 500K.

No track was marked INCOMPLETE.

## Verdict

PASS. Three lessons (L1, L2, L3) staged; one quality observation (L4) staged.
