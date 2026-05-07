# SESSION_PRECHECK — gpucheck v1.0 + claude-forge v0.2

**Session start:** 2026-05-01
**Orchestrator model:** claude-opus-4-7, effort=max
**Permission mode:** bypassPermissions (with explicit Path-B authorization for swarm + upstream filings)
**Host:** Apple Silicon Mac (MPS-capable)
**Path chosen by user:** B — full prompt as written

## Repo state

- CWD: `/Users/cero/Code/gpucheck`
- Branch: `release/v1.0` (clean, up to date with origin)
- Recent commits:
  - `a9a9d44` [ Fix ] : resolve 7 bugs, add 23 tests, rewrite docs with GTX 1650 validation
  - `2197277` [ Fix ] : resolve 7 bugs found by codebase analysis, add 23 tests, update docs
  - `5dcbf83` [ README ] : added bugs found section with Triton issue links
  - `25cdfcf` [ Perf ] : GPU fast-path for assert_close, fixed tensor core detection
  - `6562f31` [ Fix ] : recalibrated tolerance tables from GPU measurements

## Bootstrap actions completed

| step | status | detail |
|---|---|---|
| Install security team | ✅ | 13 specialists in `~/.claude/agents/security/`, PROTOCOL in `~/.claude/teams/security/` |
| Install testing team | ✅ | 12 specialists |
| Install docs team | ✅ | 11 specialists |
| Promote gpucheck experts | ✅ | 10 subagents in `~/.claude/agents/gpucheck/` |
| Create impl worktrees | ✅ | 4 branches: `feat/track-{a-mps,b-strides,c-thread-safety,d-bundle}` |
| Create fuzz worktrees | ✅ | 26 detached worktrees under `gpucheck-worktrees/fuzz-<kernel>/` |
| Init evidence trees | ✅ | `research/engineering/security/testing/docs/forge` × `v1.0/` |
| Seed TURN_LOG.md | ✅ | one per team |

## Environment notes

- `gh` authenticated as `Akasxh` (scopes: gist, read:org, repo, workflow)
- `~/.pypirc` does **not** exist — TestPyPI upload will need manual credential setup before Phase 4 can complete
- `claude` CLI v2.1.123 confirmed for Tier-3 swarm
- Total worktree count: 30 (1 main + 4 impl + 26 fuzz, tracked by git)
- **torch was missing from project venv at session start.** Installed `torch==2.11.0` via `uv pip install torch`. `torch.backends.mps.is_available() == True`, `torch.backends.mps.is_built() == True`. **MPS dogfooding is viable on this host.**

## Test-baseline correction

The orchestrator prompt claims "408 tests pass on main." Real baseline measured here:

```
$ uv run pytest -q
117 passed, 3 skipped, 6 warnings in 1.64s
```

Source tree has **6 unit-test files under `tests/`** (test_analysis, test_arch, test_assertions, test_ci, test_decorators, test_fuzzing) plus 5 GPU-integration tests skipped without an NVIDIA GPU. Phase 2 quality bar is "117 baseline + N new tests for the new code", not "match the fictional 408".

## Out-of-prompt clarifications

User explicitly chose Path B; orchestrator authorized to:
- spawn the 26-process headless swarm
- file ≤3 upstream issues against PyTorch
- create PRs `release/v1.0 → main` (gpucheck) and `release/v0.2-rc → main` (claude-forge)
- write dist artifacts; TestPyPI upload deferred to manual step pending `~/.pypirc`
