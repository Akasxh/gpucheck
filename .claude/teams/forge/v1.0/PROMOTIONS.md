# PROMOTIONS — Phase 1 candidates (NOT yet promoted)

Per charter: hold promotion until Phase 2/3 has actually exercised these drafts. Phase 4 (final promotion) will move passing drafts from `DRAFTS/<name>/SKILL.md` to `~/.claude/skills/<name>/SKILL.md` and update `~/.claude/agent-memory/forge-lead/MEMORY.md`.

## Candidate skills

| # | Name | Draft path | Eval pass-rate | Recommended for promotion in Phase 4 | Phase 2/3 caller |
|---|---|---|---|---|---|
| 1 | `mps-kernel-debugging` | `DRAFTS/mps-kernel-debugging/SKILL.md` | 3/3 | YES (load-bearing for Phase 2 MPS dispatch work) | gpucheck arch/ + decorators extension |
| 2 | `metal-shader-profiling` | `DRAFTS/metal-shader-profiling/SKILL.md` | 3/3 | YES (companion to MPS dispatch when perf debugging) | gpucheck profiling integration |
| 3 | `hatch-testpypi-release` | `DRAFTS/hatch-testpypi-release/SKILL.md` | 3/3 | YES (gpucheck v1.0 release flow) | release/v1.0 branch tag-and-publish |

## Pre-promotion checklist (Phase 4)

For each skill, before moving to `~/.claude/skills/`:

1. Confirm name still has no collision under `~/.claude/skills/` and `~/.claude/agents/`.
2. Re-read SKILL.md for any Phase 2/3 lessons that should be folded back (e.g., if mps-kernel-debugging was triggered and a new failure mode was discovered, append it).
3. Append catalog bullet to `~/.claude/agent-memory/forge-lead/MEMORY.md` with `helpful_count=0`, `harmful_count=0`, `authored_at=<promotion date>`, `last_triggered=null`.
4. Notify Akash with terse "promoted N skills" line.

## Skipped (this phase)

- **PyTorch Backend Protocol stub generation** — marked PARTIAL in `GAP_INVENTORY.md`. Existing `engineer` skill + `feature-dev` plugin can scaffold this when explicitly prompted; not worth a dedicated skill until repeated friction is observed across 2+ sessions.

---

## Phase 4 promotion executed (2026-05-01)

All 3 Phase-1 candidates promoted to `~/.claude/skills/`:

- `mps-kernel-debugging/SKILL.md` — exercised by Track A's MPS backend implementation + the dogfood swarm
- `metal-shader-profiling/SKILL.md` — referenced in research SYNTHESIS §3 (Apple Metal docs); not directly invoked but its existence steered Track A away from xcrun shell-out (security N1 disposition)
- `hatch-testpypi-release/SKILL.md` — informed Phase 4 build/tag flow; TestPyPI upload deferred to manual step pending ~/.pypirc

Promotion verdict: **3/3 ELIGIBLE** (all PASSed forge:test in Phase 1, all referenced or exercised in Phase 2-4).
