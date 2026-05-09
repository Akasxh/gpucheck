# Agentic development record

This directory is gpucheck's open record of how the codebase is built using Claude Code as a multi-agent workforce. It exists for two audiences:

1. **A future Claude Code session opening this repo cold.** Read this file first, then `SESSION_PROMPT_v3.md` (the latest), then the most recent `SESSION_N_REPORT.md`. That's enough context to continue the methodology without re-deriving it.
2. **Anyone reproducing the workflow.** Each `SESSION_PROMPT_vN.md` is a paste-able orchestrator prompt. Each `DELTA_*.md` explains why the prompt evolved between versions. Reports record what each session actually shipped.

## What gets shipped per session

- **Session 1 (2026-05-01)** — produced `gpucheck v1.0.0rc1` in PR #2 and `claude-forge v0.2-rc` in PR #1. Architecture defect (subagents tried to spawn subagents) capped wall-clock at ~1h20m. Report: [`SESSION_1_REPORT.md`](SESSION_1_REPORT.md).
- **Session 2 (2026-05-07)** — merged PR #2 → `v1.0.0` final, opened PR #3 (`v1.1` — per-(kernel,dtype) MPS overlay, Apple-tile fuzzing, xfail expansion 12→43, silent-downcast catcher, systemic refactors), shipped `claude-forge v0.2` final + `BENCHMARKS_v0.2.md`.
- **Sessions thereafter** — see the most recent `SESSION_N_REPORT.md`.

## Files

| File | Purpose |
|---|---|
| [`SESSION_PROMPT_v3.md`](SESSION_PROMPT_v3.md) | Latest paste-able orchestrator prompt. Use this. |
| [`SESSION_PROMPT_v2.md`](SESSION_PROMPT_v2.md) | Previous prompt. Kept for the architectural fix it introduced (orchestrator-dispatches-everyone). |
| [`SESSION_PROMPT_v1.md`](SESSION_PROMPT_v1.md) | First prompt. Kept for completeness; do not re-use as-is. |
| [`DELTA_v2_to_v3.md`](DELTA_v2_to_v3.md) | Why v3 differs from v2 (compounding-sessions narrative). |
| [`DELTA_v1_to_v2.md`](DELTA_v1_to_v2.md) | Why v2 differs from v1 (fake-parallelism diagnosis + fix). |
| [`SESSION_1_REPORT.md`](SESSION_1_REPORT.md) | Session 1 metrics + honest carry-over. |
| [`LAUNCH.md`](LAUNCH.md) | Pre-flight checklist before pasting a session prompt. |
| [`INVENTORY.md`](INVENTORY.md) | Available skills + plugins + teams snapshot. |

## Methodology in one paragraph

A single Claude Code session in this repo runs as a multi-team workforce: a research team investigates load-bearing facts, an engineering team ships code in plan-then-build phases with verifier and reviewer loops, security and testing teams audit independently, a docs team handles changelogs and migration, and a capability forge authors any new skills the workforce needs. All of this is dispatched directly by the main orchestrator thread (subagents cannot spawn subagents in Claude Code). Communication is file-on-disk, turn-based, gated by adversarial review. Sessions compound: lessons from session N feed into session N+1's research, and the same workforce ships both `gpucheck` and its own framework `claude-forge`.

## How to start the next session

1. Open a fresh Claude Code session in this repo (`cd ~/Code/gpucheck && claude`).
2. Read this file, then read `SESSION_PROMPT_v3.md`, then read the latest `SESSION_N_REPORT.md`.
3. If the prompt is still the right shape for the work ahead, paste it. If not, write a `SESSION_PROMPT_v4.md` that addresses what's changed, save it here, then paste it.

## Provenance

These files are mirrored from the working notes at `/Users/cero/Desktop/yc-session/` and committed here so the methodology is reproducible from a clean clone. Original prompts authored by Akash with Claude assistance.
