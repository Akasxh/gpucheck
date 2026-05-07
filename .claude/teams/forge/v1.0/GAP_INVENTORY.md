# GAP_INVENTORY — Phase 2 capability scan

Inputs scanned:
- `~/.claude/skills/` (103 user-installed skill packages)
- `~/.claude/plugins/installed_plugins.json` (8 plugins)
- `~/.claude/plugins/marketplaces/claude-plugins-official/plugins/` (35 first-party)
- `~/.claude/plugins/marketplaces/claude-plugins-official/external_plugins/` (16 external)
- gpucheck CLAUDE.md (lists "no AMD ROCm or Intel XPU support" but not MPS)

Method: keyword diff against the four Phase 2 capabilities. Verdicts:

| # | Phase-2 capability | Verdict | Evidence |
|---|---|---|---|
| 1 | **MPS dispatch** (running gpucheck against `torch.mps`, device selection, mps event timing) | **GAP** | No skill mentions `torch.mps`, `MPSDevice`, or `mps.synchronize`. `cosmos-policy` covers EGL rendering only; nothing in installed plugin set. gpucheck `arch/` only covers SM60-SM120 (NVIDIA Pascal-Blackwell). |
| 2 | **Metal shader debugging** (Metal kernel correctness, profiling via Xcode Instruments / `xctrace`) | **GAP** | `swift-lsp` plugin covers `.swift` LSP only — not Metal/`xctrace`/`metal-tt`/Instruments traces. No skill mentions `xcrun metal`, `MTLCaptureManager`, or `metal-debugger`. |
| 3 | **PyTorch backend stub generation** (Backend Protocol + concrete impls scaffolding) | **PARTIAL** | `feature-dev` plugin offers a generic 7-phase architecture scaffold, and `engineer` skill can run plan-then-build. Neither is opinionated about Python `Protocol` + dispatch-table backend stubs. The pattern (Protocol class + per-backend module + registry) is narrow enough to deserve its own skill — but a generic skill could be drafted, OR the existing `engineer` skill could be invoked with explicit instructions. Marked PARTIAL because shelf solutions exist with prompting overhead. |
| 4 | **Hatch-based PyPI release pipeline** (build wheels with `hatch build`, upload to TestPyPI, sign with sigstore) | **GAP** | No skill mentions `hatch`, `sigstore`, `pypa/gh-action-pypi-publish`, or TestPyPI. `commit-commands` and `session-report` cover git mechanics but not packaging. gpucheck `pyproject.toml` already declares hatchling, so the toolchain is committed; only the release-flow skill is missing. |

## Ranking for this session (≤3 drafts)

Charter prioritizes: `mps-kernel-debugging`, `metal-shader-profiling`, `hatch-testpypi-release`.

That maps onto rows 1+2+4 above. Row 3 (backend-stub) is PARTIAL — it can ride on the existing `engineer` skill in Phase 2 and need not be drafted now. **Decision: 3 drafts on rows 1, 2, 4.**

## Cross-collision check

Confirmed no name collisions in `~/.claude/skills/`:
- `mps-kernel-debugging` — none
- `metal-shader-profiling` — none
- `hatch-testpypi-release` — none

Confirmed no agent collisions in `~/.claude/agents/` (none of these tokens appear in `~/.claude/agents/forge-lead.md` or sibling agent files at the level scanned).

## Notes for Phase 2/3 callers

- `mps-kernel-debugging` and `metal-shader-profiling` are **macOS-only**. Skills should declare this in description (`when-to-use: macOS Apple Silicon only`) so they don't trigger on Linux/CUDA tasks. Use narrow paths/triggers.
- `hatch-testpypi-release` is Akash-flow-specific (gpucheck uses hatchling). Description should anchor on `pyproject.toml` containing `[build-system] requires = ["hatchling"]`.
