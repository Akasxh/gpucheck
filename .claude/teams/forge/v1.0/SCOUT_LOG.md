# SCOUT_LOG — Phase 2 capability scouting

Trust-tier convention (forge-lead heuristic): STRONG-PRIMARY = official upstream + verified-working; MIXED = exists but unverified for this use; NONE = not found.

## Sources queried

1. **MCP Registry** (`https://registry.modelcontextprotocol.io/v0/servers`) — official Anthropic-endorsed registry.
2. **anthropics/skills** GitHub repo (`/repos/anthropics/skills/contents/skills`) — first-party Anthropic skill catalog. 16 entries listed (algorithmic-art, brand-guidelines, canvas-design, claude-api, doc-coauthoring, docx, frontend-design, internal-comms, mcp-builder, pdf, pptx, skill-creator, slack-gif-creator, theme-factory, web-artifacts-builder, webapp-testing, xlsx).
3. **claude-plugins-official marketplace** on disk (35 first-party plugins + 16 external).
4. **103 user-installed skills** under `~/.claude/skills/`.

## Per-gap findings

### Gap 1 — MPS dispatch (`mps-kernel-debugging`)

| Source | Hit | Trust |
|---|---|---|
| MCP Registry `?search=mps` | (not queried — too generic, would return false positives) | — |
| MCP Registry `?search=metal` | empty array | NONE |
| anthropics/skills | none of 17 entries match | NONE |
| claude-plugins-official | `swift-lsp` covers `.swift` only; nothing for `torch.mps` | NONE |
| user skills | 103 ML-stack skills, none mention MPS | NONE |

**Verdict: GENUINE GAP.** Author new skill.

### Gap 2 — Metal shader debugging (`metal-shader-profiling`)

| Source | Hit | Trust |
|---|---|---|
| MCP Registry `?search=metal` | empty | NONE |
| MCP Registry `?search=xcode` | (skipped — derivable from prior empty result) | — |
| anthropics/skills | none | NONE |
| claude-plugins-official | `swift-lsp` README explicitly says "code intelligence for Swift projects" — does not handle Metal shading language `.metal` files or Instruments traces | NONE |

**Verdict: GENUINE GAP.** Author new skill.

### Gap 3 — Backend stub generation (PARTIAL — not drafting)

`feature-dev` plugin (claude-plugins-official) provides a 7-phase architecture flow that can produce a Backend Protocol scaffold when prompted. The `engineer` skill (already installed) wraps plan-then-build with a verifier loop. Either is sufficient. Skipping a dedicated skill — Phase 2 caller can invoke `engineer` with the scaffolding instructions inline.

**Verdict: shelf solution exists.** Do not draft.

### Gap 4 — Hatch + sigstore PyPI release (`hatch-testpypi-release`)

| Source | Hit | Trust |
|---|---|---|
| MCP Registry `?search=hatch` | 1 result: `com.hatchable/hatchable` — this is a webapp host, NOT the Python `hatch` build tool. False positive. | NONE (rejected) |
| MCP Registry `?search=sigstore` | (skipped — registry is sparse for tooling-DSL gaps) | — |
| anthropics/skills | none of 17 entries match | NONE |
| claude-plugins-official | no plugin covers PyPI publishing or sigstore signing | NONE |
| Authoritative primary docs available for skill body: `hatch.pypa.io` (PyPA-official), `pypi.org/help/#apitoken`, `docs.sigstore.dev`, `pypa/gh-action-pypi-publish` README on GitHub | STRONG-PRIMARY (upstream-curated) | usable as skill citation |

**Verdict: GENUINE GAP.** Author new skill, citing PyPA + sigstore upstream docs in the skill body.

## Scout summary

- 3 gaps × scout query = 3 single-query lookups (under the scout's "category-bounded" limit per persona).
- 0 shelf solutions found for the 3 prioritized gaps.
- 1 PARTIAL (Backend stub generation) deferred to existing `engineer` skill.
- All 3 prioritized gaps proceed to draft.
