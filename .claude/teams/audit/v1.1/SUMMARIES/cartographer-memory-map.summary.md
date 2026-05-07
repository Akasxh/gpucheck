# cartographer-memory-map summary (W2)

**Total memory artifacts:** 4 MEMORY.md, 8 staging, 5 PROTOCOL.md, 4 hooks, 6 scripts, 78 personas, 106 skills, 488 in gpucheck team trees, 5 root-level project files, 6 _write_audit.log.

## Per-lead MEMORY.md status (7 leads)

- **substantive (1):** `research-lead` — 262 lines, 26 H3 lessons, 2 merge-sections (last update from 2026-04-12)
- **light (3):** `engineering-lead` (32 lines), `forge-lead` (31 lines), `research-retrospector` (32 lines)
- **empty/missing (3):** `docs-lead`, `security-lead`, `testing-lead` — only staging/, no canonical MEMORY.md

## Staging files awaiting merge

8 files. Sessions: gpucheck v1.0 (2026-05-01, 6 files), upgrade-mcp-tools-deterministic (2026-05-06, 1 file), engineering's duplicate v1.0-gpucheck.md mirror.

## Disposition

**KEEP — promote in v0.3 (5):**
- docs (7 lessons, 145 lines)
- engineering v1.0 (4 lessons, 33 lines)
- engineering MCP (3 lessons, 24 lines)
- research (3 lessons, 158 lines)
- testing (5 lessons, 98 lines)

**DROP — stale (3):**
- forge stub (3 lines, no retrospector ran)
- security stub (3 lines, no retrospector ran)
- engineering v1.0-gpucheck mirror (57 lines, pure duplicate)

**DEFER:** none

## 2 structural anomalies

1. `forge/v1.0/PROMOTIONS.md` lists 3 skills as "NOT yet promoted" but `~/.claude/skills/` already has all 3 (mps-kernel-debugging, metal-shader-profiling, hatch-testpypi-release). State drift.
2. `~/.claude/agents/{security,docs,testing}/` lack inline PROTOCOL.md; only research/ and engineering/ ship one.
