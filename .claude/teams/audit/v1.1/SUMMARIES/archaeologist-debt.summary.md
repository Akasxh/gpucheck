# archaeologist-debt summary (W1)

**Top-5 hot-spot files (8/8/8/6/6 edits):**
- src/gpucheck/plugin.py
- src/gpucheck/assertions/close.py
- src/gpucheck/arch/detection.py
- src/gpucheck/sanitizers/memory.py
- src/gpucheck/assertions/tolerances.py

All show critical→high→medium→low→mypy waterfall + ad-hoc post-release fixes. **Zero `refactor:` commits in entire history.**

**tolerances.py is now ~50/50 line ownership** between pre-v1.0 and post-v1.0 eras — MPS table layered onto CUDA table with no unifying scaling law.

**Conventional commits:** lifetime 11/41=27%, post-v1.0 cutover 11/11=100%. CONTRIBUTING says "no history rewrite" — release-please/semantic-release will choke on the 30 bracket commits unless given `since v1.0.0rc1`.

**Disguised reverts (no literal reverts in 45 commits):**
- `28d808e [ Fix ] : mypy strict mode errors` is **-287 line net commit across 86 files** — silent rollback of speculative type hints
- `22780ae` moved GPU tests under `tests/gpu_integration/` (0/0 LOC pure rename) — **de-facto disabling GPU CI gate**. The "No GPU CI" weakness in CLAUDE.md is the surviving consequence.

**Lost work: none.** One dangling tree from worktree reflog noise — should `git worktree prune`.

**Test/src LOC ratio:** 0 → 0.241 → drifted DOWN to 0.229 across critical/high/medium/low sweeps (no test-first discipline) → jumped to 0.788 at HEAD. Healthy direction, but coverage % never reported in commits or CI.

**CLAUDE.md "Known Weaknesses" is STALE** — items it claims as gaps (reporting coverage, MPS, strides, thread-safety) were all delivered. Cannot trust as backlog of record.
