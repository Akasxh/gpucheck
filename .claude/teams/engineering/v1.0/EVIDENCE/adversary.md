# adversary.md

**Specialist**: engineering-adversary (post-build, pre-evaluator)
**Date**: 2026-05-01

## Audit of load-bearing assumptions vs external inputs

### A1 — SYNTHESIS §3 deadlock claim
- **Claim**: Calling `start.record(); end.record(); end.synchronize(); start.elapsed_time(end)` deadlocks on PyTorch 2.10+ Apple Silicon (pytorch#162872).
- **Verification path**: research SYNTHESIS §3 cites the issue and the github-miner crawl. We did NOT independently reproduce the deadlock on this machine (would risk permanently hanging the test session). The mitigation is "don't call this pattern at all", which we verified structurally via AST-introspected test in `test_backends.py::test_mps_backend_event_timer_uses_device_sync_not_event_sync`.
- **Verdict**: VERIFIED — the implementation correctly avoids the documented deadlock pattern; the AST test ensures future refactors can't reintroduce it.

### A2 — SYNTHESIS §7 PROVISIONAL 2× tolerance
- **Claim**: 2× multiplier is directionally correct; specific numbers are PROVISIONAL until calibrated on M-silicon.
- **Implementation**: `_MPS_TOLERANCE_MULTIPLIERS` shipped with explicit code comment marking PROVISIONAL. `compute_tolerance(device_type="mps")` applies the multiplier AFTER k_dim sqrt scaling.
- **Verdict**: REPORTED-NOT-VERIFIED quantitatively. The 2× direction is correct per SYNTHESIS; specific calibration is a follow-up Q-B per SYNTHESIS open-questions list.

### A3 — SYNTHESIS §2 12-bug xfail list
- **Claim**: The 12 entries are silent-correctness or crash bugs that cannot be tolerance-rescued.
- **Implementation**: `[tool.gpucheck.mps.xfail]` in pyproject.toml, parsed at session start, queryable via `gpucheck.is_mps_xfailed("op.subcategory")`. `test_pyproject_xfail_block_loaded_at_session_start` verifies all 12 entries are present.
- **Verdict**: VERIFIED — the xfail list IS the living document, and the test prevents accidental pruning.

### A4 — Security finding TM-E1 (`CUDA_HOME` injection)
- **Claim**: An attacker who can set `CUDA_HOME` could redirect gpucheck to execute an arbitrary binary named `compute-sanitizer`.
- **Verification path**: Track-C added `os.path.realpath` + allowlist with exact-prefix-with-separator match. Symlink attack covered by `test_find_compute_sanitizer_resolves_symlink_before_allowlist`. Lookalike attack (`/usr/local/cuda-evil`) covered by `test_is_allowed_cuda_home_rejects_lookalike_paths`.
- **Verdict**: VERIFIED — both vectors blocked.

### A5 — Security finding CFG-2 (workflow permissions)
- **Claim**: Default `GITHUB_TOKEN` has write-all scope; restricting to `contents: read` prevents accidental publishes.
- **Implementation**: Track-D added `permissions: contents: read` to `.github/workflows/ci.yml` top level.
- **Verdict**: VERIFIED — single-line, no lateral risk.

### A6 — Security finding DEP-1 (lockfile)
- **Claim**: Without a lockfile, `pip install -e ".[dev]"` resolves a different version graph each CI run.
- **Implementation**: Track-D committed `uv.lock` and the workflow now runs `uv sync --frozen --extra dev`.
- **Verdict**: VERIFIED — lockfile is committed and CI install path uses `--frozen`.

### A7 — N1-N5 prophylactics
- **N1, N2 (xcrun metal hardening)**: gpucheck v1.0 does NOT shell out to xcrun. PyTorch's MPS dispatcher does this internally. WAIVED per CHARTER.
- **N3 (mach task_info)**: gpucheck v1.0 uses `torch.mps.current_allocated_memory` and psutil RSS for memory; does NOT call mach `task_info`. WAIVED.
- **N4 (supply chain on `[mps]`)**: `[mps]` extra pins `torch>=2.6` (the floor where `torch.mps.synchronize()` is stable per SYNTHESIS §3). DEP-1 lockfile additionally pins downstream packages. PARTIALLY ADDRESSED.
- **N5 (MPS dispatch sanitizer TOCTOU)**: gpucheck v1.0 ships no MPS dispatch sanitizer (Apple does not ship one). WAIVED.

### A8 — REPORTED-NOT-VERIFIED items from SYNTHESIS
- **MSL fast-math ULP table**: REPORTED-NOT-VERIFIED. The PROVISIONAL 2× multiplier is the conservative interpretation; v1.1 should fetch the PDF and adjust.
- **`torch.mps.empty_cache()` between fuzz iterations fully eliminates pytorch#177116**: REPORTED-NOT-VERIFIED in SYNTHESIS Q-C. gpucheck v1.0 doesn't depend on this; it's an open question.

## Cross-team feedback to research

No `FEEDBACK_FROM_ENGINEERING.md` filed. All SYNTHESIS claims that engineering acted on are documented in code comments (PROVISIONAL multiplier, deadlock-safe pattern). The `_MPS_TOLERANCE_MULTIPLIERS` table cites SYNTHESIS §7 directly so the calibration follow-up has a clear pointer.

## Verdict

**PASS** — load-bearing claims are either VERIFIED in implementation/tests or explicitly carry a PROVISIONAL / WAIVED label backed by CHARTER. No load-bearing assumption is silently un-grounded.
