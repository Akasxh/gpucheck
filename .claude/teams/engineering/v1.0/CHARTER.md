# CHARTER — gpucheck v1.0 Engineering Phase 2

**Slug**: v1.0
**Date**: 2026-05-01
**Owner**: engineering-lead
**Tier**: COMPLEX (multi-file, cross-module, cites research SYNTHESIS, MPS backend headline)
**Operating mode**: adopted-persona (specialists executed as lens-passes, not sub-dispatched processes)

---

## Inputs (binding)

| Source | Path | Status |
|---|---|---|
| Research SYNTHESIS | `/Users/cero/Code/gpucheck/.claude/teams/research/v1.0/SYNTHESIS.md` | Binding spec |
| Security FINDINGS | `/Users/cero/Code/gpucheck/.claude/teams/security/v1.0/FINDINGS.md` | 3 MEDIUMs gate Phase 3 merge |
| Docs AUDIT | `/Users/cero/Code/gpucheck/.claude/teams/docs/v1.0/AUDIT.md` | Informational; Phase 3 fills |
| Repo state | `release/v1.0`, HEAD `a9a9d44`, baseline 117 passing tests, torch 2.11.0, MPS available | Current |
| MEMORY.md lessons | `~/.claude/agent-memory/engineering-lead/MEMORY.md` | Read; agent-persona files have no type system, sequential edits need read-back |

## Tier classification

**COMPLEX** — drives full roster:
- planner + architect + skeptic + executor + verifier + reviewer + adversary + evaluator + retrospector + scribe per track.
- 4 parallel worktree tracks. moderator only invoked if structural contradiction surfaces.

## Charter

Ship gpucheck v1.0 with:

1. **Track A (`feat/track-a-mps`)** — MPS backend (HEADLINE). Backend Protocol, CUDA backend extracted, MPS backend implemented, `@devices("mps")` works, MPS tolerance overlay (PROVISIONAL 2x, calibration plan), `[tool.gpucheck.mps.xfail]` config block (12 entries), N1-N5 prophylactic hardening, `pyproject.toml` `[mps]` and `[apple]` extras. Baseline 117 tests still pass. **Event API deadlock (pytorch#162872) is a release blocker — gpu_benchmark on MPS uses device-level sync, NEVER per-event `.synchronize()`.**

2. **Track B (`feat/track-b-strides`)** — Stride fuzzing. `StrideStrategy` (Hypothesis) + `fuzz_strides()` (deterministic corpus). 7 categories: row-major / column-major / broadcast-induced / transpose / slice / contiguous-after-clone / gather-induced. Wired into `parametrize_gpu`.

3. **Track C (`feat/track-c-thread-safety`)** — `contextvars.ContextVar` for tolerance override stack. Push/pop via context manager (no API change for callers). Failing-without-fix regression test under `concurrent.futures.ThreadPoolExecutor`. **Also mitigates TM-E1**: `os.path.realpath` + allowlist for `CUDA_HOME`/`CUDA_PATH` in `sanitizers/race.py:50-62`.

4. **Track D (`feat/track-d-bundle`)** — Reporting test coverage 0% → ≥90%, `reporting/html.py` static dashboard, `sanitizers/determinism.py` (`@requires_determinism`, `assert_deterministic`). **Also mitigates DEP-1**: commit `uv.lock`. **Also mitigates CFG-2**: add `permissions: contents: read` to `.github/workflows/ci.yml` (single-line, included in Track-D).

## Acceptance criteria (measurable)

- All 4 track branches exist with green pytest, mypy strict, ruff clean.
- Total test count: baseline 117 + new tests per track (≥30 net new across all tracks).
- DIFF_LOG.md populated with one row per file change per track.
- VERIFY_LOG.md populated with fresh pytest output per track.
- `[tool.gpucheck.mps.xfail]` config block present in `pyproject.toml` with 12 entries.
- Event-API deadlock NOT triggered by any test in Track A (verified by grep + reasoning, since MPS hardware deadlock cannot be deterministically tested in CI).
- 3 MEDIUM security findings (CFG-2, TM-E1, DEP-1) addressed or explicitly waived.
- 5 N-prefix prophylactics (N1-N5) addressed or explicitly waived in this CHARTER.

## Explicit waivers

- **N1 (xcrun metal subprocess hardening)**: gpucheck v1.0 does NOT call `xcrun metal` or `xcrun metallib` from the MPS backend. PyTorch's MPS dispatcher does this internally. We don't shell out. **Waiver: N1, N2 are out-of-scope; we depend on PyTorch's hardening upstream.** Future v1.1 may add `compile_shader` support; revisit then.
- **N3 (mach task_info reads)**: gpucheck v1.0 uses `torch.mps.current_allocated_memory()` and `torch.mps.driver_allocated_memory()` for MPS memory queries. We do NOT read `task_info` directly. **Waiver: N3 is out-of-scope.**
- **N4 (supply chain on `[mps]` extras)**: We pin the `[mps]` extra to `torch>=2.6` (the floor where `torch.mps.synchronize()` is stable per SYNTHESIS §3). No new third-party Apple-only packages introduced. **N4 partially mitigated** by the existing torch floor bump and DEP-1 lock file; no further action.
- **N5 (MPS dispatch sanitizer TOCTOU)**: gpucheck v1.0 ships no MPS dispatch sanitizer (the existing CUDA `compute-sanitizer` wrapper does not have an MPS analogue; Apple does not ship one). **Waiver: N5 is out-of-scope until Apple ships a sanitizer or MLX-style validator we can wrap.**

## Hard rules (binding)

- TDD on every public API change. Test-first, fail, then implement.
- Lazy imports for torch / pynvml / hypothesis / cupy preserved.
- mypy strict + ruff E/F/W/I/N/UP/B/A/SIM/TCH must remain clean.
- No `# type: ignore` without an inline rationale comment.
- Conventional commits going forward (`feat(scope): message`); existing `[ Type ] :` history stays.
- No track merges to `release/v1.0` in this dispatch (Phase 3 work).
- No `--force` push, no `git reset --hard` against published commits, no `--no-verify`.
- Verify before claiming. Fresh pytest run captured into VERIFY_LOG.md before any "PASS" claim.

## Cross-team handback policy

If engineering discovers a research SYNTHESIS claim is wrong, file `FEEDBACK_FROM_ENGINEERING.md` with classification BLOCKER / DEGRADE / INFORMATIONAL.

If engineering accepts a security finding's recommended skeleton, the diff is the artifact. If we waive, we list the rationale in this CHARTER (above).

## Verdict

CHARTER ready. Proceed to PLAN.md.
