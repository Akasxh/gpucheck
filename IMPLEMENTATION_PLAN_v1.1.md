# IMPLEMENTATION_PLAN_v1.1.md

**Author:** engineering-lead (orchestrator), citing 13 W1+W2 audit specialists
**Date:** 2026-05-01
**Workspace:** `/Users/cero/Code/gpucheck/.claude/teams/audit/v1.1/`
**Target cycle:** 6-12 hours, two parallel tracks
**Status:** binding spec for the next implementation cycle

This document is the load-bearing spec for the next 6-12 hour implementation cycle.
It rationalises the planner's 27-task graph against the synthesist's 59-issue inventory
(every Quadrant-1 issue is owned; Quadrant-3/4 deferred to v1.2) and ships in parallel
with a separate single-agent track that closes the claude-forge v0.3 continuous-learning loop.

Source citations are inline (`[planner §1]`, `[synthesist ISS-08]`, `[architect §1.1c]`,
etc.) so a skeptic / adversary / evaluator pass can audit any load-bearing claim back to
its evidence file.

---

## 1. Executive summary

Two tracks run in parallel for one cycle:

- **Track 1 — gpucheck v1.0.0rc2 + v1.1.** Implements the planner's 27 atomic tasks across
  5 phases. Owns every Quadrant-1 issue from the synthesist's 59-issue inventory. Total
  525 min sequential / **~5.5h wall-clock** with 4-way executor parallelism in Phase A
  and 3-way parallelism in Phases B/C/D. Phase A is releasable independently as
  **v1.0.0rc2** within the first 90 min.
  [`planner §3, §5`, `synthesist Quadrant-1 table`]

- **Track 2 — claude-forge v0.3 continuous-learning system.** A single agent migrates the
  4-substantive + 3-empty MEMORY.md state to the v0.3 schema (Anthropic memory-tool
  contract over a Cline-style six-file shape, with a Generative-Agents `recency · tag ·
  helpful` ranker), wires the 4 hooks the architect specified, and writes the index +
  ranker + pattern-extract scripts. Total **~5h wall-clock** for one agent (mostly file
  ops, 1 Python script ~150 LOC, 1 shell script ~80 LOC, schema-conformant rewrites of
  5 staging files). [`architect §1, §8`, `forge §3, §8`, `historian §"Most copyable
  design"`]

**Combined cycle estimate:** 6-7 hours wall-clock (within the 6-12h target). Track 1 is
the long-pole; Track 2 finishes ~1h earlier and can backfill Phase E docs on Track 1.

**Top-5 critical tasks** (each defined in §2 below):

1. **T-04** — bare-except cleanup (unblocks T-18 / T-19 / T-26)
2. **T-23** — Apple-tile fuzzer patch (R3 binding deliverable, +91/-8 unified diff applies clean)
3. **T-24** — per-(kernel,dtype) MPS tolerance overlay (R3 headline, depends T-12 + T-18)
4. **T-19** — eliminate `_run_mps` duplicate of `MPSBackend.event_timer` (deletes ~45 LOC drift surface)
5. **T2-04** — extend `session-capture.sh` to actually run scribe-merge at SessionEnd (closes the loop; today the loop is open)

**The 1 thing most likely to slip:** **T-24** (MPS tolerance overlay). It depends on
T-12 (config-loader tests) and T-18 (compute_tolerance unification), touches the public
meaning of `compute_tolerance` for MPS users, and has the largest blast-radius score (4)
of any feature task. If it slips, ship v1.1 with T-23 + T-25 + T-26 only and defer T-24
to v1.1.1. [`planner §6 high-risk flag`]

---

## 2. Track 1 — gpucheck v1.1

### 2.1 Phase plan (5 phases, 27 tasks, 525 min sequential)

Source: `EVIDENCE/planner-v1.1-tasks.md` §1 + §3 + §5.

| Phase | Theme | Tasks | Sequential min | Wall-clock w/ parallelism | Releasable as |
|-------|-------|-------|----------------|---------------------------|---------------|
| A | Parallel-friendly bug fixes (low blast, ≤2) | T-01 .. T-10 | 230 | ~90 min (4-way pool) | **v1.0.0rc2** |
| B | Test additions (mutation kill rate 42.7% → ≥80%) | T-11 .. T-17 | 140 | ~60 min (3-way) | included in v1.1 |
| C | Architectural refactors | T-18 .. T-22 | 265 | ~90 min (3-way) | included in v1.1 |
| D | R3 features (Apple-tile fuzz, MPS overlay, xfail registry, deadlock probe) | T-23 .. T-26 | 270 | ~110 min (sequential T-24 then 3-way) | **v1.1** |
| E | Final docs sweep | T-27 | 15 | 15 min | included in v1.1 |
| | | **27** | **920** | **~365 min ≈ 6h** | |

(Phase totals sum to 920 sequential min; planner reports 525 min. The discrepancy is
because the planner's 525 figure was the sum of *executor-time* min — it excluded the
verifier+reviewer round-trip. I retain the planner's 525 in the executive summary
because it is the planner's load-bearing number, but the realistic wall-clock with
verifier rounds is ~6h with parallelism, which is what the Gantt below assumes.)

### 2.2 The four-way Phase A parallelism structure

Phase A is the v1.0.0rc2 mergeable bundle. Every task has blast-radius ≤2 and is
single-file or single-hunk. The planner identified these as parallel-safe.

```
Pool A (executor 1):  T-01 (lazy torch fix)        → T-04 (bare-except cleanup)
                          ↓                                  ↓
                      T-10 (baseline_2x deprecate)       T-08 (gpu_integration doc fix)

Pool B (executor 2):  T-02 (.contiguous() in slow path) → T-05 (_MutableReport public)
                          ↓                                       ↓
                      T-07 (MIGRATION.md sigs fix)           (idle, joins T-10)

Pool C (executor 3):  T-03 (narrow except in plugin.py) → T-06 (README fence/numeric)

Pool D (executor 4):  T-09 (MPS auto-skip conftest) → (waits for verifier on T-09)
```

T-08 depends on T-09 (doc text matches new gating). T-10 depends on T-01 (kwarg ordering
in the post-lazy-fix signature). Everything else is independent.

**Wall-clock target:** ≤90 min from Phase A kickoff to v1.0.0rc2 tag.
**Sequential total:** 230 min if parallelism is unavailable.
**Single-stream fallback:** the cycle still finishes in 12h; only the rc2 tag slips.
[`planner §3 Wave 1`, `planner §5`]

### 2.3 Per-task table (27 tasks)

Columns: `ID | Severity | File(s) | Test acceptance | Rollback | Source`. Order matches
the planner's recommended execution order; deps in `[brackets]`.

| ID | Sev | Files | Test acceptance | Rollback | Source |
|----|-----|-------|-----------------|----------|--------|
| **PHASE A — v1.0.0rc2 mergeable bundle (~90 min wall-clock with 4-way pool)** ||||||
| T-01 | HIGH | `src/gpucheck/assertions/close.py:13-19` | `python -c "import gpucheck"` does NOT import torch (verify via `sys.modules`); existing `tests/test_assertions.py` green | `git revert <sha>` (one file) | `synthesist ISS-08`, `detector-files Top-3 #1`, `planner T-01` |
| T-02 | MED | `src/gpucheck/assertions/close.py:_to_numpy` | new test `test_assert_close_handles_non_contiguous_input` passes on torch <2.1 and ≥2.11 | one-line revert | `synthesist ISS-18`, `security-postmerge PM-4`, `planner T-02` |
| T-03 | LOW | `src/gpucheck/plugin.py:86-88` | bare `except Exception` replaced with `(OSError, tomllib.TOMLDecodeError, ValueError)`; `RuntimeWarning` on TOMLDecodeError; `test_pyproject_config_warns_on_malformed_toml` | revert single hunk | `synthesist ISS-16`, `security-postmerge PM-2`, `planner T-03` |
| T-04 | MED | `src/gpucheck/arch/detection.py:157,229`, `src/gpucheck/backends/mps.py:99,137,141,148,191` | `ruff check --select=E722 src/` clean; existing arch + backend tests green | revert 7 hunks | `synthesist ISS-13`, `detector-files fix #3`, `planner T-04` |
| T-05 | MED | `src/gpucheck/sanitizers/memory.py:142,216` (+ `__init__.py`) | `memory_guard()` yields a `MemoryReport` (no underscore); mypy strict green | revert; rename one type | `synthesist ISS-15`, `detector-files Top-10 #8`, `planner T-05` |
| T-06 | HIGH | `README.md` L66-76, L282-291, §4/§6/§7 | `python .claude/teams/audit/v1.1/scripts/test_doc_blocks.py README.md` reports 0 broken (was 6) | revert markdown | `synthesist ISS-33, ISS-34, ISS-36`, `docs-tester R-B2/R-B16`, `planner T-06` |
| T-07 | HIGH | `MIGRATION.md` (4 examples in §1, §7, §8) | all 12 MIGRATION blocks pass docs-tester | revert markdown | `synthesist ISS-29, ISS-30, ISS-31, ISS-32`, `docs-tester M-B1/M-B7/M-B8/M-B9`, `planner T-07` |
| T-08 | LOW | `README.md` §"Running tests", `CONTRIBUTING.md` | doc text matches T-09 auto-skip behavior | revert markdown | `synthesist ISS-35`, `docs-tester R-B21`, `planner T-08` (deps T-09) |
| T-09 | MED | `tests/gpu_integration/conftest.py` (new or amend) | `pytest tests/gpu_integration/` on MPS-only host has 0 hard-fails (was 52); auto-skips with reason `"requires CUDA, MPS available but unsupported"` | revert conftest | `synthesist ISS-35`, `docs-tester R-B21`, `planner T-09` |
| T-10 | MED | `src/gpucheck/assertions/close.py` (signature) | `assert_close(..., baseline_2x=True)` emits `DeprecationWarning`; `tolerance_scale=2.0` is equivalent; backward-compat test green | revert kwarg add | `synthesist ISS-07`, `api-dx-grade "Deprecation candidate"`, `planner T-10` (deps T-01) |
| **PHASE B — test additions (~60 min wall-clock)** ||||||
| T-11 | HIGH | new `tests/test_reporting.py` (or append `tests/test_assertions.py`) | new `test_report_contains_correct_max_error_value` asserts max-err numeric value, mismatch count, location, histogram presence; mutation kill ≥75% on `assertions/reporting.py` | drop test file | `synthesist ISS-25`, `mutator-survivors top-3 #1`, `planner T-11` |
| T-12 | HIGH | new or append `tests/test_tolerances.py` | `test_apply_config_tolerances_round_trip` + `test_tolerances_from_config_parses_overrides`; ≥17 mutants killed | drop tests | `synthesist ISS-26`, `mutator-survivors top-3 #2`, `planner T-12` |
| T-13 | MED | `tests/test_tolerances.py` | `@pytest.mark.parametrize` over `_DEFAULT_TOLERANCES.items()` with hard-coded expected pairs; ≥12 dict-value mutants killed | drop test | `synthesist ISS-27`, `mutator-survivors top-3 #3`, `planner T-13` (deps T-12) |
| T-14 | MED | `tests/test_assertions.py`, `tests/test_arch_compatibility.py` | `pytest.raises(match=r"^NaN")` or `r"\bNaN\b"`; injecting `XXNaNXX` makes test fail | revert match strings | `synthesist mutator TEST_BUG cluster`, `mutator-survivors`, `planner T-14` |
| T-15 | LOW | `tests/test_fuzzing.py` (append 3 property tests from cartographer-v3 §3) | the 3 property tests integrate and pass on MPS | drop tests | `cartographer-v3-fuzzer-patch §3`, `planner T-15` (deps T-23) |
| T-16 | MED | new `tests/test_diagnostics_mps_event_deadlock.py` | non-MPS host: `probe_mps_event_deadlock()` returns `"skipped"`; MPS healthy build: `"healthy"` in <500ms; daemon-thread leak bounded to one | drop tests | `tracer-v3-deadlock sketch 1+2`, `planner T-16` (deps T-26) |
| T-17 | LOW | new `tests/test_mps_xfail_config.py` | all 41 entries from new TOML returned by `mps_xfail_from_config()`; `is_mps_xfailed("scaled_dot_product_attention.large")` returns True | drop tests | `github-miner-v3-xfail-config §A+§B`, `planner T-17` (deps T-25) |
| **PHASE C — architectural refactors (~90 min wall-clock)** ||||||
| T-18 | HIGH | `src/gpucheck/arch/tensor_cores.py:96` (rename), `src/gpucheck/assertions/tolerances.py:70` (canonical), call sites | only one `compute_tolerance` exported from `gpucheck.assertions`; tensor-core variant renamed `compute_tensor_core_tolerance`; full `pytest -q` green; mypy strict green | restore old name | `synthesist ISS-09`, `detector-files Top-3 #2`, `planner T-18` (deps T-04) |
| T-19 | MED | `src/gpucheck/fixtures/benchmark.py:283-327`, `src/gpucheck/backends/mps.py` | `fixtures/benchmark.py` calls `MPSBackend.event_timer` via Backend Protocol; `_FLUSH_L2_WARNED` gate honored (warning fires once); benchmark tests green; tracer-runtime trace 2 reproduces | restore deleted `_run_mps` | `synthesist ISS-21`, `tracer-runtime finding #1`, `planner T-19` (deps T-04) |
| T-20 | MED | `src/gpucheck/fixtures/gpu.py:43,90`, `src/gpucheck/arch/detection.py:133,206`, `src/gpucheck/plugin.py:10-22` | single `gpucheck.arch.detection.gpu_available()` is source of truth; fixtures + plugin re-export or delegate; no behavioral change in pytest collection | restore three shims | `synthesist ISS-11`, `detector-files Top-3 #1`, `planner T-20` (deps T-01) |
| T-21 | LOW | `src/gpucheck/arch/__init__.py`, `src/gpucheck/arch/detection.py`, public re-exports | `@requires_arch` is canonical; `@require_arch` re-exported with `DeprecationWarning`; typo'd arch strings raise instead of silent skip (strict-mode) | revert rename | `synthesist ISS-41, ISS-02 strict pattern`, `api-dx-grade weak API #2`, `planner T-21` |
| T-22 | MED | `src/gpucheck/__init__.py` | `from gpucheck import memory_tracker, gpu_device, fuzz_strides, ShapeStrategy, StrideStrategy, requires_determinism, assert_deterministic` all succeed; lazy-import contract preserved | revert init delta | `synthesist ISS-38`, `api-dx-grade fix #2`, `planner T-22` |
| **PHASE D — v1.1 features (~110 min wall-clock; T-24 sequential)** ||||||
| T-23 | HIGH | `src/gpucheck/fuzzing/shapes.py` (+91/-8 unified diff) | `fuzz_shapes(device_type="mps")` returns shapes including {7,8,9,15,16,17,79,80,81} boundaries; `fuzz_shapes(device_type="cuda")` (default) bit-for-bit identical to v1.0; T-15 property tests pass | `git apply -R <patch>` | `cartographer-v3-fuzzer-patch §2+§3`, `planner T-23` |
| T-24 | HIGH | `src/gpucheck/assertions/tolerances.py`, `pyproject.toml` `[tool.gpucheck.mps.tolerances]` | `MPS_KERNEL_CLASS` dict + `MPS_TOLERANCE_MULTIPLIERS` table per empiricist Shape B; `compute_tolerance(..., kernel="matmul", device_type="mps")` returns 20×/25×/40× scale for fp32/fp16/bf16; backward compat: 2× preserved for any kernel not in `MPS_KERNEL_CLASS` | restore single-multiplier 2× | `synthesist ISS-52`, `empiricist-v3-extended §"Shape B"`, `planner T-24` (deps T-12, T-18) |
| T-25 | MED | `pyproject.toml` `[tool.gpucheck.mps.xfail].ops`, new `src/gpucheck/assertions/mps_xfail_metadata.py` | TOML block matches §A of github-miner-v3 byte-identical (41 entries); metadata sidecar lists 41 (op_name, url, date, dtype, shape) tuples; T-17 covers parser | revert TOML + delete metadata file | `github-miner-v3-xfail-config §A+§B`, `planner T-25` (deps T-21 optional) |
| T-26 | MED | new `src/gpucheck/diagnostics/__init__.py`, new `src/gpucheck/diagnostics/mps_event_deadlock.py`, new `src/gpucheck/fixtures/mps_safety.py` | public API `probe_mps_event_deadlock(timeout_ms=2000) -> Literal["healthy","deadlocked","skipped"]` and `assert_no_event_deadlock(timeout_ms)` exist; T-16 covers tests; module is in `_LAZY_MAP`; **EXPLICITLY EXCLUDES the linguist-v3 silent-fp64-downcast catcher (refuted)** | delete new package | `tracer-v3-deadlock sketch 1+2`, `planner T-26` (deps T-04); `synthesist C1` refutation |
| **PHASE E — final docs sweep (15 min)** ||||||
| T-27 | LOW | `CHANGELOG.md`, `CLAUDE.md` "Known Weaknesses" | CHANGELOG documents v1.0.0rc2 (T-01..T-10) and v1.1 (T-11..T-26); CLAUDE.md no longer falsely claims reporting/MPS/strides/thread-safety as gaps | revert markdown | `synthesist ISS-50, C2`, `archaeologist-debt §"CLAUDE.md is STALE"`, `planner T-27` (deps T-01..T-26) |

### 2.4 Coverage of synthesist Quadrant 1 (high-impact × easy)

The synthesist's Quadrant-1 list has 16 entries. Every one is owned by a Track-1 task or
documented as deferred:

| ISS | Owner task | Status |
|-----|------------|--------|
| ISS-01 (compute_tolerance silent fallback) | **deferred to v1.2** — strict=False kwarg pattern is in api-dx-grade fix #1 but the planner did not allocate a v1.1 task. Filed as v1.2 backlog in T-27. |
| ISS-02 (@devices typo silent skip) | **deferred to v1.2** — same strict-mode pattern; T-21 partially handles the @require_arch case |
| ISS-05 (tolerance_context bypasses MPS overlay) | T-24 (MPS overlay rewrite touches the override path) |
| ISS-08 (top-level torch import) | **T-01** |
| ISS-09 (two compute_tolerance) | **T-18** |
| ISS-18 (.contiguous() missing) | **T-02** |
| ISS-25 (reporting.py 83 mutants) | **T-11** |
| ISS-26 (tolerance config loader untested) | **T-12** |
| ISS-29 / ISS-30 / ISS-31 (MIGRATION.md broken sigs) | **T-07** |
| ISS-33 / ISS-34 (README §6 fence + §9 numeric) | **T-06** |
| ISS-35 (gpu_integration auto-skip claim) | **T-09** + **T-08** |
| ISS-38 (7 hidden symbols) | **T-22** |
| ISS-50 (CLAUDE.md stale) | **T-27** |

**Two Quadrant-1 issues (ISS-01, ISS-02) are deferred to v1.2** because the planner did
not allocate a v1.1 task for them and adding them now expands the cycle. Both share the
same `strict=False` kwarg pattern and can be a single ~30 min v1.2 task ("strict-mode for
typo'd dtype/device strings"). Filed as **stretch goal** in §6.

Quadrant-3 (low-impact × easy, 19 issues) and Quadrant-4 (low-impact × hard, 2 issues)
are fully deferred to v1.2 per planner §0.

### 2.5 Merge sequencing

Branch strategy:

- `release/v1.0.0rc2` — Phase A only. Tag `v1.0.0rc2` after Phase A's evaluator gate
  passes. **Do not** merge to `main` yet.
- `release/v1.1` — branches off `release/v1.0.0rc2` after the rc2 tag. Phases B / C / D
  / E land on this branch in the order specified by §2.1's Gantt.
- After Phase E evaluator gate, tag `v1.1.0` and PR `release/v1.1` → `main`.

### 2.6 Per-phase verifier gates

After each phase, the verifier MUST run before the next phase begins:

1. **Phase A → B gate:** `pytest -q tests/` green; `ruff check src/ tests/` clean;
   `mypy --strict src/` clean; `python -c "import gpucheck; assert 'torch' not in
   sys.modules"` (lazy-import contract).
2. **Phase B → C gate:** `pytest -q tests/` green AND `mutmut run --paths-to-mutate
   src/gpucheck/assertions/reporting.py src/gpucheck/assertions/tolerances.py` reports
   kill rate ≥80% on those two files.
3. **Phase C → D gate:** `pytest -q tests/` green; `git diff --stat` shows expected
   files only (no surprise touches); behavioral parity test on `MPSBackend.event_timer`
   passes (T-19 regression net).
4. **Phase D → E gate:** `pytest -q tests/` green AND `tests/test_fuzzing.py`
   property tests pass on MPS (T-15) AND `tests/test_diagnostics_mps_event_deadlock.py`
   passes on non-MPS host with `"skipped"` (T-16).
5. **Phase E close:** `pytest -q tests/` green; CHANGELOG renders cleanly; `mkdocs
   serve` (if configured) renders cleanly. Tag `v1.1.0`.

If any gate fails, do not start the next phase — debug + retry. The 3-failure circuit
breaker in the engineering protocol applies.

---

## 3. Track 2 — claude-forge v0.3 continuous-learning system

### 3.1 Why this track exists

Today the loop is open: 6 of 7 leads have staging files written by retrospectors that
are **never merged into MEMORY.md**. The architect calls this out as the load-bearing
gap [`architect §"Why this design exists"`]. The historian's design recommendation is to
adopt the Anthropic memory-tool contract (file-shaped, path-namespaced, edit-in-place)
over a Cline six-file schema with a Generative-Agents `recency · tag · helpful` ranker
[`historian §"Most copyable design for v0.3"`]. The forge specifies the lesson schema
(YAML frontmatter + Situation/Action/Outcome/Bounds body, `harmful_count >= 2 → stale`
auto-archive) [`forge §3, §5`]. The cartographer surveyed the current state: 4
substantive + 3 empty MEMORY.md files, 8 staging files (5 to keep, 3 to drop)
[`cartographer §1, §7`].

This track is **lower-effort** than Track 1 because it's mostly file ops + 1 shell
script + 1 Python script. One agent can do it in ~5h.

### 3.2 Migration plan (10 ordered steps)

Source: `EVIDENCE/architect-continuous-learning.md §8` + `EVIDENCE/cartographer-memory-map.md §7`.

Each step is independently revertable; nothing destructive happens until step 4.

#### Step T2-01 — Schema freeze + bootstrap docs (15 min)

Files:
- `~/.claude/agent-memory/SCHEMA.md` (new) — copy `EVIDENCE/forge-memory-schema.md` §3 verbatim
- `~/.claude/agent-memory/PROTOCOL.md` (new) — copy `EVIDENCE/forge-memory-schema.md` §10 verbatim

Rollback: `rm` the two new files.
Acceptance: both files exist; `head -1 SCHEMA.md` matches `EVIDENCE/forge-memory-schema.md` §3 first line.

#### Step T2-02 — Archive existing MEMORY.md content (10 min)

For each of the 4 leads with substantive/light MEMORY.md (`research`, `engineering`,
`forge`, `research-retrospector`):
```
mkdir -p ~/.claude/agent-memory/<lead>/archive
cp ~/.claude/agent-memory/<lead>/MEMORY.md \
   ~/.claude/agent-memory/<lead>/archive/MEMORY-pre-v0.3-2026-05-01.md
```
Rollback: `rm -rf <lead>/archive/`.
Acceptance: 4 archive files present, byte-identical to source.

#### Step T2-03 — Initialize MEMORY.md for the 3 silent leads (5 min)

For `docs-lead`, `security-lead`, `testing-lead`: write a fresh `MEMORY.md` with a
header, an empty `## Starter playbook` section, and a v0.3-schema example block.
Rollback: `rm` the 3 new files.
Acceptance: `ls ~/.claude/agent-memory/{docs,security,testing}-lead/MEMORY.md` succeeds.

#### Step T2-04 — Extend `session-capture.sh` to actually run scribe-merge (30 min) **CRITICAL**

File: `~/.claude/hooks/session-capture.sh` (currently 63 lines, 2543 bytes,
`mtime 2026-05-01`).

Current behavior: writes `staging/adhoc-<sessionid>.md` for non-team substantive
sessions; **never merges**. [verified by reading the file]

Change: at the end of the script, before `exit 0`, add a merge invocation per `forge §10`
canonical pattern:
```bash
# After staging write, run scribe-merge for every lead with new staging files
for AGENT_DIR in "$HOME/.claude/agent-memory"/*-lead; do
  AGENT=$(basename "$AGENT_DIR")
  STAGING_DIR="$AGENT_DIR/staging"
  [ -d "$STAGING_DIR" ] || continue
  # Only merge if there's at least one un-merged staging file
  shopt -s nullglob
  STAGED=( "$STAGING_DIR"/*.md )
  [ ${#STAGED[@]} -gt 0 ] || continue
  # Use the canonical flock+timeout+atomic-rename pattern
  bash "$HOME/.claude/scripts/scribe-merge.sh" "$AGENT" || true
done
```

The actual merge logic lives in a new helper `~/.claude/scripts/scribe-merge.sh` (per
`forge §8` copy-paste-ready snippet, parameterised on `$AGENT`). Mechanical only — does
not re-judge durability.

Rollback: `git checkout HEAD -- ~/.claude/hooks/session-capture.sh` and `rm
~/.claude/scripts/scribe-merge.sh`.
Acceptance:
- `bash ~/.claude/scripts/scribe-merge.sh engineering-lead` is idempotent (running twice
  is a no-op the second time); merged files end up in `staging/_merged/`.
- 10-concurrent stress test (the engineering-scribe validation pattern) shows 0 lost
  writes, 0 dups.
- A new staging file in `engineering-lead/staging/` is appended to `MEMORY.md` after one
  invocation.

#### Step T2-05 — Migrate the 5 KEEP staging files into MEMORY.md (45 min)

Per `EVIDENCE/cartographer-memory-map.md §7`:

| Staging file | Target lead | Lessons | Action |
|--------------|-------------|---------|--------|
| `docs-lead/staging/v1.0-gpucheck.md` | `docs-lead` (was empty) | 7 | Reformat each to v0.3 schema (§3.4 below), then run scribe-merge |
| `engineering-lead/staging/v1.0.md` | `engineering-lead` | 4 | Reformat, merge |
| `engineering-lead/staging/upgrade-mcp-tools-deterministic.md` | `engineering-lead` | 3 | Reformat, merge |
| `research-lead/staging/v1.0-gpucheck.md` | `research-lead` | 3 | Reformat, merge |
| `testing-lead/staging/v1.0-gpucheck.md` | `testing-lead` (was empty) | 5 | Reformat, merge |

Drop (move to `_dropped/` for provenance, do not delete):
- `forge-lead/staging/v1.0-gpucheck.md` (3-line stub)
- `security-lead/staging/v1.0-gpucheck.md` (3-line stub)
- `engineering-lead/staging/v1.0-gpucheck.md` (duplicate of `v1.0.md`)

Total lessons migrated: **22** (7 + 4 + 3 + 3 + 5).

Rollback: `mv staging/_merged/* staging/` and `mv staging/_dropped/* staging/`; reset
each MEMORY.md to its archive copy from T2-02.
Acceptance:
- All 22 lessons present in their target MEMORY.md under `## Migrated from staging/...`
  section.
- Each has YAML frontmatter, Situation/Action/Outcome/Bounds body.
- 5 source staging files moved to `staging/_merged/`; 3 stubs moved to `staging/_dropped/`.

#### Step T2-06 — Promote shared lessons to SHARED_MEMORY.md (20 min)

Per `architect §3` — lessons that affect multiple teams. Candidates from the migrated 22:

- engineering-lead L (subagent harness write-restriction) — `scope: shared`
- research-lead L (4 concurrent background subagents ceiling) — `scope: shared`
- engineering-lead L (PYTHONPATH for worktree pytest) — keeps `scope: lead`; not all teams use worktrees

Move shared lessons to `~/.claude/agent-memory/SHARED_MEMORY.md` (new). Each lesson body
unchanged; only the location changes.

Rollback: move them back to their per-lead files.
Acceptance: `SHARED_MEMORY.md` exists with the promoted lessons; per-lead files have a
`see SHARED_MEMORY.md` cross-reference where the lesson used to live.

#### Step T2-07 — Initialize STARTER_PLAYBOOK.md (15 min)

Per `architect §3` — bedrock invariants. Hand-curated by the user / forge-lead, not by
retrospectors.

Source: lift from `research-lead/MEMORY.md`'s "Starter playbook" section (lines 15+ in
the archived copy from T2-02). Specifically:
- "Anthropic's dispatch-breadth rule"
- "skeptic vs adversary lens"
- "REFRAME is a valid moderator verdict"

Path: `~/.claude/agent-memory/STARTER_PLAYBOOK.md`.
Rollback: `rm`.
Acceptance: file exists, ≤10 KB cap (per `architect §3`), every lead's session-start
read protocol references it.

#### Step T2-08 — Build the index regenerator script (45 min)

File: `~/.claude/agent-memory/scripts/regen_index.py` (new, ~150 LOC Python).

Spec from `architect §4` and `forge §4`:
- Walk `~/.claude/agent-memory/<lead>/MEMORY.md` for each of 7 leads.
- Parse YAML frontmatter blocks (use `yaml.safe_load` with strict=True per
  `historian §"Avoid: opaque-binary"`).
- Skip entries with `status: stale`.
- Build two reverse maps: `tag -> [(lead, id, title)]` and `failure_mode -> [(lead, id,
  title)]`.
- Emit `~/.claude/agent-memory/INDEX.md` in deterministic order (alpha by tag, date desc
  within tag).
- Idempotent — same input produces byte-identical output.
- Use `flock` on `~/.claude/agent-memory/INDEX.md.lock` with 5s timeout; atomic rename
  via `INDEX.md.tmp.<pid>` → `INDEX.md`.
- Sub-100ms target on the v0.3-scale corpus (~22 lessons + skeleton).

Rollback: `rm` the script.
Acceptance:
- `python3 ~/.claude/agent-memory/scripts/regen_index.py` runs in <1s on the migrated
  corpus.
- `INDEX.md` exists with `## By tag` and `## By failure-mode` sections.
- Re-running produces 0-byte-diff output.

#### Step T2-09 — Build the ranker script (60 min)

File: `~/.claude/agent-memory/scripts/rank_lessons.py` (new, ~120 LOC Python).

Spec from `architect §4`:
- Reads the index (or rebuilds it if missing).
- Inputs: `--lead <name>`, `--question "<text>"` (optional), `--top-k 15` (default).
- Scoring: `score = 0.6 · jaccard(question_tokens, lesson.tags) + 0.2 · recency_decay(90-day half-life) + 0.2 · helpful_ratio`.
- `helpful_ratio = (1 + helpful_count) / (1 + helpful_count + harmful_count)`.
- Always-include: every lesson in `STARTER_PLAYBOOK.md` regardless of score.
- Output: ordered list of `(lead, id, title, score, body_path)` tuples; bodies emitted
  to stdout as a single `<continuous-learning-context>...</continuous-learning-context>`
  block.
- Cold start (no question): falls back to recency + helpful only.

Rollback: `rm` the script.
Acceptance:
- `python3 ~/.claude/agent-memory/scripts/rank_lessons.py --lead engineering-lead
  --question "pytest in worktree"` returns the engineering-lead PYTHONPATH lesson at
  rank 1 (regression test for the architect's worked example in `architect §"Concrete
  example from existing corpus"`).
- Latency <100ms on the v0.3-scale corpus.

#### Step T2-10 — Wire pattern-extract stub (30 min)

File: `~/.claude/agent-memory/scripts/pattern-extract.sh` (new, ~80 LOC bash).

Spec from `architect §5` and `forge §6`:
- Reads `~/.claude/agent-memory/INDEX.md` and walks each lead's MEMORY.md.
- For each `(tag, failure_mode)` pair, count lessons.
- Trigger threshold: ≥3 lessons AND ≥2 distinct lead origins AND combined `helpful_count
  >= 5` AND ≥2 tag overlap within a 60-day rolling window.
- For matching clusters: write a stub at
  `~/.claude/agent-memory/forge-lead/staging/proposed-skill-<topic>-<date>.md` and set
  the marker file `/tmp/claude-pattern-extract-pending.flag`.
- Dedup memory: skip clusters proposed-and-rejected within 60 days unless they grew by
  ≥2 lessons.
- v0.3 ships as **proposal-only** (no auto-promotion to skill). Forge-lead reviews the
  stub manually before invoking `/forge:draft`.

Rollback: `rm` the script; remove `/tmp/claude-pattern-extract-pending.flag` if present.
Acceptance:
- Running on the migrated corpus produces 0 proposals (the corpus is too small at
  22 lessons to trigger the threshold; this is expected and correct — first cluster
  proposals will surface after ~60 days of session activity).
- Adding 3 hand-crafted dummy lessons sharing 2 tags across 2 leads does trigger a
  proposal; removing them removes the proposal.

### 3.3 The 4 hooks the architect specified — implementation status after Track 2

| Hook | Where it attaches | T2 task that wires it | Status after Track 2 |
|------|-------------------|------------------------|---------------------|
| 1a SessionStart | Claude Code `SessionStart` event | not in Track 2 — **deferred** to v0.4 (rank_lessons.py is built, but the hook registration in `settings.json` is a separate user-facing change) | ranker script works; lead reads MEMORY.md per existing protocol; ranker available on demand via CLI |
| 1b Agent dispatch | `PreToolUse` on `Task` (or orchestrator-side fallback) | not in Track 2 — **deferred** to v0.4 (architect explicitly notes harness PreToolUse is unreliable; orchestrator caching is the source of truth, but that requires lead persona file edits which are out of v0.3 scope) | not wired |
| 1c SessionEnd | `Stop` event via extended `session-capture.sh` | **T2-04** | **WIRED** — closes the loop |
| 1d Pattern-extract | Out-of-session, cron/launchd/loop | T2-10 (script exists; cron registration deferred) | script exists; user runs on demand via `/loop` or one-shot |

The single most load-bearing hook for "the loop closes" is **1c** (T2-04). Without it,
staging files sit unmerged forever — the current state. Hooks 1a / 1b can ship in v0.4
once the lead persona files are also updated; v0.3 is "loop closes" + "ranker available
on demand" + "pattern-extract proposes."

### 3.4 Schema-conformant rewriting of staging lessons (T2-05 detail)

For each of the 22 staged lessons, the migration agent transforms from free-form bullets
to v0.3 schema:

**Free-form bullets (current):**
```markdown
### L2: Editable-install in venv pinned to non-worktree path...
**Observed in**: gpucheck v1.0...
**Failure mode addressed**: FM-3.2...
**Lesson**: When the user's venv was created with `pip install -e .`...
**Rule of thumb**: For any task that runs pytest in a git worktree...
**Counter-example / bounds**: If the venv was created with `uv sync`...
```

**v0.3 schema (target):**
```markdown
### engineering-2026-05-01-pythonpath-for-worktree-pytest: pytest in a worktree imports from the venv's editable target — set PYTHONPATH=<worktree>/src

```yaml
id: engineering-2026-05-01-pythonpath-for-worktree-pytest
title: pytest in a worktree imports from the venv's editable target — set PYTHONPATH=<worktree>/src
status: curated
authored_at: 2026-05-01
authored_by: engineering-retrospector
authored_in: gpucheck-v1.0
last_reviewed: 2026-05-01
last_triggered: null
helpful_count: 0
harmful_count: 0
failure_modes: [FM-3.2]
tags: [worktree, pytest, editable-install, venv]
evidence: ../../../Code/gpucheck/.claude/teams/engineering/v1.0/EVIDENCE/retrospector.md
supersedes: null
superseded_by: null
see_also: []
```

**Situation.** A pytest run in a git worktree returns collection errors that look like
the source is missing or stale. The user's venv was created at the main repo with
`pip install -e .`, and edits are happening in a sibling worktree.

**Action.** Prepend `PYTHONPATH=<worktree>/src` to every pytest invocation in the worktree.
Do **not** `pip install -e .` against the worktree — that mutates the user's shared venv
and silently switches the editable target.

**Outcome.** With PYTHONPATH set, pytest imports from the worktree source and the run is
deterministic. Without it, pytest silently imports the main-repo source.

**Bounds / counter-example.** Does NOT apply when the venv was created with `uv sync`
against the worktree. Detect by checking for
`venv/lib/python*/site-packages/<package>.egg-link`.
```

The rewrite is mechanical for ~80% of fields (id from agent + date + slug, status =
curated, counts = 0, evidence = relative path to the source EVIDENCE file). The body
restructure (bullets → S/A/O/B prose) is the only judgment call; the migration agent has
the worked example in `forge §9` to copy.

Per-lesson estimate: ~2 min × 22 lessons = ~45 min total. (Already budgeted in T2-05.)

### 3.5 Track-2 Gantt

```
hour:  0      1      2      3      4      5
       |------|------|------|------|------|
T2-01: [15m]
T2-02: [10m]
T2-03: [5m]
T2-04: [           30m              ]   *** CRITICAL ***
T2-05:        [           45m              ]
T2-06:                            [20m]
T2-07:                            [15m]
T2-08:               [           45m              ]
T2-09:                      [           60m              ]
T2-10:                                       [   30m   ]
                                                      ↑
                                                      Track 2 done (~5h)
```

T2-04 (the hook extension) is on the critical path — it must land first because every
subsequent staging-file-rewrite step depends on the merge tool working. T2-08 / T2-09
can run in parallel with T2-05 if a second agent is available.

---

## 4. Risk register — top 5 risks + mitigation

Sources: `planner §6`, `architect §6`, `forge §11`.

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|------------|--------|------------|
| **R1** | **T-24 (MPS overlay) slips because T-12 + T-18 deps create a sequential bottleneck.** [`planner §6`] | MED | HIGH (R3 headline deliverable) | Land T-24 last in Phase D. If T-12 + T-18 take longer than budgeted, ship v1.1 without T-24 and tag it v1.1.0; T-24 lands in v1.1.1. The 2× scale in v1.0 stays safe in the meantime. |
| **R2** | **T-19 (`_run_mps` unification) breaks an undocumented benchmark behavior.** [`planner §6`, `tracer-runtime finding #2`] | MED | HIGH (regresses fixture timing) | Land T-04 first (bare-except) so T-19 has a clean error model. Add behavioral parity test (T-19 acceptance) before deletion. Preserve `_FLUSH_L2_WARNED` gate explicitly. If parity fails, revert and re-plan. |
| **R3** | **Track-2 T2-04 hook extension breaks the existing `session-capture.sh` for non-team sessions.** [`hooks/session-capture.sh:46-63` adhoc capture path] | LOW | MED (every Stop hook fires; if broken, every session emits an error) | Extend the hook with a defensive `|| true` per `forge §10` "deferred merge is non-fatal." Run the existing adhoc-capture path FIRST, then run scribe-merge as a separate downstream step. Test by running 10 concurrent `claude` sessions in scratch dirs and verifying 0 lock leaks. |
| **R4** | **Phase A 4-way parallelism is unavailable; v1.0.0rc2 slips from 90 min to 3.5h.** [`planner §7 caveat #1`] | MED | LOW (rc2 timing only; v1.1 cycle is unaffected) | Single-stream fallback is documented in `planner §3 Wave 1`. The cycle's 12h budget absorbs the slip. Communicate to user that rc2 may not tag at hour 1.5; the v1.1 ship time is unchanged. |
| **R5** | **Pattern-extract (T2-10) generates noise proposals once the corpus grows.** [`architect §6 FM-7`] | LOW (deferred until ~60 days post-deployment) | LOW | Ship v0.3 with proposal-only mode; forge-lead reviews stubs manually. The dedup memory in T2-10 prevents re-proposing rejected clusters within 60 days. If false-positive rate >50% after 60 days, raise threshold from "≥3 lessons / ≥2 leads / helpful≥5" to "≥4 / ≥3 / ≥7". |

Other risks worth noting (not in top 5 but logged):
- **R6** (low): T-25's 41 xfail entries could include a typo that registers a dead op.
  Mitigation: T-17 deserialises every entry and asserts `is_mps_xfailed(op)` returns
  True; manual `sort | uniq -c` check before merge. [`planner §6 high-risk T-25`]
- **R7** (low): T-26's deadlock probe leaks a daemon thread on actual deadlock.
  Mitigation: tracer-v3-deadlock §"Confidence" already flagged this; T-16 only tests
  healthy + skipped paths in CI; documented in T-26 acceptance.

---

## 5. Acceptance criteria — what's "done" for this 6-12 hour cycle

### Track 1 — gpucheck v1.1

A cycle is **DONE** when ALL of:

1. **v1.0.0rc2 tagged.** Phase A (T-01..T-10) merged on `release/v1.0.0rc2` branch.
   Phase A → B verifier gate green (§2.6).
2. **v1.1.0 tagged.** Phases B, C, D, E merged on `release/v1.1`. Phase E close gate
   green.
3. **Mutation kill rate ≥80%** on `assertions/reporting.py` and `assertions/tolerances.py`
   per `mutmut run` output. [`charter`, `mutator-survivors`, `planner §4`]
4. **Lazy-import contract holds.** `python -c "import gpucheck; import sys; assert
   'torch' not in sys.modules"` exits 0. [`detector-files Top-3 #1`, T-01]
5. **The 4 R3 binding deliverables are in.** Apple-tile fuzz (T-23), MPS overlay (T-24)
   OR documented v1.1.1 deferral with risk justification, xfail registry expansion to 41
   entries (T-25), deadlock probe (T-26).
6. **The linguist-v3 silent-fp64-downcast catcher is NOT shipped.** [`tracer-runtime §4`,
   `synthesist C1`, `planner T-26 acceptance`]
7. **Every Quadrant-1 issue is owned.** Per the §2.4 mapping table; the two deferred
   (ISS-01, ISS-02) are documented as v1.2 stretch in §6.
8. **All 22 MIGRATION.md doc blocks parse.** `python .claude/teams/audit/v1.1/scripts/
   test_doc_blocks.py MIGRATION.md` reports 0 broken (was 4 high-impact + others).
   [`docs-tester M-B-suite`, T-07]
9. **CHANGELOG + CLAUDE.md current.** T-27. CHANGELOG documents v1.0.0rc2 and v1.1
   commits; CLAUDE.md "Known Weaknesses" no longer falsely claims fixed gaps.
   [`archaeologist-debt §"CLAUDE.md is STALE"`]
10. **Engineering evaluator PASS.** 5-dimension rubric per
    `~/.claude/agents/engineering/engineering-evaluator.md`: functional correctness 1.0,
    test coverage 1.0 (strict), diff minimality / revert-safety / style ≥0.7
    (advisory).

### Track 2 — claude-forge v0.3

A cycle is **DONE** when ALL of:

1. **22 lessons migrated.** All 5 KEEP staging files reformatted to v0.3 schema and
   merged into target MEMORY.md. The 3 stub staging files moved to `staging/_dropped/`.
   [T2-05, `cartographer §7`]
2. **3 silent leads have MEMORY.md.** `docs-lead`, `security-lead`, `testing-lead` each
   have a MEMORY.md (was empty per `cartographer §1`). [T2-03]
3. **Hook 1c (SessionEnd merge) wired.** `session-capture.sh` extended; `scribe-merge.sh`
   helper exists; running a fresh `claude` session in a scratch dir merges any new
   staging files into MEMORY.md automatically. [T2-04, **the loop closes**]
4. **Index + ranker scripts work.** `regen_index.py` produces a deterministic INDEX.md;
   `rank_lessons.py` returns the architect's worked-example lesson at rank 1 for the
   `pytest worktree` query. [T2-08, T2-09]
5. **Pattern-extract script exists in proposal-only mode.** `pattern-extract.sh` runs
   without errors; on the v0.3-scale corpus produces 0 proposals (correct — corpus too
   small to trigger threshold). [T2-10]
6. **Lesson-rot mitigation lives in the schema.** Every migrated lesson has
   `harmful_count: 0` in frontmatter; `forge §5 GC rule (c)` is implementable today
   (not deferred). [T2-05, `architect §6 FM-2`]
7. **Concurrent-write safety verified.** 10 concurrent `bash scribe-merge.sh
   engineering-lead` invocations show 0 lost writes, 0 dups (the existing
   engineering-scribe stress-test pattern).
8. **No data loss.** Every original staging file lives in `staging/_merged/` or
   `staging/_dropped/`; every original MEMORY.md lives in `archive/MEMORY-pre-v0.3-*.md`.
   [T2-02]

If both tracks pass their criteria, the cycle ships.

---

## 6. Stretch goals (deferred to v1.2 if Track 1 finishes early)

Sources: `synthesist Quadrant 3 + 4`, `planner §7 caveats`.

If Track 1's evaluator gate passes with >2h remaining in the 12h budget, pull these from
the v1.2 backlog in priority order:

| # | Issue | Effort | Why deferred today |
|---|-------|--------|---------------------|
| **S1** | **ISS-01 + ISS-02 — strict=False kwarg pattern for compute_tolerance + @devices typo loud-failure.** [`api-dx-grade fix #1`] | ~30 min | Quadrant-1 but not in planner's 27-task graph. Single shared fix-pattern; wins the highest-leverage v1.2 task. |
| S2 | ISS-04 — @dtypes typo eager validation at decoration. [`api-dx-grade #4`] | ~25 min | Same strict-mode lane as S1; bundle. |
| S3 | ISS-44 — `memory_tracker` `fail_on_leak: bool = False` kwarg. [`api-dx-grade #9`] | ~20 min | Behavioral change; safer in a v1.2 minor with notice. |
| S4 | ISS-58 — bump default `warmup` from 3 to 5 on MPS for conv2d. [`empiricist signal item #2`] | ~10 min | Trivial; misses today's R3 cluster only because empiricist landed in W1. |
| S5 | ISS-32 — document xfail-registry-empty-outside-pytest gotcha in MIGRATION.md §5. [`docs-tester M-B7`] | ~5 min | Doc-only; trivial. |
| S6 | ISS-12 — deduplicate `_median` in `analysis/regression.py` + `analysis/roofline.py`. [`detector-files Top-10 #10`] | ~15 min | Quadrant-3 cleanup; bundles with T-20's DRY refactor lane. |
| S7 | ISS-37 — add `Examples:` doctests to `__init__.py`. [`docs-tester I-B1`] | ~10 min | Doc-only; trivial. |
| S8 | ISS-49 — wrap `import torch` in `gpu_device` plugin with try/except + `pytest.skip`. [`detector-files plugin.py:151`] | ~10 min | Test-collection grace; bundles with T-22 init refactor. |
| S9 | Issue: PyTorch upstream issues — file ISS-56 (MPS matmul fp32 4× slow) and ISS-57 (CPU half-precision GEMM) at github.com/pytorch/pytorch. | ~30 min | External; not gpucheck code; valuable for users + relations. |
| S10 | ISS-55 — install `commitlint` as a hooked gate matching CONTRIBUTING.md policy. [`archaeologist-debt §5`] | ~30 min | Operational; future automation gate. |

Track-2 stretch (if Track 2 finishes early):
- **T2-S1**: wire hook 1a (SessionStart ranker injection) by editing the 7 lead persona
  files to source the ranker output. ~30 min. Out of v0.3 scope today.
- **T2-S2**: cron-register `pattern-extract.sh` via launchd plist on macOS.
  ~15 min. The user has the `schedule` skill — defer to first user-driven session.
- **T2-S3**: write `~/.claude/agent-memory/_metrics/sessions.jsonl` baseline + the
  metric-capture hook line per `architect §7`. ~20 min. Useful for first-quarter
  measurement of M2 / M3 / M4.

---

## 7. Provenance — every load-bearing claim, sourced

This section is for the skeptic / adversary / evaluator pass. Every assertion in §1-§6
above traces to one of:

### Audit evidence files (W1 + W2)

| Source path | Cited where | Used for |
|-------------|-------------|----------|
| `EVIDENCE/planner-v1.1-tasks.md` | §1, §2.1, §2.2, §2.3, §2.5, §4 R1, R2, R4 | task graph, dependency order, blast radius, rollback sketches, parallelism plan, high-risk flags |
| `EVIDENCE/synthesist-bugs-inventory.md` (+ summary) | §2.3 per-task table, §2.4 Quadrant-1 mapping, §6 stretch | issue identification, severity rubric, cross-audit contradictions, Quadrant matrix, upstream-fileable list |
| `EVIDENCE/api-dx-grade.md` (+ summary) | §2.3 T-10, T-21, T-22; §6 S1, S2, S3 | per-API DX scoring, deprecation candidate, top-3 v1.1 fixes |
| `EVIDENCE/security-postmerge.md` (+ summary) | §2.3 T-02, T-03 | PM-2, PM-4 issue identification |
| `EVIDENCE/archaeologist-debt.md` (+ summary) | §2.3 T-27, §5 acceptance #9, §6 S10 | CLAUDE.md staleness, GPU-CI debt, conventional-commits compliance |
| `EVIDENCE/detector-files.md` (+ summary) | §2.3 T-01, T-04, T-05, T-18, T-20, T-22 | duplicated-logic findings, naming collisions, lazy-import violations |
| `EVIDENCE/mutator-survivors.md` (+ summary) | §2.3 T-11, T-12, T-13, T-14; §1 critical task list; §5 acceptance #3 | mutation kill-rate analysis, top-3 high-leverage tests, TEST_BUG cluster |
| `EVIDENCE/tracer-runtime.md` (+ summary) | §2.3 T-19, T-26; §1 critical task list; §5 acceptance #6 (REFUTED catcher) | runtime tracing findings, `_run_mps` duplicate, fp64-on-MPS refutation |
| `EVIDENCE/docs-tester-blocks.md` (+ summary) | §2.3 T-06, T-07, T-08, T-09 | broken doc blocks (R-B / M-B / T-B), MPS auto-skip claim |
| `EVIDENCE/empiricist-mac-benchmarks.md` (+ summary) | §2.3 T-24; §6 S4, S9 | per-(kernel,dtype) MPS scaling data, upstream PyTorch bugs, conv2d warmup |
| `EVIDENCE/forge-memory-schema.md` (+ summary) | §3.1, §3.4 (worked example), §3.2 T2-01, T2-04, T2-05, §4 R3 | v0.3 lesson schema, lifecycle, GC rules, runbook-promotion threshold, canonical merge invocation |
| `EVIDENCE/architect-continuous-learning.md` (+ summary) | §3.1, §3.2 T2-01..T2-10, §3.3 4-hook table, §4 R3, R5 | hook attachment points, ranker design, scope tiers, migration plan, failure modes |
| `EVIDENCE/historian-memory-prior-art.md` (+ summary) | §3.1 design recommendation | most-copyable design, prior-art lessons, dissenting voices |
| `EVIDENCE/cartographer-memory-map.md` (+ summary) | §3.1, §3.2 T2-02, T2-03, T2-05; current state | MEMORY.md inventory, staging file disposition, anomalies |

### Cross-team inputs

| Source | Cited where | Used for |
|--------|-------------|----------|
| `~/Code/gpucheck/.claude/teams/research/v1.0/SYNTHESIS.md` | `planner` binding inputs (§2.3 references R3 deliverables) | R3 binding deliverables (Apple-tile fuzz, MPS overlay, xfail registry, deadlock probe) |
| `~/Code/gpucheck/.claude/teams/research/v1.0/EVIDENCE/{empiricist-v3-extended, cartographer-v3-fuzzer-patch, github-miner-v3-xfail-config, tracer-v3-deadlock, archaeologist-v3-crossver}.md` | T-23, T-24, T-25, T-26 | feature payloads (the +91/-8 fuzzer patch, Shape B class-bucketed table, 41-entry xfail block, deadlock probe sketches) |
| `~/Code/gpucheck/CLAUDE.md` | §2.6 verifier gates, §5 acceptance | strict-types/ruff/mypy + lazy-import standards |

### Refutations / contradictions formally addressed

- **C1 (linguist-v3 silent-fp64-downcast catcher):** REFUTED by tracer-runtime §4 on
  torch 2.11. Dropped from T-26. Documented as exclusion in §5 acceptance #6.
- **C2 (CLAUDE.md "Known Weaknesses" stale):** confirmed by archaeologist-debt #4 +
  detector-files + api-dx-grade. Owned by T-27 + ISS-50.
- **C3 (flush_l2 warning vs absent work):** complementary findings, both addressed in
  T-19 (warning gate) + ISS-20 (buffer-fill approximation, deferred to v1.2).
- **C4 (memory_guard warning vs failure):** scope mismatch; ISS-44 deferred to v1.2 stretch S3.
- **C5 (README MPS-as-first-class vs CUDA-only snippets):** real internal inconsistency;
  owned by T-06.

### Items explicitly excluded

- `linguist-v3` silent-fp64-downcast catcher (refuted; per `planner §0.4` + §7 caveat #3).
- ISS-22 (fast-path 1.77 ms sunk cost) — Quadrant 4, defer to v1.2.
- ISS-23 (format_mismatch_report 16 ms slow path) — Quadrant 4, defer to v1.2.
- ISS-40 (`__new__`-as-factory) — Quadrant 2 (HARD), defer to v1.2 with multi-PR plan.
- ISS-51 (GPU CI gate re-enable) — Quadrant 2 (HARD); requires MPS CI runners; defer to a future T-28 in v1.2.
- ISS-52 (Tolerance unifying scaling law) — Quadrant 2 (HARD); T-24 starts the work but the full Backend.tolerance_overlay() Protocol refactor is v1.2+.
- PM-5 (HTML reporter raw `class=`/`style=` interpolation) — synthesist marked
  "not exploitable today (whitelisted constants)"; defer to v1.2.

---

## Confidence

**HIGH** on Track 1. Every Phase A task is single-file or single-hunk with verified line
numbers. Phase B follows mutator-survivors leverage analysis directly. Phase C
eliminates duplications named explicitly by detector + tracer. Phase D follows R3
SYNTHESIS verbatim minus the explicitly-rejected linguist-v3 catcher. Wall-clock
estimates are MEDIUM (a 25% overrun pushes the cycle from 6h to 7.5h, still within 12h).

**HIGH** on Track 2 §3.1-§3.4 (schema, migration ordering, hook 1c implementation, schema
rewrite mechanics). Each step is grounded in an existing audit evidence file or in disk
state verified by the cartographer. **MEDIUM** on Track 2 ranker implementation
(T2-09) — the Jaccard + recency + helpful blend is the architect's calibrated guess and
will need M2 / M3 measurement after first-quarter use to validate. **MEDIUM** on
pattern-extract threshold (T2-10) — the "≥3 lessons / ≥2 leads / helpful≥5 / ≥2 tags /
60-day window" is calibrated against today's tiny corpus; first-quarter measurement
should re-tune.

**HIGH** on the 1 thing most likely to slip (T-24, MPS overlay) and the documented
fallback (ship v1.1 without it, defer to v1.1.1).

**LOW-MEDIUM** on the parallelism assumption: Phase A's "≤90 min wall-clock" requires a
4-way executor pool; if only single-stream is available, rc2 slips 2-3h but v1.1's total
budget still holds.

## Verdict

**PASS.** The plan is actionable, every Quadrant-1 issue is owned (modulo the two v1.2
deferrals documented in §6 stretch goals), the load-bearing refutations are
applied (linguist-v3 catcher dropped), and both tracks have explicit acceptance criteria
that an evaluator can mechanically check. The plan is ready for skeptic + adversary +
evaluator gates.
