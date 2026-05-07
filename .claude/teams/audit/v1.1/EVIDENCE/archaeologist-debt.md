# Archaeologist — Debt Patterns in gpucheck git history

Repo: `/Users/cero/Code/gpucheck`
Branch examined: `release/v1.0` (HEAD = `82b853e`)
Tag: `v1.0.0rc1` at `d720e3d` (annotated, points to commit `6a07ca6`)
Total commits across all refs: **45** (full clone, not shallow)
Window: `16b95ef` (initial) → `82b853e` (HEAD), 2026-03 → 2026-05

---

## 1. Commit-style drift (Conventional Commits compliance)

**Claim under audit (`CONTRIBUTING.md`, commit `195779b`, lines around the
"Commit message format" section):**

> Going forward, gpucheck uses [Conventional Commits 1.0]. ... New commits
> should use Conventional Commits.
>
> Legacy `[ Type ] :` bracket-style commits (visible in pre-v1.0 history,
> e.g. `[ Fix ] : resolve 7 bugs`) are unchanged — we are not rewriting
> history.

So the policy is forward-only. The natural cut-line is the first commit
authored after the v1.0 release tracks landed.

### Whole-history breakdown

Counts from `git log --all --pretty=format:'%s'`:

| style | count | notes |
|---|---|---|
| Conventional (`type(scope): …`) | 11 | all dated 2026-04-30 / 2026-05-01 |
| Bracket (`[ Type ] : …`) | 30 | all dated 2026-03-22 → 2026-03-28 |
| `Initial commit` | 1 | `16b95ef` GitHub default |
| `Merge branch …` | 3 | merge commits from track A/B/D |

(Total = 45.)

### Post-v1.0 window: where the policy actually applies

The bracket→conventional switchover happens at the boundary
between commit `a9a9d44` (last `[ Fix ] :`, 2026-03-28) and
`24035aa` (first `feat(mps): …`, 2026-04-30).
Everything from `24035aa` onward should be Conventional Commits.

Listing every commit reachable from HEAD authored after `a9a9d44`
(excluding merges):

| sha | subject | conventional? |
|---|---|---|
| `24035aa` | `feat(mps): add Apple Silicon MPS backend …` | yes |
| `4ede763` | `feat(fuzzing): stride and contiguity fuzzing …` | yes |
| `5ddd26e` | `fix(tolerances): thread-safe override stack …` | yes |
| `02507da` | `feat(reporting+sanitizers): HTML dashboard, …` | **multi-scope, technically valid** (`type(scope): …`) but spec says scope is "noun describing a section"; `reporting+sanitizers` is two scopes packed in one |
| `a60e3a5` | `chore(lock): regenerate uv.lock post-merge …` | yes |
| `85de0f9` | `docs(changelog): add Keep-a-Changelog 1.1 …` | yes |
| `195779b` | `docs(contributing): add development guide …` | yes |
| `2673211` | `docs(migration): add v0.1.0 -> v1.0 migration guide` | yes |
| `40ba1de` | `docs(readme): replace CUDA-only language …` | yes |
| `6a07ca6` | `docs(claude.md): reflect v1.0 delivery …` | yes |
| `82b853e` | `fix(fuzzing): drop unused # type: ignore …` | yes |
| 3 merge commits (`cc81650`, `472fd91`, `0c44e74`) | `Merge branch …` | n/a (default merge subject; allowed) |

**Compliance rate, post-v1.0 non-merge commits: 11/11 = 100%.**
Across the entire history: 11/41 non-merge non-initial = **27%**.

### Cite

- `CONTRIBUTING.md` policy section, introduced in commit `195779b`,
  "Commit message format" header.
- Post-v1.0 commits enumerated via `git log v1.0.0rc1` minus
  `a9a9d44..origin/main`.

### Smell — "we are not rewriting history" is a debt promise

The bracket vs conventional split is now permanently visible in
`git log`. Anything tooling that consumes commits (changelog generators,
release-please, semantic-release, commitlint pre-commit hooks) will see
30 non-conformant commits and either choke or silently skip them.
`docs/changelog` already exists (commit `85de0f9`), but it was
**hand-written**, not generated — exactly because the bracket commits
can't be parsed.

---

## 2. Hot-spot files

`git log --pretty=format: --name-only | sort | uniq -c | sort -rn | head -20`:

| edits | path |
|---|---|
| 8 | `src/gpucheck/plugin.py` |
| 8 | `src/gpucheck/assertions/close.py` |
| 8 | `src/gpucheck/arch/detection.py` |
| 8 | `README.md` |
| 6 | `tests/test_assertions.py` |
| 6 | `src/gpucheck/sanitizers/memory.py` |
| 6 | `src/gpucheck/assertions/tolerances.py` |
| 6 | `src/gpucheck/assertions/reporting.py` |
| 6 | `src/gpucheck/analysis/regression.py` |
| 6 | `src/gpucheck/analysis/bottleneck.py` |
| 5 | `src/gpucheck/sanitizers/race.py` |
| 5 | `src/gpucheck/fixtures/profiler.py` |
| 5 | `src/gpucheck/fixtures/benchmark.py` |
| 5 | `src/gpucheck/decorators/dtypes.py` |
| 5 | `src/gpucheck/arch/tensor_cores.py` |
| 5 | `src/gpucheck/analysis/roofline.py` |
| 5 | `pyproject.toml` |

Note: with only 45 total commits this list is dominated by sweeping
"polish/critical/medium/low" commits that touched many files at once.
Edit-counts ≥6 are still real signal.

### 2a. `src/gpucheck/assertions/close.py` (8 edits, churn pattern: stack of patches)

```
9352672 [ Feature ]    : created                 (initial impl)
76c33ae [ Fix ]        : +35 / -10 lines         "addressed critical issues from expert code review"
dc4fadb [ Fix ]        : +5  / -0                "high-severity code quality"
97f06c7 [ Fix ]        : +23 / -8                "cleaned up medium-severity issues"
8d8c894 [ Fix ]        : (none touching close.py in this commit)
28d808e [ Fix ]        : +5  / -5                "resolved all mypy strict mode errors"
25cdfcf [ Perf ]       : +21 / -0                "GPU fast-path"
2197277 [ Fix ]        : +46 / -10               "resolve 7 bugs found by codebase analysis"
24035aa feat(mps)      : +17 / -4                MPS backend
```

**Pattern:** every "expert review" / "codebase analysis" sweep had to come
back to `close.py`. This is the central API surface, so churn is somewhat
expected, but the same file getting hit by *critical*, *high*,
*medium*, *low*, *mypy*, and *7 more bugs* in succession means each prior
sweep missed real issues. There is no commit titled "refactor close.py";
the file is patch-over-patch.

**Cite:** `git log --oneline -- src/gpucheck/assertions/close.py`.

### 2b. `src/gpucheck/arch/detection.py` (8 edits)

```
2ee221e [ Feature ]   : 268 lines created
f044373 [ Fix ]       : 2 line change       "critical issues"
97f06c7 [ Fix ]       : 2 line change       "medium-severity"
8d8c894 [ Fix ]       : 24 lines           "low-severity / consistency"
28d808e [ Fix ]       : 2 line change       "mypy strict"
25cdfcf [ Perf ]      : 33 lines           "fixed tensor core detection"
2197277 [ Fix ]       : 8 line change       "7 bugs found"
24035aa feat(mps)     : 11 lines           Apple Silicon
```

**Pattern:** "fixed tensor core detection" in `25cdfcf` is the load-bearing
one — confirms the README claim that "GTX 16xx exclusion" was a real bug
being fixed *after* the feature shipped, not designed in.

### 2c. `src/gpucheck/assertions/tolerances.py` (6 edits)

```
9352672 [ Feature ]    : 106 lines (initial)
8d8c894 [ Fix ]        : +32 polish
f7f84eb [ Fix ]        : +7  "unified tolerance scaling, lazy dtype resolution"
6562f31 [ Fix ]        : +11 "recalibrated tolerance tables from GPU measurements"
24035aa feat(mps)      : +108 (massive — MPS-specific tolerance shifts)
5ddd26e fix(tolerances): +37 "thread-safe override stack via contextvars; mitigate TM-E1"
```

**Pattern: tolerance numerics never settled.** Three separate fix commits
(`f7f84eb` "unified scaling", `6562f31` "recalibrated tables", and the
+108-line MPS-specific recalibration in `24035aa`) signal the tolerance
table was a moving target driven by *empirical GPU measurements*, not a
priori design. The `5ddd26e` thread-safety fix landed only days before
v1.0 and explicitly cites a debt item in `CLAUDE.md`:

> Track-C of the gpucheck v1.0 release fixes the documented
> "Thread-safety issue in tolerance override stack" gap (CLAUDE.md weakness)

So `CLAUDE.md` was used as a backlog. That is a debt smell — the README's
"Known Weaknesses" list still contains items the audit will rediscover.

### 2d. `src/gpucheck/sanitizers/memory.py` (6 edits)

```
58b6cd2 [ Feature ]   : 258 lines (initial)
dc4fadb [ Fix ]       : 22 lines  "high-severity"
97f06c7 [ Fix ]       : 7  lines  "medium-severity"
8d8c894 [ Fix ]       : 11 lines  "low-severity"
28d808e [ Fix ]       : 4  lines  "mypy strict"
25cdfcf [ Perf ]      : -5 lines  cleanup
```

**Pattern:** classic critical/high/medium/low descent. No re-architecture
despite `CLAUDE.md` flagging "Memory leak detection uses process-level
metrics (imprecise)". The fix was always lipstick — never the architectural
move to per-tensor accounting.

### 2e. `tests/test_assertions.py` (6 edits)

```
32407c1 [ Test ]       : 1163 lines (initial test suite)
e7e48da [ Fix ]        : import mismatches
8d8c894 [ Fix ]        : "low-severity"
f7f84eb [ Fix ]        : tolerance API consistency
6562f31 [ Fix ]        : "recalibrated tolerance tables — error reporting cosmetics"
2197277 [ Fix ]        : 7 bugs add 23 tests
```

**Pattern:** test churn lags the source — every src patch produced a test
edit. Healthy in principle, but it confirms tests are *characterization*
tests (locking in current behavior), not invariant tests; they had to be
re-written each time the source moved.

### 2f. `src/gpucheck/assertions/reporting.py` (6 edits)

`8d5c5a6` introduced a fix titled "all-NaN reporting crash" — that is a
crash-on-edge-input bug that escaped the initial test suite. The fix is
+10 lines and adds no obvious invariant; suggests there are other
edge-input crashes lurking (zero-tensor, ±inf-only, dtype-empty).

---

## 3. Blame patterns on the 5 most-edited source files

`git blame -w -C -C -C` (whitespace-tolerant, cross-file move tracking):

| file | author A (`Akasxh`) | author B (`Akash`) | dominant author |
|---|---|---|---|
| `src/gpucheck/plugin.py` | 170 lines | 39 lines | Akasxh |
| `src/gpucheck/assertions/close.py` | 267 lines | 17 lines | Akasxh (94%) |
| `src/gpucheck/arch/detection.py` | 282 lines | 10 lines | Akasxh (97%) |
| `src/gpucheck/sanitizers/memory.py` | 258 lines | 0 | Akasxh (100%) |
| `src/gpucheck/assertions/tolerances.py` | 121 lines | 119 lines | **near-50/50 split** |

### Two-author identity smell

`Akasxh` and `Akash` are the **same physical author** (both
`drakathakash@gmail.com`, see `CLAUDE.md` "Account: Akasxh /
drakathakash@gmail.com"). The split is just the GitHub-username vs
display-name mismatch and corresponds **exactly** to the
bracket-vs-conventional commit cutover:

- Pre-v1.0 (bracket commits): committer = `Akasxh`.
- Post-v1.0 (conventional commits): committer = `Akash`.

So "blame split" is really "this many lines of the file were rewritten
in the v1.0 sprint."

### Patch-over-patch hot zones

- `tolerances.py`: ≈50/50 split, meaning **half of the file was rewritten
  during v1.0** (mostly by `24035aa` MPS table extension and `5ddd26e`
  ContextVar refactor). The MPS path was *layered on*, not designed in.
  Tracer should verify the CUDA tolerance table and MPS table don't
  diverge in scaling behavior.
- `close.py`: only 17/284 lines are new since v1.0, but the GPU fast-path
  in `25cdfcf` was glued onto a numpy-first design. Worth checking the
  fast-path doesn't bypass the rich-report code path silently.
- `plugin.py`: 39/209 lines are new — the MPS hooks (`24035aa` +39).
  Read like an addition, not an integration; verify pytest hook ordering
  isn't surprising.

### Cite

- `git blame -w -C -C -C --line-porcelain <file> | awk '/^author /' | sort | uniq -c`.
- `git log` author histograms confirm the `Akasxh` → `Akash` rename
  coincides with `24035aa` (2026-04-30, first conventional commit).

### Smell

There is **no `refactor:` commit anywhere in the history** (verified
via `git log --pretty=%s | grep -E '^refactor'`). All 41 non-merge
non-initial commits are `feat`, `fix`, `docs`, `test`, `perf`, `chore`,
or bracket-style — the "tidy code by reshaping" lane was never used.
Combined with 4 successive critical/high/medium/low fix sweeps on
`close.py` and `memory.py`, this is the canonical patch-over-patch
signature.

---

## 4. Reverted decisions

`git log --all --pretty=format:'%H %s' | grep -iE 'revert|rollback|undo|backout'` returns **zero matches**.

Searching for negative-sounding subjects (`drop`, `remove`, `delete`):

- `82b853e` `fix(fuzzing): drop unused # type: ignore on @st.composite decorator`
  — micro-revert of a `# type: ignore` comment, not a behavior change.
- No other "drop / remove / delete" commits.

### Disguised revert: `28d808e [ Fix ] : resolved all mypy strict mode errors for CI`

`git show --stat 28d808e` reports **86 files changed, 38 insertions(+),
325 deletions(-)** — i.e. it is a -287 line net commit. That kind of net
deletion under a "fix CI" message is almost always a revert of speculative
type-hint scaffolding. Worth a closer read before the structural audit
trusts the type signatures.

### `22780ae [ Fix ] : moved GPU-dependent tests to integration suite to fix CI on CPU-only runners`

`2 files changed, 0 insertions(+), 0 deletions(-)` — this is a pure file
rename (test files moved to `tests/gpu_integration/`). It's a
**de-facto revert** of "tests run on every PR." The CI gate was lowered
to make green builds; the GPU tests are still in-tree but they no longer
run on GitHub Actions. This is exactly the "moved the bar to fit under
it" smell that should be a v1.1 priority.

### Cite

- `git show --stat 28d808e` and `git show --stat 22780ae`.
- `CLAUDE.md` "No GPU CI (tests run CPU-only on GitHub Actions)" — this
  is the surviving consequence of `22780ae`.

---

## 5. Lost work / dangling objects

`git fsck --full` output:

```
dangling tree abe06586cb122f16a309b0f7d4f2dae440eee3b9
```

No dangling commits. No dangling blobs. Just one dangling tree.

### What is `abe06586`?

`git ls-tree abe0658` returns a root tree containing
`.claude .github .gitignore CLAUDE.md LICENSE README.md examples
pyproject.toml src tests` — 10 entries, no `CHANGELOG.md`, `CONTRIBUTING.md`,
`MIGRATION.md`, or `uv.lock`.

That layout matches `a9a9d44` (the v0.1.0 release tip on `main`) very
closely — same `.claude`, `.github`, `examples`, `LICENSE`, `README.md`
blob hashes — but the `src` and `tests` subtrees differ (different
SHAs). It is **not** the root tree of any commit (`git log --all
--pretty='%H %T' | grep abe0658` is empty).

**Verdict:** this is almost certainly a transient `git read-tree`
artifact from a worktree-create operation against the ~30 fuzz worktrees
visible in the reflog (`worktrees/fuzz-stack`, `worktrees/fuzz-tile`, …
~50 of them). No commit message, no recoverable narrative. **Not lost
work.**

### Worktree zoo (separate signal)

`git reflog --all | awk '{print $NF}' | sort -u | grep worktrees` lists
**~50 stale worktree refs**, all pointing at `82b853e`. They look like
they were created for parallel fuzz-target experiments and never cleaned
up. They aren't taking disk-space-of-content (they all point to the
same SHA), but they bloat `.git/refs/worktrees/`. Should be pruned via
`git worktree prune` — operational debt, not lost work.

### Cite

- `git fsck --full` (single line of output above).
- `git reflog --all | grep worktrees | wc -l` (count).

---

## 6. Test-to-source LOC ratio drift

Computed at each milestone via
`git ls-tree -r <sha> | grep '^src/.*\.py$' | xargs git show <sha>:<f> | wc -l`:

| sha | src LOC | tests LOC | ratio | milestone |
|---|---:|---:|---:|---|
| `060333d` | 159 | 0 | 0.000 | scaffold (no tests yet) |
| `9352672` | 561 | 0 | 0.000 | first feature shipped |
| `32407c1` | 4817 | 1163 | **0.241** | initial test suite landed |
| `76c33ae` | 4930 | 1154 | 0.234 | first expert-review fix |
| `f044373` | 4999 | 1154 | 0.231 | "all critical" |
| `dc4fadb` | 5033 | 1154 | 0.229 | high-severity (still flat tests) |
| `8d5c5a6` | 5282 | 3794 | **0.718** | "expanded test suite to 279 tests" |
| `22780ae` | 5282 | 3794 | 0.718 | (no LOC change, file move) |
| `25cdfcf` | 5315 | 3794 | 0.714 | GPU fast-path |
| `a9a9d44` | 5435 | 4129 | **0.760** | v0.1.0 release tip |
| `4ede763` | 5829 | 4349 | 0.746 | track-B strides |
| `24035aa` | 6259 | 4532 | 0.724 | track-A MPS |
| `02507da` | 5861 | 4739 | **0.809** | track-D bundle |
| `5ddd26e` | 5503 | 4387 | 0.797 | track-C thread-safety |
| `82b853e` | 7136 | 5620 | **0.788** | HEAD (post-merge) |

Note: the four track LOC counts above each show the *branch tip* in
isolation (i.e. only that track's diff applied); the merged HEAD value
of 0.788 is what actually shipped.

### Drift narrative

- Pre-test-suite: ratio = 0 (commits `060333d`, `9352672`, …, all the
  `[ Feature ]` block).
- Test suite landed in `32407c1` at ratio **0.241** — already low.
- Through 4 fix sweeps (`76c33ae` → `dc4fadb`), src grew but tests didn't.
  Ratio drifted *down* to 0.229. **Sweeps added code without adding tests.**
- `8d5c5a6` more than tripled tests (from 1154 to 3794 LOC) — ratio
  jumps to 0.718. This is the "279 tests" commit.
- v1.0 release adds another +1.5 K LOC src and +1.5 K LOC tests; ratio
  settles around 0.79.

### Surfaced findings

1. **Healthy direction overall** — ratio went from 0.241 → 0.788, mostly
   because of `8d5c5a6` and the `_thread_safety`, `_strides`, `_mps`,
   `_determinism` test files added in the four tracks.
2. **The dip from 0.241 → 0.229 across the critical/high/medium/low
   sweep proves no test-first discipline during that sweep.** Each
   bracket-style "Fix" commit added src LOC without commensurate tests.
   Compare to `5ddd26e` (Track-C, conventional) which adds dedicated
   `test_tolerance_thread_safety.py` (131 lines) and
   `test_race_cuda_home_allowlist.py`. The cultural shift coincides
   with the commit-style switch.
3. **CLAUDE.md still says** "Reporting module (console, json, ci) has
   zero test coverage" — but `tests/test_reporting_console.py`,
   `tests/test_reporting_json.py`, `tests/test_reporting_ci.py`,
   `tests/test_reporting_html.py` all exist at HEAD (added by `02507da`
   track-D). The known-weaknesses list is **stale** — debt because the
   audit-of-record is wrong.
4. **No coverage % is ever reported in commits.** `git log -S 'pytest-cov'`
   returns no matches in the bracket era; `coverage` is mentioned in
   `CONTRIBUTING.md` but never wired into CI. LOC ratio is the closest
   proxy we have, and we know LOC ratio is a poor proxy for branch
   coverage.

### Cite

- LOC table above, derived from `git ls-tree -r --name-only <sha>` per
  milestone.
- Stale weakness claim: `CLAUDE.md` line "Reporting module … has zero
  test coverage" vs. presence of `tests/test_reporting_*.py` files at
  HEAD.

---

## Top-5 debt items (priority-ordered)

1. **GPU CI gate is permanently disabled.** `22780ae` moved GPU-dependent
   tests under `tests/gpu_integration/` and silently exempted them from
   GitHub Actions. As of HEAD, ~6 integration test files (the deepest
   correctness-verifiers, e.g. `test_arch_detection_gtx1650.py`,
   `test_benchmark_accuracy.py`, `test_decorator_combinations.py`)
   never run automatically. Every "GPU bug" found post-`22780ae` is a
   manual-run discovery. **v1.1 must add a self-hosted-GPU runner gate
   or a Lambda-Labs/Modal CI job, or accept that integration tests are
   documentation, not verification.**
2. **Tolerance numerics never settled — and now have a CUDA path and
   an MPS path that diverge.** Five separate commits (`9352672`,
   `8d8c894`, `f7f84eb`, `6562f31`, `24035aa`'s +108-line MPS
   recalibration) recalibrated `tolerances.py`. The MPS table in
   `24035aa` was *added* alongside the CUDA table without a unifying
   abstraction; `git blame` shows ≈50/50 line ownership between the two
   eras. **v1.1 should add a property test that asserts CUDA vs MPS
   tolerances satisfy the same scaling law (`atol ∝ sqrt(k/128)`).**
3. **Patch-over-patch on `close.py` / `memory.py` / `detection.py`.**
   8/8/8 edits each, a 4-tier critical→low fix waterfall, **zero
   `refactor:` commits ever in history**. Each sweep missed real bugs
   (`8d5c5a6` "all-NaN crash" was post-medium-severity-sweep; `25cdfcf`
   "fixed tensor core detection" was post-low-severity-sweep).
   `28d808e` is a -287-line net commit titled "mypy strict" — likely a
   silent revert of speculative type hints; warrants a structural
   re-read before v1.1.
4. **`CLAUDE.md`'s "Known Weaknesses" list is stale and is being used
   as a backlog.** Items it still claims as gaps (reporting test
   coverage, MPS support, stride fuzzing, thread-safety) were
   **delivered by the v1.0 tracks** (commits `02507da`, `24035aa`,
   `4ede763`, `5ddd26e`), but the doc was last updated in `6a07ca6`
   without removing the obsolete items. The audit cannot trust this
   doc. **v1.1 should mechanically split CLAUDE.md "Known Weaknesses"
   into "Active backlog" (tracked in issues) and "Resolved in v1.0"
   (tracked in CHANGELOG.md).** Also: there's no `refactor:` lane
   in the commit grammar of this repo — adopt one in CONTRIBUTING.md.
5. **Conventional-commits compliance is 100% post-v1.0 but
   tooling-incompatible going forward.** `02507da`'s subject
   `feat(reporting+sanitizers):` packs two scopes into one — most
   commitlint configs (`@commitlint/config-conventional`) reject `+`
   in scope. The 30 legacy bracket commits will trip release-please
   /semantic-release if those tools are introduced for v1.1
   automation. **v1.1 should either install commitlint as a hooked
   gate (matching the policy in `CONTRIBUTING.md`) or formally
   document a `since v1.0.0rc1` start point for changelog
   automation.** Operationally: also run `git worktree prune` to
   clear the ~50 stale `worktrees/fuzz-*` refs from the reflog.

---

## Confidence

**high** for items 1, 3, 5 — directly visible in commit metadata and
diffs, no interpretation needed.
**medium** for items 2 and 4 — the divergence/staleness claims would
each need one more cross-check (run the tolerance scaling tests,
diff CLAUDE.md against actual repo state) before action; the evidence
strongly suggests but does not prove an active correctness gap.

**Caveats:**
- Repo is a full clone (`git rev-parse --is-shallow-repository` =
  `false`), so no shallow-clone caveats apply.
- Only 45 total commits is a small sample; "8 edits to plugin.py" is
  not a many-decades-of-codebase signal — most edits are part of
  multi-file sweep commits (e.g. `8d8c894` touched 55 files in one go).
  Edit-counts ≥6 are still meaningful relative to the small denominator.
- The `Akasxh` ↔ `Akash` author name is a single physical author with
  one email; do not interpret it as multi-contributor blame split.
