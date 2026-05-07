# EVIDENCE — docs-retrospector

**Persona:** docs-retrospector (cross-session — extract 3-7 binding
lessons for the docs-lead `MEMORY.md` playbook).
**Run:** Phase 1 audit, gpucheck v1.0.

> Per PROTOCOL.md §"Session close — Retrospection + handback", this
> file is the staging input. Lead merges into
> `~/.claude/agent-memory/docs-lead/MEMORY.md` only at session close.

## Lessons

### L1. **Audit before draft is non-negotiable when the cross-team substrate is missing.**

Phase 1 was charged with drafting CHANGELOG / CONTRIBUTING / MIGRATION
even though `<cwd>/.claude/teams/research/v1.0/SYNTHESIS.md` and
`<cwd>/.claude/teams/engineering/v1.0/DIFF_LOG.md` did not yet exist.
The right move was to **draft skeletons with explicit
`{{TODO Phase 3: cite SYNTHESIS / DIFF_LOG}}` markers** rather than
either (a) refusing to draft, or (b) inventing entries.

Operationalize: when a docs charter requires drafts that depend on
cross-team data not yet on disk, every uncertainty must be a
`{{TODO Phase 3: ...}}` marker, never a glib placeholder. This
matches DocAgent (truthfulness > coverage) and the protocol's
"never invent" rule.

### L2. **`git log` is part of the source of truth.**

The CLAUDE.md "Git Conventions" claim "conventional commits
(type(scope): description)" is contradicted by 25/25 of the most
recent commits, which use `[ Type ] :` brackets. This was caught by
the docs-skeptic, NOT by the docs-reviewer's text-vs-source pass.

Operationalize: when a doc claims a *process* convention (commits,
branches, releases), validate against repository history
(`git log`, `git branch -a`, `git tag`), not just against other
docs. Add this to the docs-reader/reviewer checklist.

### L3. **The "lead-paragraph hero number" is a high-risk accuracy locus.**

`README.md:12` claims "8 real bugs" but the same README's "Bugs found"
table lists only 5. The lead-paragraph is the most-quoted text in any
project; an unbacked number there is more damaging than the same
number missing from a deep section. Skeptic-level adversarial review
is required for hero claims; reviewer's normal pass missed this.

Operationalize: every numeric claim in a README's first 30 lines must
have a clickable / followable backing in the same README, OR a
referenced source. Add to the docs-reviewer checklist.

### L4. **API-shape decisions must be flagged as such, not buried.**

The MPS pivot is not "add a few CUDA → MPS string substitutions". It
forces a decision on `GPUInfo` shape — extend in place, sibling
dataclass, or Protocol — that engineering must make and docs must
mirror. The first draft of MIGRATION_v0_to_v1.md presented this as a
single "API change" sentence; revising to enumerate Options A/B/C
with explicit `{{TODO Phase 3: confirm}}` markers raised quality.

Operationalize: when a docs gap stems from an unmade engineering
decision, present the option-space explicitly in the migration draft.
Do not collapse to a single guess.

### L5. **CUDA-only language audit must extend to docstrings, constants, and dataclass shapes — not just prose.**

The auditor's first pass focused on README/CLAUDE.md prose. The reader
pass surfaced a parallel set of stale items inside source docstrings
(14 cases) and constant tables (`_KNOWN_SPECS`, `SM_TO_ARCH`,
`_TENSOR_CORE_GEN`, `_DTYPE_TC_MIN_GEN`). Documentation widening for a
new backend must address all four layers: prose, docstring text,
exported constants, and dataclass shape.

Operationalize: when widening a project from one backend to N, the
audit checklist must include (1) prose, (2) docstrings, (3) exported
constants and lookup tables, (4) dataclass / Protocol field shapes.

### L6. **`pytest` hooks and module dunders should NOT inflate the missing-docstring count.**

The reader's initial count of 24 missing public docstrings was
correctly trimmed to 19 by the skeptic — `gpucheck.__getattr__` is a
module dunder (not user-facing), and the four `pytest_*` hooks are
conventionally exempt. Inflated counts undermine the audit's
credibility downstream.

Operationalize: distinguish three buckets: (a) user-facing public
symbols, (b) framework hooks (pytest, asyncio, etc.) where convention
allows skipping, (c) module dunders. Only (a) is a "must fix" item.

### L7. **Deliverable file paths and line numbers are the audit's contract with engineering.**

The whole point of the AUDIT.md format is that engineering can do
`grep -n` or jump-to-line and act on each item without re-deriving
the finding. Every entry in `AUDIT.md §A` and §B includes both the
absolute path and the 1-based line number. This is non-negotiable.

Operationalize: NO bullet in any AUDIT.md may exist without a
file:line. If the finding is genuinely cross-cutting (e.g. "CUDA-only
language" appearing many places), enumerate every site rather than
collapsing to a single bullet.

## Cross-session takeaways for `MEMORY.md`

(Lead merges 3-7 of these into the playbook at session close.)

- Phase 1 (audit + skeleton) is a legitimate stop on the way to
  shipped docs; it has its own evaluator gate.
- Cross-team handback markers (`{{TODO Phase 3: cite XXX_LOG.md}}`)
  are the right tool when dependencies are not yet on disk.
- `git log` is source of truth for *process* conventions; never
  trust prose alone.
- API-shape decisions get enumerated option-spaces, not guesses.
- The four-layer language widening checklist (prose / docstrings /
  constants / dataclass shape).
- The (a)/(b)/(c) docstring-bucket distinction.
- File:line citations are mandatory for every audit bullet.

## Failure modes avoided in this Phase 1

- **FM-1.2 (invention):** Avoided. Every concrete claim is sourced;
  uncertainty is marked `{{TODO Phase 3}}`.
- **FM-2.3 (style drift):** Avoided. Keep-a-Changelog format strict;
  CONTRIBUTING uses repo's actual commit convention.
- **FM-3.3 (accuracy regression):** Mitigated. Skeptic caught two
  pre-existing accuracy bugs (8 vs 5 bugs; commit-style claim) that
  are independent of the v1.0 pivot.
- **FM-3.1 (over-completion):** Avoided. Did not draft CODE_OF_CONDUCT
  or SECURITY which were out of clean scope; flagged for deferral.
- **FM-3.2 (broken examples):** Avoided. Drafts contain only
  illustrative code, no claim-to-runnable snippets.

## Gaps for the next session

- `examples/` directory was not read in Phase 1. Phase 3 must
  inventory each example file and re-run on both CUDA and MPS hosts.
- Sphinx/MkDocs site decision still open (out-of-scope here).
- Tolerance numbers for MPS still depend on research SYNTHESIS
  sub-question 7.
- CODE_OF_CONDUCT.md and SECURITY.md still open.
