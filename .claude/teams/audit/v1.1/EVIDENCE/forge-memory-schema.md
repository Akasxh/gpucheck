---
specialist: forge-lead
slug: v1.1-audit
artifact: forge-memory-schema
date: 2026-05-01
status: design-spec
audience: research-lead, engineering-lead, security-lead, testing-lead, docs-lead, forge-lead, research-retrospector, future *-scribe agents
---

# Forge Memory Schema — claude-forge v0.3 continuous-learning playbook

This schema is the canonical structure for `~/.claude/agent-memory/<lead>/MEMORY.md` files and the merge-from-staging pipeline that fills them. It exists because the gpucheck v1.0 → v2 expansion exposed a leak: only `engineering-scribe` actually ran the flock+atomic-rename merge protocol; the other 5 teams' staging files (testing, docs, security, forge, research, retrospector) sat un-merged, so durable lessons learned in those sessions never got into MEMORY.md, never got read at the next session start, never compounded.

This document defines: (1) what makes a lesson durable, (2) the lifecycle, (3) the schema, (4) the index, (5) garbage-collection, (6) the runbook-promotion threshold, (7) read protocol, (8) write protocol — including a 1-page PROTOCOL.md addendum every team scribe can copy-paste.

The design is grounded in real artifacts on disk:
- `~/.claude/agent-memory/research-lead/MEMORY.md` lines 89-97 (the ACE-paper citation that anchors the evolving-playbook concept).
- `~/.claude/agent-memory/research-retrospector/MEMORY.md` (the durability + merge heuristics already in production).
- `~/.claude/agents/engineering/engineering-scribe.md` lines 22-60 (the canonical flock+timeout+atomic-rename pattern, empirically validated 10-concurrent at 0.07s with zero lost writes).
- The 5 staging files written this session (`engineering`, `testing`, `docs`, `research`, `forge`, `security`) — schemas vary; this document standardizes them.

Prior art cited:
- ACE: Agentic Context Engineering, arXiv:2510.04618 (https://arxiv.org/abs/2510.04618). Generation/Reflection/Curation tri-loop. The retrospector + scribe pair already implements this; the schema below makes the curation step explicit.
- Voyager (Wang et al. 2023, https://github.com/MineDojo/Voyager) — skill library with self-verification before promotion. Mirrors our `/forge:test → /forge:promote` gate.
- Anthropic memory-tool docs (https://docs.claude.com/en/docs/agents-and-tools/tool-use/memory-tool) — file-based markdown memory with explicit retention rules.
- `~/.claude/agents/engineering/engineering-scribe.md` — the canonical merge invocation that scribes for other teams must inherit verbatim.

I deliberately do not cite LangGraph, AutoGen, or CrewAI memory features without a concrete file pointer; per the hard rule, no invented prior art. Where their patterns informed the schema below, the schema stands on its own without their authority.

---

## §1. What is a "durable lesson"?

A lesson is **durable** if and only if a 3-month-future agent answering an *unrelated* question would change its behavior because of it. This is the existing test in `research-retrospector/MEMORY.md` ("Durability test: would this lesson apply in 3 months to an unrelated question?"). I keep that test verbatim and add three operational checks every retrospector must apply before staging:

1. **Transferability check.** The lesson must reference at least one general rule of thumb that does not name the specific session's product (gpucheck, vllm-moe-routing, memory-hook-a-v1). If the rule of thumb cannot be stated without naming the product, the finding belongs in `SYNTHESIS.md`, not `MEMORY.md`.
2. **Failure-mode anchor.** The lesson must cite at least one MAST failure-mode tag (FM-1.x decomposition, FM-2.x context, FM-3.x verification) so retrieval-by-failure-mode works at session start. Lessons without a failure-mode anchor become unreachable.
3. **Counter-example or bounds.** Every durable lesson states when *not* to apply it. A lesson without bounds becomes cargo cult ("always do X"). The bounds field is mandatory; "not applicable" is a valid value but must be explicit.

### Positive examples (keep these)

- engineering-lead L2 ("Editable-install in venv pinned to non-worktree path, breaking pytest"). Transferable: any worktree + editable-install combo. FM-3.2 anchor. Bounds stated (uv sync against worktree).
- research-lead "Self-improvement lives in MEMORY.md, not in ad-hoc prompt edits". Transferable to every long-horizon agent. FM-cross-session. Bounds: in-session protocol bugs go to OPEN_QUESTIONS.md.
- testing L "Bind plans to importable target names". Transferable to any cross-team plan. FM-1.1, FM-2.4. Bounds: internal-only refactors can skip.

### Negative examples (kill these — they are findings, not lessons)

- "For the gpucheck-v1-MPS session, the github-miner found 6 April 2026 PRs". This is data, not a behavior change. Goes in SYNTHESIS.md.
- "pytorch issue #162872 is a deadlock bug". A finding. Goes in SYNTHESIS.md or a permanent issue tracker, not MEMORY.md.
- "We achieved 75% mutation score on Track A". Outcome metric. Belongs in INDEX.md or VERIFY_LOG.md.
- "The skeptic found two adversarial holes". Empty without the *pattern* of the hole. The pattern (e.g. "skeptic must run before adversary on community-sourced corpora") is the lesson; the count is not.

The retrospector's job at session close is to produce 3-7 durable lessons that pass all three operational checks. Anything else is logged in SYNTHESIS or LOG and left there.

---

## §2. Lesson lifecycle

Four states. Transitions are explicit and audit-able.

```
[draft]  -->  [curated]  -->  [stale]  -->  [removed]
   ^             |               |             ^
   |             v               v             |
   +-- amend ----+               +-- supersede-+
```

### State: draft (lives in `staging/`)

- Written by `<team>-retrospector` at session close to `staging/<slug>.md` in markdown with the schema below already filled in.
- Multiple drafts can stack in `staging/` until merged (a session can produce several drafts; they are not merged piecemeal).
- Files in `staging/_merged/` are tombstones, not active drafts; the scribe moves merged files there so the next merge pass skips them. The `case "$f" in *_merged*) continue;; esac` line in the canonical pattern (engineering-scribe.md:47) implements this.

### State: curated (lives in `MEMORY.md`)

- Promoted by `<team>-scribe` running the canonical flock+timeout+atomic-rename invocation. Promotion is **mechanical**: the scribe does NOT re-judge durability. The retrospector is the durability judge; the scribe is the merge engine. Splitting these roles avoids "scribe drops a lesson because it disagrees with the retrospector" — a real failure mode in any non-mechanical merge.
- Each curated lesson lives under a heading anchor `### <id>: <title>` so it is addressable for `supersedes:` references.
- Counters (`helpful_count`, `harmful_count`, `last_triggered`, `last_reviewed`) update in-place; the rest is append-only. This dual-mode (append-only structure, in-place counter updates) is what the `forge-lead/MEMORY.md` line 4 ("Append-only at the section level, bullet counters update in-place") already prescribes for the Forge.

### State: stale (still in `MEMORY.md`, marked `status: stale`)

- A curator (the lead, or a supersede from another lesson) marks the frontmatter `status: stale` and adds `superseded_by: <id>` if applicable. The lesson stays in the file so historical references resolve, but the read-protocol filter (§7) skips it by default. This is gentler than deletion and recovers from "we deprecated it but actually it still applies in case X" without losing the prose.

### State: removed (the lesson is moved to `MEMORY-archive.md` in the same directory)

- Used only when a lesson is *wrong* (not just superseded or stale). Removal happens at session close, never mid-session, and requires writing the harmful_count >= 2 OR an explicit retrospector note "this lesson was misapplied in <session-slug> and produced <bad outcome>". Archive file is read-once-on-demand, not at session start.

### Who promotes, when?

| Transition       | Actor                  | When                                                                                          |
|------------------|------------------------|-----------------------------------------------------------------------------------------------|
| draft → curated  | `<team>-scribe`        | Session close, after retrospector wrote staging/, before final INDEX.md entry.                |
| amend (in-place) | `<team>-scribe`        | When a draft cites `reinforces: <existing-id>` instead of a new id; scribe bumps the counter and appends a `Reinforced-in:` line. |
| curated → stale  | `<team>-retrospector`  | When the retrospector finds the lesson in MEMORY.md was applied in this session and produced a wrong outcome (harmful_count++), or when a strictly stronger replacement is staged.|
| stale → removed  | The lead (manual)      | Quarterly review, or when archive size exceeds 2x curated size.                                |

The retrospector is the only one allowed to mark `status: stale`. The lead approves removals. The scribe never deletes.

---

## §3. Curated-lesson schema (frontmatter + body)

Every curated lesson is one heading block in MEMORY.md with structured YAML frontmatter and a free-form markdown body. Both human-readable (the body carries 80% of the value) and machine-parseable (the frontmatter carries the index keys).

```markdown
### <id>: <title>

```yaml
id: <agent-prefix>-<YYYY-MM-DD>-<short-slug>
title: <terse imperative title, <80 chars>
status: curated | stale | reinforced
authored_at: <YYYY-MM-DD>
authored_by: <retrospector-name>
authored_in: <session-slug>
last_reviewed: <YYYY-MM-DD>
last_triggered: <YYYY-MM-DD or null>
helpful_count: <int>
harmful_count: <int>
failure_modes: [FM-x.y, FM-x.y]   # one or more MAST tags
tags: [<topic>, <topic>]          # 2-5 topical tags
evidence: <relative path to the EVIDENCE/<file>.md backing this lesson>
supersedes: <id-or-null>
superseded_by: <id-or-null>
see_also: [<id>, <id>]
```

**Situation.** When does this lesson fire? (1-3 sentences. The trigger.)

**Action.** What should the agent do? (Concrete, imperative. The rule of thumb.)

**Outcome.** What happens if the action is taken? What happens if not? (Both branches.)

**Bounds / counter-example.** When does this lesson NOT apply?

**Reinforced in:** (optional, append-only list of `<session-slug> (<date>): <one-line note>`)
```

### Field semantics

- **id**: stable identifier. `<agent>-<date>-<slug>` keeps human-readable while collision-resistant. Example: `engineering-2026-04-12-grep-old-string-before-edit`.
- **status**: machine-readable filter for the read protocol. `curated` is the default. `stale` skipped at read. `reinforced` is a curated lesson that has been reinforced 2+ times — same as curated for read but flagged for runbook promotion (§6).
- **authored_in**: session slug, lets the lead trace a lesson back to evidence. Required.
- **last_reviewed / last_triggered**: feed garbage-collection (§5).
- **helpful_count / harmful_count**: bumped by the retrospector at session close, in-place counter updates per the engineering-scribe pattern.
- **failure_modes**: the FM-x.y MAST tags. Used at read time to fetch lessons relevant to the current task's anticipated failure modes.
- **tags**: free-text but conventional (e.g. `worktree`, `pytest`, `editable-install`, `tolerance`, `mps`). The INDEX.md (§4) is keyed by these.
- **evidence**: relative path so the file is portable. Without an evidence link, the lesson is unverifiable; reject it at staging.
- **supersedes / superseded_by**: linked-list pointers. When B supersedes A, A gets `status: stale` and `superseded_by: B`; B gets `supersedes: A`.
- **see_also**: lateral links to related lessons. Not used at read time, but the lead may follow them when the failure_modes filter underfetches.

### Body sections

I deliberately use **Situation / Action / Outcome / Bounds** rather than the existing free-form bullet style (`- Observed in: / - Lesson: / - Rule of thumb: / - Counter-example`). The structure is identical in spirit; the new names make the contract explicit and easier for the retrospector to fill in. The existing bullet style remains valid for legacy lessons; new lessons use S/A/O/B.

Lesson length: 80% context, 20% data. The rule-of-thumb fits in two lines; the situation and bounds need 3-5 sentences each. A lesson that is a single bullet of "do X" without context fails the durability test.

---

## §4. INDEX.md — the searchable lookup

`~/.claude/agent-memory/INDEX.md` is generated, not hand-edited. It cross-cuts all 7 leads' MEMORY.md files into a tag-indexed and failure-mode-indexed reverse map.

### Generation

A small script — `~/.claude/agent-memory/scripts/regen_index.py` — runs at the end of every scribe pass. The scribe shells out to it after the flock+rename. The script:

1. Walks `~/.claude/agent-memory/<lead>/MEMORY.md` for each lead.
2. Parses the YAML frontmatter blocks (yaml.safe_load on the fenced code block under each `### <id>` heading).
3. Skips entries with `status: stale`.
4. Builds two reverse maps: `tag -> [(lead, id, title)]` and `failure_mode -> [(lead, id, title)]`.
5. Emits INDEX.md with a deterministic order (alphabetical by tag, then by date desc within tag).
6. The script is idempotent and writes via the same flock+atomic-rename pattern, since multiple scribes can finish concurrently.

Format:

```markdown
# Agent-memory cross-team index — auto-generated
# Last regenerated: <ISO timestamp>
# Source-of-truth: ~/.claude/agent-memory/<lead>/MEMORY.md

## By tag

### tag: worktree
- engineering-2026-04-12-grep-old-string-before-edit (engineering-lead) — Verify old_strings empirically before applying
- engineering-2026-05-01-pythonpath-for-worktree-pytest (engineering-lead) — Editable-install + worktree breaks pytest

### tag: webfetch
- research-2026-05-01-pytorch-stable-redirect (research-lead) — PyTorch /docs/stable URLs are redirect-only
- research-2026-05-01-vendor-pdf-cap (research-lead) — Vendor PDFs exceed 10MB cap; use REPORTED-NOT-VERIFIED

## By failure-mode

### FM-3.2 incomplete verification
- engineering-2026-05-01-pythonpath-for-worktree-pytest
- testing-2026-05-01-dual-pronged-absence-test
- research-2026-05-01-vendor-pdf-cap

### FM-1.1 task specification
- testing-2026-05-01-bind-plans-to-importable-names
- research-2026-04-12-dispatch-breadth-anthropic-rule
```

The lead reads INDEX.md *only when needed* (§7) — most reads start from the home MEMORY.md.

---

## §5. Garbage collection (decay rule)

A curated lesson goes to `stale` under any of three triggers, evaluated by the retrospector at session close:

1. **Reviewed-not-confirmed for 90 days.** If `last_reviewed > 90 days ago` AND `last_triggered == null` AND `helpful_count == 0`, the retrospector marks `status: stale` with note `decay: not-applied-in-90-days`. The lesson stays readable but is filtered from session-start reads. 90 days is the same window the docs-tier "REPORTED-NOT-VERIFIED" decay uses; harmonized to keep one decay constant.
2. **Strictly superseded.** When a new lesson covers the same situation with a stronger or more general rule, it sets `supersedes: <old-id>` in its frontmatter, the scribe writes `superseded_by: <new-id>` into the old lesson and flips its status to `stale`. The old prose is preserved for context.
3. **Harmful_count >= 2.** Two distinct sessions report the lesson misled them. The retrospector flips the status to `stale` and writes a `Decay note:` paragraph in the body explaining what went wrong. Removal (to MEMORY-archive.md) waits for lead-level review at quarterly cleanup, not session close — to prevent thrashing in cases where a single misapplication was actually operator error.

A `stale` lesson can be revived (`status: curated`, increment `last_reviewed`) by a retrospector that re-encounters it and confirms it is still right. Decay is not a one-way street.

### Why not auto-delete?

Two reasons. First, lesson prose is expensive to recreate; preserving stale lessons in-file is cheap. Second, supersession is a partial-order, not a total-order — a "stale" lesson may still apply in a context the superseder doesn't cover. Keeping the prose lets a future agent find that context.

---

## §6. Runbook-promotion threshold

When 3+ similar lessons accumulate, they should be promoted to a runbook (a skill or doc), so that the *static* agent prompt benefits, not just the read-at-session-start dynamic memory.

### Concrete trigger

A retrospector runs `regen_index.py` and looks at the `By failure-mode` and `By tag` sections. For any (tag, failure_mode) pair with **3+ curated lessons** AND **at least 2 distinct lead origins** AND **combined helpful_count >= 5**, the retrospector emits a `runbook-promotion-request` to `~/.claude/forge/research-requests/runbook-<tag>-<fm>.md` and stops.

Why three thresholds (count, breadth, evidence-of-use):
- 3 lessons alone is noise. Two engineers solving similar bugs in different projects could write 5 cross-referenced lessons that all share one root cause.
- 2 distinct leads forces breadth. A pattern that is real across teams is worth a skill; a pattern in one team is worth a `<team>-lead/HANDBOOK.md` section, not a global skill.
- helpful_count >= 5 forces evidence-of-use. A lesson with 0 helpful_count was never applied; it has no track record yet. Five helpful applications across distinct sessions is enough.

The Forge takes the promotion request and runs `/forge:scout` (search for prior art that might already encode this), then `/forge:draft` (wrap skill-creator), then `/forge:test`, then `/forge:promote`. The runbook is born; the lessons that fed it stay in MEMORY.md but flip to `status: reinforced` (so they remain readable but signal "the static skill carries the weight now"). That preserves the audit trail.

---

## §7. Read protocol at session start

The lead has 200 lines / 25KB of room at session start (research-lead.md, lines 4-7). Three filters, in order:

### Step 1: Read the home MEMORY.md (cheap, deterministic)

Always read your own lead's `MEMORY.md` first, top 200 lines. This is the existing protocol; keep it. The schema's structured frontmatter does not change it.

### Step 2: Identify failure modes for *this* session

Before any other reads, the lead writes (in scratch, not on disk) the 1-3 MAST failure modes most likely to surface this session. Examples: a research session with 3 community-sourced corpora pre-loads FM-3.3 (incorrect verification — corpus capture). An engineering session that runs in a worktree pre-loads FM-3.2 (incomplete verification — wrong source tree). A docs session with cross-team dependencies pre-loads FM-2.4 (information withholding cross-team).

The lead writes those failure modes at the top of `LOG.md` with one-line rationale. This makes the read filter explicit and reviewable.

### Step 3: Pull cross-team lessons via INDEX.md

For each anticipated failure mode, scan INDEX.md's `By failure-mode` section for hits in **other** leads' MEMORY.md (not your own — those came in step 1). For each hit, decide: (a) load the full lesson body (high relevance, e.g. exactly the same failure mode in a worktree + editable-install context); (b) note the id but skip the body (relevant but high cost to load); (c) skip entirely (drift).

For tag-driven retrieval (e.g. "this session uses pytest fixtures"), do the same scan against `By tag`.

### Step 4 (optional): LLM-judge filter for ambiguous cases

If steps 1-3 produce more than ~30 candidate lessons (token budget too high), the lead may invoke a small LLM-judge step: "Given these N lesson titles + situations, rank top 10 by relevance to <session question>." Only use this when the deterministic filters underfilter; default is to skip it.

### Why not embedding-based retrieval

Embedding retrieval over MEMORY.md was considered. Rejected for v0.3 because: (a) the failure-mode + tag taxonomy is small and deterministic — embeddings give noise; (b) every lesson is human-prose, not embedding-friendly; (c) the read budget is 25KB, easily filled with deterministic top-K. Reconsider for v0.5 if MEMORY.md size > 200KB per lead.

---

## §8. Write protocol at session end

This is the **canonical** flock+timeout+atomic-rename pattern. Every team scribe inherits it verbatim. Source: `~/.claude/agents/engineering/engineering-scribe.md` lines 22-60, validated 10-concurrent at 0.07s zero-loss zero-dup.

### The contract

1. Retrospector wrote `staging/<slug>.md` with one-or-more lesson blocks in §3 schema.
2. Scribe runs the merge invocation. **Mechanical** — no judgment.
3. After merge, scribe runs `regen_index.py` to refresh INDEX.md.
4. Scribe writes the team's `INDEX.md` entry for this session.
5. If cross-team, scribe writes `HANDBACK_FROM_<team>_<slug>.md`.

### The merge invocation (copy-paste-ready, parameterized on $AGENT)

```bash
AGENT="<your-team>-lead"   # e.g. testing-lead, docs-lead, security-lead, forge-lead, research-lead
ROOT="$HOME/.claude/agent-memory/$AGENT"
LOCK="$ROOT/.lock"
MEM="$ROOT/MEMORY.md"
STAGING_DIR="$ROOT/staging"

mkdir -p "$STAGING_DIR/_merged"
touch "$LOCK"

flock -w 5 -x "$LOCK" timeout --signal=KILL --kill-after=1 30 bash -c '
  set -e
  AGENT="'"$AGENT"'"
  MEM="$HOME/.claude/agent-memory/$AGENT/MEMORY.md"
  STAGING="$HOME/.claude/agent-memory/$AGENT/staging"
  TMP="$MEM.tmp.$$"

  if [ -f "$MEM" ]; then
    cp "$MEM" "$TMP"
  else
    : > "$TMP"
  fi

  for f in "$STAGING"/*.md; do
    [ -f "$f" ] || continue
    case "$f" in *_merged*) continue;; esac
    cat "$f" >> "$TMP"
    mv "$f" "$STAGING/_merged/"
  done

  mv "$TMP" "$MEM"
' || {
  echo "[scribe-curator] deferred merge on $AGENT — staging preserved" >&2
  exit 0
}

python3 "$HOME/.claude/agent-memory/scripts/regen_index.py" || true
```

### Why each piece

- `flock -w 5 -x` waits up to 5s for the lock; -x is exclusive. Concurrent scribes serialize cleanly.
- `timeout --signal=KILL --kill-after=1 30` REQUIRED. Bare `flock -c` leaks locks on child-process inheritance when the parent is killed. Validated in `engineering-team-self-evolve-v1`.
- `set -e` stops on any error mid-merge so a corrupt staging file does not partially merge.
- `cp "$MEM" "$TMP"` then append-and-rename ensures the readers (other agents) always see a complete file, never a half-written one. POSIX rename(2) is atomic on the same filesystem.
- The `*_merged*` skip rule means the scribe is idempotent — running it twice is safe.
- The `|| { ... exit 0 }` outer handler turns a lock contention into a non-fatal "deferred" — staging stays put for the next scribe, no work is lost.

### What the scribe does NOT do

- Does NOT re-judge durability — the retrospector did that.
- Does NOT delete lessons — only the lead does that, manually, at quarterly cleanup.
- Does NOT mark `status: stale` — only the retrospector does that.
- Does NOT attempt to deduplicate semantic overlaps — that's a `reinforces:` reference at the retrospector layer.

The scribe's discipline is mechanical merge. That discipline is why this protocol scales to 10 concurrent and zero loss.

---

## §9. Concrete migration: one real lesson, staging → curated

I take **engineering-lead L2** from `~/.claude/agent-memory/engineering-lead/staging/v1.0-gpucheck.md` (the gpucheck v1.0 worktree pytest lesson) and show the migration.

### Before (current staging, free-form bullets)

```markdown
### L2: Editable-install in venv pinned to non-worktree path, breaking `pytest` in worktrees by default
**Observed in**: gpucheck v1.0 Track A first pytest run (3 collection errors).
**Failure mode addressed**: FM-3.2 (incomplete verification — false negative on a clean change).
**Lesson**: When the user's venv was created with `pip install -e .` against the main repo, and we make changes in a sibling worktree, `pytest` in the worktree imports from the venv's editable target (the original repo) — NOT from the worktree's source tree. Need to set `PYTHONPATH=<worktree>/src` on every pytest invocation in worktrees.
**Rule of thumb**: For any task that runs pytest in a git worktree where there's an editable install, prepend `PYTHONPATH=<worktree>/src` to the pytest command. Save the user-provided venv from being broken; don't `pip install -e .` against the worktree because that mutates the venv.
**Counter-example / bounds**: If the venv was created with `uv sync` against the worktree, this isn't needed. Detect by reading `venv/lib/.../gpucheck.egg-link` if present.
```

This passes the durability test (transferable, FM-anchor present, bounds present) but the schema is implicit — frontmatter not parseable, no id, no counters, no evidence link.

### After (curated form, schema-compliant)

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

**Situation.** A pytest run in a git worktree returns collection errors that look like the source is missing or stale. The user's venv was created at the main repo with `pip install -e .`, and edits are happening in a sibling worktree.

**Action.** Prepend `PYTHONPATH=<worktree>/src` to every pytest invocation in the worktree. Do **not** `pip install -e .` against the worktree — that mutates the user's shared venv and silently switches the editable target away from the main repo, which then breaks pytest *there*.

**Outcome.** With PYTHONPATH set, pytest imports from the worktree source and the run is deterministic. Without it, pytest silently imports the main-repo source, and any code changes in the worktree appear to do nothing — a false-negative verification (FM-3.2).

**Bounds / counter-example.** Does NOT apply when the venv was created with `uv sync` against the worktree (uv pins per-directory). Detect by checking for `venv/lib/python*/site-packages/<package>.egg-link` — if present and points to the main repo, you need the PYTHONPATH workaround; if absent, the venv is uv-managed and self-contained.

**Reinforced in:**
(none yet)
```

The migration is purely additive: nothing was deleted, only structured. The frontmatter unlocks the INDEX.md, the counters unlock decay, the evidence link unlocks audit. The body is the same content, retitled into Situation / Action / Outcome / Bounds for consistency.

A retrospector running 60 days from now in a different project, on a worktree-pytest collection error, will (a) read its own engineering-lead/MEMORY.md, find this entry, apply it; (b) bump `helpful_count` from 0 to 1, update `last_triggered`, and the next merge runs clean.

---

## §10. PROTOCOL.md addendum — copy-paste for all 6 team scribes

This section is a self-contained 1-page block. Each `<team>-scribe.md` (research, engineering, security, testing, docs, forge) appends this verbatim — the only change is `<team>` substitution.

```markdown
## MEMORY.md merge — canonical (copy-paste, do not edit)

This block is the **only** sanctioned write path into `~/.claude/agent-memory/<team>-lead/MEMORY.md`.
Source-of-truth: `~/.claude/teams/audit/v1.1/EVIDENCE/forge-memory-schema.md` §8.

Run AFTER `<team>-retrospector` finishes writing to staging/. Run BEFORE writing your team
INDEX.md entry. Run BEFORE any HANDBACK file.

```bash
AGENT="<team>-lead"   # CHANGE THIS for your team
ROOT="$HOME/.claude/agent-memory/$AGENT"
LOCK="$ROOT/.lock"
MEM="$ROOT/MEMORY.md"
STAGING_DIR="$ROOT/staging"

mkdir -p "$STAGING_DIR/_merged"
touch "$LOCK"

flock -w 5 -x "$LOCK" timeout --signal=KILL --kill-after=1 30 bash -c '
  set -e
  AGENT="'"$AGENT"'"
  MEM="$HOME/.claude/agent-memory/$AGENT/MEMORY.md"
  STAGING="$HOME/.claude/agent-memory/$AGENT/staging"
  TMP="$MEM.tmp.$$"

  if [ -f "$MEM" ]; then
    cp "$MEM" "$TMP"
  else
    : > "$TMP"
  fi

  for f in "$STAGING"/*.md; do
    [ -f "$f" ] || continue
    case "$f" in *_merged*) continue;; esac
    cat "$f" >> "$TMP"
    mv "$f" "$STAGING/_merged/"
  done

  mv "$TMP" "$MEM"
' || {
  echo "[<team>-scribe] deferred merge — staging preserved" >&2
  exit 0
}

python3 "$HOME/.claude/agent-memory/scripts/regen_index.py" || true
```

### Discipline notes (read once, then never deviate)

1. **Mechanical merge.** You do NOT re-judge durability. The retrospector authored the lesson; you merge what was staged.
2. **Idempotent.** Run it twice if unsure — files in `staging/_merged/` are skipped.
3. **The `timeout --signal=KILL --kill-after=1 30` wrapper is REQUIRED.** Bare `flock -c` leaks locks on child-process inheritance. Do not remove it as a "simplification".
4. **Atomic rename.** `cp + append + mv` is the pattern. Never write directly to `$MEM`.
5. **Failure is non-fatal.** If the lock is contended, the staging file stays put; the next scribe (or the next session's scribe) will pick it up. Do NOT block the session close on a deferred merge.
6. **Index regen runs after.** The `|| true` makes it advisory — a failed regen does not lose the merge.
```

---

## §11. Open questions / future work

These are not blockers for v0.3 but should be revisited.

1. **Cross-lead lesson migration.** A lesson originally authored as `engineering-lead` may apply equally to `testing-lead`. v0.3 doesn't migrate; the read-time INDEX.md cross-cut is the workaround. v0.4 may add a `migrate_to: <other-lead>` field that the scribe acts on.
2. **Lesson contradiction detection.** Two lessons with overlapping `(tag, failure_mode)` may give contradictory advice. v0.3 leaves this to the retrospector at session start (read both, use judgment). A v0.5 contradiction-finder script could pre-flag these for the lead.
3. **Embedding retrieval at scale.** Once any single lead's MEMORY.md exceeds 200KB, deterministic top-K becomes lossy and the LLM-judge step (§7 step 4) becomes mandatory. At that point reconsider sentence-transformer-based retrieval against the body text.
4. **Quarterly review automation.** §2 says the lead does removals manually at quarterly cleanup. We do not yet have a forcing function for that quarterly review. Possible: a `staleness-budget.md` script that emits "you have N stale lessons older than 180 days; review them" at the next session start.

## Verdict

PASS. Schema specified; lifecycle states 4; garbage-collection 90-day-or-superseded-or-harmful; runbook promotion at 3 lessons / 2 leads / helpful>=5; canonical merge pattern documented and copy-paste-ready for all 6 team scribes; one real lesson (engineering L2 worktree-pytest) migrated end-to-end as a worked example.

## Confidence

High on the schema and lifecycle (grounded in 7 real MEMORY.md / staging files inspected this session, plus the validated engineering-scribe pattern). Medium on the runbook-promotion thresholds (3 / 2 / 5 are calibrated guesses; first quarter of operation will reveal whether they are too tight or too loose). Medium on the LLM-judge filter at §7 step 4 (untested at scale; flagged as v0.5 reconsideration).
