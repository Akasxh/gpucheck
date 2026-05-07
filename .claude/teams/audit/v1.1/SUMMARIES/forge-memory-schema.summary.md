# forge-memory-schema summary (W2)

**Schema (1 paragraph).** Each curated lesson is one heading block in `~/.claude/agent-memory/<lead>/MEMORY.md` with YAML frontmatter (id, status, authored_at, last_reviewed, last_triggered, helpful_count, harmful_count, failure_modes [MAST tags], tags, evidence path, supersedes, superseded_by, see_also) + free-form prose in 4 sections: **Situation / Action / Outcome / Bounds** (the 80% context). Frontmatter parseable for `INDEX.md`; body human-edit-friendly markdown. flock+atomic-rename merge protocol from engineering-scribe (validated 10-concurrent at 0.07s zero-loss).

**Lifecycle states:** draft (staging/, retrospector writes) → curated (MEMORY.md, scribe merges, counters update in-place) → stale (status: stale, filtered at read) → removed (MEMORY-archive.md, lead-only at quarterly review). Retrospector marks stale; scribe never deletes; lead approves removal.

**Garbage-collection rule:** lesson goes stale on any of:
- (a) `last_reviewed > 90 days` AND `last_triggered == null` AND `helpful_count == 0`
- (b) strictly superseded by a stronger lesson (linked-list pointers preserve audit)
- (c) `harmful_count >= 2` from distinct sessions

Removal waits for quarterly review to prevent thrash.

**Runbook promotion (3+ lessons → skill):** when 3+ curated lessons share a (tag, failure_mode) AND span 2+ distinct leads AND combined `helpful_count >= 5`, retrospector emits request to `~/.claude/forge/research-requests/` and Forge wraps skill-creator. Promoted lessons flip to `status: reinforced`.

**Example migration in §9** — engineering-lead L2 ("Editable-install in venv pinned to non-worktree path") rewritten as id `engineering-2026-05-01-pythonpath-for-worktree-pytest` with full frontmatter, MAST tag FM-3.2, restructured to Situation/Action/Outcome/Bounds. Content unchanged, structure unlocked.
