---
specialist: engineering-architect
slug: continuous-learning-v0.3
charter: design the continuous-learning SYSTEM for claude-forge v0.3
not_in_scope: lesson schema (forge-lead owns that)
input_evidence:
  - /Users/cero/Code/gpucheck/.claude/teams/research/v1.0/SYNTHESIS.md
  - /tmp/yc-recon/claude-forge/BENCHMARKS_v0.2.md
  - /Users/cero/.claude/agent-memory/{research,engineering,forge,research-retrospector}-lead/MEMORY.md (existing corpus)
  - /Users/cero/.claude/hooks/session-capture.sh (existing Stop hook)
  - /Users/cero/.claude/settings.json (existing hook registration)
date: 2026-05-01
---

# Architect — continuous-learning system for claude-forge v0.3

## Why this design exists

Today the system has the *raw materials* of continuous learning but no closed loop:

- 7 lead memory directories (`~/.claude/agent-memory/{research,engineering,forge,security,testing,docs,research-retrospector}-lead/`).
- 4 of 7 have a curated `MEMORY.md`; 3 are silent (security, testing, docs).
- Every lead has a `staging/` subdir holding raw `<slug>-<project>.md` retrospector outputs that have never been merged into the parent `MEMORY.md`. v1.0-gpucheck staging files exist for 6 of 7 leads, sized 118 bytes (security stub) → 9.8 KB (testing).
- One Stop hook (`session-capture.sh`) exists but only writes to research-lead staging on non-team sessions; team retrospectors short-circuit it.
- No SessionStart loader. Each lead reads "first 200 lines" of its own MEMORY.md by static instruction in the persona file — there is no relevance ranker, no cross-team injection, no pattern-extraction.

The v0.3 charter is to make the loop actually close: lessons from session N feed session N+1's dispatch decisions, and the system gets quantifiably better at recurring questions.

The schema (lesson fields, frontmatter shape, validation rules) is forge-lead's deliverable and is referenced as "the v0.3 lesson schema" throughout. The architect commits to where lessons LIVE, when they MOVE, and who READS them.

---

## §1. Hooks — where the loop attaches to Claude Code

Four hook attachment points, each with one job. Numbering matches the user's prompt (a/b/c/d).

### 1a. SessionStart — load relevant lessons into context

- **Hook point**: Claude Code `SessionStart` hook (fires after the persona's static prompt is injected, before the first user turn).
- **Job**: read the calling lead's `MEMORY.md`, the cross-team `SHARED_MEMORY.md`, and (if the session is invoked with a `slug` and a `question`) run the **ranker** (§4) to inject a top-N relevant subset into context.
- **Output**: a system message of the form `<continuous-learning-context>...</continuous-learning-context>` containing the ranked lessons, plus a citation footer naming each lesson's slug-of-origin and date.
- **Scope**: per-lead memory only is loaded by default; cross-team `SHARED_MEMORY.md` is loaded when the question's keywords overlap shared tags (see §3, §4).
- **Latency budget**: ≤500ms wall-clock; the ranker runs offline-indexed (see §4) so SessionStart doesn't block on full-corpus scan.
- **Failure**: if MEMORY.md is missing/malformed, the hook degrades gracefully — log to LOG.md, inject empty context, do not block the session.

### 1b. Agent dispatch — filter for the dispatched lead

- **Hook point**: a `PreToolUse` hook matched on `Task` (the Agent dispatch tool).
- **Job**: when the orchestrator dispatches a sub-agent, intercept the dispatch payload, run the ranker against the **dispatched lead's** memory dir (not the orchestrator's), and **inject the ranked lesson set into the dispatch prompt** as a `<lessons-from-prior-sessions>` block.
- **Why both 1a and 1b**: 1a covers the main-thread session; 1b covers sub-agents. Without 1b, sub-agents would be cold-started with no lesson context and would repeat known mistakes. This is the single most important hook for "cloud is always improving in its output" — it's the dispatch path that fans out to specialists.
- **Caveat (load-bearing)**: the existing research-lead MEMORY.md (line 236-241) documents that **subagent PreToolUse hooks do NOT reliably fire in v2.1.101**. We cannot rely on harness-level interception. The fallback is **synthesis-by-orchestrator**: the orchestrator's own SessionStart hook reads MEMORY.md for every lead it might dispatch, caches them, and **prepends** the ranked subset to the Agent prompt at dispatch time. This is application-layer enforcement, not harness-layer, and it works regardless of harness PreToolUse reliability.
- **Output**: a prepended block in the sub-agent's first message; identical schema to 1a's `<continuous-learning-context>`.

### 1c. SessionEnd — run retrospector → scribe → MEMORY.md merge

- **Hook point**: Claude Code `Stop` hook (already registered; we extend `session-capture.sh`).
- **Job**: at session close, decide one of three branches:
  1. **Team session detected** (an `EVIDENCE/retrospector.md` was written this session): trigger the **scribe-merge** sub-step — read the `staging/<slug>-<project>.md` file the retrospector populated, dedup against `MEMORY.md` (§3 dedup rules), append.
  2. **Non-team session, substantive** (≥10 tool calls, not a chat-only session): run a **lite-retrospector** that produces 0-3 lesson candidates, writes them to `~/.claude/agent-memory/research-lead/staging/adhoc-<sessionid>.md` for the next research session to dedup. (This is what the current `session-capture.sh` already does.)
  3. **Trivial session**: skip.
- **Critical change vs today**: the current Stop hook writes to staging but **never automatically merges into MEMORY.md**. The merge step is performed manually-or-never. v0.3's hook **runs scribe-merge automatically** at session end, behind a `flock` on `~/.claude/agent-memory/<lead>/MEMORY.md`. This is what closes the loop.

### 1d. Idle / scheduled — pattern extraction

- **Hook point**: a separately scheduled task (the user has the `loop` and `schedule` skills available; v0.3 ships a default `~/.claude/scripts/pattern-extract.sh` that can be invoked from either, or run as a nightly cron). NOT a Claude Code in-session hook.
- **Job**: scan the union of all `MEMORY.md` files across leads, group lessons by tag-overlap (using the schema's `tags` field, which forge-lead defines), find clusters of 3+ lessons sharing tags within a 60-day window, and **propose a skill draft** to the forge-lead (writes a stub to `~/.claude/agent-memory/forge-lead/staging/proposed-skill-<topic>-<date>.md`).
- **Why scheduled, not in-session**: full-corpus scan is O(N) over all lessons across 7 leads. At v0.3 scale (<1000 lessons total) this is fast; at v1.0 scale (10K+) it's a 30-second job that has no place in an interactive session. Scheduling decouples it from latency-critical paths.
- **Trigger threshold**: see §5.

---

## §2. Triggers — what fires each hook

| Hook | Claude Code event | Concrete trigger |
|---|---|---|
| 1a SessionStart | `SessionStart` (Anthropic-defined) | Always, at session start, regardless of session type. |
| 1b Agent dispatch (orchestrator-side fallback) | Lead's own SessionStart (caches all 7 MEMORY.md files into the lead's working set) + injection at every `Task` tool emission | Always at SessionStart for caching; at every `Task` emission for injection. |
| 1b Agent dispatch (harness-side, optional) | `PreToolUse` matching `Task` | Best-effort; v0.3 documents that this fires unreliably and the application-layer fallback is the source of truth. |
| 1c SessionEnd | `Stop` (Anthropic-defined) | Always. Hook decides team-vs-adhoc-vs-trivial branch internally. |
| 1d Pattern extraction | None (out-of-session) | (a) cron / launchd nightly, AND (b) on-demand via `claude /loop` if the user wants faster cadence, AND (c) auto-triggered by 1c when the merge brings the cluster count over threshold. |

The "trigger from 1c" path matters: when scribe-merge appends a lesson and that lesson's tags push a cluster over the §5 threshold, the merge script writes a marker file `/tmp/claude-pattern-extract-pending.flag`. The next 1d run sees the flag and prioritizes that cluster. This is the *closed* loop without requiring 1d to scan the full corpus on every merge.

---

## §3. Scope — per-lead vs cross-team vs global

Three tiers. The schema's `scope` field (forge-lead) decides which tier a lesson lives at.

### Per-lead memory (current pattern, kept)

- Path: `~/.claude/agent-memory/<lead>/MEMORY.md`
- Owner: the lead's retrospector (writes), the lead's scribe (dedups + merges).
- Read at: that lead's SessionStart only.
- Holds: lessons specific to that lead's protocol (e.g., research's "REPORTED-NOT-VERIFIED tier", engineering's "PYTHONPATH for worktree pytest").
- ~80% of all lessons live here.

### Cross-team shared memory (NEW in v0.3)

- Path: `~/.claude/agent-memory/SHARED_MEMORY.md`
- Owner: any retrospector that produces a lesson tagged `scope: shared`. The scribe routes it here instead of (or in addition to) the per-lead file.
- Read at: every lead's SessionStart (all 7 leads load this in addition to their own).
- Holds: lessons that affect multiple teams. Examples from current corpus that should have been shared:
  - "Subagent harness has a write-restriction" (BENCHMARKS_v0.2.md §1) — every team needs this.
  - "Persistent monitors have unbounded cost" (BENCHMARKS_v0.2.md §3) — every team that spawns monitors.
  - "Credit caps silently truncate Agent returns" (BENCHMARKS_v0.2.md §4) — every team.
  - "4 concurrent background subagents is the parallel-team empirical ceiling" (research-lead/MEMORY.md L243-248) — currently only research-lead sees this; engineering and testing both need it.
- Size cap: 50 KB hard. When over cap, oldest-by-`last_referenced` lessons are demoted to per-lead memory of their originating team.

### Global / starter-playbook (NEW in v0.3)

- Path: `~/.claude/agent-memory/STARTER_PLAYBOOK.md`
- Owner: hand-curated by the user / forge-lead; retrospectors do NOT write here.
- Read at: every lead's SessionStart, **always**, with no ranker filtering (it's small and load-bearing).
- Holds: bedrock invariants — "Anthropic's dispatch-breadth rule", "skeptic vs adversary lens", "REFRAME is a valid moderator verdict". These are currently embedded in research-lead/MEMORY.md as the "Starter playbook" section; v0.3 lifts them to global so engineering, testing, etc. inherit them.
- Size cap: 10 KB. If over cap, this is a signal that something is over-promoted — demote.

### Routing rules (load-bearing for the schema)

The schema's `scope` field is one of `{lead, shared, global}`. The retrospector sets it; the scribe enforces the routing on merge. A lesson with `scope: shared` written by engineering-retrospector lands in `SHARED_MEMORY.md`, NOT `engineering-lead/MEMORY.md`. A lesson with `scope: global` is **rejected at merge** with an error — only the user/forge-lead promotes a lesson to global, and that's a manual step. This prevents global memory from drifting under retrospector churn.

---

## §4. Ranker — picking which lessons to inject at SessionStart

A naive "load first 200 lines" ranker is what we have today. It's wrong: it loads by file order, not by relevance to *this* session's question. A long-running team accumulates lessons; the most-recent are not the most relevant.

### Design: hybrid tag-match + recency + manual-pinning

Three signals, scored, top-K returned.

- **Signal A: tag overlap (60% weight)**. The schema (forge-lead) defines a `tags` field on each lesson. The session's question, when known (via `slug` + `QUESTION.md`), is keyword-tokenized; tokens that match a lesson's tags contribute to the lesson's score. Implementation: scikit-learn's `CountVectorizer` over the union of `(question_tokens, lesson_tags)` and Jaccard similarity. Cheap (<10ms per lesson, runs at SessionStart).
- **Signal B: recency-decay (20% weight)**. Half-life 90 days. A lesson observed 30 days ago scores 0.79; 90 days ago scores 0.5; 365 days ago scores 0.06. Prevents the corpus from being dominated by ancient lessons that may no longer apply.
- **Signal C: helpfulness counter (20% weight)**. The schema includes `helpful_count` and `harmful_count` (forge-lead). When a lesson is *referenced* during a session (the lead writes "applying lesson X from MEMORY.md" in LOG.md), that's a helpful_count increment. When the retrospector says "lesson X turned out to be wrong / contradicted by this session", that's a harmful_count. Score multiplier: `(1 + helpful) / (1 + helpful + harmful)`.

**Always-include exception**: lessons in `STARTER_PLAYBOOK.md` are always included regardless of score (they're load-bearing invariants). Cap of 10KB ensures this is feasible.

**Top-K**: K=15 lessons per per-lead-memory load; K=10 for shared memory load. Token budget per lesson averages ~600 tokens (the existing schema is lesson_body ≈ 4 paragraphs + 2 bullet lists). Total injection budget per SessionStart: 15×600 + 10×600 + 5×600 (starter) = 18000 tokens ≈ 9% of a 200K context window. Acceptable.

**Implementation**: a small Python script `~/.claude/scripts/rank_lessons.py` invoked by the SessionStart hook. Rebuild a JSON-serialized index of `(lesson_id, tags, observed_date, helpful, harmful, body_path)` whenever scribe-merge runs (it touches MEMORY.md anyway). At SessionStart, the ranker reads only the index, scores, then reads the top-K lesson bodies from disk. Sub-100ms.

**Cold start**: when no `slug` / `QUESTION.md` exists (chat session, not a team session), Signal A's score is 0 for all lessons. The ranker falls back to recency + helpfulness only. Still useful, but degraded.

### Rejected ranker designs

- **Vector embedding similarity (rejected)**: would need to embed every lesson + every question, requires an embedding service or local model, adds 200+ms latency at SessionStart. The corpus at v0.3 scale is small (<1000 lessons) — Jaccard on tags is sufficient. Revisit at v1.0+ if precision suffers.
- **LLM-based "ask Claude which lessons matter" (rejected)**: cost (one model call per session start), latency (>1s), and circular (we'd be using Claude to decide what Claude reads). The deterministic ranker is auditable; the LLM ranker is not.
- **Static "first 200 lines" (current, rejected for v0.3)**: file-order is not relevance.

---

## §5. Pattern-extraction loop — promoting lessons to skills

The user's request: "notice 'this is the 4th lesson about MPS event-timing — promote to a skill'."

### Concrete trigger

**Threshold**: ≥3 lessons across ≥2 sessions sharing ≥2 tags within a 60-day rolling window.

- "≥2 sessions" prevents one over-eager retrospector spawning 5 sub-lessons in one session from triggering a false positive.
- "60-day window" is calibrated against the recency-decay half-life from §4 — within a half-life, the cluster is "active" not "historical."
- "≥2 tags" prevents single-tag clusters (e.g., everything tagged `pytorch`) from triggering. Two-tag overlap is more specific (`{pytorch, mps_event}`).

### Mechanism

The pattern-extraction script (1d) scans the lesson corpus index. For each candidate cluster:

1. Check the cluster against existing skills (read `~/.claude/skills/*/SKILL.md` frontmatter for `tags` overlap). If a skill already covers the topic, **increment the skill's `helpful_count` instead of proposing a new skill**. This is critical — the system should reinforce existing skills, not duplicate them.
2. If no existing skill covers it: write a stub at `~/.claude/agent-memory/forge-lead/staging/proposed-skill-<slug>-<date>.md` with the cluster's lessons, tag set, and a one-line proposal.
3. Set the marker `/tmp/claude-pattern-extract-pending.flag` so the next forge-lead session sees the proposal.

### Why this isn't auto-promotion

The pattern-extractor only **proposes**. The forge-lead reads the proposal, applies its existing gap-investigation protocol (`forge-lead/MEMORY.md`'s "Failed gap investigations" section pattern), runs the skill-creator eval harness, and decides. This is a queue, not a pipeline. Auto-promoting clusters to skills without human-or-forge review is how lesson-corpus rot turns into skill-corpus rot.

### Concrete example from existing corpus

Currently across the 4 active MEMORY.md files, lessons about subagent harness behavior:
- research-lead L207-210: "Adopted-persona pattern 2 is universal..."
- research-lead L236-241: "Claude Code subagent PreToolUse hooks do NOT reliably fire..."
- research-lead L243-248: "4 concurrent background subagents is the parallel-team empirical ceiling"
- engineering-lead L11-17: "Agent persona files have no type system — verify old_strings empirically..."

If tagged consistently (forge-lead schema decision), these 4 lessons across 2 sessions within a 30-day window cluster on `{subagent, harness}` tags → trigger threshold met → skill proposal: "subagent-harness-quirks" reference card. Forge-lead would then decide whether to author a new skill or fold into PROTOCOL.md.

---

## §6. Failure modes and mitigations

### FM-1. Memory bloat

- **Symptom**: research-lead/MEMORY.md is already 35 KB; left unchecked, it'll be 200 KB by v1.0 and exceed the SessionStart load budget.
- **Mechanism**: every retrospector appends, no garbage collection.
- **Mitigation**: scribe-merge enforces a **per-MEMORY.md size cap of 50 KB** (per-lead) / 50 KB (shared) / 10 KB (global). When over cap, the lowest-scoring lessons (Signal C, helpfulness ratio) are demoted to `~/.claude/agent-memory/<lead>/archive/MEMORY-archive-<YYYYQQ>.md` and removed from the active file. The ranker doesn't read archives but pattern-extraction does (so old lessons can still resurrect into a skill cluster).

### FM-2. Lesson rot (most important — calling out per the prompt)

- **Symptom**: a lesson written 6 months ago about a Claude Code v2.1.101 bug is still injected at SessionStart, but the bug was fixed in v2.1.150. The lead applies an obsolete workaround.
- **Mechanism**: lessons have no expiry. The schema's `Counter-example / bounds` field is text-only and unverified.
- **Mitigation (load-bearing)**: **two complementary mechanisms.**
  1. **Recency-decay in the ranker (§4 Signal B)** automatically de-weights old lessons. A 365-day-old lesson scores 0.06 — it's effectively never injected unless tags match perfectly and helpfulness is huge.
  2. **Negative-feedback recording (NEW)**: the schema (forge-lead) MUST include a `harmful_count` field. When a session applies a lesson and the retrospector flags "this lesson was followed and produced wrong outcome", that's a `harmful_count++`. Once `harmful_count > helpful_count` and total ≥3, the scribe demotes the lesson to archive automatically (no human review). This is the system's auto-correction reflex — without it, the corpus only ever grows monotonically wrong.

This is the **most important failure mode**. Memory bloat is annoying; lesson rot is corrosive — it actively makes the system *worse* than no memory by injecting confidently wrong patterns.

### FM-3. Contradictory lessons

- **Symptom**: Lesson A says "always pin torch>=2.11"; lesson B says "do NOT pin torch>=2.11" (this exact contradiction is in the gpucheck v1.1 SYNTHESIS.md §6).
- **Mitigation**: scribe-merge runs a **contradiction check** before merging — diff the new lesson's `Rule of thumb` against existing lessons with overlapping tags using a simple negation-keyword detector ("never" vs "always", "do" vs "don't"). On match, the merge is **deferred** to a `~/.claude/agent-memory/<lead>/CONFLICTS.md` file for the next session's lead to resolve. The conflict file is read at SessionStart with high salience.

### FM-4. Lessons that contradict the user's evolving preferences

- **Symptom**: Akash's CLAUDE.md says "Show diffs before applying them"; a lesson written 6 months ago says "auto-apply when bypassPermissions is set". The lesson stays; the preference changes.
- **Mitigation**: **CLAUDE.md takes precedence over MEMORY.md, always.** The SessionStart hook reads CLAUDE.md FIRST and prepends an explicit instruction: "If a lesson in `<continuous-learning-context>` contradicts your current operating preferences in CLAUDE.md, follow CLAUDE.md and surface the contradiction to LOG.md as a `harmful_count` candidate for the lesson." This routes user-preference drift back into the harmful-count → archive pipeline.

### FM-5. Cross-team lesson pollution

- **Symptom**: a docs-team lesson about Sphinx config is loaded into engineering-team's session and wastes context budget.
- **Mitigation**: the §3 scope routing prevents this by default — docs-team's lesson lives in `docs-lead/MEMORY.md`, not shared. The retrospector has to *explicitly* set `scope: shared` to cross teams, and the schema (forge-lead) requires a justification field for shared scope.

### FM-6. Race on MEMORY.md write (concurrent sessions)

- **Symptom**: BENCHMARKS_v0.2.md showed 92 concurrent processes with 6 leads in parallel. Two retrospectors trying to merge into the same `MEMORY.md` corrupt the file.
- **Mitigation**: scribe-merge uses **`flock` on a sentinel file** (`~/.claude/agent-memory/<lead>/MEMORY.md.lock`) with a 30-second timeout, then **atomic rename** (`mv MEMORY.md.tmp MEMORY.md`). The existing engineering-lead/MEMORY.md mentions this pattern at L1-3 — this design adopts it as the universal rule for all 7 leads.

### FM-7. Pattern-extractor false positives (skill-graveyard)

- **Symptom**: every cluster of 3 lessons spawns a "proposed skill" that the forge-lead has to evaluate. Backlog grows; nothing gets authored.
- **Mitigation**: the pattern-extractor maintains a **dedup memory**: if a cluster was proposed and rejected within the last 60 days, it's not re-proposed unless the cluster size grows by ≥2. This is a back-off mechanism, not strict suppression — genuinely growing clusters do re-trigger.

---

## §7. Metrics — is the system actually working?

Five metrics, each with a target and a measurement mechanism. Captured by an extension to the existing `session-capture.sh` hook into a `~/.claude/agent-memory/_metrics/sessions.jsonl` log.

| Metric | Definition | How to measure | v0.3 target | What "broken" looks like |
|---|---|---|---|---|
| M1: Lessons-per-session | Count of lessons appended at session-end (by scribe-merge) | `wc -l staging/<slug>.md` ÷ session-count | 0.5–2 / session | <0.1 = retrospectors not running; >5 = retrospectors over-eager (lesson rot risk) |
| M2: Lesson-application rate | Fraction of injected lessons that the lead actually references in LOG.md or evidence files | Grep LOG.md for "lesson", count matches ÷ injected count | ≥30% | <10% = ranker is injecting irrelevant lessons |
| M3: Repeat-question kill-rate | When the same question (by tag-cluster) recurs in a later session, fraction where the prior lesson resolves the issue without new investigation | Compare recurring questions' wall-clock time, before vs after | ≥40% wall-clock reduction on the second occurrence | 0% = lessons are not transferable, raw memorization not pattern-extraction |
| M4: Time-to-resolution drop | For tasks tagged identically across sessions, slope of wall-clock-to-completion over session number | Linear regression on `(session_n, wallclock)` per tag-cluster | Negative slope on ≥60% of clusters | Positive slope = lessons make sessions slower, kill the system |
| M5: Skill-promotion rate | Pattern-extractor proposals that become authored skills, per quarter | Forge-lead's `Authored skills catalog` section, dated entries | 1–3 / quarter | 0 = pattern-extractor is dead or proposals are all bad; >5 = forge-lead is rubber-stamping |

M3 is the single most important metric — it directly measures the user's stated goal ("cloud is always improving in its output"). M2 is the early warning: if M2 is low, M3 will be low next quarter.

---

## §8. Migration plan — bringing the 7 leads into v0.3 schema

Current state recap:
- 4 leads with curated MEMORY.md: research, engineering, forge, research-retrospector
- 3 leads with **no** MEMORY.md: security, testing, docs (only have staging files)
- 6 leads have v1.0-gpucheck staging files: research, engineering, forge, security, testing, docs (sized 118 B → 9.8 KB)
- 1 staging file is stale and may need archival: engineering-lead/staging/v1.0.md (separate from v1.0-gpucheck.md)

Order of operations (each step is a separate PR / commit, validated independently):

### Step 1: Schema freeze (forge-lead's deliverable)

Forge-lead publishes the v0.3 lesson schema. Architect waits. NOTHING in this migration plan can run until the schema exists, because every step writes lessons in the new shape.

### Step 2: Archive old free-form MEMORY.md content (no data loss)

For each of the 4 leads with existing MEMORY.md:
- Copy current MEMORY.md → `~/.claude/agent-memory/<lead>/archive/MEMORY-pre-v0.3-2026-05.md` (read-only).
- The active MEMORY.md is rewritten in v0.3 schema in step 4.

### Step 3: Initialize MEMORY.md for the 3 silent leads

For security, testing, docs leads: write a fresh `MEMORY.md` with:
- Header and ownership comment.
- Empty `## Starter playbook` section.
- v0.3 schema-compliant frontmatter.

This unblocks step 4.

### Step 4: Run scribe-merge over every staging file in v0.3 schema mode

For each of the 6 staging files:
- Load the staging file.
- Reformat each lesson to v0.3 schema (forge-lead's schema → required fields per lesson).
- Run the contradiction-check (§6 FM-3).
- Write to the parent MEMORY.md under a new section `## Migrated from staging/<filename> at <date>`.
- Move the staging file to `staging/_migrated/<filename>` (don't delete — provenance).

This is a one-time bulk migration; can be scripted (`~/.claude/scripts/migrate-staging-to-v0.3.sh`). Estimated 30 minutes to write, 5 minutes to run across all 6 files.

### Step 5: Promote shared lessons

For each lesson migrated in step 4, evaluate its `scope` field (set during reformatting in step 4). Lessons tagged `scope: shared` are MOVED from the per-lead MEMORY.md to `SHARED_MEMORY.md`. Candidates from the existing corpus:

- engineering-lead's "Subagent harness has a write-restriction" → shared.
- research-lead's "4 concurrent background subagents ceiling" → shared.
- BENCHMARKS_v0.2.md §3 "monitor lifetimes" → shared (pre-existing as observation, formalize as lesson).

### Step 6: Initialize SHARED_MEMORY.md and STARTER_PLAYBOOK.md

- `SHARED_MEMORY.md`: starts with the lessons promoted in step 5.
- `STARTER_PLAYBOOK.md`: hand-curated by the user / forge-lead from the existing research-lead "Starter playbook" section (currently embedded at top of research-lead/MEMORY.md, lines 15+). This is a manual lift-and-shift — no automation.

### Step 7: Install the v0.3 hooks

Two hook script changes:
- Extend `~/.claude/hooks/session-capture.sh` to run scribe-merge (currently it only writes staging).
- Add a new SessionStart hook script that runs the ranker.

Update `~/.claude/settings.json` `hooks` block. Critical: settings change is one diff to one file, reviewable.

### Step 8: Build the index and the ranker

- Write `~/.claude/scripts/rank_lessons.py` and `~/.claude/scripts/build_index.py`.
- Run `build_index.py` once over the migrated corpus.
- Smoke-test the SessionStart hook against a fresh `claude` session in a scratch directory.

### Step 9: Schedule pattern-extraction

- Write `~/.claude/scripts/pattern-extract.sh`.
- Schedule via launchd plist on macOS (or cron on Linux). User has the `schedule` skill — use it.
- First run is dry-run (`--propose-only`, no marker file); review proposals before going live.

### Step 10: Establish metrics baseline

- Initialize `~/.claude/agent-memory/_metrics/sessions.jsonl`.
- Add the metric-capture line to `session-capture.sh`.
- Document the dashboard in `~/.claude/scripts/show_metrics.sh`.

### Order rationale

The dependency chain is: schema (1) → archive (2) → init (3) → migrate (4) → promote (5) → shared/starter (6) → hooks (7) → ranker (8) → pattern-extract (9) → metrics (10). Steps 1-6 are data migration; 7-10 are runtime. If any step fails, all downstream steps halt — no partial deployment because a partial deployment with no scribe-merge is *worse than today* (lessons are written to staging and never merged, current state).

### Rollback

Each step writes to a new file/path; nothing destructive happens until step 4 (the staging→MEMORY merge). Step 4 keeps the originals (move to `_migrated/`, don't delete). Steps 7-10 are reversible by reverting `settings.json` to a tagged baseline. Total rollback time: ≤5 minutes by design.

---

## §9. Hook + trigger flow diagram

```
                           ┌──────────────────────────────────────────────┐
                           │              CLAUDE CODE SESSION              │
                           └──────────────────────────────────────────────┘
                                          │
              ┌───────────────────────────┼───────────────────────────┐
              │                           │                           │
              ▼                           ▼                           ▼
   ┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
   │  SessionStart hook   │   │   Tool-use loop      │   │     Stop hook        │
   │      (1a)            │   │   incl. Task         │   │       (1c)           │
   │                      │   │     dispatch         │   │                      │
   │  rank_lessons.py     │   │       (1b)           │   │  session-capture.sh  │
   │  reads:              │   │  PreToolUse on Task: │   │  branches:           │
   │   STARTER_PLAYBOOK   │   │   prepends ranked    │   │   team session?      │
   │   SHARED_MEMORY      │   │   lesson set into    │   │     → scribe-merge   │
   │   <lead>/MEMORY      │   │   the dispatched     │   │   substantive adhoc? │
   │  injects top-K       │   │   sub-agent's prompt │   │     → lite-retro     │
   │  into context        │   │  (orchestrator-side  │   │   trivial? → skip    │
   │                      │   │   fallback if        │   │                      │
   │  ≤500 ms             │   │   PreToolUse fails)  │   │  flock + atomic-mv   │
   └──────────┬───────────┘   └──────────┬───────────┘   └──────────┬───────────┘
              │                          │                          │
              ▼                          ▼                          ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │                     ~/.claude/agent-memory/                              │
   │                                                                          │
   │   STARTER_PLAYBOOK.md   SHARED_MEMORY.md   <lead>/MEMORY.md              │
   │     (10 KB, manual)       (50 KB, scoped)    (50 KB, per-lead)           │
   │                                                                          │
   │   <lead>/staging/<slug>.md   ←  retrospector writes                      │
   │   <lead>/archive/...md       ←  scribe demotes (size cap or harmful)     │
   │   <lead>/CONFLICTS.md        ←  contradiction-check defers               │
   │   forge-lead/staging/proposed-skill-*.md  ←  pattern extraction          │
   │   _metrics/sessions.jsonl    ←  every session appends                    │
   │   .index.json                ←  scribe-merge rebuilds                    │
   └──────────────────────────────────────────────────────────────────────────┘
              ▲                                                  │
              │                                                  ▼
              │                                       ┌──────────────────────┐
              │                                       │ Pattern-extractor    │
              │                                       │       (1d)           │
              │                                       │                      │
              │  /tmp/claude-pattern-extract-pending  │ pattern-extract.sh   │
              └───────────────────────────────────────│ scans .index.json    │
                                                      │ clusters by tags     │
                                                      │ ≥3 lessons + 60d +   │
                                                      │ ≥2 tags overlap +    │
                                                      │ no existing skill    │
                                                      │ → proposed-skill-*.md│
                                                      │                      │
                                                      │ runs: nightly cron   │
                                                      │       OR /loop       │
                                                      │       OR triggered   │
                                                      │       by 1c flag     │
                                                      └──────────────────────┘
```

Read top→bottom for one session's lifecycle; read bottom-up for the cross-session learning loop. The closed loop is: Session-end (1c) writes lessons → next Session-start (1a) reads them → dispatched specialists (1b) get filtered subset → next retrospector either reinforces (helpful_count++) or contradicts (harmful_count++) → 1c merges that signal back. Pattern-extractor (1d) runs orthogonally and feeds the forge.

---

## §10. Open design questions (for plan-skeptic to attack)

1. **Helpful_count detection mechanism**. The §4 ranker depends on `helpful_count`, but how does the system *detect* that a session "applied" a lesson? Options: (a) the lead writes "applying lesson X" verbatim in LOG.md and the scribe greps for it; (b) the retrospector explicitly cites lessons-applied in a `cross-references` section; (c) an LLM judge reads LOG.md vs injected lessons. Currently I've assumed (a)+(b). (c) is more reliable but adds a model call per session-end. Recommend: ship with (a)+(b), measure detection rate as M2, escalate to (c) only if M2 is unreliable.

2. **Where does adhoc-session memory go**. The current Stop hook writes adhoc lessons to `research-lead/staging/`. v0.3 should they go to a new `adhoc-lead/` or stay routed to research? Routing to research makes adhoc lessons influence research's MEMORY.md inappropriately. Recommend: introduce `~/.claude/agent-memory/general-lead/MEMORY.md` for adhoc/non-team sessions; route the existing hook there. Forge-lead should sign off on whether `general-lead` deserves a full lead identity.

3. **Schema fields the architecture depends on** (forge-lead, please ensure these exist):
   - `tags: list[str]` — for ranker tag-overlap.
   - `scope: {lead, shared, global}` — for routing.
   - `observed_date: ISO8601` — for recency-decay.
   - `helpful_count: int`, `harmful_count: int` — for harmful auto-archive and ranker.
   - `cluster_id: str` (optional) — populated by pattern-extractor when the lesson contributed to a skill proposal; lets the system unwind a proposal back to its constituents.
   - A `bounds: str` field is already standard in current corpus and should be preserved.

4. **Ranker context budget under attack**. K=15 lessons × 600 tokens = 9000 tokens per per-lead load, but the existing research-lead MEMORY.md has lessons up to 1500 tokens (the orchestration-full-activation ones). At worst case this is 22,500 tokens — 11% of context — borderline. Mitigation: a per-lesson size cap of 1000 tokens enforced at scribe-merge time (truncate with `...truncated, see archive` link).

5. **Whether 1d (pattern-extractor) is in-scope for v0.3 or should be deferred to v0.4**. Cost: ~150 LOC of script, 1 launchd plist. Benefit: skill proposals start flowing 60 days post-deployment. Architect's recommendation: ship a stub (script that only logs, no proposals) in v0.3 to seed the data; defer real proposal generation to v0.4 once the corpus has enough density to cluster usefully. This is a hedged recommendation; happy to defer to forge-lead.

---

## Verdict

The system attaches at four hook points (SessionStart load, dispatch filter, SessionEnd merge, scheduled pattern-extraction), routes lessons by three scopes (per-lead, shared, global), and ranks by three signals (tag overlap 60%, recency 20%, helpful/harmful counter 20%). The single most important failure mode is **lesson rot**, mitigated by the harmful_count → auto-archive reflex; this is non-negotiable, otherwise the system gets confidently worse over time. Migration is 10 sequenced steps with zero data loss until step 4 and full rollback ≤5 min through step 10. Schema dependencies on forge-lead are explicit in §10; the architect commits everything that does NOT touch the schema.

## Confidence

High on §1-§4 (hook anatomy, scope tiers, ranker design) — these are deterministic infrastructure choices grounded in the existing 7-lead memory layout, the existing Stop hook, and the existing retrospector→scribe pattern. High on §6 FM-2 mitigation — harmful_count auto-archive is the only mechanism that prevents corpus rot, and it falls out of the schema. Medium on §5 (pattern-extraction trigger threshold) — the "≥3 lessons, ≥2 sessions, 60-day window, ≥2 tags" threshold is a calibrated guess; it should be re-tuned at 90-day post-deployment review based on M5 (skill-promotion rate) and the false-positive rate of proposals. Medium on §7 metric targets — these are first-pass numbers; M3's "≥40% wall-clock reduction" is the single most important target and the one most likely to need adjustment after first-quarter measurement.
