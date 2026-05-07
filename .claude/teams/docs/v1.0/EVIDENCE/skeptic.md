# EVIDENCE — docs-skeptic

**Persona:** docs-skeptic (close gate, FM-3.3 — attack documentation
quality: inaccuracies, gaps, stale content, over-documentation).
**Method:** Adversarial pass over the auditor's findings — what is the
auditor missing, what is over-claimed, what's the worst single
unaddressed risk?

## Attack 1 — "stale CUDA language" claim is too narrow

The `EVIDENCE/reviewer.md` and `EVIDENCE/reader.md` both say the README
is "CUDA-only" and needs MPS widening. But the issue is wider: the
codebase is also **NVIDIA-only** in subtler ways:

1. `arch/detection.py:14-48` — every architecture name is an NVIDIA
   marketing tag. Documentation that just says "supports MPS" without
   addressing the `architecture` field on `GPUInfo` will mislead users
   who write `if gpu.architecture == "Hopper"`. The `GPUInfo` dataclass
   shape (`detection.py:104-122`) is **NVIDIA-shaped** —
   `compute_capability: tuple[int, int]` is meaningless on Apple GPUs;
   `tensor_core_generation: int | None` is mismapped to MPS (Apple has
   matmul-accelerated SoCs but the concept is not "tensor cores").
2. `analysis/roofline.py:40-46` — `_KNOWN_SPECS` lacks Apple entries.
   Phase 3 docs need to commit to representative Apple Silicon FLOP/s
   and bandwidth numbers (M1: ~2.6 TFLOP/s FP32, ~68 GB/s; M2 Pro:
   ~6.8 TFLOP/s FP32, ~200 GB/s; M3 Max: ~28 TFLOP/s FP32, ~400 GB/s,
   approximate). Documentation must source these from Apple's published
   GPU specs, **not folk numbers from blogs**.
3. `sanitizers/race.py:99-127` — `_build_wrapper_script` shells out to
   `compute-sanitizer`, which is NVIDIA-only. There is no MPS
   equivalent. Honest docs should say so explicitly: "compute-sanitizer
   is unavailable on Apple Silicon; use Xcode's Metal Validation
   Layer for memory/race issues there".

**Auditor risk:** if the docs only widen "CUDA → CUDA + MPS" surface
language but leave `GPUInfo` shape unaddressed, users hit
`AttributeError`-shaped surprises on MPS. AUDIT.md must call this out
as an **API-shape decision** that engineering must make and docs must
mirror.

## Attack 2 — "20 missing docstrings" might be over-claimed

The reader counts 20 truly user-facing missing docstrings. Defensible
breakdown:

- 4 dtype groups (`FLOAT_DTYPES`, ...). These are advertised in
  README:150 and re-exported from `gpucheck` top-level. **Real public
  API; missing doc is real.**
- 4 shape groups (`SMALL_SHAPES`, ..., `EDGE_SHAPES`). Same.
- 4 fuzzing constants (`TILE_SIZES`, `PRIMES`, `POWER_OF_2_BOUNDARIES`,
  `LARGE_DIMS`). Re-exported via `fuzzing/shapes.py:235-242`. Real
  public API. **Real gap.**
- 4 `JSONReporter` mutators. CLAUDE.md says reporting has "zero test
  coverage" — these are public methods imported via the lazy map at
  `reporting/__init__.py:8-13`. **Real gap.**
- 2 `MemoryTracker.start/stop`. Real public API per
  `fixtures/__init__.py:8-17`. **Real gap.**
- 1 `_MutableReport.to_report` — this one has a leading underscore;
  arguably **NOT public**. Skeptic flags: drop from MISSING list, or
  reclassify as "leaks via memory_guard yield, deserves a doc anyway".
  Recommendation: keep it (memory_guard yields the mutable; the
  `to_report()` method is observable to users).
- 1 `gpucheck.__getattr__` — module dunder, not public API. **Drop**
  from MISSING. Reduce headline count.

**Revised MISSING count: 19** (drop `__getattr__`).

Plus 4 pytest hooks (`pytest_addoption`, `pytest_configure`,
`pytest_collection_modifyitems`, `pytest_terminal_summary` in
`plugin.py`). Pytest convention treats hook docstrings as optional;
**skeptic allows skipping these**. Final headline: **19**.

## Attack 3 — the v0.1.0 → v1.0.0 jump skips RC

Reader and reviewer both flag `__version__ = "0.1.0"` (`__init__.py:8`)
and `pyproject.toml:7`. But the charter says "Phase 3 fills entries
from DIFF_LOG" and references `[1.0.0rc1]`. There is no draft of the
intermediate `[1.0.0]` entry. Skeptic recommends:

- CHANGELOG_DRAFT.md should include **both** `[Unreleased]` and
  `[1.0.0rc1]` placeholder sections so Phase 3 has a clear template
  for either path (RC-then-stable or RC-as-final).
- MIGRATION should call out that `0.1.0 → 1.0.0` skips the
  conventional `[0.x.0]` pre-release path; users may be surprised. Add
  a section "Why the version jump?".

## Attack 4 — examples/ directory is unaudited

Reviewer focused on README + CLAUDE.md text. The
`examples/` directory (`README.md:454-456`) contains 6 files including
`triton_layernorm_bug.py` and `triton_matmul_bug.py`. None of these
were read by the docs-reader pass. **Risk:** docstrings inside example
files might be CUDA-only, breaking the "real reproducer" claim if a
Mac user runs them.

Recommendation for AUDIT.md: add a "Phase 3 docs-tester must run
`examples/` on both CUDA and MPS hosts" item.

## Attack 5 — the "511 test configurations / 8 real bugs" hero number is fragile

The number "8 real bugs" appears in:
- `README.md:12` (lead paragraph).
- `CLAUDE.md` Strengths.

The README "Bugs found" table (`README.md:328-334`) lists only 5. The
GitHub issue links go to `triton#9838` and `triton#9839` (2 distinct).
The remaining 6 are "torch.baddbmm", "torch.bmm", "cuFFT" (3 in the
table) plus 3 unlisted ones. Skeptic's question: **where are the other
3 bugs documented?**

If they are in `examples/` or in unmerged commit notes, the README
lead paragraph is **technically misleading**. AUDIT.md must flag this
as a P0 accuracy claim that engineering needs to either:

(a) expand the "Bugs found" table to 8 rows, with links/file refs,
(b) reword "8 real bugs" to "8 distinct issues, of which 5 are
    listed below" with provenance for the other 3, or
(c) reduce the lead-paragraph number to match what is documentable.

This is independent of the MPS pivot and is a real accuracy risk
*today*.

## Attack 6 — over-documentation in CONTRIBUTING_DRAFT?

User asked for a "full structure: dev setup, test layout, PR process,
conventional commits, expert system, security policy". Skeptic warning:
**do not invent expert-system details**. The CLAUDE.md table maps
modules to experts, but the personas live in `~/.claude/agents/` —
those are *user-private*, not part of the published repo. Public
CONTRIBUTING.md should reference the **conventional-commits prefix
shape that is visible in git log**, NOT the private agent system.

The git log shows `[ Fix ] :`, `[ Feature ] :`, `[ Docs ] :`,
`[ Test ] :`, `[ Perf ] :`, `[ README ] :` — these are
**bracket-style**, not strict Conventional Commits 1.0
(`type(scope): description`). Skeptic flags:

- CONTRIBUTING.md should document the **actual** convention used in
  this repo (`[ Type ] :` brackets) and either (a) standardize on
  Conventional Commits going forward (breaking change), or (b)
  document the existing convention as the project standard.
- CLAUDE.md says "conventional commits (type(scope): description)"
  but the actual git log does **not** follow that. **CLAUDE.md is
  inaccurate today.**

This is a real accuracy bug in CLAUDE.md — `git log --oneline -25`
shows 0 commits matching `type(scope):` and 25/25 matching
`[ Type ] :`. AUDIT.md gets a new P1 item.

## Attack 7 — security policy

User charter mentions CODE_OF_CONDUCT.md but NOT SECURITY.md. Skeptic
notes: a v1.0 release tag without a SECURITY.md or `security@` contact
is a soft red flag for OSS consumers. The session has a security team
(`.claude/teams/security/`), so docs should defer rather than draft.
AUDIT.md should flag SECURITY.md as a **deferred-to-security-team**
item, not a docs-team blocker.

## Skeptic verdict

The audit is broadly correct but missed:
- API-shape implications of MPS for `GPUInfo` and `architecture`
  (Attack 1).
- `examples/` directory unaudited (Attack 4).
- "8 bugs vs 5 in table" accuracy bug, **independent of v1.0** (Attack 5).
- The actual commit-message convention contradicting CLAUDE.md (Attack 6).
- `_KNOWN_SPECS` Apple GPU entries needed for roofline accuracy (Attack 1).

These are all incorporated into AUDIT.md.

The 20-missing-docstring claim should be **19** (drop
`gpucheck.__getattr__`). The 4 pytest hooks are conventionally OK to
leave undocumented, so the headline is **19 missing public-symbol
docstrings**.

The drafts (CHANGELOG / CONTRIBUTING / MIGRATION) should be honest
about the parts that depend on engineering DIFF_LOG and research
SYNTHESIS — neither of which exists yet — and use explicit
`{{TODO Phase 3}}` markers rather than glib placeholders.
