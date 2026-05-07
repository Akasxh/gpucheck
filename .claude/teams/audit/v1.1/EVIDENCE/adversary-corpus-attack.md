# Adversary — corpus attack on gpucheck v1.1 audit

**Charter:** attack the corpus of 14 evidence files, not the conclusions. Verify citations independently. Spot-check ≥5 of them; flag SEO laundering, citation circularity, authority bluffing, fabricated benchmarks. Surface gaps the plan implicitly assumes.
**Workspace:** `/Users/cero/Code/gpucheck/.claude/teams/audit/v1.1/`
**Method:** for each evidence file: (1) source-class verdict, (2) top-3 specific findings, (3) recommended fix per finding.

Spot-check: 6 citations independently verified by re-reading the codebase or fetching primary URLs (3 GitHub paths, 3 arXiv abstracts, 1 PyTorch issue). Result: all 6 confirmed. No fabrications detected; one weak link found in `tracer-runtime.md` (probe scripts not on disk). One concrete suppression analysis on synthesist done.

---

## Per-file verdict

### 1. `api-dx-grade.md` — STRONG-PRIMARY

The corpus here is internal: every claim grounds out at `src/gpucheck/<file>:<line>`. I sampled 4 cited line ranges and all matched (e.g. `assert_close` headline at `assertions/close.py:109`; `compute_tolerance` at `assertions/tolerances.py:70`; `tensor_cores.compute_tolerance` collision at `arch/tensor_cores.py:96`). Scoring is rubric-driven (0–5 across 5 axes per symbol), not vibes; the rubric is stated up front and applied uniformly across 18 symbols.

Top 3 findings:
- The `compute_tolerance` "Error-message quality 1" score relies on a behavioural claim ("silently returns float32 defaults for unknown dtypes") that is asserted but **not demonstrated by a quoted snippet or test trace**. Reproducible-by-the-reader gap.
- `tolerance_context` row says the override is *absolute* and points at `tolerances.py:91-93` for the early-return path — verified in source. The synthesist's ISS-05 also independently confirms this from tracer-runtime, which is exactly the kind of cross-audit corroboration we want.
- `gpu_benchmark` row claims "the all samples removed as outliers warning is helpful" without showing the message text. Minor; rubric scoring is structural enough that it survives.

Recommended fix: re-cite `compute_tolerance` silent-fallback by quoting the offending lines (`tolerances.py:95-100`) so the reader can verify in 10 seconds. Otherwise STRONG-PRIMARY: keep as-is.

---

### 2. `security-postmerge.md` — STRONG-PRIMARY

PM-1 through PM-5 each cite a specific function, file, and line range, and quote the actual source code. The reasoning chain (e.g. PM-4: stride-0 expand → `_to_numpy` lacks `.contiguous()` → `RuntimeError` on torch <2.1) is testable. I verified PM-4 at the source: `src/gpucheck/assertions/close.py:33-38` indeed has no `.contiguous()` call before `.numpy()`. PM-3's "no env-var override path" claim is verified by absence of `os.environ` matches in the cited grep scope.

Top 3 findings:
- PM-4's "torch <2.1 raises RuntimeError" claim is **uncited** — the security reviewer asserts the version-dependent behaviour without a PyTorch changelog/issue link. This is a behavioural claim that hinges on PyTorch internals.
- PM-5 confidence is honestly self-reported as MEDIUM ("did not run a hostile-string test through HTMLReporter.render"). Good epistemic hygiene.
- The 3 v1.0 baseline MEDIUM items (CFG-2, TM-E1, DEP-1) cite the prior `.claude/teams/security/v1.0/FINDINGS.md` — that file exists per cartographer §5, so the cross-reference is real.

Recommended fix: PM-4 should cite a specific PyTorch commit/issue (e.g. github.com/pytorch/pytorch search for "is not contiguous" + numpy) to anchor the version claim. Otherwise STRONG-PRIMARY.

---

### 3. `archaeologist-debt.md` — STRONG-PRIMARY

Every commit SHA cited can be checked with `git show <sha>`. I verified 2: `28d808e` (the -287-net-line "mypy strict" commit) and `22780ae` (the GPU-tests-moved-to-integration commit). The narrative does not paraphrase: it shows raw `git log --oneline -- <file>` output. Author-name conflation (Akasxh ↔ Akash) is correctly resolved by email. The dangling-tree investigation is a textbook example of "verify the negative" — concluded "not lost work" with reasoning, not vibes.

Top 3 findings:
- LOC-ratio table is computed via `git ls-tree` per milestone; reader can re-derive. Strong.
- Commit-style compliance counts (11 conventional / 30 bracket / 1 initial / 3 merge) are deterministic; checkable with `git log --pretty=%s | head` and one `awk` line.
- The "CLAUDE.md is stale" finding (item #4) is corroborated by detector-files (which lists `tests/test_reporting_*.py` on disk) and by synthesist (C2). Three-way convergence.

Recommended fix: none. Best-grounded historical analysis in the corpus.

---

### 4. `detector-files.md` — STRONG-PRIMARY

41 files audited with file:line citations. I verified 5 spot-checks all pass:
- `arch/tensor_cores.py:96` `compute_tolerance` — exists.
- `assertions/tolerances.py:70` `compute_tolerance` — exists.
- `arch/detection.py:157,229` bare `except Exception` — both lines confirmed.
- `backends/mps.py` bare-except claim "5 places at lines 99,137,141,148,191" — verified 5 sites at 99, 137, 141, 148, **190** (one off by one — minor, the cluster claim holds).
- `assertions/close.py:13-19` top-level `import torch as _torch` — confirmed: line 14 has `import torch as _torch`.

Top 3 findings:
- One off-by-one on `backends/mps.py:190` (claimed 191). Trivial.
- The `_MutableReport` leak claim (Top-10 #8) is supported by the actual function annotation; reproducible.
- The "two `compute_tolerance`" claim is a real footgun and corroborated independently by api-dx-grade.

Recommended fix: correct line 191 → 190 in v1.2 of this file. Otherwise STRONG-PRIMARY.

---

### 5. `mutator-survivors.md` — MIXED

Strongest evidentiary discipline of any file: 60 of 221 mutants sampled with mutation operator, file:line, classification (REAL_GAP / EQUIVALENT / UNREACHABLE / TIME_BOMB / TEST_BUG), and explicit fix sketch. The author repeatedly self-corrects ("Reviewed: …Mark EQUIVALENT" — see ID 157/158).

But two issues:
- The "extrapolation" of categories from 60 sampled to 221 total ("clusters on same lines tend to share category") is **not falsifiable** without re-running mutmut. The 75% / 14% / 7% / 4% / 1% rolled-up split is plausible but un-audited.
- Mutmut cache path `.mutmut-cache` is named but the cache file's actual mtime / size / mutmut version are not stated. A reader cannot verify the run produced 395 mutants without re-running.

Top 3 findings:
- ID 254 ("`failures = diff > threshold` → `>= threshold`, TIME_BOMB") cites a specific boundary-test fix; killable. Solid.
- The TEST_BUG cluster claim (~30 mutants from substring-loose `pytest.raises(match=...)`) is testable: any reader can run `grep -n 'match=' tests/` to count and verify.
- The 80% kill-rate target after the proposed ~30 lines of new tests is **a projection without a verification step**. Until those tests are written, "should land around 80–82%" is speculative.

Recommended fix: re-run mutmut after Phase B tests land and update the kill-rate claim from projection to measurement. Until then **downgrade the 80% claim to "projected"** in the synthesist and planner files. The bulk of the evidence is solid; the projection is the soft spot.

---

### 6. `tracer-runtime.md` — MIXED — **EVIDENCE GAP**

Trace 1 + Trace 2 are detailed and the file:line citations into the source are accurate. The tracer correctly **refutes** the linguist-v3 silent-fp64-downcast hypothesis with explicit construction-path probes — that's gold-standard. Pytorch#162872 deadlock citation verified live: real issue, "MPS deadlock when calling Event.synchronize()" with labels module: deadlock, module: mps.

But:
- The header says `Probe scripts: /tmp/trace_runtime.py, /tmp/trace_silent_downcast.py`. **Both files do NOT exist on disk.** I checked: `/tmp/trace*.py` is empty. By contrast, the empiricist's `/tmp/mac_bench-*.py` files all exist. So either the probe scripts were never written, were deleted post-hoc, or live elsewhere. **The "/tmp/trace_runtime.py lines 49-110" cross-references are unreproducible.**
- All timing tables (1.44 ms fast-path, 21.5 ms slow-path, 25.4 ms fixture call) are reported as "median of 10–30 iterations" with no raw samples preserved. Reader cannot reproduce or audit variance.
- Conversely: H4 (silent fp64 downcast refutation) is so well-specified ("`torch.tensor(0.5, device='mps', dtype=torch.float64)` raises `TypeError: Cannot convert a MPS Tensor to float64...`") that any reader can copy-paste it on torch 2.11. Asymmetric reproducibility within the same file.

Top 3 findings:
- **Probe scripts missing from `/tmp/` is the load-bearing gap.** Either re-create them as `.claude/teams/audit/v1.1/scripts/trace_runtime.py` or downgrade the timing claims to "approximate, not reproducible."
- The H4 fp64 refutation is fully reproducible — keep as the gold standard for future tracers.
- Hidden-cost #4 (fixture bypasses `MPSBackend.event_timer`) is independently verified by the source: `fixtures/benchmark.py:283-327` does its own MPS path. Real.

Recommended fix: copy the probe scripts into the workspace under `scripts/` and re-run; pin a numbers-table with the new run. Until then, the timing claims should be footnoted "single-run, scripts not preserved." STRONG-PRIMARY on H4 refutation; WEAK on quantitative timings.

---

### 7. `docs-tester-blocks.md` — STRONG-PRIMARY

This is the densest reproducible-evidence file in the corpus. Every triple-backtick block is extracted to a numbered `/tmp/doctest_<id>.py`, executed, exit code reported, error message quoted. The "M-B7 wrong fence language" claim is the single most operator-friendly finding I've seen: the reader can navigate to README L66-76 and see the bug.

Top 3 findings:
- M-B8 (`fuzz_strides` wrong signature) verifies because the file:line citations match the actual source signature `fuzz_strides(shape, dtype, *, n=None, ...)`.
- M-B16 / R-B16 numeric mismatch (README claims `+12.0%` / `d=4.21`; actual `+11.7%` / `d=7.48`) is testable by running the example. Strong.
- The "52 hard-fails on MPS" claim from R-B21 is a measured outcome with exit-code evidence. Strong.

Recommended fix: none — this is the model the rest of the corpus should aim for.

---

### 8. `empiricist-mac-benchmarks.md` — STRONG-PRIMARY (with caveats)

I verified the harness scripts exist:
- `/tmp/mac_bench-mps_kernels.py` — present
- `/tmp/mac_bench-mlx_matmul.py` — present
- `/tmp/mac_bench-mps_4k_sanity.py` — present
- `/tmp/mac_bench-mps_isolate.py`, `mac_bench-mps_diag.py`, `mac_bench-mps_repro.py` — all present

And the canonical artifact at `.claude/teams/audit/v1.1/mac_benchmarks.json` is on disk (~71 KB, 126-row JSON, machine-readable).

The bug-found-mid-run admission ("Original lambda-factory pattern timed only input allocation, not kernel — inflated MPS by ~25×. Caught via 4096³ sanity probe; fixed mid-run; final numbers post-fix") is exactly the candor we want — it strengthens credibility, doesn't weaken it. The 3.54 TFLOPs fp32 / 14.1 TFLOPs fp16/bf16 numbers at 4096³ are within published M5-class envelope.

Top 3 findings:
- 10 measurements per cell is on the **low** end. The author flags conv2d N4_64_128 (CV 80–135%) as a known weakness needing 5+ warmups. Honest. Charter-aligned with charter's adversary question on this point.
- The MLX comparison (12 cells) is the cross-check that matters: the corpus would be compromised if it relied on a single timing harness; with MLX as a sanity oracle, the 4× MPS-fp32 anomaly at 1024³ is hard to fake. Strong.
- The "94 TFLOPs at 4096³" early-error → root-cause → fix story is **the reproducibility check made manifest**. Compare to tracer-runtime which has no preserved scripts.

Recommended fix: re-run with WARMUP=5, N=20 once an audit slot is open (the author already requests this in §"Follow-ups"). For v1.1 acceptance, the existing run is sufficient; mark the conv2d 80-135% CV cell as "advisory, do not use for regression-detection".

---

### 9. `forge-memory-schema.md` — STRONG-PRIMARY

Cites real files: `~/.claude/agent-memory/research-lead/MEMORY.md:89-97`, `~/.claude/agents/engineering/engineering-scribe.md:22-60`, `~/.claude/agent-memory/research-retrospector/MEMORY.md`. The cartographer-memory-map file independently confirms these all exist on disk. The flock+atomic-rename pattern is named explicitly and the file structure is auditable. The author **explicitly disclaims** undue prior art ("I deliberately do not cite LangGraph, AutoGen, or CrewAI memory features without a concrete file pointer; per the hard rule, no invented prior art") — best-practice citation hygiene.

Top 3 findings:
- ACE / Voyager / Anthropic memory tool docs are cited with URLs; all three are verifiable (I'll note ACE was verified separately under historian §1).
- The "validated 10-concurrent at 0.07s" engineering-scribe claim cites the agent file but **does not show the 10-concurrent test trace**. A reader has to take it on the engineering-scribe file's authority. Weak link.
- Section §9's worked-example migration (engineering L2 worktree-pytest) is concrete and reproducible — best schema-level finding.

Recommended fix: forge-lead should cite the specific session/log where 10-concurrent at 0.07s was measured (BENCHMARKS_v0.2.md per architect file?). Otherwise the schema is solid.

---

### 10. `architect-continuous-learning.md` — STRONG-PRIMARY

Ten-section design with file:line citations into existing infrastructure: `~/.claude/hooks/session-capture.sh`, `~/.claude/settings.json`, `~/.claude/agent-memory/research-lead/MEMORY.md` lines 236-241 / 243-248. The architect explicitly flags **its own dependency on forge-lead's schema** (§10 open question 3 lists required fields). Honest about what it doesn't own.

Top 3 findings:
- The "PreToolUse hooks do NOT reliably fire in v2.1.101" claim cites research-lead/MEMORY.md L236-241 — verifiable via cartographer's filesystem inventory.
- The fall-back "synthesis-by-orchestrator" is presented as **the load-bearing path**, not the harness path. That's the right framing.
- Lesson-rot mitigation (§6 FM-2) is the most important specifically-architected guardrail: harmful_count auto-archive once `harmful > helpful AND total ≥ 3`. Concrete trigger.

Recommended fix: none — this is a design doc, not a measurement, and it stays disciplined about its evidence base.

---

### 11. `historian-memory-prior-art.md` — STRONG-PRIMARY

I independently verified:
- **LangGraph BaseStore** at `libs/checkpoint/langgraph/store/base/__init__.py` — file exists, defines `BaseStore`, `Item`, `SearchItem`, `IndexConfig`, `TTLConfig`, `PutOp`, `GetOp`, `SearchOp`. All matches.
- **AutoGen Memory** at `python/packages/autogen-core/src/autogen_core/memory/_base_memory.py` — file exists, defines `Memory` ABC with `update_context`, `query`, `add`, `clear`, `close`. All five methods present as claimed.
- **Letta** at `letta/functions/function_sets/base.py` — file exists, defines `core_memory_append` (lines 354-364), `core_memory_replace` (366-378), `archival_memory_insert` (316-341), `archival_memory_search` (343-379). All four functions present. Note: I learned from the verify that `archival_memory_insert` and `archival_memory_search` are interface stubs (`NotImplementedError` in this base file). The historian's text "Letta v1 architecture (2025) deprecated the old MemGPT-style heartbeat/send_message pattern in favour of native reasoning tokens, but kept the three-tier memory exactly" is consistent with these being protocol stubs that concrete subclasses implement.
- **arXiv 2510.04618 (ACE)** — title "Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models", primary author Qizheng Zhang, ICLR 2026. Matches.
- **arXiv 2502.12110 (A-MEM)** — title "A-MEM: Agentic Memory for LLM Agents", primary author Wujiang Xu, NeurIPS 2025, Zettelkasten claim confirmed. Matches.
- **arXiv 2504.19413 (Mem0)** — title and authors match; benchmark numbers (26% / 91% / 90%) match the historian's report.

Six-for-six. The historian's own caveat — "*adversary: please verify the Mem0 benchmark numbers against an independent re-evaluation; vendor-self-reported*" — is exactly the right epistemic flag, and I record here that the benchmarks are still vendor-claimed. The number itself comes from arXiv 2504.19413 §5 (LOCOMO benchmark). Independent corroboration would be a 3rd-party reproduction I haven't found; the original paper is real, the numbers are as reported, but they remain authors-self-reported.

Top 3 findings:
- 6 spot-check verifications all pass. Strong.
- Cursor "explicitly does NOT ship native memory" is corroborated by the cited URL `https://cursor.com/docs/rules`. Real.
- The single soft point — Mem0 vendor-reported numbers — is **flagged by the historian itself**. Adversary cannot do better than that without an independent benchmark in hand.

Recommended fix: none. Best-cited file in the corpus.

---

### 12. `cartographer-memory-map.md` — STRONG-PRIMARY

Filesystem inventory only. Every claim is `wc -l`, `stat`, or `find` output. I verified counts pass: e.g. `~/.claude/skills/` count of 106 was sampled and matches; the `audit/v1.1/EVIDENCE/` directory count and mtimes match my own `ls` here. Charter explicitly disclaims interpretation ("No interpretation of lesson semantics; only path, size, mtime, schema, provenance"); discipline holds throughout.

Top 3 findings:
- "Forge `PROMOTIONS.md` says NOT yet promoted but `~/.claude/skills/` already contains the 3 drafts" is a **structural contradiction surfaced for human review**. Exactly the right altitude — surface, do not resolve. Solid.
- 8 staging-file disposition table is reproducible (file by file, lines, lessons-count, source-evidence presence). Solid.
- Audit-of-audit: the `audit/v1.1/_write_audit.log` 8 lines / mtime 2026-05-06 entry confirms the in-flight audit is being audit-logged. Self-auditing.

Recommended fix: none.

---

### 13. `planner-v1.1-tasks.md` — STRONG-PRIMARY (with one citation-laundering risk)

27 tasks, each citing a specific upstream summary file (`SUMMARIES/<x>.summary.md`). Acceptance criteria are concrete and machine-checkable (e.g. T-01: `python -c "import gpucheck"` does not import torch by `sys.modules` check). Dependencies are explicit. The planner explicitly **drops** the linguist-v3 catcher per tracer-runtime's refutation — exactly the kind of corpus-gated reasoning we want.

Top 3 findings:
- Citation-laundering risk: the planner cites `SUMMARIES/<x>.summary.md`, which paraphrase the EVIDENCE files. I checked: the summaries are short re-writes of the evidence (cartographer §5 confirms the `SUMMARIES/` dir exists with 14 .summary.md files of 1.3-2.2 KB each). When the planner says "src: api-dx-grade.summary.md fix #2", the chain is `planner → summary → evidence → source`. **I walked one chain end-to-end** (T-22 promote 7 hidden symbols → api-dx-grade Fix 2 → claim about `_LAZY_MAP` missing 7 symbols → verified against `src/gpucheck/__init__.py:75-105`). It bottoms out at primary source. So the laundering risk is **low** but architecturally present: a future planner update could drift if summaries drift from evidence.
- T-09 ("MPS auto-skip gate to gpu_integration") rests on R-B21 + the `22780ae` archaeologist finding — both verified.
- T-26 explicitly excludes the linguist-v3 catcher with a citation to tracer-runtime §4. Cross-audit-corpus consistency held.

Recommended fix: when the implementation phase begins, the executor should read EVIDENCE/<file>.md (not SUMMARIES/<file>.summary.md) for any claim that drives a code change — the summaries are paraphrase, the evidence files are primary.

---

### 14. `synthesist-bugs-inventory.md` — STRONG-PRIMARY

59 issues with severity / impact / ease / file:line / fix sketch / conflicts. Quotes are pulled verbatim from upstream evidence (I cross-checked: ISS-08 quote about `assertions/close.py:13-19` matches detector-files Top-10 #1 word-for-word). 5 contradictions explicitly named (C1–C5) with verdicts. The "Cross-audit contradictions" section is the right place to surface inter-evidence disagreements — and the synthesist did not cherry-pick which to surface.

I asked: did the synthesist suppress a 6th contradiction? Walking the corpus:
- linguist-v3 vs tracer-runtime fp64 → C1 ✓
- CLAUDE.md stale vs reality → C2 ✓ (3-way: archaeologist + detector + api-dx)
- flush_l2 warning vs absent work → C3 ✓
- memory leak warn vs fail → C4 ✓
- README MPS-first vs CUDA-only → C5 ✓
- Mem0 vendor numbers — **flagged by historian, not promoted to a contradiction**, which is reasonable since it isn't a v1.1 implementation decision.
- Mutator 80% projection vs measurement — not a contradiction, just a projected metric. Synthesist did not surface it. Defensible omission.
- The "two `compute_tolerance`" + "two `MemoryReport`" findings are distinct issues (ISS-09, ISS-10), not contradictions per se.

I find no suppressed 6th. The 5 are reasonable.

Top 3 findings:
- The 4-quadrant impact × ease matrix (Quadrant 1 = High × Easy "FIX FIRST") is a useful synthesis layer that does not over-claim. Strong.
- The "Mac/Metal-specific cluster" segmentation is structural, not vibes — driven by where the code path runs, not what it claims to do.
- The 2 fileable upstream candidates (ISS-56 MPS matmul anomaly, ISS-57 PyTorch CPU half-precision GEMM) are both backed by empiricist's measured data and the JSON artifact. Strong.

Recommended fix: minor — note in the rolled-up tally that ISS-25's 80% kill-rate target is **projected** (not measured) until Phase B tests land.

---

## Source-class verdict roll-up

| File | Verdict |
|---|---|
| api-dx-grade.md | STRONG-PRIMARY |
| security-postmerge.md | STRONG-PRIMARY |
| archaeologist-debt.md | STRONG-PRIMARY |
| detector-files.md | STRONG-PRIMARY |
| mutator-survivors.md | MIXED (projection unverified) |
| **tracer-runtime.md** | **MIXED (probe scripts missing)** |
| docs-tester-blocks.md | STRONG-PRIMARY |
| empiricist-mac-benchmarks.md | STRONG-PRIMARY (caveats) |
| forge-memory-schema.md | STRONG-PRIMARY |
| architect-continuous-learning.md | STRONG-PRIMARY |
| historian-memory-prior-art.md | STRONG-PRIMARY |
| cartographer-memory-map.md | STRONG-PRIMARY |
| planner-v1.1-tasks.md | STRONG-PRIMARY |
| synthesist-bugs-inventory.md | STRONG-PRIMARY |

**12/14 STRONG-PRIMARY, 2/14 MIXED, 0 WEAK.**

---

## Top 3 weakest citations across the corpus

1. **tracer-runtime.md `/tmp/trace_runtime.py` and `/tmp/trace_silent_downcast.py` — files do not exist on disk.** Every cited line range ("`/tmp/trace_runtime.py` lines 49-110", "lines 230-283") is unverifiable. This is the single biggest reproducibility gap. Empiricist preserved its scripts; tracer did not. Recommend: re-run probes, write scripts to `.claude/teams/audit/v1.1/scripts/`, re-emit trace numbers as a 2-row table (single-run vs canonical-run).

2. **mutator-survivors.md "80–82% kill-rate" projection.** No verification step is planned within the same evidence file. Until Phase B tests land and mutmut re-runs, this is a forecast, not a measurement. The synthesist (ISS-25) and planner (T-11..T-14) inherit the projection without flagging it as such.

3. **security-postmerge.md PM-4 "torch <2.1 raises RuntimeError on stride-fuzzed `.numpy()`" claim is uncited.** The behavioural claim is plausible (torch's contiguity machinery did change in this band) but no PR / changelog / issue is cited. A reader cannot verify the version boundary.

(Honourable mention — Mem0 vendor-reported benchmarks in historian §1: flagged by the historian itself, no further action available without an independent reproduction.)

---

## Citation-laundering walk

I traced **planner T-22 → SUMMARIES/api-dx-grade.summary.md → EVIDENCE/api-dx-grade.md (Fix 2) → src/gpucheck/__init__.py:75-105**. The chain bottoms out at primary source (the actual `_LAZY_MAP` definition in the codebase). **No SEO-laundering**, but a structural risk: the planner cites *summaries* not *evidence*. If the executor implements from summaries, drift is possible. Recommend: at implementation time, executors verify against EVIDENCE/, not SUMMARIES/.

I traced **synthesist ISS-25 → mutator-survivors.md "Top-3" #1 → ID 270/283/284/290/292-295/310-318/328-337/343 → reporting.py line numbers in actual source**. Chain holds. No laundering.

I traced **synthesist C1 (fp64 refutation) → tracer-runtime H4 → 4 explicit construction probes**. The probes are reproducible from the prose alone (`torch.tensor(0.5, device='mps', dtype=torch.float64)`). Chain holds even though the script itself is missing.

---

## Astroturf / community integrity

No HN, Reddit, X, Stack Overflow, Twitter, Medium, or Substack citations in the corpus. The community-source attack vector is **not present**. The only external community reference is `forum.cursor.com / "Add Persistent Memory in Cursor" (thread 57497)` in historian §"Issue-tracker findings", and it is correctly framed as observational ("long-running community thread tracking the BYO-MCP memory pattern") not load-bearing. Pass.

---

## Staleness

- All cited file:line references in src/gpucheck/ are against `release/v1.0` HEAD `82b853e`. Cartographer confirms the audit branch is current. Pass.
- arXiv papers cited are 2023–2026 — all current; ACE / A-MEM are 2025–2026.
- PyTorch issue 162872 is OPEN as of audit date; deadlock module:mps labels confirmed.
- One staleness gotcha: the docs-tester runs against torch 2.11.0 but the project README claims to support older torch. PM-4's "torch <2.1" path is therefore **not verified by the audit** — the audit ran on a single torch version. This is a real coverage gap, not a citation problem.

---

## Corpus concentration

By author / persona:
- 14 distinct specialist personas, each writing one file. No single persona dominates the corpus. **No concentration risk.**
- The most cross-cited evidence file is `assertions/close.py` (the source file, not an audit) — appears in api-dx-grade, detector-files, mutator-survivors, security-postmerge, tracer-runtime, synthesist. That's appropriate concentration: it's the headline API.
- The `synthesist` file aggregates the other 8 Wave-1 audits but does not cite itself. Pass.

---

## Coverage gaps the plan implicitly assumes

1. **No torch-version matrix.** Every audit ran on torch 2.11. The planner's T-02 ("Add `.contiguous()` for torch <2.1") rests on PM-4's uncited claim. The fix is correct in spirit (force contiguity is harmless on torch 2.11+) but **the bug class is not actually verified to exist on the supported version range**. Recommend: implementer should confirm by running the existing test suite against torch 2.0/2.1 in CI.
2. **No CUDA validation.** Every measurement is on Apple Silicon (MPS). Empiricist explicitly says CPU AMX wins on small fp32 cells; nobody re-ran on a CUDA box for Phase D's CUDA-touching tasks (T-23 fuzz expansion, T-24 tolerance overlay's CUDA path). Acceptance criteria say "CUDA bit-for-bit identical to v1.0" but no audit demonstrates v1.0's CUDA behaviour as a baseline.
3. **No Windows / non-macOS platform check.** All file-system audits and timing measurements are on Darwin 25.4.0. The plan implicitly assumes the same code runs identically on Linux. Cartographer's filesystem inventory cannot speak to this.
4. **Mutator's 80% projection is the only metric the planner inherits without a verification step.** Phase B should include "re-run mutmut after T-11..T-14 land; commit the post-fix kill-rate to PR description" as an acceptance gate.
5. **The tracer's missing /tmp/ probes mean the 1.44 ms / 21.5 ms / 25.4 ms numbers are single-run.** If any v1.1 task is justified by a *latency budget* derived from these (T-19 `_run_mps` unification: "tracer-runtime trace 2 reproduces same hot-step distribution") the executor should re-measure post-refactor on the same hardware.

---

## Verdict

The corpus **substantially supports the plan**. 12 of 14 files are STRONG-PRIMARY; the 2 MIXED files are mixed for distinct reasons (one missing artifacts, one un-verified projection). Source verification spot-checks (6) all passed. No SEO-laundering or astroturf detected. No 6th-contradiction suppression by the synthesist. Citation chains bottom out at primary sources (codebase line numbers, real arXiv papers, real GitHub paths, real PyTorch issues).

**Claims requiring re-sourcing or re-measurement before "high confidence":**

- tracer-runtime quantitative timings (1.44 ms / 21.5 ms / 25.4 ms etc.) — re-run with preserved scripts.
- mutator-survivors 80% projected kill-rate — measure post-Phase-B.
- security-postmerge PM-4 "torch <2.1 raises RuntimeError" — cite a PyTorch commit/changelog or run an old-torch CI job.

**Most likely gap to bite v1.1 implementation:** the missing `/tmp/trace_runtime.py` probes. T-19 (eliminate `_run_mps` duplicate of `MPSBackend.event_timer`) explicitly cites tracer-runtime trace 2 as its regression baseline. If the executor refactors and the new path is 100 µs slower, there's no original artifact to compare against — only a rounded median in the prose. The executor will either (a) re-derive a baseline before the refactor, or (b) ship blind. Both are recoverable; (a) is what the implementer should do.

## Confidence

**Medium-high.** Six independent-source spot-checks all passed; one named gap (tracer probes) and one named projection (mutator kill-rate) are concrete and addressable. The corpus is healthy enough to ship Phase A as v1.0.0rc2; Phase B should be gated on a measurable post-fix mutmut re-run; Phase C/D should add a CUDA-host re-measurement step.
