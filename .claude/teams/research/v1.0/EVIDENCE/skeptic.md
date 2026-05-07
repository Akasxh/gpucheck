---
specialist: research-skeptic
slug: v1.0
started: 2026-05-01T03:44:30Z
completed: 2026-05-01T03:45:00Z
tool_calls_count: 0
citations_count: 8
confidence: high
---

# Skeptic — adversarial review of the synthesis reasoning

I read all Round 1 evidence files plus `synthesist.md`. I am attacking the
reasoning chains, not the corpus (that's the adversary's job).

## Attack 1 — "The fuzzing playbook transfers verbatim to MPS" is overconfident

The synthesis (and cartographer §3, empiricist §1) claims gpucheck's CUDA
fuzzing strategy applies to MPS unchanged because shape adversariality is
device-independent. **Counter**: the priority order
`degenerate > non-tile-aligned > prime > power-of-2 > large > mixed`
was calibrated against NVIDIA tile sizes (32/64/128) used by cuBLAS/cuDNN/Triton.
Apple's MPSGraph uses **different tile sizes** internally (we don't know them —
Apple doesn't publish, and the Metal kernels in MLX use different SIMD
groupings). "Non-tile-aligned" against NVIDIA tiles may not be non-tile-aligned
against Apple's tiles. The fuzzer might miss MPS bugs by hitting Apple's
tile-aligned shapes.

**Resolution**: this is a real concern but does NOT block v1.0. Mitigation:
ship the fuzzer with the existing tile set AND add an MPS-specific tile
hypothesis (e.g. add 16 and 256 to the tile set for the MPS path) AND treat
the v1.0 fuzz run on M-silicon as a calibration that will surface any
MPS-specific tile constants. Confidence on "fuzzing transfers" downgrades from
High to **MEDIUM-HIGH**, with a note in SYNTHESIS that MPS tile sizes may
emerge from real test runs.

## Attack 2 — "2× CUDA tolerance is the right starting multiplier" is hypothesis dressed as recommendation

Empiricist §2.3 admits the 2× factor is borrowed from FlashAttention's
methodology, not measured on M-series. The synthesis adopts it as a
recommendation. **Counter**: 2× could be too tight (under-detect drift,
spurious failures) or too loose (under-detect bugs the test should catch).
Empiricist §3 acknowledges this risk; SYNTHESIS must not present 2× as
calibrated truth.

**Resolution**: SYNTHESIS labels the 2× starting point as PROVISIONAL until
M-machine calibration. The v1.0 release plan must include a calibration step
(run the fuzz suite, observe drift histograms per dtype, fit an empirical
quantile, publish the resulting overlay). Until then, confidence on the
specific 2× value is **MEDIUM**, not high.

## Attack 3 — "PyTorch MPS docs silent on determinism" is a negative result, treat with care

Librarian §5 notes the silence is "load-bearing" — that it tells us the docs do
not promise determinism. **Counter**: silence can also mean "the contract is
inherited from CUDA equivalents" or "the docs page didn't get written for MPS
yet". Negative space is not a contract.

**Resolution**: SYNTHESIS phrases this carefully — "PyTorch's MPS docs do not
publish a determinism contract for MPS as of 2.11; gpucheck must therefore not
assume one. This may change in 2.12+; revisit." Confidence on the
practical conclusion (gpucheck must not assume MPS determinism) is **HIGH** because empirical evidence (issues #181936 BERT/RoBERTa #170837, #177116 corruption) corroborates from the other direction.

## Attack 4 — The 12-bug xfail list might be incomplete

The synthesis selects 12 issues from the top of the github-miner list. **Counter**:
the corpus has 255 open MPS issues. Cherry-picking the top 12 by reactions or
recency is a heuristic, not a guarantee that it's the right 12. Bugs that
matter for gpucheck might be in the long tail.

**Resolution**: the 12 are chosen for impact = (load-bearing kernel) ×
(silent-correctness or crash) × (recently active or high-reaction). This is a
defensible heuristic and matches Anthropic's published "scaling rule" advice on
not over-dispatching. Mitigation: SYNTHESIS reframes the 12 as "**starting
xfail set**" not "complete xfail set", and the gpucheck-MPS release plan
includes a periodic re-mining of the issue tracker (quarterly cadence
suggested). Confidence on 12-as-starting-set is **HIGH**; confidence on
12-as-complete-set is **LOW** (and we don't claim the latter).

## Attack 5 — H4 ("MLX already does this") was dismissed too fast

The synthesist drops H4 to 0.05. **Counter**: H4 wasn't asking "does MLX cover
PyTorch MPS"; it was asking "is gpucheck-MPS undifferentiated given existing
Apple Silicon tooling". MLX has its own validation harness; llama.cpp has
test-backend-ops. If Akash's user is doing all their work in MLX, gpucheck
adds nothing.

**Resolution**: gpucheck's user is the **PyTorch user** working on Apple
Silicon. They are using `torch.Tensor(device='mps')`, not `mlx.core.array`.
MLX's harness does not test PyTorch MPS. So the differentiation is real for
the actual user base. H4 stays low. **NO change to synthesis**, but I'm
adding this rationale here in case the reframed argument is helpful later.

## Attack 6 — "511 test configs / 8 bugs" record was rebased to "2 verified externally"

Archaeologist §2 quietly downgrades the README claim. The synthesis carries
both numbers. **Counter**: this is a documentation-precision risk — the README
says 8, the rigorous count is 2. If the engineering team uses "8" in v1.0 marketing
without sharpening the language, they inherit a small credibility risk.

**Resolution**: SYNTHESIS recommends the docs team (downstream) sharpen the
language — "8 bugs found, 2 filed and verified upstream". This is a docs
concern, not a research conclusion. NO change to v1.0 ship/no-ship.

## §7. Unstated assumptions

A1. We assume Akash has access to an Apple Silicon machine to calibrate. No
session evidence verified this; if Akash is on Linux/CUDA only, gpucheck-MPS
ships as untested code. **Action**: SYNTHESIS adds an explicit "calibration
machine availability" prerequisite.

A2. We assume gpucheck v1.0 already has a "shipping process" — release notes,
PyPI publish, docs update. If not, MPS adds new release-process work the
research did not scope. **Action**: out-of-scope; flag for engineering-lead.

A3. We assume xfail/skip is a clean mechanism in pytest. It is. No issue.

## §8. Verdict

The synthesis stands. **5 of 6 attacks are absorbed by careful labeling
(PROVISIONAL, MEDIUM confidence on calibration, "starting set" rather than
"complete set")**. Attack 1 (fuzzing transfer) requires a small footnote in
SYNTHESIS; Attack 2 requires the 2× to be labeled PROVISIONAL; the rest are
already implicit in the synthesist's open-questions §4.

I find no reason to recommend a moderator debate (no load-bearing
contradiction). I find no reason to delay v1.0 (the open-bug count is
manageable as xfails, and the API surface is stable).

**Status: PASS for "high confidence with explicit medium-confidence caveats on
the tolerance multipliers and the M-generation calibration prerequisite".**

## Confidence

High on this verdict — every attack has a primary citation or a defensible
mitigation; none of them flip the v1.0 decision.
