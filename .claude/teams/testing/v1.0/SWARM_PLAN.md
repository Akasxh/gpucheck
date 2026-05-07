# SWARM_PLAN — gpucheck v1.0 26-Kernel Fuzzer Swarm

**Slug**: testing/v1.0
**Owner**: testing-lead
**Date**: 2026-05-01
**Source**: `swarm/launch-swarm.sh` (orchestrator pre-staged), SYNTHESIS.md xfail
list, FINDINGS.md security gates
**Phase**: pre-flight only — **NO LAUNCH IN THIS DISPATCH**

The 26-process kernel-fuzzer swarm runs at the Phase 2 → Phase 3
transition, **after** engineering-lead merges Track A (MPS backend) to
`release/v1.0`. This file is the pre-flight checklist + divergence
classifier + confidence bar.

---

## Pre-flight verdict

**READY** — but **BLOCKED on Track A merge**. The launcher exists and is
correct; the only missing precondition is a buildable MPS backend in
each worktree. Specifically:

| Check | Status | Evidence |
|---|---|---|
| Launcher script present | PASS | `swarm/launch-swarm.sh` exists, 122 lines, executable bit OK |
| Kernel list complete | PASS | 26 kernels enumerated (matmul × 3, attention, FA-v1/v2, scatter/gather, etc.) |
| Wave size sane | PASS | `WAVE_SIZE=6` (resp. macOS process limits + Anthropic rate limit) |
| Wave delay sane | PASS | `WAVE_DELAY_S=15` |
| Per-kernel runtime budget | PASS | hard 8 minutes per kernel |
| Output dir exists | PASS | `~/Code/gpucheck/.claude/teams/testing/v1.0/swarm/` exists, `logs/` will be created |
| `claude` CLI in PATH | NEEDS CHECK | run `which claude` before launch |
| 26 worktrees provisioned | NEEDS CHECK | `~/Code/gpucheck-worktrees/fuzz-<kernel>/` for each kernel |
| Each worktree has gpucheck installed | NEEDS CHECK | `pip install -e ".[dev,mps]"` per worktree |
| `torch.mps.is_available()` returns True | NEEDS RUNTIME CHECK | swarm prompt enforces SKIPPED + exit 0 if False |
| Track A backend importable in each worktree | **BLOCKED** | engineering-lead must merge first |
| pyproject.toml `[tool.gpucheck.mps.xfail]` populated | **BLOCKED** | depends on Track A |

Phase 2 → 3 transition gate: when engineering-lead's Track A lands on
`release/v1.0`, run the pre-flight checks listed below, then
launch.

---

## Launch command (post-merge, do NOT run now)

```bash
# Pre-flight (run from gpucheck repo root)
bash /Users/cero/Code/gpucheck/.claude/teams/testing/v1.0/swarm/launch-swarm.sh
```

Optional subset / dry-run:
```bash
# Subset
bash /Users/cero/Code/gpucheck/.claude/teams/testing/v1.0/swarm/launch-swarm.sh matmul-fp16 softmax layernorm

# Dry run (prints prompts, doesn't dispatch)
DRYRUN=1 bash /Users/cero/Code/gpucheck/.claude/teams/testing/v1.0/swarm/launch-swarm.sh
```

---

## Expected output artifacts

After the swarm completes (~30 min wall-clock, 5 waves of ~6 min each):

```
.claude/teams/testing/v1.0/swarm/
├── RESULTS_relu.md
├── RESULTS_softmax.md
├── RESULTS_layernorm.md
├── RESULTS_matmul-fp32.md
├── RESULTS_matmul-fp16.md
├── RESULTS_matmul-bf16.md
├── RESULTS_attention.md
├── RESULTS_cross_entropy.md
├── RESULTS_gelu.md
├── RESULTS_silu.md
├── RESULTS_rmsnorm.md
├── RESULTS_rope.md
├── RESULTS_conv2d.md
├── RESULTS_batchnorm.md
├── RESULTS_groupnorm.md
├── RESULTS_gemm-3d.md
├── RESULTS_flash-attn-v1.md
├── RESULTS_flash-attn-v2.md
├── RESULTS_scatter.md
├── RESULTS_gather.md
├── RESULTS_index_select.md
├── RESULTS_topk.md
├── RESULTS_argmax.md
├── RESULTS_nll_loss.md
├── RESULTS_kl_div.md
├── RESULTS_cosine_sim.md
├── swarm.jsonl                  # one JSON line per kernel
└── logs/
    ├── relu.log
    ├── relu.pid
    ├── ... (one .log + .pid per kernel)
```

### `RESULTS_<kernel>.md` schema

Each `RESULTS_<kernel>.md` MUST contain (per the launcher prompt):

```markdown
# <kernel> swarm results

- Iterations attempted: <N>
- Iterations completed: <M>
- Divergences found: <D>
- Top 3 minimal repros:
  1. shape=<...> dtype=<...> stride=<...> max_rel_err=<...>
  2. ...
  3. ...
- MPS-vs-CPU max relative error: <...>
- MPS-vs-CUDA-mock max relative error: <...> | N/A (mocked)
- Recommended upstream filing target: pytorch/pytorch | triton-lang/triton | none
- Determinism mode: <eager|graph|deterministic algorithms enabled>
- Iteration seed range: [seed_start, seed_end]
- Notes: <freeform>
```

### `swarm.jsonl` schema

```json
{
  "kernel": "matmul-fp16",
  "iterations_attempted": 250,
  "iterations_completed": 248,
  "divergences": 4,
  "top_repros": [
    {"shape": [128, 8192, 128], "dtype": "float16", "stride_class": "contiguous",
     "max_rel_err": 0.125, "seed": 42}
  ],
  "max_rel_err_vs_cpu": 0.125,
  "max_rel_err_vs_cuda_mock": null,
  "filing_target": "triton-lang/triton",
  "deterministic_mode": true,
  "wall_seconds": 412,
  "exit_status": "ok"
}
```

---

## Divergence classifier

Triages every divergence reported in `swarm.jsonl` into one of three
buckets. Engineering-lead consumes this classification to decide whether
to file upstream, recalibrate `[tool.gpucheck.mps.tolerances]`, or
suppress.

### Bucket 1: FILABLE-UPSTREAM (highest confidence)

A divergence belongs here iff **all** of:

1. **≥3 reproductions** across distinct seeds — same shape, same dtype, same stride class, max_rel_err within 10% of each other.
2. **Deterministic seed**: re-running with the captured seed produces the SAME max_rel_err to within 5% (catches non-determinism flavors A and B from SYNTHESIS §4).
3. **max_rel_err > 10× the dtype tolerance**, where the dtype tolerance is `compute_tolerance(dtype, device="mps")[1]` (rtol). 10× is the SYNTHESIS H1+H3 threshold for "implementation bug" vs "precision floor".
4. **Kernel deterministic mode honored where possible**:
   - PyTorch: `torch.use_deterministic_algorithms(True)` set at process start
   - Seeds set: `torch.manual_seed`, `torch.mps.manual_seed`, `random.seed`, `np.random.seed`
   - If kernel doesn't support deterministic mode (e.g., scatter on MPS), this requirement is waived AND classification confidence drops to MEDIUM.
5. **Not on the SYNTHESIS xfail list** (those are already filed). Specifically check against `SYNTHESIS_TOP12` from `tests/test_xfail_registry.py`.
6. **CPU reference baseline matches CUDA-known-good**: when `swarm.jsonl[i].max_rel_err_vs_cuda_mock` is non-null, the divergence is bounded relative to a known-good CUDA result, not just CPU.

If bucket 1: produce a draft upstream issue with the
RESULTS_<kernel>.md content + minimal repro Python snippet.

### Bucket 2: TOLERANCE-RECALIBRATION

A divergence belongs here iff:

1. **≥10 reproductions across diverse shapes** (not just one outlier).
2. **max_rel_err within 1×–10× of dtype tolerance** (precision-floor band per SYNTHESIS §7).
3. **Reproduces deterministically** (rules out flavor A — non-determinism).
4. **Distribution is consistent**: P99 of |MPS-CPU| across the 10+ shapes is ≤2× the median. Wide P99/median spread suggests bucket 1, not bucket 2.

If bucket 2: produce a recalibration patch for
`pyproject.toml [tool.gpucheck.mps.tolerances]`. Specifically: set new
overlay = `max(2× CUDA, P99 measured)` per SYNTHESIS Sub-Q 7.

### Bucket 3: FALSE POSITIVE

A divergence belongs here iff **any** of:

1. <3 reproductions and re-run with same seed produces different max_rel_err — flavor A non-determinism, expected per SYNTHESIS §4.
2. The shape is on the SYNTHESIS xfail list — already covered.
3. The error magnitude is bounded by `compute_tolerance(dtype, device="mps", k_dim=K)` correctly applied — gpucheck didn't apply the k_dim scaling in the first place.
4. The CPU reference itself is non-deterministic across runs (rare; check first by running CPU reference 3×).

If bucket 3: log to `OPEN_QUESTIONS.md` for retrospection but no action.

---

## Confidence bar table (machine-readable)

| Confidence | Reproductions | Deterministic | max_rel_err vs tol | xfail-listed | Action |
|---|---|---|---|---|---|
| HIGH (file upstream) | ≥3 | YES | >10× | NO | Draft GitHub issue |
| MEDIUM (file upstream, mark "investigation needed") | ≥3 | NO (kernel can't deterministic) | >10× | NO | Draft issue, flag |
| MEDIUM (recalibrate) | ≥10 | YES | 1×–10× | NO | Update tolerance overlay |
| LOW (open question) | <3 | UNKNOWN | any | NO | OPEN_QUESTIONS.md |
| ALREADY-KNOWN | any | any | any | YES | xfail registry already covers |
| FALSE POSITIVE | <3 + non-determ | NO | any | NO | drop |

---

## Aggregator script (post-swarm)

After the swarm completes, run a classifier to triage `swarm.jsonl`:

```bash
# To be implemented as `tools/swarm_triage.py` in the same dispatch as Phase 3 close.
python tools/swarm_triage.py \
  --input .claude/teams/testing/v1.0/swarm/swarm.jsonl \
  --xfail-toml pyproject.toml \
  --out .claude/teams/testing/v1.0/swarm/TRIAGE.md
```

`TRIAGE.md` schema:

```markdown
# Swarm triage — v1.0

## FILABLE_UPSTREAM (N=<n>)
| Kernel | Shape | Dtype | Stride | max_rel_err | Repros | Issue draft |
|---|---|---|---|---|---|---|
| matmul-fp16 | (128, 8192, 128) | float16 | contiguous | 0.125 | 5 | drafts/matmul-fp16-K8192.md |

## TOLERANCE_RECALIBRATION (N=<n>)
| Kernel | Dtype | Old atol | Old rtol | New atol | New rtol | Source |
|---|---|---|---|---|---|---|
| layer_norm | float32 | 2e-4 | 2e-4 | 4e-4 | 4e-4 | swarm/RESULTS_layernorm.md P99 |

## FALSE_POSITIVE (N=<n>)
| Kernel | Reason |
|---|---|

## ALREADY_KNOWN (N=<n>) — xfail registry covers
| Kernel | xfail entry | Issue |
|---|---|---|
| softmax | softmax.large_attention | pytorch#96602 |
```

---

## Per-kernel test contract (the swarm prompt enforces)

Each kernel-fuzzer process:
- Confirms `torch.mps.is_available()` is True; if not, marks SKIPPED + exit 0.
- Runs 250 iterations (per launcher prompt).
- For each iteration: samples shape × dtype × stride class.
- Compares MPS vs CPU max_rel_err.
- Captures minimal repro for divergences.
- Writes RESULTS_<kernel>.md + appends `swarm.jsonl` line.
- Halts at 8 min budget, writes partial results, exits 0.

---

## Hard rules during swarm execution

1. Per-kernel runtime budget: **8 minutes hard**. Process self-terminates.
2. **Determinism**: every iteration sets seed via `gpucheck.fixtures.seeded_rng` or equivalent. Seeds recorded in RESULTS file.
3. **No invented divergences**. If `torch.mps` raises NotImplementedError, log UNSUPPORTED and continue.
4. **Process error → halt**: any subprocess error is a failure to investigate, not silently skipped.
5. **Output discipline**: every kernel produces RESULTS file + jsonl line, even if 0 divergences (proves the kernel ran).
6. **No source-tree mutation**: swarm processes only write to `swarm/` subdirectory.

---

## Risk register (swarm-specific)

| Risk | Likelihood | Mitigation |
|---|---|---|
| Anthropic rate-limit on 26 parallel claude calls | MEDIUM | Wave-based dispatch (6 at a time, 15s spacing) |
| M-Mac thermal throttling skews timings | MEDIUM | Use median-of-N for elapsed_ms; record temps if `powermetrics` available |
| MPS deadlock from pytorch#162872 | LOW (Track A fixes via `torch.mps.synchronize()`) | If reproduces, this IS a bucket 1 finding (Track A regression!) |
| Worktree state drift between launches | LOW | Each worktree pinned to `release/v1.0` HEAD; `git status --porcelain` checked pre-launch |
| Some kernels not on MPS (CTC loss per pytorch#160828) | HIGH | Per swarm prompt: "If `torch.mps` cannot run the op, mark UNSUPPORTED" |
| Kernel-fuzzer process leaks GPU memory | MEDIUM | `torch.mps.empty_cache()` between iterations per SYNTHESIS §1 #177116 mitigation |
| Two kernel processes contend for MPS | LOW (waves of 6, MPS arbitrates serially) | If contention surfaces, drop WAVE_SIZE to 3 |
| `claude -p` truncates output | LOW | RESULTS file written to disk before LLM finishes; verify file exists post-run |

---

## Pre-flight checklist (run BEFORE invoking the launcher)

```bash
# 1. claude CLI is in PATH
which claude

# 2. all 26 worktrees exist
for k in relu softmax layernorm matmul-fp32 matmul-fp16 matmul-bf16 attention \
         cross_entropy gelu silu rmsnorm rope conv2d batchnorm groupnorm gemm-3d \
         flash-attn-v1 flash-attn-v2 scatter gather index_select topk argmax \
         nll_loss kl_div cosine_sim; do
  test -d "$HOME/Code/gpucheck-worktrees/fuzz-$k/" || echo "MISSING: $k"
done

# 3. each worktree has gpucheck installed (sanity-check first 3)
for k in relu matmul-fp16 attention; do
  (cd "$HOME/Code/gpucheck-worktrees/fuzz-$k/" && python -c "from gpucheck.arch.backend import detect_backend; print(detect_backend().name)")
done

# 4. MPS available
python -c "import torch; assert torch.backends.mps.is_available(), 'MPS not available'"

# 5. xfail registry populated
python -c "
import tomllib
cfg = tomllib.loads(open('pyproject.toml').read())
ops = cfg.get('tool',{}).get('gpucheck',{}).get('mps',{}).get('xfail',{}).get('ops', [])
assert len(ops) >= 12, f'xfail registry has only {len(ops)} entries; SYNTHESIS requires 12'
print(f'xfail registry: {len(ops)} entries OK')
"

# 6. swarm dir is writable
test -w "$HOME/Code/gpucheck/.claude/teams/testing/v1.0/swarm/" || echo "swarm dir not writable"

# 7. dry-run smoke
DRYRUN=1 bash $HOME/Code/gpucheck/.claude/teams/testing/v1.0/swarm/launch-swarm.sh relu | head -10
```

If all pass, launch. If any fails, fix BEFORE launching — partial swarm
runs poison the data.

---

## Post-swarm gate

After the swarm completes:

1. Verify all 26 RESULTS_<kernel>.md exist + `swarm.jsonl` has 26 lines.
2. Run `tools/swarm_triage.py` (to-be-implemented).
3. Triage with the bucket classifier (above).
4. For Bucket 1: open issues against `pytorch/pytorch` or `triton-lang/triton`. Track in `.claude/teams/testing/v1.0/UPSTREAM_ISSUES.md`.
5. For Bucket 2: PR against `pyproject.toml [tool.gpucheck.mps.tolerances]` with new overlay; re-run swarm subset (5 affected kernels) to confirm no new divergences.
6. For Bucket 3 / xfail-already-known: log only; no action.
7. Lead writes `EVIDENCE/swarm.md` summarising counts and verdict.
8. Lead updates this file's "Pre-flight verdict" to POST-SWARM-COMPLETE.

---

## File pointers

- `.claude/teams/testing/v1.0/swarm/launch-swarm.sh` — orchestrator-staged launcher
- `EVIDENCE/testing-planner.md` §A8 — xfail registry
- `EVIDENCE/testing-property.md` §A6/A8 — properties that the swarm output cross-checks
- `.claude/teams/research/v1.0/SYNTHESIS.md` Sub-Q 1, 2, 7 — xfail list + tolerance bar
- `.claude/teams/security/v1.0/FINDINGS.md` — N1 metal subprocess threat (does not apply to swarm processes; they don't compile shaders)
