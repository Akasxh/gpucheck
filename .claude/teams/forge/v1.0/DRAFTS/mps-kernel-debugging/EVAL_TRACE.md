# EVAL_TRACE — mps-kernel-debugging

Lightweight trigger + behavior eval per `/forge:test` (skill-creator non-interactive limitation noted in MEMORY.md — manual validation in lieu of automated harness).

## Eval 1: trigger on user phrasing

User prompt: "I added `devices=['cuda', 'mps']` to my parametric test and the MPS row throws NotImplementedError on `aten::scaled_dot_product_attention`. Why?"

Expected: skill triggers (description matches "MPS dispatch", "this works on CUDA but fails on Mac").
Expected first-action: route to **Step 2** (op-coverage check), explain `PYTORCH_ENABLE_MPS_FALLBACK=0`, suggest restructuring to a supported op or skipping with `@pytest.mark.skipif`.
Verdict: **PASS** — description keywords + procedure step match cleanly.

## Eval 2: behavior — tolerance shift question

User prompt: "My fp16 matmul passes on CUDA with rtol=1e-3 but fails on MPS with `mismatched=412/8192`."

Expected: skill triggers, routes to **Step 3** (tolerance multiplier table), prescribes `device_tol_multiplier={"mps": 2.0}` for fp16.
Verdict: **PASS** — exact mapping of dtype + multiplier table to the question.

## Eval 3: anti-trigger on Linux session

User prompt (in a Linux + CUDA session): "Why is my Triton kernel slow on A100?"

Expected: skill does NOT trigger (Apple-Silicon-only constraint in description).
Verdict: **PASS** — `when-to-use` field anchors on macOS Apple Silicon, would not surface for an A100 question.

Trigger eval: 3/3 PASS. Skill is promotable per charter.
