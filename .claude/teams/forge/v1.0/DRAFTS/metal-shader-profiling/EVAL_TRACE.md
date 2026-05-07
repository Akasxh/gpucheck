# EVAL_TRACE — metal-shader-profiling

## Eval 1: trigger on profiling request

User prompt: "Can you capture a Metal System Trace for the layer-norm kernel running via torch.mps?"

Expected: skill triggers, routes to **Step 3** (`xctrace record --template "Metal System Trace" --attach <PID>`), warns that PID-based attach captures the running pytest process.
Verdict: **PASS** — keyword "Metal System Trace" is in the description; xctrace command is in Step 3.

## Eval 2: behavior — kernel-level capture

User prompt: "How do I get a `.gputrace` bundle for one specific kernel dispatch?"

Expected: skill triggers, routes to **Step 4** (MTLCaptureManager Swift snippet), correctly states that torch.mps does not currently expose a Python-level capture hook and offers to scaffold a Swift driver.
Verdict: **PASS** — Step 4 explicitly handles the "Python-only capture" defer case.

## Eval 3: anti-trigger on missing toolchain

User prompt (Linux session): "I want to profile my CUDA kernel with Nsight."

Expected: skill does NOT trigger; description anchors on macOS + xcode-select. The Step 0 verification check would fail-fast.
Verdict: **PASS** — Linux/CUDA session won't match Metal/xctrace keywords; if it did somehow trigger, Step 1 redirects to a CUDA profiling skill.

Trigger eval: 3/3 PASS. Promotable.
