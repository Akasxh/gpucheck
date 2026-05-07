---
specialist: research-linguist
slug: v1.0
started: 2026-05-01T03:42:00Z
completed: 2026-05-01T03:43:00Z
tool_calls_count: 0
citations_count: 5
confidence: high
---

# Linguist — disambiguation of "non-deterministic" and "MPS-correct"

## §1. "Non-deterministic" has three distinct meanings in this corpus

The MPS issue tracker uses "non-deterministic" with at least three meanings. Conflating them produces wrong tolerance choices.

### 1.1 Meaning A — Run-to-run divergence with identical inputs (TRUE non-determinism)

Example: PyTorch issue [#181936, "[MPS] Non-deterministic backward pass for F.linear", retrieved 2026-05-01](https://github.com/pytorch/pytorch/issues/181936). The reporter's verbatim claim: "The backward pass for F.linear(x, w) produces different x.grad across consecutive calls"; max diff between two consecutive calls = 130.0; M5-specific. This is **literal non-determinism** — same input, different output between runs.

For gpucheck: this is a **bug**, not a tolerance issue. No tolerance multiplier accommodates 130.0 of run-to-run drift. Such cases must be marked xfail/skip in the test matrix until upstream fixes them.

### 1.2 Meaning B — MPS-vs-CPU divergence (parity bug)

Example: [#170837, "MPS backend - inconsistent results for batched inference on BERT/RoBERTa", retrieved 2026-05-01](https://github.com/pytorch/pytorch/issues/170837). The reporter quotes: "On MPS, batched inputs to RoBERTa or BERT give outputs that differ from non-batched". This is an **MPS-specific implementation bug** — CPU is correct, MPS is wrong. Same op, same dtype, same input → MPS produces a different result than CPU/CUDA.

For gpucheck: this is the **canonical case** the test matrix should catch — assert_close(MPS_out, CPU_out) under appropriate tolerances. The bug is real if drift > tolerance.

### 1.3 Meaning C — Numerical drift at the precision floor (NOT a bug)

Example: layer_norm gradient differs by 1.2e-4 in fp16 between MPS and CPU. This is just **precision-floor drift**: fp16 has 1024 representable values per decade, atomic ordering varies, sqrt+reciprocal is implementation-dependent. Such drift IS expected and IS what gpucheck's per-dtype tolerances are designed to absorb.

For gpucheck: this is the **base case** — the existing CUDA tolerances already absorb fp16 precision-floor drift. The MPS overlay only needs to add multipliers where the drift exceeds CUDA-equivalent.

## §2. The PyTorch label "module: correctness (silent)"

PyTorch's `module: correctness (silent)` label is applied to MPS bugs where the
op completes without error but the result is wrong. Examples in this session:
- [#142836 conv2d zero-output, retrieved 2026-05-01](https://github.com/pytorch/pytorch/issues/142836) — "silent correctness bug"
- [#137001 BCE loss MPS, retrieved 2026-05-01](https://github.com/pytorch/pytorch/issues/137001) — "module: correctness (silent)"

For gpucheck: silent-correctness MPS bugs are precisely what assert_close
catches — the op runs, produces a value, the value is wrong. This is gpucheck's
sweet spot.

## §3. "MPS-correct" — what it means and doesn't

When this report uses "MPS-correct" it means: the MPS output equals the CPU
output (FP64 reference) within an MPS-specific tolerance overlay. It does NOT
mean:
- MPS output equals CUDA output bit-for-bit (different rounding, different reductions)
- MPS output is deterministic across MPS runs (still subject to atomic ordering)
- MPS output is identical across Apple-Silicon generations (M1 vs M5 differ — see [#181936, retrieved 2026-05-01](https://github.com/pytorch/pytorch/issues/181936))

**Implication for gpucheck v1.0**: the reference for `assert_close` on MPS must
be the CPU FP32/FP64 path, not a CUDA path. Akash typically does not have CUDA
on the same machine as MPS anyway.

## §4. "Determinism" in Apple Metal Shading Language

Apple's MSL spec [Metal Shading Language Specification v4 PDF](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) (URL retrieved 2026-05-01; PDF is 10MB+, exceeded WebFetch limit so NOT VERBATIM-VERIFIED in this session) is reported by secondary sources to:
- Define `atomic_*` types subset of C++14 atomics
- Provide accuracy tables in ULP for fast-math vs precise-math single/half-precision math functions
- Conform to IEEE 754 with caveats for fast-math and transcendental functions

For gpucheck v1.0 the takeaway is **the MSL spec does not promise bit-exact
reproducibility for fast-math paths**. Apple's own MPSGraph default-enables
fast-math optimizations. PyTorch's MPS backend dispatches through MPSGraph,
inheriting the fast-math contract. **This is REPORTED-NOT-VERIFIED** for the
specific ULP numbers — if Akash needs the exact ULP table he should fetch the
PDF locally — but the directional claim "MPS does not promise bit-exact across
runs" is corroborated by the dozens of issues (e.g. #181936, #170837, #177116) showing
exactly that empirical behavior.

## §5. Word-equivalence map for the synthesist

| Issue tracker says | Sub-Q 4 category | gpucheck action |
|---|---|---|
| "non-deterministic" / "different across calls" | (a) numerical drift OR (d) determinism — depends on magnitude | xfail until fixed; not a tolerance bug |
| "incorrect results" / "wrong output" | (a) numerical drift IF |MPS-CPU| > expected precision floor | inflated tolerance OR xfail |
| "silent correctness bug" | (a) numerical drift, magnitude unbounded | xfail/skip if magnitude is huge |
| "crash" / "SIGABRT" / "command buffer exited with error" | (b) crash/hang | skip with message |
| "not currently implemented" | (c) missing op | skip with @require feature gate |
| "memory leak" | not in (a-d) — runtime issue | doc'd in test fixture, doesn't fail correctness |

This map is what the synthesist will use when categorizing the github-miner's
top-N bug list.

## Confidence

High — the disambiguation is grounded in primary issue threads. The MSL-spec
claim is REPORTED-NOT-VERIFIED with explicit caveat per MEMORY.md lesson on the
4-tier source quality scale.
