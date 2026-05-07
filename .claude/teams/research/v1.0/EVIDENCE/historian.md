---
specialist: research-historian
slug: v1.0
started: 2026-05-01T03:42:00Z
completed: 2026-05-01T03:43:30Z
tool_calls_count: 5
citations_count: 9
confidence: medium
---

# Historian — prior art on Metal correctness validation and FlashAttention/Triton-on-Metal

## §1. llama.cpp Metal precedent (load-bearing for Sub-Q 5)

llama.cpp's `ggml-metal` backend is the most-cited reference architecture for
"a CPU/Metal dispatcher that ships in production." Pattern observed across
multiple primary sources:

### 1.1 supports_op() returns false → automatic CPU fallback

llama.cpp implements per-op support checks. When the Metal backend's
`supports_op()` returns false for a given `GGML_OP`, the runtime transparently
falls back to CPU. Confirmed via:
- [llama.cpp issue #10845, "Eval bug: ggml_metal_encode_node: error: unsupported op 'IM2COL'", retrieved 2026-05-01](https://github.com/ggml-org/llama.cpp/issues/10845)
- [llama.cpp commit 62bfef5, "metal : disable FA kernel for HS=256 (#7556)", retrieved 2026-05-01](https://github.com/ggml-org/llama.cpp/commit/62bfef5194d5582486d62da3db59bf44981b7912) — explicit example of disabling a Metal kernel for a specific head-size after correctness regressions
- [DeepWiki "Metal Backend (Apple) | ggml-org/llama.cpp", retrieved 2026-05-01](https://deepwiki.com/ggml-org/llama.cpp/) — documents the supports_op pattern
- [stable-diffusion.cpp issue #1040, "Metal backend fails with GGML_OP_DIAG_MASK_INF", retrieved 2026-05-01](https://github.com/leejet/stable-diffusion.cpp/issues/1040) — third-party project hitting same fallback

### 1.2 Per-op Metal kernels via PSO pipelines

`ggml-metal-device.cpp` exposes `ggml_metal_library_get_pipeline_*()` factories.
[WebFetch retrieved 2026-05-01] confirms families:
- `..._unary()` — handles SCALE, FILL, CLAMP, SQR, SQRT, SIN, COS, LOG, LEAKY_RELU, TANH, RELU, SIGMOID, GELU variants, SILU, ELU, NEG, ABS, SGN, STEP, HARDSWISH, HARDSIGMOID, EXP, SOFTPLUS, EXPM1, FLOOR, CEIL, ROUND, TRUNC, XIELU.
- `..._mul_mv()`, `..._mul_mm()` — matrix multiplications.
- Pool, softmax, sum, cumsum, RWKV, SSM specialized kernels.

The pattern that translates to gpucheck: **enumerate a Metal-supported op set,
gate tests per-op, fall back gracefully where MPS doesn't have the op**.
This is exactly what gpucheck's `@require_arch` does for CUDA architectures.

### 1.3 Validation: tests/test-backend-ops.cpp

llama.cpp's primary correctness harness compares Metal output to CPU output via
NMSE (Normalized Mean Squared Error) and per-op tolerances. The tolerance values
themselves were not extractable in this session (file is 9525 lines and was past
the WebFetch summarization threshold), but the search corpus and DeepWiki page
confirm: **per-op + per-dtype tolerance constants, threshold-checked NMSE**.
Reference: [HF mirror of test-backend-ops.cpp, retrieved 2026-05-01](https://huggingface.co/spaces/Steven10429/apply_lora_and_quantize/blob/main/llama.cpp/tests/test-backend-ops.cpp).

This is the same pattern gpucheck's `assert_close` already uses (per-dtype atol/rtol
overlay), with one twist: **NMSE is normalized by reference magnitude**, which
handles the matmul-error-scales-with-K case more gracefully than ATOL+RTOL alone.
For gpucheck v1.0 MPS we keep ATOL+RTOL+`k_dim` scaling but recognize NMSE as a
v1.1 enhancement.

## §2. MLX precedent (load-bearing for Sub-Q 6)

[MLX repo `mlx/backend/metal/scaled_dot_product_attention.cpp`, retrieved 2026-05-01](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/scaled_dot_product_attention.cpp) confirms MLX ships **fused Metal SDPA**:

- `sdpa_vector` for short sequences (≤8 tokens)
- `sdpa_full_self_attention_metal` for longer sequences
- Steel attention kernels for head_dims 64/80/96/128/256
- Multi-dtype support (kernel name selected via `get_type_string(q.dtype())`)

[MLX `mlx/backend/metal/kernels/`, retrieved 2026-05-01](https://github.com/ml-explore/mlx/tree/main/mlx/backend/metal/kernels) confirms additional Metal kernels:
- `scaled_dot_product_attention.metal` (fused FA-style)
- `layer_norm.metal`, `rms_norm.metal`, `softmax.metal`, `logsumexp.metal`
- `conv.metal`, `binary.metal`, `reduce.metal`, `scan.metal`, `sort.metal`
- `rope.metal`, `arange.metal`, `random.metal`, `fft.metal`

**Implication for gpucheck**: MLX has hand-written Metal kernels for the same op
families gpucheck targets. PyTorch's MPS backend, by contrast, lags MLX —
PyTorch issue [#179294, "[MPS] scaled_dot_product_attention (SDPA) improvements", retrieved 2026-05-01](https://github.com/pytorch/pytorch/issues/179294) confirms PyTorch's MPS SDPA has **no native backward** (uses the device-agnostic "math" decomposition), and that "a Metal kernel implementation exists but is never invoked in the current codebase". MLX is ahead.

## §3. FlashAttention on Metal

There is no upstream FlashAttention port to Metal. The community projects most
relevant:
- **MLX's fused SDPA** (above) — closest to FlashAttention semantics, in-house.
- **pmetal** ([github.com/Epistates/pmetal](https://github.com/Epistates/pmetal)) — community Metal shaders incl. FlashAttention ports from FLA Triton, retrieved 2026-05-01 via WebSearch.
- **ZMLX** ([github.com/Hmbown/ZMLX](https://github.com/Hmbown/ZMLX)) — Triton-style kernel toolkit on top of MLX with fused MoE decode, retrieved 2026-05-01 via WebSearch.

These confirm: Apple-Silicon FlashAttention exists, but only via MLX or via
community projects that wrap MLX/raw Metal. **Not via PyTorch's MPS path**.

## §4. Triton on Metal

No native Triton-Metal backend yet. Confirmed:
- [Triton issue #4824, "Adding Metal Backend to Triton", retrieved 2026-05-01](https://github.com/triton-lang/triton/issues/4824) — open RFC, no merged backend.
- [Triton issue #3443, "Build Triton on MacOS with Apple silicon", retrieved 2026-05-01](https://github.com/triton-lang/triton/issues/3443) — open, build broken on macOS.
- [Triton discussion #1796, "MPS backend support", retrieved 2026-05-01](https://github.com/triton-lang/triton/discussions/1796) — community inquiry, no roadmap.
- [vllm-metal](https://github.com/vllm-project/vllm-metal) — community vLLM plugin uses MLX as the compute backend explicitly because Triton-Metal is unavailable, retrieved 2026-05-01.

**Net**: as of 2026-05-01, there is no Triton-Metal backend. Triton kernels do
not run on MPS. Anyone wanting Triton-equivalent fused ops on Apple Silicon goes
through MLX or hand-rolled Metal.

## §5. What this means for gpucheck v1.0

1. The "find Triton bugs" muscle gpucheck has does **not** apply on MPS — no Triton.
2. The "compare a kernel against a reference" muscle DOES apply — it's the same
   pattern llama.cpp uses, the same MLX uses internally, and what gpucheck already
   does for CUDA.
3. PyTorch MPS lags MLX in fused-kernel coverage for SDPA, so gpucheck-MPS will
   surface PyTorch issues, not Apple ones — a useful position.
4. `torch.compile` on MPS is "early prototype" per [PyTorch issue #150121, retrieved 2026-05-01](https://github.com/pytorch/pytorch/issues/150121): "attempt to use it to accelerate end-to-end network is likely to fail". gpucheck-MPS in v1.0 should test eager-mode only.

## Confidence

Medium-high. llama.cpp pattern verified through multiple primary sources. MLX
kernel inventory verified by file listing. FlashAttention/Triton-Metal claims
based on issue trackers and search summaries — the community projects (pmetal,
ZMLX) were not deeply audited and are flagged as MIXED, not load-bearing for
the v1.0 decision.
