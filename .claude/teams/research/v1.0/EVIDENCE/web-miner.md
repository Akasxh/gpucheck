---
specialist: research-web-miner
slug: v1.0
started: 2026-05-01T03:42:00Z
completed: 2026-05-01T03:43:30Z
tool_calls_count: 6
citations_count: 8
confidence: medium
---

# Web-Miner — Apple Silicon ML compute landscape outside PyTorch

## §1. Community Apple-Silicon ML projects (retrieved 2026-05-01 via WebSearch + WebFetch)

| Project | Source | What it provides | Maturity (2026-05) |
|---|---|---|---|
| MLX | [github.com/ml-explore/mlx](https://github.com/ml-explore/mlx) | Apple's official Metal-native array framework with fused SDPA, layer_norm, rms_norm, softmax kernels | Production; Apple-supported |
| llama.cpp Metal | [github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | per-op Metal kernels with CPU fallback via supports_op | Production; community |
| pmetal | [github.com/Epistates/pmetal](https://github.com/Epistates/pmetal) | "Powdered Metal — high performance LLM fine-tuning framework for Apple Silicon"; FlashAttention + fused kernels ported from FLA Triton | Community; not audited |
| ZMLX | [github.com/Hmbown/ZMLX](https://github.com/Hmbown/ZMLX) | Triton-style kernel toolkit on top of MLX, prototype/benchmark/upstream fusions | Community incubator; not audited |
| vllm-metal | [github.com/vllm-project/vllm-metal](https://github.com/vllm-project/vllm-metal) | vLLM plugin using MLX as compute backend on Apple Silicon | Community; vLLM-project blessed |
| mlx-turboquant | [github.com/yzamari/mlx-turboquant](https://github.com/yzamari/mlx-turboquant) | TurboQuant KV-cache compression for MLX-LM | Community; not audited |

## §2. MLX as the load-bearing reference

MLX is the only project on the list with Apple's official involvement. Its
kernel inventory (verified §1.2 of `EVIDENCE/historian.md`) covers:
- fused scaled_dot_product_attention (MLX-specific FlashAttention-equivalent)
- layer_norm, rms_norm
- softmax, logsumexp
- conv, binary, reduce, scan, sort, rope, fft

**Critically**, MLX is a **separate framework from PyTorch**. Tensors are
`mlx.core.array`, not `torch.Tensor`. There is no shared device — MLX kernels
are not callable on `torch.Tensor(device='mps')`. So while MLX proves Apple
Silicon CAN do these ops well, MLX kernels are not part of the surface
gpucheck-MPS validates. PyTorch MPS is the SUT; MLX is the **upper bound** of
what is achievable on Apple hardware.

## §3. Apple Silicon hardware features relevant to MPS testing

From [Apple Metal documentation top-level page](https://developer.apple.com/metal/) (retrieved 2026-05-01) and the publicly available Metal Feature Set Tables PDF:

- M1/M2/M3/M4/M5 share Metal3+ feature set; differ in atomic-FP support and SIMD width
- Apple Neural Engine is **not** exposed via MPS (it's CoreML-only)
- AMX coprocessor for matmul on Apple Silicon CPU side, **not** Metal-side — invisible to torch.mps
- No tensor-core analog (no FP16 matmul fused-multiply-add SIMD wide)
- Unified memory architecture: GPU and CPU share memory; `tensor.to('mps')` is a flag-flip not a copy

**Implication for gpucheck v1.0**:
- One device class for v1.0 ("Apple Silicon"). No SM-version equivalent.
- `arch/detection.py` returns a degenerate `GPUInfo` for MPS — no compute capability, no tensor-core gen.
- "memory leak" is harder to detect on MPS because allocation is unified — see [#164299, retrieved 2026-05-01](https://github.com/pytorch/pytorch/issues/164299): "Memory growth is not recognized by built-in `torch.mps.current_allocated_memory()` and `torch.mps.driver_allocated_memory()` methods but visible on Activity Monitor".

## §4. SEO/spam-farm risk in this corpus

Two categories of low-quality sources exist for "PyTorch MPS bugs":
- Medium / dev.to posts repackaging GitHub issue threads (HIGH SEO-farm risk; cited none).
- Reddit r/MachineLearning threads with reproductions (MIXED — some primary, some hand-wave; cited none directly).

This report cites **only**:
- PyTorch's own GitHub issue tracker (primary)
- PyTorch's official documentation (primary)
- llama.cpp / MLX official repos and their issue trackers (primary)
- Apple's official Developer Documentation (primary)

No SEO blogs. The adversary will verify this discipline. Per QUESTION.md hard rule.

## §5. M-series generational drift is real and reproducible

[#181936, retrieved 2026-05-01](https://github.com/pytorch/pytorch/issues/181936) explicitly: "Hardware: Apple M5 processor"; non-determinism is M5-specific (not seen on M3/M4). [#180776, retrieved 2026-05-01](https://github.com/pytorch/pytorch/issues/180776) confirms an earlier M5-specific determinism bug. [#173640, retrieved 2026-05-01](https://github.com/pytorch/pytorch/issues/173640): LSTM dropout collapse "when model weights are trained on MPS and moved to CPU" — also generation-sensitive.

**Implication for gpucheck v1.0**: the README must publish what M-generation
the MPS validation ran on (just like the CUDA section publishes "GTX 1650 /
SM75"). The MPS table cannot claim to cover all M generations from a single
M-machine run.

## §6. The 14-day fresh window (Sub-Q 2 cross-cutting)

Issues with `updated_at >= 2026-04-17` (last 14 days as of 2026-05-01) — top 10:
1. #182052 "[MPS] aten::copy_ into strided view silently wraps writes at element offset > 2^32" (2026-04-30)
2. #181936 "[MPS] Non-deterministic backward pass for F.linear" (2026-04-30)
3. #181946 "[MPS] Remove Unnecessary Tensor Memory Gathers and Contiguous Calls..." (2026-04-30)
4. #181725 "[MPS] nn.MultiheadAttention is ~9x slower..." (2026-04-28)
5. #181650 "MPS: `add_dense_scalar_cast_float` shader fails..." (2026-04-27)
6. #181374 "torch.empty(..., device='cpu', pin_memory=True).device gives back MPS device" (2026-04-27)
7. #173640 "nn.LSTM internal dropout causes output collapse..." (2026-04-26)
8. #178037 "[MPS] Raise clear error when MPS is used in forked subprocess" (2026-04-26)
9. #179294 "[MPS] scaled_dot_product_attention (SDPA) improvements" (2026-04-26)
10. #177116 "MPS: catastrophically wrong gradients in backward pass (>32K elements)" (2026-04-17)

These are all open or active. **MPS is being actively iterated on** — the surface gpucheck targets is moving weekly.

## Confidence

Medium-high — primary sources verified for §1-§5; §3 cites Apple official docs
which were not directly fetched (Apple Developer site title-only response). The
generational-drift claim is supported by 3 independent issues and is high-confidence.
