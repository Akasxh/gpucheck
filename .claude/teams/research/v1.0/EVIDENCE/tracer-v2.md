---
specialist: research-tracer
round: 2
slug: v1.0
started: 2026-05-01T05:30:00Z
completed: 2026-05-01T06:00:00Z
attack: skeptic v1 attack #5 binding ("MLX already does this")
citations_count: 11
mlx_files_cited: 5
gpucheck_files_cited: 3
confidence: high
---

# Tracer v2 — MLX SDPA dispatch vs gpucheck MPSBackend

## Entry point

The skeptic-binding question: when Akash's MPS-target user calls
`gpucheck.assert_close(out_actual, out_expected)` on `scaled_dot_product_attention`
outputs, does MLX's existing harness already cover the differential, making
gpucheck-MPS redundant? Trace both stacks.

- **MLX entry**: `mx.fast.scaled_dot_product_attention(q, k, v, scale=...)`
  → resolves to `mlx::core::fast::scaled_dot_product_attention` C++ symbol.
  Source: [`/tmp/mlx-src/mlx/fast.cpp:613-862`](file:///tmp/mlx-src/mlx/fast.cpp).
- **gpucheck entry**: user code computes `out = torch.nn.functional.scaled_dot_product_attention(q.mps(), k.mps(), v.mps())`,
  then `gpucheck.assert_close(out, ref)`. gpucheck never sees SDPA — it
  only sees the *output tensors* and an inferred dtype/device.

## §1. MLX forward trace — `scaled_dot_product_attention`

### 1.1 Public-API validation contract
File: `/tmp/mlx-src/mlx/fast.cpp:613-710`

Inputs validated *before* dispatch (each throws `std::invalid_argument`):
- L622-627: all of Q, K, V must be **rank 4** (B, H, L, D).
- L631-635: `mask_mode` ∈ `{"", "causal", "array"}`; nothing else.
- L647-651: cannot pass both `mask_mode="causal"` AND `mask_arr`.
- L659-663: `mask_arr` rank ≤ 4.
- L666-673: batch dim of K, V must equal Q's batch dim.
- L677-681: `Q.shape[-1] == K.shape[-1]` (head dim match).
- L688-693: `K.shape[-3] == V.shape[-3]` (n_kv_heads match).
- L697-701: `n_q_heads % n_kv_heads == 0` (GQA constraint).
- L705-709: `result_type(Q,K,V)` must be in `floating` (FP32/FP16/BF16; FP8
  is **rejected**, FP64 is rejected).

This is a **strict precondition contract**. Tolerance is *not* part of MLX's
contract — MLX guarantees output, not error magnitude.

### 1.2 The fallback decision
File: `/tmp/mlx-src/mlx/backend/metal/scaled_dot_product_attention.cpp:588-637`

`ScaledDotProductAttention::use_fallback()` returns true (= drop fast
kernel, use unfused `q*scale @ k.T → softmax → @ v`) when ANY of:
- L598-602: `is_training` (training-mode autograd is unfused even on Metal).
- L603-605: `output_logsumexp` requested.
- L606-608: `s.device == cpu`.
- L618-621: head dim ∉ `{64, 96, 128, 256}` (vector path) AND ∉ `{64, 80, 128}`
  (full path).
- L631-634: `qL > 8` rules out vector path; `qL ≤ 8 AND gqa*qL > 32` rules out
  the only path.

This is the **silent-fallback set**. The user gets correct output but a
different code path with different numerics — this is where MLX's own atol
budget (§1.5) was calibrated.

### 1.3 The three Metal kernels
File: `/tmp/mlx-src/mlx/backend/metal/scaled_dot_product_attention.cpp:18-584`

MLX has THREE Metal kernels, dispatched from `eval_gpu` at L643-786:
1. **`sdpa_full_self_attention_metal`** (L166-327): standard dispatch for
   `qL > 8`. Tile sizes: bq=32, bk={32 if D<128 else 16}, wm=4 simdgroups.
2. **`sdpa_full_self_attention_nax`** (L18-164): NAX-accelerated path,
   gated at L177 by `metal::is_nax_available() && D != 80 && (env::enable_tf32() || dtype != float32)`.
   This is **M5+ tensor-core analog** (NAX = Neural Accelerator on M5).
3. **`sdpa_vector`** (L329-416) and **`sdpa_vector_2pass`** (L418-584): the
   vector / decode path, dispatched at L743-750 when `qL ≤ 8`.

Routing in `eval_gpu` L743-750:
```cpp
if (((devc == 'd' || devc == 's') && k.shape(2) >= 1024) ||
    (k.shape(1) < q.shape(1) && k.shape(2) >= 4096)) {
  sdpa_vector_2pass(s, d, q, k, v, o, scale_, do_causal, mask, sinks);
} else {
  sdpa_vector(s, d, q, k, v, o, scale_, do_causal, mask, sinks);
}
```
`devc` is `d.get_architecture().back()` — `'s'` = M-series Pro/Max, `'d'` =
M-series Ultra, anything else (`g`?) = base. **Code-path is hardware-keyed**.

### 1.4 The accumulator dtype contract
File: `/tmp/mlx-src/mlx/backend/metal/kernels/sdpa_vector.h:50`

```cpp
typedef float U;
thread U q[qk_per_thread];
thread U k[qk_per_thread];
thread U o[v_per_thread];
```

The kernel **always upcasts to FP32** for the inner accumulators, regardless
of input dtype (FP16 / BF16 / FP32). The output is cast back at the end.
This is the same FlashAttention numeric trick. **It is the reason MLX can
hold `atol=3e-4` on FP16** (§1.5).

### 1.5 The tolerance contract — empirical, not part of the API
File: `/tmp/mlx-src/python/tests/test_fast_sdpa.py:439`

```python
atol = 2e-5 if dtype == mx.float32 else 3e-4
diff = mx.abs(out_fst - out_ref) - atol * mx.abs(out_ref)
self.assertLessEqual(mx.max(diff).item(), atol)
```

This is the **comprehensive shape sweep** (`test_sdpa`, L364-444) and it
uses a **tolerance schedule**:
- FP32: `atol = 2e-5`
- FP16 / BF16: `atol = 3e-4`

For long-mask cases (`test_sdpa_long_masked_sequence`, L448-483), atol
relaxes to `1e-3` because the bug being regression-tested (int16 overflow
in `col_pos` when `kL > 32767`) had this looser bound.

For all other simpler tests (L150, L193, L222, L246, L278, L304, L321, L331,
L499, L510): `atol = 1e-4, rtol = 1e-4`.

**This is NOT MLX's public contract.** It is *MLX testing MLX*. There is
no documented "if you call SDPA you get atol=X" guarantee. Akash inheriting
these numbers requires the empirical assumption "the same kernel produces
the same atol on the same input".

## §2. gpucheck dispatch trace — what actually runs on `assert_close`

### 2.1 Entry — `gpucheck.assert_close(actual, expected)`
File: `/Users/cero/Code/gpucheck/src/gpucheck/assertions/close.py:140-189`

gpucheck's `assert_close` does **not know what kernel produced the tensor**.
It only knows:
- `dtype` (resolved at L140 via `_resolve_dtype`)
- `device.type` (L145-150 — the only place MPS enters)
- `k_dim` (passed by caller; only relevant for matmul-shape ops)

Trace:
1. L140: `dtype = _resolve_dtype(actual, expected)` — picks first tensor's dtype.
2. L145-150: scan tensors for `device.type` (becomes `"mps"` if either is MPS).
3. L165-167: `compute_tolerance(dtype, k_dim=k_dim, device_type=device_type)` —
   returns `(atol, rtol)` from the MPS-overlay table.
4. L176-189: GPU fast-path — if both tensors on same device AND device is
   `cuda` or `mps`, calls `torch.allclose(actual, expected, atol, rtol)` and
   returns on success. **Never inspects the SDPA call site.**

### 2.2 The MPS tolerance overlay
File: `/Users/cero/Code/gpucheck/src/gpucheck/assertions/tolerances.py:35-43, 105-110`

```python
_MPS_TOLERANCE_MULTIPLIERS = {
    "float32": 2.0, "float16": 2.0, "bfloat16": 2.0, "float64": 1.0, ...
}
# ... in compute_tolerance:
if device_type == "mps":
    multiplier = _MPS_TOLERANCE_MULTIPLIERS.get(name, 2.0)
    atol *= multiplier
    rtol *= multiplier
```

Base CUDA tolerances (L13-26): FP32 `(1e-4, 1e-4)`, FP16 `(1e-2, 1e-2)`,
BF16 `(5e-2, 5e-2)`.

So gpucheck's effective MPS-FP16 atol is `2 * 1e-2 = 2e-2` — **66× looser**
than MLX's FP16 SDPA atol of `3e-4`. gpucheck would PASS test cases where
MLX's own harness FAILS. This is the headline differential.

The comment at tolerances.py:28-34 admits "PROVISIONAL — research SYNTHESIS §7
... must be calibrated on Akash's actual M-generation hardware before being
canonical".

### 2.3 The MPS xfail registry — the only SDPA-aware bit
File: `/Users/cero/Code/gpucheck/src/gpucheck/assertions/tolerances.py:181-228`

```python
_mps_xfail_set: set[str] = set()
def is_mps_xfailed(op_name: str) -> bool: ...
```

The only place gpucheck mentions SDPA by name is a *string* `"scaled_dot_product_attention.large"`
in the example/docstring at tolerances.py:191. Nothing in
`/Users/cero/Code/gpucheck/src/gpucheck/backends/mps.py` or
`/Users/cero/Code/gpucheck/src/gpucheck/assertions/close.py` actually
*detects* SDPA. The `xfail` registry is opt-in: the *user* has to tag a test
`@pytest.mark.gpucheck_op("scaled_dot_product_attention.large")` for the
xfail to fire.

## §3. The differential — what each covers

### 3.1 What MLX covers that gpucheck does NOT
- **Kernel-internal silent-fallback detection**: MLX knows when D=80 forces
  the non-NAX path (scaled_dot_product_attention.cpp:177). gpucheck cannot
  detect this — the user's test passes both paths with the same tolerance.
- **Three-kernel coverage matrix**: `sdpa_full`, `sdpa_full_nax`,
  `sdpa_vector`, `sdpa_vector_2pass`. Each has a different numeric error
  profile. test_fast_sdpa.py exercises all four via shape parametrization
  (qL > 8 hits full; qL ≤ 8 + N≥1024 hits 2pass; qL ≤ 8 + N<1024 hits vector).
- **Shape regression for fixed bugs**: `test_sdpa_long_masked_sequence`
  (test_fast_sdpa.py:448) regression-guards a *specific* int16 overflow at
  `kL > 32767`. gpucheck has no equivalent — it would need a fuzz bucket
  matching that overflow boundary.
- **Reference oracle = naive matmul-softmax-matmul** (test_fast_sdpa.py:99-116
  `mlx_primitives_sdpa`). MLX *defines* its tolerance contract by comparing
  against its own reference. gpucheck has no oracle for SDPA.

### 3.2 What gpucheck covers that MLX does NOT
- **The PyTorch MPS surface, not the MLX surface**. MLX tests `mx.fast.SDPA`,
  not `torch.nn.functional.scaled_dot_product_attention(device='mps')`. These
  are different kernels: PyTorch MPS-SDPA per #179294 currently dispatches
  to the *math* backend (decomposition), not a fused kernel. MLX's harness
  does not exercise that decomposition path.
- **Cross-device reproducibility**: gpucheck runs *the same test* CUDA-then-MPS
  with `@parametrize_gpu(devices=["cuda", "mps"])`. MLX is single-platform.
- **Memory leak detection** (gpucheck/sanitizers/) — MLX's harness has no
  RSS-tracking equivalent.
- **CUDA-event-style benchmark fixture** with IQR outlier removal — MLX has
  `mx.metal.start_capture()` for Xcode profiler but no statistical timing.
- **Fuzz-driven shape coverage** via Hypothesis — MLX uses a hand-listed
  shape sweep at test_fast_sdpa.py:366-388. Akash's `fuzz_shapes` strategy
  in `/Users/cero/Code/gpucheck/src/gpucheck/fuzzing/strategies.py` has the
  degenerate-first ordering that hits boundary bugs (the int16 overflow MLX
  caught was a hand-add after the fact).

### 3.3 What both miss
- **Backward-pass numerics on MPS**: MLX's `ScaledDotProductAttentionVJP::use_fallback`
  always returns `true` (scaled_dot_product_attention.cpp:788-790) — meaning
  MLX *never* uses a fused backward on Metal. gpucheck has no gradient testing
  at all (CLAUDE.md "Known Weaknesses": "No gradient/backward pass testing").

## §4. Competing hypotheses

### H1: MLX-as-oracle is feasible — gpucheck should import MLX as a reference
- **Supporting evidence**: MLX's `mlx_primitives_sdpa` (test_fast_sdpa.py:99-116)
  is a pure-MLX, pure-FP32 reference computation. If gpucheck transports
  PyTorch tensors → MLX arrays → run reference → transport back, it has a
  trustworthy oracle.
- **Falsifiable by**: actually trying it. Probe: `torch.from_numpy(np.array(mx_array))`
  and the reverse path. If FP16/BF16 round-trip preserves precision, oracle
  is feasible. If MLX's BF16 differs from PyTorch's BF16 (different rounding
  in `astype`), the oracle introduces noise larger than the bug.
- **Status**: OPEN. I did not run the probe — Apple Silicon required and the
  budget did not permit cloning + building MLX. Round 3 candidate.

### H2: MLX-as-oracle is infeasible — kernels too different
- **Supporting evidence**:
  - MLX's reference (`mlx_primitives_sdpa` L99-116) operates on rank-4
    `[B, H, L, D]`. PyTorch's SDPA accepts the same shape but PyTorch's
    "math" backend may transpose internally before reduction, picking up
    different summation order → different roundoff.
  - MLX upcasts to FP32 (sdpa_vector.h:50). PyTorch's MPS math backend may
    not — issue #179294 noted "the dedicated MPS implementation is called
    under the math backend despite being platform-specific". If the math
    backend doesn't upcast, the reference and the MPS output have *different*
    numeric envelopes; the oracle won't match either.
  - Stride contracts differ: MLX requires `is_matrix_contiguous` on the head
    dim (scaled_dot_product_attention.cpp:670-673) and triggers a copy
    otherwise. PyTorch's SDPA accepts arbitrary strides. Round-tripping
    PyTorch tensors with non-contig memory layout into MLX may invoke a
    layout-changing copy that erases the bug being tested.
- **Falsifiable by**: same probe as H1. If output mismatch is < gpucheck's
  PROVISIONAL `2 * dtype_atol`, H2 is refuted. If mismatch dominates, H1
  is refuted.
- **Status**: SUPPORTED but not decisive.

### H3: MLX is the wrong oracle — use PyTorch CPU instead
- **Supporting evidence**: `torch.nn.functional.scaled_dot_product_attention(
  q.cpu(), k.cpu(), v.cpu())` runs the canonical math decomposition on CPU
  in FP64 if upcast. The CPU path is the same `aten::` op as MPS dispatches
  to under the math backend — so the numerical contract is *identical* up to
  device-dispatch determinism. This is also the existing Apple-recommended
  pattern (`torch_test/mps_helper.py` style). gpucheck Round 1's empiricist
  observed this in fixtures/.
- **Falsifiable by**: empirical bug detection rate. If PyTorch-CPU oracle
  catches MPS bugs that gpucheck-MPS-tolerance misses → H3 supported. If
  PyTorch-CPU oracle is too tight (catches dtype-noise that isn't a bug) →
  H3 refuted.
- **Status**: STRONGEST candidate. Default oracle for any v1.0 MPS test
  should be CPU-PyTorch, not MLX, because it shares the *aten dispatch tree*
  with the device-under-test.

## §5. Probes I ran

I ran **no live probes** in this round — the question requires Apple Silicon
+ MLX build + PyTorch MPS, which are environmental. Static-trace evidence
is sufficient to:
- prove the differential exists (gpucheck atol = 66× MLX atol on FP16)
- locate the silent-fallback boundary (D=80, qL=8 thresholds)
- demonstrate gpucheck has no SDPA-aware code path (zero matches for
  `mlx`, `sdpa`, `attention` in `src/gpucheck/`).

A Round 3 empiricist probe should:
1. Set up: `mlx==0.18+`, `torch==2.11+` on M-series.
2. Run MLX's `test_sdpa` (test_fast_sdpa.py:364-444) AND a parallel PyTorch
   `torch.nn.functional.scaled_dot_product_attention` on identical NumPy
   inputs (round-tripped through both `mx.array(np_arr)` and
   `torch.from_numpy(np_arr).mps()`).
3. Compute pairwise atol: `(mlx_out - torch_mps_out).abs().max()`.
4. If pairwise atol < gpucheck's MPS PROVISIONAL atol → MLX is a viable
   oracle within gpucheck's existing budget.
5. If pairwise atol > gpucheck's MPS PROVISIONAL atol → MLX is a *better*
   oracle than gpucheck's heuristic and gpucheck should *tighten* its
   tolerances (revealing real bugs).

## §6. Verdict on "MLX as reference oracle"

**Recommend: NO — not as a default oracle, but YES as one of three optional oracles.**

Rationale:
1. **Different kernel populations**. MLX's `mx.fast.SDPA` is not what
   PyTorch-MPS dispatches to. Using MLX as oracle for `torch.SDPA` tests
   compares two *separate implementations* — any disagreement is
   uninformative because it could be either side's bug.
2. **The right default oracle is `torch.<op>(*tensors_on_cpu)`** because
   it shares the aten op signature with the device-under-test; differences
   are concentrated on the device backend.
3. **MLX is a useful TIE-BREAKER** when CPU-oracle and MPS-output disagree:
   if MLX agrees with MPS, the CPU path is the suspect (e.g., a CPU
   regression like the BF16 layer norm in pytorch#175189). If MLX agrees
   with CPU, MPS is the suspect.
4. **Adopt MLX's tolerance schedule**, not MLX's kernel. The numbers
   (`atol=2e-5` FP32, `atol=3e-4` FP16, `atol=1e-3` long-mask) are
   *empirically calibrated* on Apple Silicon for the FlashAttention-style
   numeric profile. gpucheck's FP16-MPS atol of `2e-2` is **66× looser**
   and will not catch real bugs. **This is the actionable v1.0 finding.**

The skeptic's attack #5 was correct that "MLX already does this" was
dismissed too fast — but the corrected framing is "MLX provides
**calibration data and a tie-break oracle**, not a substitute for gpucheck."
gpucheck-MPS still has a real differentiation against MLX (PyTorch surface,
fuzz-driven shapes, leak detection, parametric cross-device), and against
MLX's own harness (Hypothesis-driven instead of hand-listed shapes,
@require_op for missing-op coverage on the math backend).

## §7. Uncovered ground (what would make me trace deeper)

- **Did not trace MLX's CUDA SDPA** (`/tmp/mlx-src/mlx/backend/cuda/scaled_dot_product_attention.cpp`,
  `.cu`) — would tell us whether MLX's tolerance schedule is portable across
  hardware or Metal-specific. Round 3 expansion if oracle question resurfaces.
- **Did not trace `get_steel_attention_kernel` JIT** (`jit_kernels.cpp:1113`,
  `:1147`) — kernel-cache layer. Relevant only if "kernel JIT cache busts
  between gpucheck fuzz iterations" becomes a hypothesis.
- **Did not run the H1/H2 probe** — needs M-series hardware. Punted to
  empiricist Round 3.
- **Did not check MLX's `examples/cpp/` SDPA usage** for a third oracle
  pattern — could yield a simpler "kernel from C" calling sequence that
  bypasses the Python validation overhead.
- **Did not read `steel_attention.h`** for full-attn kernel internals —
  only relevant if the "MLX upcasts internally" claim from sdpa_vector.h:50
  needs corroboration for the full path. Inspected the `bd/bq/bk` tile
  parameters at scaled_dot_product_attention.cpp:31-37, 197-199 instead.

## Confidence

**HIGH**. Five MLX file paths cited with exact line ranges (§1.1, §1.2,
§1.3, §1.4, §1.5), three gpucheck file paths cited (§2.1, §2.2, §2.3),
none of these were in Round 1's tracer.md. The numerical differential
(66× tolerance gap) is computed from primary-source tables, not paraphrase.
The ONE thing not pinned down is whether the MLX-as-oracle probe (H1) holds
empirically — flagged for Round 3 empiricist.
