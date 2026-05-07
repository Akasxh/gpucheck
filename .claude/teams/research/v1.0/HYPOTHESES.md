# Hypotheses — gpucheck v1.0 MPS Backend

## H1 — "Ship it; CUDA-shaped tolerance, MPS-shaped exclusions"
**Prior: 0.40**

MPS in PyTorch 2.5+ is mature enough for gpucheck's surface (matmul, softmax,
norms, attention, conv2d, common losses). Correctness gaps are concentrated in
specific kernels (FP16 matmul drift, BF16 reductions, log_softmax tail behavior)
that can be enumerated and given inflated tolerance multipliers. Adding MPS as
the v1.0 headline expands gpucheck's TAM (every Apple Silicon developer) at low
incremental engineering cost. Same fuzzing playbook (degenerate > non-tile-aligned
> prime > power-of-2 > large > mixed) applies because MPS shares the same matmul
tile sensitivities.

## H2 — "Wait one cycle; MPS is still rough on the kernels we test"
**Prior: 0.30**

The `module: mps` issue tracker carries hundreds of open numerical-drift and
crash bugs, including high-impact ones in the exact kernels gpucheck targets
(SDPA, layer_norm, conv2d backward). Shipping MPS in v1.0 will mean either
(a) blanket-relaxing tolerances to the point that real bugs hide, or
(b) carving out so many xfails that the matrix is mostly skips. v1.0 should
ship CUDA-only with MPS as a v1.1 follow-up after the open-bug count drops
below a threshold (e.g. < 50 numerical-drift bugs).

## H3 — "Ship MPS, but as a *correctness oracle*, not a parity backend"
**Prior: 0.20**

Reframe: MPS is not equivalent to CUDA, but its *differences* from CUDA are
exactly what fuzzing should surface. Use gpucheck-MPS to find Apple's MPS bugs
the same way gpucheck-CUDA found Triton bugs — by running parametric fuzz across
dtypes/shapes and reporting drift > tolerance against a CPU FP64 reference.
Tolerance defaults are not "what MPS-CUDA disagree by" but "what MPS-CPU
disagree by within FP precision". This is a different product framing.

## H4 — "Apple already has the right tooling; gpucheck-MPS is undifferentiated"
**Prior: 0.10**

MLX (Apple's own array framework) ships with parity tests, and llama.cpp's
Metal backend has its own validation harness. Adding MPS to gpucheck duplicates
work that's already done by people closer to the metal. Better v1.0: invest
in stride/contiguity fuzzing, gradient testing, or NCCL multi-GPU — gaps in the
existing CUDA story.

## Decision criteria
Hypothesis wins if:
- H1: ≥3 sub-questions land "MPS is usable with bounded tolerance multipliers"
  AND open-bug audit shows clusters that map to clean xfail rules
- H2: ≥3 sub-questions land "tolerances would have to inflate >10x" or "open
  crashes block the test matrix"
- H3: H1 and H2 evidence are mixed AND there's a clean reframe to "find MPS
  bugs, not parity-check them"
- H4: Independent harnesses cover what gpucheck would add, with no gap

## Note on prior probabilities
These are seeded BEFORE specialists run. Skeptic and moderator will revise.
