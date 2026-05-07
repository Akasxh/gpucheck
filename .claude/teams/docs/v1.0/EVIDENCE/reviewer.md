# EVIDENCE — docs-reviewer

**Persona:** docs-reviewer (Phase B inner / FM-2.3, FM-3.3 — spec
compliance, accuracy vs source).
**Method:** Walk every numbered claim in `README.md` and `CLAUDE.md`,
cross-check against `EVIDENCE/reader.md` ground truth, flag
contradictions.

## README.md — claim-by-claim audit

### Hero paragraph and code sample
- `README.md:3` — "**pytest for GPU kernels.**" Generic; OK.
- `README.md:10` — "You write a CUDA kernel, eyeball `torch.allclose`...".
  **STALE (P1):** the framing is CUDA-only. Phase 3 must add "or a
  Metal kernel" once MPS lands.
- `README.md:12` — "tested gpucheck against Triton tutorials and PyTorch
  CUDA ops with 511 test configurations and found 8 real bugs". Numbers
  are reproduced verbatim in CLAUDE.md `Strengths` (CLAUDE.md says "8
  real bugs in Triton/PyTorch with 511 test configs"). **Internally
  consistent.** OK to keep.
- `README.md:19-29` — Code sample uses `@devices("cuda:0")`. **STALE
  (P0):** v1.0 will offer `@devices("mps")` or `@devices()` auto-detect
  including MPS. Sample should also show MPS to legitimize the new
  backend.

### §"Installation" (`README.md:31-49`)
- `README.md:37-43` — Optional extras (`torch`, `hypothesis`, `all`).
  Match `pyproject.toml:36-47`. **OK.**
- No mention of `pip install gpucheck[mps]` or `[apple]`. **GAP for
  v1.0** — engineering may or may not introduce a new extra; flag for
  Phase 3.

### §"Step by step usage guide" (`README.md:51-298`)
- `README.md:53` — "Every code block here has been tested on a real
  NVIDIA GeForce GTX 1650 (Turing, SM75, 4GB VRAM) running PyTorch
  2.11.0 with CUDA 13.0". **STALE (P0):** v1.0 host is Apple Silicon
  per `SESSION_PRECHECK.md:6`. Either:
  - dual-validate every snippet on Apple M-series + GTX 1650 in Phase 3,
  - or split into "validated on CUDA / GTX 1650" + "validated on MPS /
    Apple Silicon" subsections.
- `README.md:55-82` — `detect_gpu()` example shows GTX 1650 output
  literally. After v1.0 ships MPS detection, this output will look
  different on a Mac. Add a parallel MPS sample.
- `README.md:97` — `device="cuda"`. Should be `device="cuda"` OR
  `device="mps"`; documenting both.
- `README.md:122` — `@devices("cuda:0")` — same MPS gap.
- `README.md:140-148` — `parametrize_gpu(devices=("cuda:0",))` — same.
- `README.md:150` — "Predefined groups are available: ... `EDGE_SHAPES`".
  Cross-checked against `decorators/dtypes.py:99-101` and
  `decorators/shapes.py:18-47`: groups exist. But these constants have
  **no docstrings** (`reader.md` MISSING list). Fine for README to
  enumerate them; source-side gap is the issue.
- `README.md:159-167` — `k_dim` math claim:
  > "the tolerance scales by `sqrt(4096/128) = 5.66x`, which matches
  > the CUTLASS error accumulation model"
  Cross-check `assertions/tolerances.py:67`:
  ```
  atol = atol * math.sqrt(max(k_dim, 1) / 128.0)
  ```
  And `arch/tensor_cores.py:127`:
  ```
  k_scale = math.sqrt(max(k_dim, 1) / 128.0)
  ```
  **Math matches.** OK.
- `README.md:177` — "CUDA events, which are more accurate than
  wall-clock timing". v1.0 must qualify: "On CUDA we use CUDA events;
  on MPS we use `torch.mps.Event` (PyTorch ≥ 2.x); on CPU we use
  `time.perf_counter()`."
- `README.md:188-190` — "On our GTX 1650, a 256x256 float32 matmul
  takes about 0.055ms median with 0.005ms standard deviation" — this is
  a hardware-specific claim. **Keep as illustrative**, but call it
  illustrative; don't read as universal.
- `README.md:251` — "uses `torch.cuda.memory_stats()` when available
  and falls back to `pynvml`". Verified against `sanitizers/memory.py:
  36-66`. **OK** today, **STALE for v1.0** — engineering Track A will
  presumably add `torch.mps.current_allocated_memory()` per research
  QUESTION sub-question 3.
- `README.md:263-273` — `require_arch`/`require_capability`. Verified
  against `arch/compatibility.py:58, 95`. **OK.**
  But the Supported architectures list (`README.md:273`) is
  CUDA-tag-only ("Volta SM70 ... Blackwell SM120"). **STALE (P0):** v1.0
  must extend or document explicitly that `@require_arch` is CUDA-only
  and that MPS users should use a parallel decorator (TBD by
  engineering).
- `README.md:269` — "On our GTX 1650 (Turing, SM75), tests marked
  `@require_arch("Hopper")` are skipped". Verified against
  `arch/compatibility.py:82-87`. **OK.**
- `README.md:271` — GTX 16xx tensor-core caveat. Verified against
  `arch/detection.py:95-100`. **OK** (matches code).
- `README.md:286-289` — Mann-Whitney U regression sample output.
  Cross-check `analysis/regression.py:276-282`:
  ```
  REGRESSION DETECTED: {change_pct:+.1f}% (p={pvalue:.4f}, Cohen's d={cohen_d:.2f})
  ```
  Matches. **OK.**
- `README.md:295` — `GPUSpecs(peak_flops=3.5e12, peak_bandwidth=128e9)
  # GTX 1650`. Real GTX 1650 is closer to FP32 ≈ 2.98 TFLOP/s and BW
  ≈ 128 GB/s (boost 1665 MHz × 896 cores × 2 = 2.98 TFLOP/s). The
  3.5e12 figure is illustrative-rounded but slightly **HIGH (P2 nit)**.
  Phase 3 may correct, but it's an example, not load-bearing.

### §"Tolerance table" (`README.md:300-322`)
- `README.md:304-312` — Default tolerances. Cross-check
  `assertions/tolerances.py:12-24`:
  ```
  "float64": (1e-10, 1e-7),
  "float32": (1e-4, 1e-4),
  "float16": (1e-2, 1e-2),
  "bfloat16": (5e-2, 5e-2),
  "float8_e4m3fn": (0.125, 0.125),
  "float8_e5m2": (0.25, 0.25),
  "tf32": (5e-4, 5e-4),
  ```
  **Match — every row.** OK.
- `README.md:314-320` — `[tool.gpucheck.tolerances]` example. Cross-check
  `assertions/tolerances.py:91-110` (`tolerances_from_config`). The shape
  matches. **OK.**

### §"Bugs found" (`README.md:324-338`)
- `README.md:328-334` — 5 bugs in table. The README opening claims "8
  real bugs" (`README.md:12`); the table only enumerates 5. **MINOR
  INCONSISTENCY (P1):** either expand the table to 8 (preferred) or
  reword to "5 representative bugs from a total of 8". CLAUDE.md
  `Strengths` says "8 real bugs" — same gap.

### §"Tested hardware and software" (`README.md:340-368`)
- `README.md:344` — "NVIDIA GeForce GTX 1650 (Turing, SM75, 4GB GDDR6,
  no tensor cores)". **STALE (P0):** v1.0 must add Apple Silicon (M1 or
  M2 or M3, whichever the engineer ran on) and update the test stack.
- `README.md:347-352` — "Ubuntu 22.04+ ... CUDA 13.0 ... PyTorch
  2.11.0+cu130 ... Triton 3.6.0". Pinned versions; OK as-is, augment
  with "macOS 14+ / PyTorch 2.x with MPS / Metal 3" row in Phase 3.
- `README.md:354-358` — "120 unit tests ... 235 GPU integration tests
  ... 408 tests passing". After Phase 3 the engineering team will have
  added MPS tests — counts will change.
- `README.md:360-364` — Architecture detection "verified for Turing
  SM75 only". OK.
- `README.md:367` — "AMD ROCm and Intel XPU are not supported yet."
  **STALE (P1):** Apple Silicon will be supported. Rewrite to "Apple
  Silicon supported via MPS as of v1.0; AMD ROCm and Intel XPU planned".
- `README.md:368` — "Google TPU is not in scope". OK.

### §"Comparison" (`README.md:370-382`)
- Generic claims; OK. Phase 3 might add a "MPS support" row to compare
  vs. competitors.

### §"Project structure" (`README.md:386-427`)
- ASCII tree matches actual src layout per `EVIDENCE/detector.md`
  module map. **OK.**

### §"Contributing" (`README.md:429-456`)
- Three-line contributing block. **GAP (P1):** v1.0 should link to
  full `CONTRIBUTING.md` (drafted in this Phase as
  `CONTRIBUTING_DRAFT.md`).

## CLAUDE.md — claim-by-claim audit

- `CLAUDE.md` "Project Overview" — "pytest plugin for GPU kernel testing"
  matches `__init__.py:1`. OK.
- "PyPI: gpucheck v0.1.0" — matches `pyproject.toml:7`. **STALE for
  v1.0:** must bump.
- "Architecture" tree matches actual layout. OK.
- "Build & Test" — `pip install -e ".[dev]"` / `pytest --tb=short -q`
  / `ruff check src/ tests/` / `mypy src/`. Verified against
  `pyproject.toml:42-47, 60-85`. **OK.**
- "Key Design Decisions"
  - "Lazy imports everywhere: torch/pynvml never imported at collection
    time" — verified at `__init__.py:11-45` (lazy), `decorators/dtypes.py
    :108-145` (deferred to collection), `fixtures/__init__.py:8-25`. **OK.**
  - "Dual backend: pynvml preferred over torch for detection (lighter)"
    — verified `arch/detection.py:259-282`. **OK.**
  - "Tolerance model: Base tolerances per dtype, scaled by sqrt(k/128)
    for matmul ops" — verified `assertions/tolerances.py:64-67`. **OK.**
  - "GPU fast-path: assert_close checks torch.allclose on-device first,
    falls back to numpy for rich reporting" — verified
    `assertions/close.py:163-176`. **OK** today; needs MPS device-type
    extension in v1.0 (currently checks `actual.device.type == "cuda"`
    only at line 168).
  - "Statistical benchmarking: CUDA events + L2 flush + IQR outlier
    removal" — verified `fixtures/benchmark.py:147-243`. **OK** today;
    needs MPS branch in v1.0.
  - "Shape fuzzing priority: degenerate > non-tile-aligned > prime >
    power-of-2 boundary > large > mixed" — verified
    `fuzzing/shapes.py:107-114`. **OK.**
- "Strengths"
  - "Found 8 real bugs in Triton/PyTorch with 511 test configs" — same
    8-vs-5 README inconsistency note.
  - "Architecture detection: Pascal through Blackwell (SM60-SM120)" —
    verified `arch/detection.py:14-28`. **OK.**
  - "Tensor core generation tracking with GTX 16xx exclusion" —
    verified `arch/detection.py:95-100`. **OK.**
- **"Known Weaknesses & Gaps"** — entire list (`CLAUDE.md` section)
  is in flight for v1.0:
  - "No stride/contiguity fuzzing" — Track B (`feat/track-b-strides`)
    closes this.
  - "No AMD ROCm or Intel XPU support" — partial: Apple Silicon
    via MPS is Track A, not ROCm/XPU.
  - "Thread-safety issue in tolerance override stack" — Track C
    (`feat/track-c-thread-safety`) closes this.
  - **Phase 3 must rewrite this entire bullet list** as items move from
    weaknesses to features.
- "Code Standards" — verified against `pyproject.toml:60-85`. **OK.**
- "Git Conventions" — "conventional commits", "Account: Akasxh /
  drakathakash@gmail.com". Verified against `.git/config`. **OK.**
- "Expert System" — table maps modules to expert agents; cross-check
  against the experts directory referenced by SESSION_PRECHECK
  (`.claude/teams/SESSION_PRECHECK.md:27`: "Promote gpucheck experts:
  10 subagents in ~/.claude/agents/gpucheck/"). Implicitly **OK.**

## Cross-doc consistency check

| Claim | README.md | CLAUDE.md | Source | Verdict |
|---|---|---|---|---|
| Bug count | "8 real bugs" / table of 5 | "8 real bugs" | n/a | README internal inconsistency (5 vs 8) |
| Tolerance table | 7 rows | (refs) | 7 rows in `tolerances.py:12-24` | OK |
| k_dim formula | sqrt(k/128) | sqrt(k/128) | sqrt(max(k,1)/128.0) at `tolerances.py:67` | OK |
| Architectures | "Volta..Blackwell" | "Pascal through Blackwell (SM60-SM120)" | `detection.py:14-28` lists Pascal=(6,0), so CLAUDE.md is more accurate | README slightly narrower; both correct |
| Lazy imports | (implicit) | "everywhere" | `__init__.py:11-45` | OK |
| MPS support | not mentioned | not mentioned | not implemented | both consistent today; both stale post-Phase-3 |

## Stale-claim consolidated list (P0/P1)

Inputs to `AUDIT.md`:

P0 (must address before v1.0 RC tag):
- `README.md:10` (CUDA-only framing)
- `README.md:23` `@devices("cuda:0")` in hero sample
- `README.md:53` GTX-1650-only validation claim
- `README.md:122, 145` `cuda:0` in subsequent samples
- `README.md:177` "CUDA events"
- `README.md:251` `torch.cuda.memory_stats()`
- `README.md:273` SM-tag-only architecture list
- `README.md:344` GTX 1650 only in tested-hardware
- `README.md:367` "AMD ROCm and Intel XPU not supported yet"
  (omits MPS, presents fallback as no-op)
- `CLAUDE.md` "Known Weaknesses" all bullets (§ entire list)
- `__init__.py:8` `__version__ = "0.1.0"` not bumped
- `pyproject.toml:7` same

P1 (release-quality):
- `README.md:328-334` 5-row "Bugs found" vs "8 real bugs" lead
- `README.md:295` `GPUSpecs(peak_flops=3.5e12, ...)` over-rounded for GTX 1650
- `README.md:430-456` 3-line Contributing block (replace with link to
  CONTRIBUTING.md)
- 20 missing docstrings (per `EVIDENCE/reader.md`)
