# EVIDENCE — docs-detector

**Persona:** docs-detector (Phase A, FM-1.1 — auto-detect project environment).
**Run:** Phase 1 audit, gpucheck v1.0.

## Project profile

- **Name:** `gpucheck`
- **Repo:** `/Users/cero/Code/gpucheck` (branch `release/v1.0`, clean)
- **Author:** Akash <drakathakash@gmail.com>
- **License:** Apache-2.0
- **PyPI:** `gpucheck` v0.1.0 (per `pyproject.toml:7`)
- **Python:** `>=3.10` (`pyproject.toml:11`)
- **Build backend:** hatchling (`pyproject.toml:1-3`)
- **Pytest entry-point:** `gpucheck.plugin` registered as `pytest11`
  (`pyproject.toml:49-50`).

## Language / framework

| Aspect | Detected |
|---|---|
| Language | Python (sole) |
| Type discipline | mypy strict (`pyproject.toml:77-85`) |
| Lint discipline | ruff with E/F/W/I/N/UP/B/A/SIM/TCH (`pyproject.toml:74-76`) |
| Test framework | pytest >= 7 (`pyproject.toml:31`) |
| Plugin entry-point | `gpucheck.plugin` (`pyproject.toml:49-50`) |
| Optional deps | `torch`, `cupy`, `triton`, `hypothesis`, `dev` (`pyproject.toml:36-47`) |

## Doc-format detection

| Artifact | Present | Path | Notes |
|---|---|---|---|
| `README.md` | yes | `/Users/cero/Code/gpucheck/README.md` (461 lines) | Mature, GTX-1650-anchored, CUDA-only language. |
| `CLAUDE.md` | yes | `/Users/cero/Code/gpucheck/CLAUDE.md` | Project memory; CUDA-only. |
| `CHANGELOG.md` | **NO** | — | Phase 1 must draft. |
| `CONTRIBUTING.md` | **NO** | — | Phase 1 must draft. |
| `MIGRATION.md` | **NO** | — | Phase 1 must draft (v0 → v1). |
| `CODE_OF_CONDUCT.md` | **NO** | — | Phase 1 flags. |
| `SECURITY.md` | **NO** | — | Phase 1 flags. |
| `LICENSE` | yes | Apache-2.0 SPDX | OK. |
| `examples/` | yes | 6 runnable examples (`README.md:454`) | OK, but never re-validated for MPS. |
| `docs/` site (Sphinx/MkDocs) | **NO** | — | Out-of-scope for Phase 1, note for Phase 3. |

## Module map

```
src/gpucheck/
  __init__.py            (lazy public API, 91 LOC)
  plugin.py              (pytest hooks, fixtures, terminal summary)
  assertions/  close.py · tolerances.py · reporting.py
  decorators/  dtypes.py · shapes.py · devices.py · parametrize.py
  fixtures/    benchmark.py · gpu.py · profiler.py
  fuzzing/     shapes.py · inputs.py · strategies.py
  sanitizers/  memory.py · race.py
  arch/        detection.py · compatibility.py · tensor_cores.py
  analysis/    roofline.py · regression.py · bottleneck.py
  reporting/   console.py · json.py · ci.py
```

5605 source LOC total. Module boundaries match CLAUDE.md's claimed
architecture (CLAUDE.md §Architecture).

## Cross-team handoff signals

- `SESSION_PRECHECK.md` (`.claude/teams/SESSION_PRECHECK.md:7-9`) marks
  the host as **Apple Silicon, MPS-capable** and notes 4 impl
  worktrees: `feat/track-{a-mps,b-strides,c-thread-safety,d-bundle}`.
- Research team's `QUESTION.md` (`.claude/teams/research/v1.0/QUESTION.md:9`)
  states the headline v1.0 feature is an **Apple-Silicon MPS backend**.
- `.claude/teams/engineering/v1.0/DIFF_LOG.md` does **not yet exist** —
  Phase 3 docs must wait for engineering to populate it. Phase 1 only
  drafts skeletons and audits stale claims.

## CI surface

- GitHub Actions badge in README (`README.md:8`) but no GPU CI per
  CLAUDE.md "Known Weaknesses" §3 ("No GPU CI (tests run CPU-only)").
- JUnit XML / GitHub annotations / PR comment generation present in
  `src/gpucheck/reporting/ci.py` but CLAUDE.md notes "Reporting module
  has zero test coverage".

## Audience

- **Primary:** GPU kernel authors (CUDA / Triton / Metal-MPS) who write
  pytest tests.
- **Secondary:** OSS maintainers reading bug reports filed against
  Triton (#9838, #9839) using gpucheck.
- **Tertiary:** CI reviewers consuming JUnit XML / PR comment outputs.

## Detector verdict

Single-language Python pytest plugin, mature module structure,
README-only docs. Top-level governance docs (CHANGELOG / CONTRIBUTING /
MIGRATION / CoC / SECURITY) are **all missing**. Phase 1 has authority
to draft skeletons; Phase 3 fills entries from real DIFF_LOG. The MPS
v1.0 pivot is the dominant narrative shift the docs must absorb.
