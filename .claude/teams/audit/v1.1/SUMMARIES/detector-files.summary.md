# detector-files summary (W1)

**Files audited:** 41 .py under src/gpucheck/

**Top-3 systemic issues:**
1. **Duplicated logic across modules.** GPU detection 3× (fixtures/gpu.py:43,90 / arch/detection.py:133,206 / plugin.py:10-22). _median twice (analysis/regression.py:204 + analysis/roofline.py:310). Three gpu_available shims. Drift risk on every fix.
2. **Naming collisions.** Two different `compute_tolerance` (assertions/tolerances.py:70 vs arch/tensor_cores.py:96) with incompatible signatures. Two `MemoryReport` (fixtures/profiler.py:28 vs sanitizers/__init__.py:14). Import-time footguns.
3. **Bare except violations.** arch/detection.py:157,229 + 5 places in backends/mps.py (lines 99,137,141,148,191) — directly violates CLAUDE.md "no bare except" rule.

**Top-3 highest-impact single-file findings:**
1. **assertions/close.py:13-19 — top-level `import torch as _torch`.** Only top-level torch import in src/. **Defeats the lazy-import contract** advertised in CLAUDE.md. Major.
2. **decorators/parametrize.py:135-139** — uses try/except TypeError to detect callback arity. A 3-arg skip predicate that internally raises TypeError gets silently downgraded.
3. **sanitizers/memory.py:142,216** — `memory_guard` (public) yields a `_MutableReport` (private). Underscore-prefixed type leaks into public API.
