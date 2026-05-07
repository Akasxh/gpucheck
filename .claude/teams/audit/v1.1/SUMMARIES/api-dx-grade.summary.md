# api-dx-grade summary (W1 — landed first)

**Per-axis means (n=18 symbols):**
- Discoverability 3.7 / Type safety 3.4 / Ergonomics 4.0 / Error-msg 3.4 / Deprecation 3.8

**Weakest 3 APIs:**
1. ShapeStrategy + StrideStrategy (2.8) — `__new__`-as-factory blocks isinstance/mypy
2. @require_arch (3.0) — naming inconsistency with requires_determinism (require vs requires); silent typo skip
3. compute_tolerance, tolerance_context, memory_tracker (3.2) — silent fp32 fallback on unknown dtype

**Top 3 v1.1 fixes (additive, no v1.0 breakage):**
1. Loud failure on typo'd strings — compute_tolerance, @devices, register_mps_xfail, @dtypes. Add strict=False kwarg now, flip in v1.2.
2. Promote 7 hidden symbols to top-level _LAZY_MAP — memory_tracker, gpu_device, fuzz_strides, ShapeStrategy, StrideStrategy, requires_determinism, assert_deterministic.
3. Ergonomic shortcuts — tolerance_context(scale=2.0), @xfail_on_mps("op.sub") decorator.

**Preserve:** Backend/EventTimer Protocol pair, DeterminismError message text, fuzz_shapes priority-ordered design.

**Deprecation candidate:** baseline_2x: bool on assert_close — magic boolean blocks non-2x scales; replace with tolerance_scale: float | None, keep with DeprecationWarning in v1.1.

(Full audit at EVIDENCE/api-dx-grade.md)
