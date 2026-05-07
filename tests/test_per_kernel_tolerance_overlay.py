"""Per-(kernel_class, dtype) MPS tolerance overlay (T-24, v1.1 calibration).

Pins the multipliers from `_MPS_KERNEL_DTYPE_MULTIPLIERS` (sourced from
`.claude/teams/audit/v1.1/drift_histogram_5k.json` analysed in
`EVIDENCE/calibration-final.md`) and the resolution order encoded in
`_resolve_mps_multiplier`. Every assertion in this module is CPU-only.

Three pinning shapes:

* Per-cell pinning — each measured (kernel_class, dtype) cell maps to its
  expected multiplier on top of the v1.0 base atol/rtol.
* Backward-compat pinning — calls without `kernel_class=` MUST produce the
  exact v1.0 result (the flat ``_MPS_TOLERANCE_MULTIPLIERS`` overlay).
* Fallback-ordering pinning — unknown (kernel_class, dtype) pairs must
  fall through wildcard → DEFAULT-class → flat → hard 2.0.
"""

from __future__ import annotations

import pytest

from gpucheck.assertions import KernelClass, compute_tolerance
from gpucheck.assertions.tolerances import (
    _DEFAULT_TOLERANCES,
    _MPS_KERNEL_DTYPE_MULTIPLIERS,
    _MPS_TOLERANCE_MULTIPLIERS,
    _resolve_mps_multiplier,
)


# ---------------------------------------------------------------------------
# Per-cell pinning — every measured cell from drift_histogram_5k.json
# ---------------------------------------------------------------------------

_MEASURED_CELLS: list[tuple[KernelClass, str, float]] = [
    # (kernel_class, dtype_name, expected_multiplier)
    (KernelClass.MATMUL, "float32", 16.0),
    (KernelClass.MATMUL, "float16", 20.0),
    (KernelClass.MATMUL, "bfloat16", 32.0),
    (KernelClass.CONV2D, "float32", 4.0),
    (KernelClass.CONV2D, "float16", 8.0),
    (KernelClass.CONV2D, "bfloat16", 12.0),
]


@pytest.mark.parametrize(("kernel_class", "dtype", "mult"), _MEASURED_CELLS)
def test_compute_tolerance_per_kernel_dtype_overlay(
    kernel_class: KernelClass, dtype: str, mult: float,
) -> None:
    """Each measured (kernel_class, dtype) cell scales atol+rtol by `mult`."""
    base_atol, base_rtol = _DEFAULT_TOLERANCES[dtype]
    atol, rtol = compute_tolerance(
        dtype, device_type="mps", kernel_class=kernel_class,
    )
    assert atol == pytest.approx(base_atol * mult)
    assert rtol == pytest.approx(base_rtol * mult)


@pytest.mark.parametrize(
    "kernel_class",
    [KernelClass.NORM, KernelClass.REDUCTION, KernelClass.POINTWISE],
)
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_norm_reduction_pointwise_apply_2x_via_wildcard(
    kernel_class: KernelClass, dtype: str,
) -> None:
    """NORM / REDUCTION / POINTWISE classes resolve via the dtype wildcard
    row to a flat 2× multiplier — the v1.1 calibration confirmed all
    norm-protected and pointwise dtypes sit under FA-2× at P99.9.
    """
    base_atol, base_rtol = _DEFAULT_TOLERANCES[dtype]
    atol, rtol = compute_tolerance(
        dtype, device_type="mps", kernel_class=kernel_class,
    )
    assert atol == pytest.approx(base_atol * 2.0)
    assert rtol == pytest.approx(base_rtol * 2.0)


# ---------------------------------------------------------------------------
# Backward compatibility — omitting `kernel_class` reproduces v1.0 exactly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "dtype",
    ["float32", "float16", "bfloat16", "float64", "float8_e4m3fn", "float8_e5m2", "tf32"],
)
def test_omitting_kernel_class_matches_v1_0_flat_overlay(dtype: str) -> None:
    """v1.0 callers (no `kernel_class=`) MUST get the legacy flat overlay
    multiplier verbatim — the per-(kernel_class, dtype) table is opt-in.
    """
    base_atol, base_rtol = _DEFAULT_TOLERANCES[dtype]
    flat_mult = _MPS_TOLERANCE_MULTIPLIERS[dtype]
    atol, rtol = compute_tolerance(dtype, device_type="mps")
    assert atol == pytest.approx(base_atol * flat_mult)
    assert rtol == pytest.approx(base_rtol * flat_mult)


def test_v1_0_signature_is_unchanged_when_kernel_class_none() -> None:
    """Explicit `kernel_class=None` is identical to omitting the kwarg."""
    omitted = compute_tolerance("float32", device_type="mps")
    explicit_none = compute_tolerance(
        "float32", device_type="mps", kernel_class=None,
    )
    assert omitted == explicit_none


def test_non_mps_device_ignores_kernel_class() -> None:
    """For device_type != 'mps', the overlay must not run — kernel_class
    is silently ignored and CUDA tolerances are returned untouched.
    """
    cuda = compute_tolerance("float32", device_type="cuda")
    cuda_with_kc = compute_tolerance(
        "float32", device_type="cuda", kernel_class=KernelClass.MATMUL,
    )
    assert cuda == cuda_with_kc


def test_kdim_scaling_still_applies_under_per_kernel_overlay() -> None:
    """k_dim sqrt-scale and the per-kernel-class multiplier compose
    multiplicatively — the order is k_dim THEN MPS overlay.
    """
    base_atol, _ = _DEFAULT_TOLERANCES["float32"]
    atol, _rtol = compute_tolerance(
        "float32", k_dim=512, device_type="mps", kernel_class=KernelClass.MATMUL,
    )
    # k_dim=512 → sqrt(512/128) == 2.0; MATMUL/fp32 multiplier == 16.0.
    expected = base_atol * 2.0 * 16.0
    assert atol == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Fallback ordering — wildcard / DEFAULT-class / flat / hard-2x
# ---------------------------------------------------------------------------

def test_norm_with_fp8_falls_through_to_wildcard() -> None:
    """(NORM, 'float8_e4m3fn') has no exact entry but the wildcard row
    `(NORM, '*')` matches at step 2 → 2.0× multiplier.
    """
    assert _resolve_mps_multiplier(
        KernelClass.NORM, "float8_e4m3fn",
    ) == pytest.approx(2.0)


def test_default_kernel_class_falls_through_to_flat_table() -> None:
    """`KernelClass.DEFAULT` has no exact rows and no wildcard row, so
    resolution drops through to the flat `_MPS_TOLERANCE_MULTIPLIERS` —
    matching the v1.0 behaviour for whichever dtype is requested.
    """
    for dtype, expected in _MPS_TOLERANCE_MULTIPLIERS.items():
        assert _resolve_mps_multiplier(
            KernelClass.DEFAULT, dtype,
        ) == pytest.approx(expected)


def test_unknown_dtype_under_known_class_uses_class_wildcard() -> None:
    """An unrecognised dtype string under NORM still hits the wildcard row,
    not the flat-table fallback.
    """
    assert _resolve_mps_multiplier(
        KernelClass.NORM, "complex128",
    ) == pytest.approx(2.0)


def test_unknown_dtype_with_no_class_uses_hard_two_fallback() -> None:
    """Unknown dtype, no kernel_class → step 5: hard 2.0 fallback."""
    assert _resolve_mps_multiplier(None, "complex128") == pytest.approx(2.0)


def test_kernel_class_str_enum_accepts_string_value() -> None:
    """`KernelClass` is a `str`-Enum: the value-equality contract lets
    integrations stash the string in pyproject.toml and round-trip it.
    """
    assert KernelClass.MATMUL == "matmul"
    assert KernelClass.CONV2D.value == "conv2d"


def test_overlay_table_count_pins_dict_size() -> None:
    """If a row is added or removed in the future, this test forces a
    deliberate update — pins the v1.1 ship-state to 9 entries
    (3 MATMUL + 3 CONV2D + 3 wildcards).
    """
    assert len(_MPS_KERNEL_DTYPE_MULTIPLIERS) == 9


def test_tolerance_context_overrides_short_circuit_overlay() -> None:
    """When `tolerance_context` pushes an override, the per-kernel-class
    overlay must NOT be applied (the override is the user's explicit
    decision and out-ranks any device-type rules).
    """
    from gpucheck.assertions import tolerance_context

    with tolerance_context(atol=1e-9, rtol=1e-9):
        atol, rtol = compute_tolerance(
            "float32",
            device_type="mps",
            kernel_class=KernelClass.MATMUL,
        )
    assert atol == 1e-9
    assert rtol == 1e-9
