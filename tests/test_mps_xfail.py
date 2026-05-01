"""MPS xfail registry config-loader (Track A)."""

from __future__ import annotations

from gpucheck import is_mps_xfailed, mps_xfail_list, register_mps_xfail
from gpucheck.assertions.tolerances import (
    apply_mps_xfail_config,
    mps_xfail_from_config,
    reset_mps_xfail,
)

# The 12 entries we ship in pyproject.toml per SYNTHESIS §7.
_EXPECTED_XFAIL_OPS = {
    "scaled_dot_product_attention.large",
    "scaled_dot_product_attention.backward",
    "layer_norm.backward.shape1",
    "batch_norm.backward.channels_last",
    "conv2d.large_channels",
    "conv2d.backward.channels_last_format",
    "F.linear.backward.bf16_3d_nobias_m5",
    "softmax.large_attention",
    "avg_pool2d.backward.channels_last",
    "binary_ops.uint16_uint32_uint64",
    "BCE_loss",
    "matmul.backward.over_32K_elements",
}


def test_mps_xfail_from_config_extracts_ops_list() -> None:
    cfg = {
        "tool": {
            "gpucheck": {
                "mps": {
                    "xfail": {"ops": ["softmax.large_attention", "BCE_loss"]},
                }
            }
        }
    }
    assert mps_xfail_from_config(cfg) == {"softmax.large_attention", "BCE_loss"}


def test_mps_xfail_from_config_returns_none_for_empty() -> None:
    assert mps_xfail_from_config({}) is None
    assert mps_xfail_from_config({"tool": {"gpucheck": {}}}) is None


def test_apply_mps_xfail_replaces_existing_registry() -> None:
    # Save the current registry so we can restore it (the plugin populated
    # it from pyproject.toml at session start; other tests rely on that).
    from gpucheck.assertions.tolerances import _mps_xfail_set

    saved = set(_mps_xfail_set)
    try:
        reset_mps_xfail()
        register_mps_xfail("phantom.op")
        assert is_mps_xfailed("phantom.op")
        apply_mps_xfail_config({
            "tool": {"gpucheck": {"mps": {"xfail": {"ops": ["softmax.large_attention"]}}}}
        })
        assert not is_mps_xfailed("phantom.op")
        assert is_mps_xfailed("softmax.large_attention")
    finally:
        reset_mps_xfail()
        register_mps_xfail(*saved)


def test_pyproject_xfail_block_loaded_at_session_start() -> None:
    """The 12 SYNTHESIS §7 entries must populate the registry once the
    plugin's pytest_configure has run (which it has, since we're running
    inside pytest).
    """
    actual = set(mps_xfail_list())
    missing = _EXPECTED_XFAIL_OPS - actual
    assert not missing, (
        f"pyproject.toml [tool.gpucheck.mps.xfail] is missing entries: {missing}; "
        f"the SYNTHESIS §7 living-document list must be populated."
    )


def test_register_mps_xfail_at_runtime() -> None:
    register_mps_xfail("some.runtime.op")
    try:
        assert is_mps_xfailed("some.runtime.op")
    finally:
        # Don't pollute other tests. (reset_mps_xfail clears all entries
        # including the pyproject-loaded ones, so we instead rebuild from
        # config.)
        from gpucheck.assertions.tolerances import _mps_xfail_set

        _mps_xfail_set.discard("some.runtime.op")
