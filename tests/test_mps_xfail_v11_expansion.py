"""v1.1 MPS xfail registry expansion tests (T-25, github-miner-v3).

The v1.0 registry shipped 12 entries. v1.1 expands the registry from
the Round-2/Round-3 PyTorch issue long-tail audit (see
``.claude/teams/research/v1.0/EVIDENCE/github-miner-v3-xfail-config.md``
§A for the canonical TOML block).

These tests cover only the v1.1 expansion; the v1.0-era assertions
live in ``test_mps_xfail.py`` and remain in force.
"""

from __future__ import annotations

import sys
from pathlib import Path

from gpucheck import is_mps_xfailed, mps_xfail_list
from gpucheck.assertions.tolerances import (
    apply_mps_xfail_config,
    mps_xfail_from_config,
    register_mps_xfail,
    reset_mps_xfail,
)

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover -- 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef,unused-ignore]


# Total entry count after the Round-3 long-tail expansion (12 v1 + 39 v1.1).
# This is the canonical count produced by §A of the github-miner-v3 evidence
# file after de-dup against v1's existing list. If a future re-mine adds or
# removes entries, update this number alongside the TOML edit.
_EXPECTED_TOTAL = 51

# Subset of v1.1 additions chosen from the binding input §B as
# "high-value" — they each map to a verified-OPEN PyTorch issue, span
# different bug families (silent-correctness, OOB, prefix-sum drift,
# scatter, model-level), and are the entries the engineering lead
# explicitly flagged for the regression watch.
_V11_HIGH_VALUE_ENTRIES = {
    "copy_.strided_view_offset_2pow32_wrap",
    "binary_ops.uint16_uint32_uint64",
    "avg_pool1d.prefix_sum_drift_long_seq",
    "scatter_add_.nonzero_offset_slice",
    "model.voxtral_asr_full_pipeline",
}


def _load_pyproject_xfail_ops() -> list[str]:
    """Read ``pyproject.toml`` and return the raw ``ops`` list verbatim.

    Reuses the same parsing path the plugin uses at session start
    (``apply_mps_xfail_config(data)`` with ``data = tomllib.load(f)``)
    so this test exercises the real config-loading mechanism rather
    than mirroring the data inline.
    """
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with pyproject.open("rb") as f:
        data = tomllib.load(f)
    parsed = mps_xfail_from_config(data)
    assert parsed is not None, "pyproject.toml must define [tool.gpucheck.mps.xfail]"
    return sorted(parsed)


def test_pyproject_xfail_block_has_expected_total_after_v11_expansion() -> None:
    """The TOML block must contain exactly the post-v1.1 entry count."""
    ops = _load_pyproject_xfail_ops()
    assert len(ops) == _EXPECTED_TOTAL, (
        f"expected {_EXPECTED_TOTAL} entries after v1.1 long-tail expansion, "
        f"found {len(ops)}; if intentional, update _EXPECTED_TOTAL and the "
        f"CHANGELOG entry together."
    )


def test_pyproject_xfail_entries_are_unique() -> None:
    """Op names must be unique — duplicate keys would silently mask bugs."""
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with pyproject.open("rb") as f:
        data = tomllib.load(f)
    ops_list = data["tool"]["gpucheck"]["mps"]["xfail"]["ops"]
    assert len(ops_list) == len(set(ops_list)), (
        f"duplicate op names in pyproject.toml: "
        f"{[o for o in ops_list if ops_list.count(o) > 1]}"
    )


def test_v11_high_value_entries_are_registered_at_session_start() -> None:
    """The five high-value v1.1 entries must populate the live registry.

    The plugin's ``pytest_configure`` hook calls
    ``apply_mps_xfail_config`` against the parsed pyproject before
    tests run, so by the time this test executes the registry should
    contain every entry in the TOML block.
    """
    actual = set(mps_xfail_list())
    missing = _V11_HIGH_VALUE_ENTRIES - actual
    assert not missing, (
        f"v1.1 high-value xfail entries missing from live registry: {missing}; "
        f"the pyproject [tool.gpucheck.mps.xfail] block did not load correctly."
    )


def test_is_mps_xfailed_returns_true_for_strided_copy_2pow32() -> None:
    """Public ``is_mps_xfailed`` must report the flagship Tier-1 entry.

    pytorch#182052 — aten::copy_ silently wraps writes into strided
    views when the destination element offset exceeds 2^32. This is
    the most-recent and highest-severity entry in the v1.1 expansion
    (silent data corruption, no workaround), so it serves as a smoke
    test for the public lookup path.
    """
    assert is_mps_xfailed("copy_.strided_view_offset_2pow32_wrap"), (
        "pytorch#182052 (copy_ strided wrap > 2^32) must be xfailed on MPS"
    )


def test_apply_mps_xfail_round_trip_with_full_v11_block() -> None:
    """Re-applying the pyproject config must keep the registry consistent.

    Saves the live registry, clears it, re-applies from the freshly
    parsed pyproject, and asserts the same 51-entry set comes back.
    Restores the registry on exit so we don't pollute downstream tests.
    """
    from gpucheck.assertions.tolerances import _mps_xfail_set

    saved = set(_mps_xfail_set)
    try:
        reset_mps_xfail()
        register_mps_xfail("phantom.v11_expansion_op")
        assert is_mps_xfailed("phantom.v11_expansion_op")

        pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
        with pyproject.open("rb") as f:
            data = tomllib.load(f)
        apply_mps_xfail_config(data)

        # The phantom entry must be evicted (apply replaces, not merges).
        assert not is_mps_xfailed("phantom.v11_expansion_op")
        # And the v1.1 high-value entries must all be back.
        for op in _V11_HIGH_VALUE_ENTRIES:
            assert is_mps_xfailed(op), f"{op} missing after re-apply"
        assert len(mps_xfail_list()) == _EXPECTED_TOTAL
    finally:
        reset_mps_xfail()
        register_mps_xfail(*saved)
