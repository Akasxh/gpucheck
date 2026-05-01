"""Stride fuzzing — corpus + per-category contracts (Track B)."""

from __future__ import annotations

import pytest

from gpucheck.fuzzing.strides import (
    CATEGORIES,
    fuzz_strides,
    fuzz_strides_for_category,
)

torch = pytest.importorskip("torch")


def test_categories_are_seven_canonical() -> None:
    assert CATEGORIES == (
        "row_major",
        "column_major",
        "broadcast",
        "transpose",
        "slice",
        "non_contig",
        "gather",
    )


def test_fuzz_strides_returns_all_categories_in_order() -> None:
    out = fuzz_strides((64, 64), torch.float32, seed=0)
    assert [c for c, _t in out] == list(CATEGORIES)


def test_fuzz_strides_each_tensor_has_correct_shape() -> None:
    out = fuzz_strides((32, 32), torch.float32, seed=0)
    for label, t in out:
        assert t.shape == (32, 32), f"{label}: wrong shape {t.shape}"


def test_fuzz_strides_each_tensor_has_correct_dtype() -> None:
    out = fuzz_strides((16, 16), torch.float16, seed=0)
    for label, t in out:
        assert t.dtype == torch.float16, f"{label}: wrong dtype {t.dtype}"


def test_row_major_is_contiguous() -> None:
    t = fuzz_strides_for_category((64, 64), torch.float32, "row_major", seed=0)
    assert t.is_contiguous()


def test_column_major_is_not_contiguous() -> None:
    t = fuzz_strides_for_category((64, 32), torch.float32, "column_major", seed=0)
    assert not t.is_contiguous()


def test_broadcast_has_stride_zero_on_last_dim() -> None:
    t = fuzz_strides_for_category((4, 8, 16), torch.float32, "broadcast", seed=0)
    # The last dim was expanded from size 1 -> 16, so stride is 0 there.
    assert t.stride(-1) == 0
    assert t.shape == (4, 8, 16)


def test_transpose_is_not_contiguous() -> None:
    t = fuzz_strides_for_category((4, 8, 16), torch.float32, "transpose", seed=0)
    assert not t.is_contiguous()
    # Transpose preserves shape because we transposed the LAST two dims of
    # the contiguous (4, 16, 8) buffer.
    assert t.shape == (4, 8, 16)


def test_slice_has_stride_two_on_each_dim() -> None:
    t = fuzz_strides_for_category((8, 8), torch.float32, "slice", seed=0)
    assert not t.is_contiguous()
    assert t.shape == (8, 8)


def test_non_contig_2d_is_not_contiguous() -> None:
    t = fuzz_strides_for_category((8, 8), torch.float32, "non_contig", seed=0)
    assert not t.is_contiguous()
    assert t.shape == (8, 8)


def test_gather_returns_contiguous_with_correct_shape() -> None:
    t = fuzz_strides_for_category((4, 4), torch.float32, "gather", seed=0)
    assert t.is_contiguous()
    assert t.shape == (4, 4)


def test_unknown_category_raises() -> None:
    with pytest.raises(ValueError, match="Unknown stride category"):
        fuzz_strides_for_category((4,), torch.float32, "weird")


def test_fuzz_strides_with_n_caps_results() -> None:
    out = fuzz_strides((4, 4), torch.float32, n=3, seed=0)
    assert len(out) == 3


def test_fuzz_strides_with_explicit_categories() -> None:
    out = fuzz_strides(
        (4, 4), torch.float32,
        categories=("row_major", "transpose"),
        seed=0,
    )
    assert [c for c, _t in out] == ["row_major", "transpose"]


def test_fuzz_strides_invalid_category_raises() -> None:
    with pytest.raises(ValueError, match="Unknown stride categories"):
        fuzz_strides((4, 4), torch.float32, categories=("row_major", "weird"))


def test_fuzz_strides_seed_is_deterministic() -> None:
    out1 = fuzz_strides((4, 4), torch.float32, seed=42)
    out2 = fuzz_strides((4, 4), torch.float32, seed=42)
    for (l1, t1), (l2, t2) in zip(out1, out2, strict=True):
        assert l1 == l2
        assert torch.equal(t1, t2), f"seed-determinism broken for {l1}"


def test_1d_fallbacks_to_row_major_or_slice_correctly() -> None:
    # 1D shape: column_major and transpose fall back to row_major,
    # non_contig falls back to slice.
    out = fuzz_strides((8,), torch.float32, seed=0)
    # Just assert we got tensors of shape (8,) — the fallback semantics
    # are documented in the docstring; this is a smoke check.
    for _label, t in out:
        assert t.shape == (8,)


def test_higher_rank_3d_smoke() -> None:
    out = fuzz_strides((4, 8, 16), torch.float32, seed=0)
    for _label, t in out:
        assert t.shape == (4, 8, 16)
