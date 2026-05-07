"""v1.1: MPS / Apple-Silicon tile-set extension property tests.

These tests pin cartographer-v2.md §3's master tile table into the
fuzzer's output and assert that the CUDA path is unaffected (so the
NVIDIA users of v1.0 are not regressed by the new ``device_type`` knob).

See ``.claude/teams/research/v1.0/EVIDENCE/cartographer-v3-fuzzer-patch.md``
for design rationale and source citations.
"""

from __future__ import annotations

from gpucheck.fuzzing.shapes import fuzz_shapes


class TestFuzzShapesMPSTiles:
    """v1.1 MPS path: surface Apple-canonical tile boundaries."""

    def test_mps_pool_includes_8x8_mma_fragment_boundary(self) -> None:
        """8x8 simdgroup_matrix is the Apple MMA fragment (cartographer-v2 §S4).

        cartographer-v2 establishes 8 as the load-bearing Apple tile that
        is absent from CUDA's set. The MPS fuzzer must produce shapes
        whose dimensions touch the 8-edge ({7, 8, 9}) so kernels that
        special-case the fragment boundary are exercised.
        """
        result = fuzz_shapes(
            ndim=2, n=200, max_size=512, seed=0, device_type="mps"
        )
        # Every value in {7, 8, 9} must appear in at least one shape.
        observed_dims = {d for shape in result for d in shape}
        assert {7, 8, 9}.issubset(observed_dims), (
            f"MPS fuzzer dropped 8x8-MMA-fragment boundary; "
            f"observed dims = {sorted(observed_dims)}"
        )

    def test_mps_pool_includes_head_dim_80_boundary(self) -> None:
        """FA head_dim=80 is Apple-only (cartographer-v2 §S8 / steel_attention.metal).

        NVIDIA FlashAttention does not ship a fast path for head_dim=80,
        but MLX does. The MPS fuzzer must surface {79, 80, 81} so a
        gpucheck-driven SDPA test on Apple Silicon will hit the
        head_dim=80 dispatcher path.
        """
        result = fuzz_shapes(
            ndim=2, n=200, max_size=512, seed=0, device_type="mps"
        )
        observed_dims = {d for shape in result for d in shape}
        assert {79, 80, 81}.issubset(observed_dims), (
            f"MPS fuzzer dropped head_dim=80 Apple-only boundary; "
            f"observed dims = {sorted(observed_dims)}"
        )

    def test_cuda_path_unaffected_by_mps_extension(self) -> None:
        """v1.0 CUDA users must not regress.

        With device_type='cuda' (the default), the deterministic fuzzer
        pool must equal the v1.0 pool — no MPS-only dims (8, 9, 15, 17,
        79, 80, 81) leak into CUDA's deterministic shapes. This test
        pins the CUDA/MPS boundary so a future refactor cannot
        accidentally widen the CUDA pool (which would hide CUDA-specific
        bugs behind extra noise).

        Note: ``fuzz_shapes`` may pad with random shapes if ``n``
        exceeds the deterministic-pool size; we exercise only the
        deterministic portion (n=20 < CUDA's deterministic pool of 26
        at max_size=512) so this test pins the curated tile/boundary
        constants, not the random padding RNG output.
        """
        n_deterministic = 20  # < CUDA deterministic pool size (26 at max_size=512)
        cuda_result = fuzz_shapes(
            ndim=2,
            n=n_deterministic,
            max_size=512,
            seed=0,
            device_type="cuda",
        )
        cuda_dims = {d for shape in cuda_result for d in shape}
        # MPS-only deterministic boundaries that must NOT appear in CUDA.
        # (We pick values that are unique to MPS: 8 and 80 are not in
        # the CUDA tile set, primes list, or pow2-boundaries list.)
        mps_only = {8, 9, 15, 17, 79, 80, 81}
        leaked = mps_only & cuda_dims
        # Default CUDA path should equal explicit "cuda" path.
        default_result = fuzz_shapes(
            ndim=2, n=n_deterministic, max_size=512, seed=0
        )
        assert default_result == cuda_result
        # Note: 16 is not asserted here because _mixed_shapes uses
        # 16 as a filler — that's a v1.0 behaviour we preserve.
        assert not leaked, (
            f"CUDA fuzzer leaked MPS-only dims {leaked}; "
            f"this regresses NVIDIA users."
        )
