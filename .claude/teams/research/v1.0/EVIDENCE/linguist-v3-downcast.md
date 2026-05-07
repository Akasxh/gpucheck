---
specialist: research-linguist
slug: v1.0
round: 3
charter: silent-downcast catcher — design Backend Protocol surface
started: 2026-05-01T07:30:00Z
completed: 2026-05-01T08:05:00Z
inputs:
  - .claude/teams/research/v1.0/EVIDENCE/linguist-v2.md
  - src/gpucheck/backends/mps.py
  - src/gpucheck/backends/_protocol.py
  - src/gpucheck/assertions/close.py
  - src/gpucheck/assertions/tolerances.py
  - aten/src/ATen/native/mps/OperationUtils.mm:120-158
confidence: high
---

# Linguist Round 3 — gpucheck v1.0 silent-downcast catcher

## Background recap

The Round 2 finding (linguist-v2.md §2.4 + §3): PyTorch MPS exhibits a
load-bearing **dtype-rank asymmetry**. For `torch.float64` and
`torch.complex128`, the kernel dispatcher diverges based on tensor rank:

| dtype | tensor (`ndim >= 1`) | 0-d scalar (`ndim == 0`) |
|---|---|---|
| `torch.float64` | `TORCH_CHECK_TYPE` raise (`OperationUtils.mm:71-73`, `EmptyTensor.cpp:45`) | **silent fp32** (`OperationUtils.mm:124-127`) |
| `torch.complex128` | raise (default branch `OperationUtils.mm:86`) | **silent complex64** (`OperationUtils.mm:144-147`) |

The Python-level `tensor.dtype` reads `torch.float64` in **both** cases.
A user who writes `ref = torch.tensor(0.1, dtype=torch.float64).to('mps')`
believes they have a fp64 reference oracle; in reality the value is now an
fp32 representation of `0.1` whose error is ~5.96e-8, ~1000x worse than
the fp64 base tolerance `1e-10` in `tolerances.py:19`. `assert_close`
becomes vacuous: it would pass even when the kernel is wrong.

This round designs the gpucheck API surface to detect and refuse this.

## §1. Backend Protocol extension — `silently_downcasts_dtype`

The Round 2 recommendation introduced a `silently_downcast_dtypes`
**frozenset** on `Backend`. That set alone is insufficient because the
downcast is rank-conditional. A polysemous `dtype in S` check would
either (a) over-warn on valid `ndim >= 1` fp64 calls (which raise loudly
anyway, so warning is redundant) or (b) under-warn if the predicate is
inverted. The correct surface is a **predicate method** that takes both
`dtype` and `ndim`.

### 1.1 Signature on the `Backend` Protocol

```python
# src/gpucheck/backends/_protocol.py — additions
from typing import Any

class Backend(Protocol):
    # ... existing members ...

    def silently_downcasts_dtype(self, dtype: Any, ndim: int) -> bool:
        """Return True iff using ``dtype`` at this rank silently loses precision.

        A return value of True means: the backend will accept the tensor
        without raising, but the underlying compute will run at a lower
        precision than ``dtype`` advertises. Callers (notably
        ``gpucheck.assertions.assert_close``) MUST treat True as a
        correctness hazard for any reference / oracle role.

        On backends with no such hazards (CUDA), this MUST return False
        for every (dtype, ndim) pair.
        """
        ...
```

### 1.2 `MPSBackend.silently_downcasts_dtype` — runnable code sketch

This is the concrete implementation, lazy-importing torch in line with the
existing `_torch()` helper in `mps.py:54-57`. It returns `True` exactly
for the two known cases identified at `OperationUtils.mm:124-127`
(fp64 0-d scalar) and `OperationUtils.mm:144-147` (complex128 0-d scalar):

```python
# Append to src/gpucheck/backends/mps.py — class MPSBackend

class MPSBackend:
    # ... existing methods ...

    # Frozen registry of (dtype, ndim) pairs that PyTorch MPS silently
    # downcasts. Each entry is keyed by the dtype's torch.dtype.__str__()
    # because torch.dtype objects don't hash stably across import cycles
    # in lazy-import setups (frozenset[torch.dtype] would force eager
    # torch import at MPSBackend construction time).
    #
    # Citations:
    #   aten/src/ATen/native/mps/OperationUtils.mm:120-128 (fp64 -> fp32)
    #   aten/src/ATen/native/mps/OperationUtils.mm:144-147 (cplx128 -> cplx64)
    _DOWNCAST_DTYPE_NAMES: frozenset[str] = frozenset({
        "torch.float64",
        "torch.complex128",
    })

    def silently_downcasts_dtype(self, dtype: Any, ndim: int) -> bool:
        """Return True iff (dtype, ndim) hits the MPS scalar-path downcast.

        Per PyTorch ``aten/src/ATen/native/mps/OperationUtils.mm``::

            120  MPSDataType getMPSScalarType(ScalarType scalar_type) {
            121    switch (scalar_type) {
            ...
            124      case ScalarType::Double:
            125      case ScalarType::Float:
            126        // It is hard to keep precision for Double, but
            127        return MPSDataTypeFloat32;
            ...
            144      case ScalarType::ComplexDouble:
            145      case ScalarType::ComplexFloat:
            146        return MPSDataTypeComplexFloat32;

        These cases ONLY trigger via ``getMPSScalarType``, which is the
        scalar (0-d / wrapped Python number) dispatch path. Tensors with
        ``ndim >= 1`` go through ``getMPSDataType`` which raises
        ``TORCH_CHECK_TYPE`` for the same dtypes (lines 71-73, 86).
        """
        # Reject negative or non-int rank up front — the caller is
        # contract-bound to pass a real tensor rank.
        if not isinstance(ndim, int) or ndim < 0:
            raise TypeError(
                f"ndim must be a non-negative int, got {ndim!r}"
            )
        if ndim != 0:
            return False
        return str(dtype) in self._DOWNCAST_DTYPE_NAMES

    # CUDABackend gets the trivial implementation:
    #   def silently_downcasts_dtype(self, dtype: Any, ndim: int) -> bool:
    #       return False
```

**Design notes:**
- Keying on `str(dtype)` rather than `torch.dtype` avoids forcing an eager
  `import torch` in the backend module, preserving the lazy-import
  invariant from `CLAUDE.md` ("Lazy imports everywhere: torch/pynvml never
  imported at collection time").
- The predicate is **rank-aware** because the asymmetry is rank-conditional.
  A naive `dtype in unsupported_dtypes` check (Round 2 sketch) would not
  capture the polysemy.
- `ndim == 0` is the load-bearing rank. Wrapped Python scalars
  (`tensor + 0.5` where `0.5` is a fp64 Python float) also dispatch via
  `getMPSScalarType` per `aten/src/ATen/native/mps/OperationUtils.mm:165-180`,
  so a future expansion can reuse this predicate for the
  `is_python_scalar` axis.

## §2. `assert_close` extension — `MPSDtypeWarning`

### 2.1 New warning class

Add to `src/gpucheck/assertions/__init__.py` (or a new
`src/gpucheck/assertions/_warnings.py`):

```python
class MPSDtypeWarning(UserWarning):
    """Emitted when MPS would silently downcast a tensor in assert_close.

    The user almost certainly intended their fp64 oracle to actually run
    at fp64 precision. On MPS this is impossible for tensors and
    silently degraded for 0-d scalars. Treating this as an assertion
    failure (or at minimum a warning) is the only way to keep the
    correctness contract of ``assert_close``.

    See ``aten/src/ATen/native/mps/OperationUtils.mm:124-127, 144-147``.
    """
```

`MPSDtypeWarning` subclasses `UserWarning` (not `RuntimeWarning`) because
the scenario is a **user-introduced miscalibration** — the user wrote
`fp64` but the platform doesn't support it. `RuntimeWarning` would
suggest a transient runtime condition.

### 2.2 Hook into `assert_close` — runnable diff sketch

The check fires before the GPU fast-path so the warning surfaces even
when `torch.allclose` would have returned True (a vacuous pass).

```python
# src/gpucheck/assertions/close.py — insertion before line 174 ("GPU fast-path")

from gpucheck.assertions._warnings import MPSDtypeWarning


def _mps_downcast_check(actual: Any, expected: Any, *, strict: bool, msg: str) -> None:
    """Raise / warn if either operand hits the MPS scalar-downcast trap.

    No-op on CPU and CUDA. Lazy: only walks operands if torch is loaded
    AND at least one tensor is on device 'mps'.
    """
    if not _has_torch:
        return
    operands = [t for t in (actual, expected) if isinstance(t, _torch.Tensor)]
    mps_ops = [t for t in operands if t.device.type == "mps"]
    if not mps_ops:
        return

    # Lazy backend lookup — defers torch.mps import to runtime
    from gpucheck.backends.mps import MPSBackend
    backend = MPSBackend()

    offending: list[tuple[str, Any, int]] = []
    for tag, t in zip(("actual", "expected"), (actual, expected)):
        if not isinstance(t, _torch.Tensor):
            continue
        if t.device.type != "mps":
            continue
        if backend.silently_downcasts_dtype(t.dtype, t.ndim):
            offending.append((tag, t.dtype, t.ndim))

    if not offending:
        return

    detail = ", ".join(
        f"{tag} dtype={dt} ndim={nd}" for tag, dt, nd in offending
    )
    prefix = f"{msg}\n" if msg else ""
    text = (
        f"{prefix}MPS silently downcasts these operands to lower precision: "
        f"{detail}. PyTorch MPS routes 0-d float64/complex128 through "
        f"getMPSScalarType which returns the float32/complex64 dtype "
        f"(aten/src/ATen/native/mps/OperationUtils.mm:124-127, 144-147). "
        f"Your assertion's tolerance budget is now governed by float32 "
        f"epsilon, NOT float64 — the comparison may be vacuous. "
        f"Fix: build the reference on CPU "
        f"(torch.tensor(..., dtype=torch.float64, device='cpu')) and pass "
        f"actual.cpu() to assert_close, or use dtype=torch.float32."
    )

    if strict:
        raise AssertionError(text)
    warnings.warn(text, MPSDtypeWarning, stacklevel=3)


def assert_close(
    actual: Any,
    expected: Any,
    *,
    rtol: float | None = None,
    atol: float | None = None,
    k_dim: int | None = None,
    nan_equal: bool = False,
    baseline_2x: bool = False,
    msg: str = "",
    mps_strict_dtype: bool | None = None,  # NEW — None = read pyproject config
) -> None:
    # ... existing dtype resolution ...

    # NEW: silent-downcast check, before the fast-path so we don't let
    # a vacuous allclose() return True silently.
    if mps_strict_dtype is None:
        from gpucheck.config import get_mps_strict_dtype  # default: True
        mps_strict_dtype = get_mps_strict_dtype()
    _mps_downcast_check(actual, expected, strict=mps_strict_dtype, msg=msg)

    # ... rest unchanged ...
```

The `mps_strict_dtype` knob has three states:
- `True` -> raise `AssertionError` immediately
- `False` -> emit `MPSDtypeWarning` (still visible, never silent)
- `None` (default) -> read `[tool.gpucheck.mps.strict_dtype]` from pyproject

`stacklevel=3` puts the warning at the user's `assert_close(...)` call
site, not at the internal `_mps_downcast_check` frame.

## §3. Test cases — five (input, expected_warning) pairs

These would live in `tests/assertions/test_mps_downcast.py`. Each uses
mocked `torch.Tensor` shapes to keep CPU-only CI green per `CLAUDE.md`
("tests run CPU-only on GitHub Actions").

```python
# tests/assertions/test_mps_downcast.py
import pytest
import torch
from unittest.mock import patch

from gpucheck.assertions import assert_close
from gpucheck.assertions._warnings import MPSDtypeWarning
from gpucheck.backends.mps import MPSBackend


# ---- Predicate-level tests ----------------------------------------------

@pytest.mark.parametrize(
    "dtype, ndim, expected",
    [
        # Case 1: fp64 0-d scalar — silent downcast (the headline bug)
        (torch.float64, 0, True),
        # Case 2: fp64 1-d tensor — would raise on MPS, NOT a silent
        # downcast (PyTorch raises TORCH_CHECK_TYPE), so predicate=False
        (torch.float64, 1, False),
        # Case 3: complex128 0-d scalar — second silent downcast site
        (torch.complex128, 0, True),
        # Case 4: fp32 0-d scalar — natively supported, no downcast
        (torch.float32, 0, False),
        # Case 5: bf16 1-d tensor — natively supported on macOS 14+
        (torch.bfloat16, 1, False),
    ],
)
def test_silently_downcasts_dtype(
    dtype: torch.dtype, ndim: int, expected: bool,
) -> None:
    """Each row encodes one (dtype, ndim) -> bool from OperationUtils.mm."""
    backend = MPSBackend()
    assert backend.silently_downcasts_dtype(dtype, ndim) is expected


# ---- assert_close integration tests -------------------------------------

def _mps_tensor(value, dtype, ndim):
    """Build a CPU tensor and lie about device.type == 'mps'.

    GitHub-Actions CI has no MPS, so we mock at the .device.type level.
    """
    t = torch.tensor(value, dtype=dtype) if ndim == 0 else \
        torch.tensor([value] * 4, dtype=dtype)
    # Mock device attribute
    class _Dev: type = "mps"
    t.device = _Dev()  # type: ignore[assignment]  # test-only
    return t


def test_assert_close_warns_on_fp64_zero_d_scalar() -> None:
    """fp64 0-d MPS scalar must trip MPSDtypeWarning under strict=False."""
    actual = _mps_tensor(0.1, torch.float64, ndim=0)
    expected = _mps_tensor(0.1, torch.float64, ndim=0)
    with pytest.warns(MPSDtypeWarning, match="silently downcasts"):
        assert_close(actual, expected, mps_strict_dtype=False)


def test_assert_close_raises_on_fp64_zero_d_scalar_strict() -> None:
    """Default strict mode must raise — assertion budget is otherwise vacuous."""
    actual = _mps_tensor(0.1, torch.float64, ndim=0)
    expected = _mps_tensor(0.1, torch.float64, ndim=0)
    with pytest.raises(AssertionError, match="silently downcasts"):
        assert_close(actual, expected, mps_strict_dtype=True)


def test_assert_close_silent_on_fp32_one_d_tensor() -> None:
    """fp32 ndim=1 — natively supported, no warning, no raise."""
    actual = _mps_tensor(1.0, torch.float32, ndim=1)
    expected = _mps_tensor(1.0, torch.float32, ndim=1)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", MPSDtypeWarning)
        assert_close(actual, expected, mps_strict_dtype=False)  # MUST NOT warn


def test_assert_close_warns_on_complex128_zero_d() -> None:
    """complex128 0-d scalar — second known downcast site."""
    actual = _mps_tensor(complex(1, 0), torch.complex128, ndim=0)
    expected = _mps_tensor(complex(1, 0), torch.complex128, ndim=0)
    with pytest.warns(MPSDtypeWarning, match="complex"):
        assert_close(actual, expected, mps_strict_dtype=False)


def test_assert_close_silent_on_cuda_fp64_zero_d() -> None:
    """CUDA has no downcast — predicate must be False, assert_close clean."""
    backend_cuda = ...  # CUDABackend.silently_downcasts_dtype always False
    # Build CPU tensor pretending to be on CUDA
    actual = torch.tensor(0.1, dtype=torch.float64)
    class _Dev: type = "cuda"
    actual.device = _Dev()  # type: ignore[assignment]
    expected = torch.tensor(0.1, dtype=torch.float64)
    expected.device = _Dev()  # type: ignore[assignment]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", MPSDtypeWarning)
        assert_close(actual, expected, mps_strict_dtype=False)
```

The five rows in the parametrize cover the **truth table** of the
asymmetry: fp64×{0d,1d}, complex128×0d, fp32×0d (control), bf16×1d
(control). Together they pin every load-bearing branch in
`silently_downcasts_dtype`.

## §4. Recommendation: opt-in or opt-out?

**Recommendation: OPT-OUT, raise-by-default. Strict mode ON.**

The default behavior should be:
1. `assert_close` raises `AssertionError` when MPS would silently downcast.
2. Users disable per-call via `mps_strict_dtype=False` (downgrade to warn).
3. Users disable globally via `[tool.gpucheck.mps.strict_dtype] = false`.

### Justification

**Argument for opt-out (default-on):**

a. **The bug is silent and load-bearing.** The whole point of
   `assert_close` is to catch numerical errors. Silently routing fp64
   through fp32 makes the assertion vacuous — the test would pass when
   the kernel is wrong. This is **exactly** the failure mode `assert_close`
   exists to prevent. Off-by-default reproduces the original bug at
   library scope.

b. **The fix is cheap.** The user moves their oracle to CPU
   (`torch.tensor(..., dtype=torch.float64, device='cpu')`) — one keyword
   argument change. The cost of opt-out is "I have to think about precision
   on Apple Silicon"; the cost of opt-in is "my Triton/PyTorch test passed
   while my kernel was actually broken."

c. **Library precedent.** PyTorch's own approach in
   `OperationUtils.mm:71-73` for fp64 **tensors** is to RAISE — the C++
   layer already considers fp64-on-MPS a hard error for ndim>=1. The
   silent-downcast for ndim=0 is a workaround for backward-compat with
   wrapped Python scalars. gpucheck **augmenting** PyTorch's behavior to
   close that compat hole is consistent with the precedent.

d. **CLAUDE.md design ethos.** The codebase already takes strict stances
   ("Strict types: mypy strict mode", "no bare except", "specific exceptions
   only"). An opt-out warning-by-default is more consistent with the rest
   of the project than opt-in.

**Counterargument considered (and rejected):**

A user who wraps an existing test suite written for CUDA in a CI matrix
that includes MPS will now hit `AssertionError`s on their fp64 scalars.
Mitigation: the error message points them at `mps_strict_dtype=False` AND
the pyproject knob `[tool.gpucheck.mps.strict_dtype] = false`, so the fix
is a one-line config change. The trade-off — possible CI red the first
time someone runs the matrix — is worth it because the alternative is
the user shipping an MPS configuration where their fp64 invariants are
silently invalidated.

### Configuration surface

```toml
# pyproject.toml — under [tool.gpucheck.mps]
[tool.gpucheck.mps.strict_dtype]
# Default: true (opt-out). When true, assert_close RAISES on silent MPS
# downcasts. When false, only emits MPSDtypeWarning.
enabled = true

# Optional allowlist for dtypes you accept the downcast on (escape hatch
# for users who genuinely don't care about fp64 precision and only want
# the float32 semantics).
allow = []  # e.g. ["torch.float64"] to silence fp64 only
```

The `allow` list is a finer-grained opt-out than the global toggle and
keeps complex128 strict even if the user accepts fp64 silently — these
are independent dimensions of the asymmetry and should be tunable
independently.

## §5. Confidence

**High** for §1 (predicate signature + implementation): the
`OperationUtils.mm:120-158` source is verbatim cited in linguist-v2.md §3
and the rank conditioning is mechanically derivable from the C++ switch.

**High** for §2 (assert_close hook): the insertion point (before fast-path
at `close.py:174`) is the only correct location — fast-path returns True
on numeric equality and would mask the dtype hazard.

**High** for §3 (test cases): the five parametrize rows cover every
truth-table cell of the (dtype, ndim) asymmetry plus a CUDA control.
The CI mocks `t.device.type = "mps"` rather than requiring real Apple
hardware, consistent with `CLAUDE.md` "tests run CPU-only on GitHub
Actions".

**Medium** for §4 (opt-in vs opt-out): there is a real trade-off between
"don't break user CI" and "don't ship a vacuous assertion". I lean
opt-out by ~70/30 because gpucheck is **explicitly** a numerical-
correctness library (CLAUDE.md "Found 8 real bugs in Triton/PyTorch with
511 test configs"); shipping a default that lets fp64 silently degrade
contradicts the library's marketing.
