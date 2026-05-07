---
specialist: research-linguist
slug: v1.0
round: 2
started: 2026-05-01T04:10:00Z
completed: 2026-05-01T04:35:00Z
tool_calls_count: 18
new_citations_count: 9
confidence: high
---

# Linguist Round 2 — `torch.mps.*` type-system contracts

The Round 1 adversary correctly flagged that the Apple MSL spec was
REPORTED-NOT-VERIFIED. This round shifts focus from the spec to the **PyTorch
source-of-truth**: the Python wrappers in `torch/mps/`, the type stubs in
`torch/_C/__init__.pyi.in`, and the C++ runtime gates in
`aten/src/ATen/mps/` and `aten/src/ATen/native/mps/`. Every claim below is
backed by a fresh file:line citation that does not appear in `linguist.md` or
any other Round 1 evidence.

## §1. Public-API surface — verbatim signatures (PyTorch `main`, retrieved 2026-05-01)

### 1.1 `torch/mps/__init__.py`
Every public callable in the module, with its source-stated signature:

| Symbol | Signature (verbatim from source) | File:line |
|---|---|---|
| `device_count` | `def device_count() -> int` | `torch/mps/__init__.py:21` |
| `synchronize` | `def synchronize() -> None` | `torch/mps/__init__.py:25` |
| `get_rng_state` | `def get_rng_state(device: int \| str \| torch.device = "mps") -> Tensor` | `torch/mps/__init__.py:29` |
| `set_rng_state` | `def set_rng_state(new_state: Tensor, device: int \| str \| torch.device = "mps") -> None` | `torch/mps/__init__.py:39` |
| `manual_seed` | `def manual_seed(seed: int) -> None` | `torch/mps/__init__.py:50` |
| `seed` | `def seed() -> None` | `torch/mps/__init__.py:62` |
| `empty_cache` | `def empty_cache() -> None` | `torch/mps/__init__.py:66` |
| `set_per_process_memory_fraction` | `def set_per_process_memory_fraction(fraction) -> None` (note: untyped param) | `torch/mps/__init__.py:72` |
| `current_allocated_memory` | `def current_allocated_memory() -> int` | `torch/mps/__init__.py:103` |
| `driver_allocated_memory` | `def driver_allocated_memory() -> int` | `torch/mps/__init__.py:114` |
| `recommended_max_memory` | `def recommended_max_memory() -> int` | `torch/mps/__init__.py:125` |
| `compile_shader` | `def compile_shader(source: str)` (note: no return annotation) | `torch/mps/__init__.py:135` |
| `load_metallib` | `def load_metallib(source)` (note: BOTH untyped) | `torch/mps/__init__.py:160` |
| `is_available` | `def is_available() -> bool` | `torch/mps/__init__.py:198` |

### 1.2 `torch/mps/event.py` — `class Event`

| Method | Signature (verbatim) | File:line |
|---|---|---|
| `__init__` | `def __init__(self, enable_timing: bool = False) -> None` | `torch/mps/event.py:14` |
| `record` | `def record(self) -> None` | `torch/mps/event.py:23` |
| `wait` | `def wait(self) -> None` | `torch/mps/event.py:27` |
| `query` | `def query(self) -> bool` | `torch/mps/event.py:31` |
| `synchronize` | `def synchronize(self) -> None` | `torch/mps/event.py:35` |
| `elapsed_time` | `def elapsed_time(self, end_event: "Event") -> float` | `torch/mps/event.py:41` |

### 1.3 `torch/mps/profiler.py`

`ProfilerMode = Literal["interval", "event", "interval,event"]`
(`torch/mps/profiler.py:9` — load-bearing literal type, but normalisation at
`profiler.py:34` does `.lower().replace(" ", "")` which lets non-conforming
strings through to the C++ layer if they happen to lowercase to one of the
three).

| Function | Signature | File:line |
|---|---|---|
| `start` | `def start(mode: ProfilerMode = "interval", wait_until_completed: bool = False) -> None` | `torch/mps/profiler.py:14` |
| `stop` | `def stop() -> None` | `torch/mps/profiler.py:38` |
| `profile` | `def profile(mode: ProfilerMode = "interval", wait_until_completed: bool = False) -> Iterator[None]` | `torch/mps/profiler.py:43` |
| `is_metal_capture_enabled` | `def is_metal_capture_enabled() -> bool` | `torch/mps/profiler.py:64` |
| `is_capturing_metal` | `def is_capturing_metal() -> bool` | `torch/mps/profiler.py:69` |
| `metal_capture` | `def metal_capture(fname: str) -> Iterator[None]` | `torch/mps/profiler.py:74` |

## §2. Type-stub bugs in PyTorch's MPS surface

Stub file: `torch/_C/__init__.pyi.in:2072–2096` (the entire MPS native-binding
stub block).

### 2.1 BUG #1 — Missing stubs for `_mps_compileShader` and `_mps_loadMetallib*`

The Python wrapper `torch.mps.compile_shader` calls `torch._C._mps_compileShader`
at `torch/mps/__init__.py:154`, but the stub block at
`torch/_C/__init__.pyi.in:2072–2096` does **not** declare it. Same for
`_mps_loadMetalllib` (note the triple-`l` typo carried in the binding name —
called from `torch/mps/__init__.py:181`) and `_mps_loadMetallibFromPath`
(`torch/mps/__init__.py:185`). Result: `mypy --strict` against gpucheck code
that imports `torch.mps.compile_shader` will hit `Returns Any` because the
called binding is unknown.

### 2.2 BUG #2 — Two functions in `torch.mps` are partially or fully untyped

`torch/mps/__init__.py:72`:
```python
def set_per_process_memory_fraction(fraction) -> None:
```
The `fraction` parameter has no annotation. The runtime guard at line 102
(`if not isinstance(fraction, float)`) means the contract is "must be `float`,
not even `int`". A correct stub would be `fraction: float`.

`torch/mps/__init__.py:160`:
```python
def load_metallib(source):
```
Both `source` and the return type are missing. The body branches on
`isinstance(source, (bytes, bytearray))` vs `isinstance(source, (str, os.PathLike))`,
so the correct annotation is
`source: bytes | bytearray | str | os.PathLike[str]` returning whatever
`_mps_loadMetalllib` / `_mps_loadMetallibFromPath` returns (also unstubbed —
see §2.1).

### 2.3 BUG #3 — `Event.elapsed_time` units are correct, but precision claim is unverifiable from the stub alone

Native impl: `aten/src/ATen/mps/MPSEvent.mm:221–238` —
```cpp
double MPSEventPool::elapsedTime(id_t start_event_id, id_t end_event_id) {
  ...
  const uint64_t start_time = start_event->getCompletionTime();
  const uint64_t end_time = end_event->getCompletionTime();
  ...
  return double(end_time - start_time) * 1e-6;
}
```
The `1e-6` factor converts **nanoseconds** (`mach_absolute_time`-derived) to
**milliseconds**. The Python stub `_mps_elapsedTimeOfEvents(...) -> _float`
(`torch/_C/__init__.pyi.in:2092`) and the wrapper return-type `float`
(`torch/mps/event.py:41`) are both correct on **return type** but neither
documents the **unit**. Worse: there is no precision guarantee in the
type system, so a caller can't tell from the signature whether they are
being handed microseconds or milliseconds. Compare CUDA's
`torch.cuda.Event.elapsed_time` which has the same bug.

### 2.4 Asymmetry between `getMPSDataType` and `getMPSScalarType`

This is a finding the Round 1 work did not catch and which has a load-bearing
consequence for assertions:

- `aten/src/ATen/native/mps/OperationUtils.mm:49–88` — `getMPSDataType` for
  **tensors** raises `TORCH_CHECK_TYPE(false, ...)` for `kDouble` and any
  unmapped dtype (so `kComplexDouble` falls through to the `default:` and
  errors out as well — line 86).
- `aten/src/ATen/native/mps/OperationUtils.mm:120–158` — `getMPSScalarType`
  for **0-d scalars** silently downcasts: `case ScalarType::Double:` falls
  through to `case ScalarType::Float:` returning `MPSDataTypeFloat32` (line
  124–127), and `case ScalarType::ComplexDouble:` falls through to
  `case ScalarType::ComplexFloat:` returning `MPSDataTypeComplexFloat32`
  (line 144–147).

So a 0-d `torch.tensor(1.0, dtype=torch.float64)` does NOT raise on MPS — it
silently runs as fp32. A 1-d `torch.tensor([1.0], dtype=torch.float64)`
raises. **This precision loss is invisible at the Python type level**
because in both cases `tensor.dtype == torch.float64`. gpucheck's
`assert_close` could be fooled if the user expects fp64 reference precision
but is silently being given fp32.

## §3. dtype × MPS support matrix (verified against PyTorch `main`, 2026-05-01)

Sources combined: `getMPSDataType` switch
(`aten/src/ATen/native/mps/OperationUtils.mm:49–88`), `getMPSScalarType`
switch (`OperationUtils.mm:120–158`), `EmptyTensor.cpp:45` runtime check,
`supportedFloatingType` predicate (`OperationUtils.h:612–614`),
`supportedFloatingOrComplexType` (`OperationUtils.h:620–625`).

| `torch.dtype` | C10 `ScalarType` | MPS tensor support | MPS 0-d scalar support | Notes |
|---|---|---|---|---|
| `torch.float32` | `Float` | yes (`MPSDataTypeFloat32`) | yes | The native fast path. |
| `torch.float64` | `Double` | **NO — raises `TORCH_CHECK_TYPE`** at `OperationUtils.mm:71–73` and at `EmptyTensor.cpp:45` | **YES — silently downcast to fp32** at `OperationUtils.mm:124–127` | The asymmetry of §2.4. fp64 tensors error; fp64 scalars run as fp32. |
| `torch.float16` | `Half` | yes (`MPSDataTypeFloat16`) | yes | All Apple Silicon supports half. |
| `torch.bfloat16` | `BFloat16` | yes on macOS 14+ (`MPSDataTypeBFloat16`) | yes on macOS 14+ | Comment at `OperationUtils.h:611` "MPS yet to support double types, but starting from MacOS 14, supports bfloat16". On macOS 13, ops like `add` work but other paths can fail at the Metal-shader-template level — not a clean Python-level error. |
| `torch.complex32` (= `kComplexHalf`) | `ComplexHalf` | yes (`MPSDataTypeComplexFloat16`) | yes | Op coverage is partial — many kernels skip complex. |
| `torch.complex64` | `ComplexFloat` | yes (`MPSDataTypeComplexFloat32`) | yes | Op coverage is partial. |
| `torch.complex128` | `ComplexDouble` | **NO — raises** (default: branch in `getMPSDataType` at `OperationUtils.mm:86`, plus the `kComplexDouble` term in the `EmptyTensor.cpp:45` check) | **YES — silently downcast to complex64** at `OperationUtils.mm:144–147` | Same asymmetry as fp64. |
| `torch.int8` | `Char` | yes (`MPSDataTypeInt8`) | yes | |
| `torch.int16` | `Short` | yes (`MPSDataTypeInt16`) | yes | |
| `torch.int32` | `Int` | yes (`MPSDataTypeInt32`) | yes | |
| `torch.int64` | `Long` | yes (`MPSDataTypeInt64`) | yes | Mapping exists but per Apple docs Metal historically lacked native int64; perf is degraded for many int64 ops. |
| `torch.uint8` | `Byte` | yes (`MPSDataTypeUInt8`) | yes | |
| `torch.uint16/32/64` | unsigned | yes (`MPSDataTypeUInt16/32/64`) | yes | Modern additions; check if dtype is even importable on caller's PyTorch. |
| `torch.bool` | `Bool` | yes (`MPSDataTypeBool`) | yes | |
| `torch.float8_e4m3fn` | `Float8_e4m3fn` | **NO** — falls through to default branch in `getMPSDataType` (`OperationUtils.mm:86`) | **NO** — same | Apple Silicon has no FP8 hardware. gpucheck's `MPSBackend.arch_info` already sets `supports_fp8=False` (`src/gpucheck/backends/mps.py:209`). |
| `torch.float8_e5m2` | `Float8_e5m2` | **NO** — same | **NO** — same | Same. |

**Key cross-cuts:**
1. `supportedFloatingType` (`OperationUtils.h:612–614`) is the load-bearing
   predicate for many op-level early returns; it accepts ONLY
   `{kFloat, kHalf, kBFloat16}`. So even though `getMPSDataType` would map
   `kHalf`/`kBFloat16` for elementwise, **kernels that gate on
   `supportedFloatingType` will reject `complex` and `int*` paths** that the
   dtype-mapping function would happily accept. Callers don't see this in
   the Python-level dtype.
2. `supportedFloatingOrComplexType` (`OperationUtils.h:620–625`) extends the
   set with `{kComplexFloat, kComplexHalf}`. This is the predicate used by
   linalg ops; complex128 is excluded.

## §4. Naming conventions and language-idiom audit

### 4.1 Drift: parameter name vs docstring on `Event.elapsed_time`
`torch/mps/event.py:41` — the parameter is `end_event` but the C++-binding
parameter ordering convention used elsewhere (e.g. CUDA `Event.elapsed_time`)
calls it `other`. The doc string says "after the event was recorded and
before the end_event was recorded" — uses `end_event`, which matches code.
No drift here, just naming idiosyncrasy worth flagging when writing a
gpucheck Protocol that wraps both CUDA and MPS Events: their parameter names
disagree.

### 4.2 Triple-`l` typo carried forward
The native binding name is `_mps_loadMetalllib` (`MetaLLLib` — three `l`s).
This is called from `torch/mps/__init__.py:181`. There is also
`_mps_loadMetallibFromPath` with the correct two-`l` spelling. A pyi stub
that fixes the typo would break the binding lookup. The Python public name
is `load_metallib` (correct spelling). gpucheck must NOT fix the typo when
shadowing.

### 4.3 Name-mangling escape hatch in `Event.elapsed_time`
`torch/mps/event.py:45`: `return torch._C._mps_elapsedTimeOfEvents(self.__eventId, end_event.__eventId)`
— the `end_event.__eventId` access depends on Python name mangling
(`_Event__eventId`). This is fine within the class, but it means a
**Protocol cannot structurally type-check Event** by demanding a public
`event_id` attribute — the class deliberately makes the id name-mangled.
gpucheck's `EventTimer` Protocol (`src/gpucheck/backends/_protocol.py:18–32`)
is correctly designed: it does not expose ids, only `elapsed_ms`.

### 4.4 Type-hint escape hatches in MPS code
- `torch/mps/__init__.py:13`:
  `_default_mps_generator: torch._C.Generator = None  # type: ignore[assignment]`
  — sentinel-None pattern with a `type: ignore`. Common in PyTorch but a
  hint that the typing here is permissive.
- `torch/mps/profiler.py:35,40,46,66,71,77,80`: every `torch._C` call is
  decorated with `# type: ignore[attr-defined, no-any-return]`. The
  profiler bindings are entirely unstubbed.

### 4.5 Untyped public API in profiler module
`torch.mps.profiler.is_metal_capture_enabled` and `is_capturing_metal` both
declare `-> bool` but their bodies return whatever
`torch._C._mps_isCaptureEnabled()` and `_mps_isCapturing()` return. Stubs at
`torch/_C/__init__.pyi.in:2093–2094` say `_bool`. Match — but the
load-bearing conversion to a real `bool` is missing in case the C++ binding
ever returns a numpy-like.

## §5. Recommendations: what gpucheck's Backend Protocol should override

gpucheck has a private Protocol in `src/gpucheck/backends/_protocol.py:18–32`
(EventTimer) and `:36–73` (Backend). Round 1 already exposed
`elapsed_ms: float`. Round 2 type-system findings let me sharpen the
Protocol so mypy-strict catches bugs before they reach the runtime.

### 5.1 Add an explicit dtype-support contract

Append to `Backend`:
```python
unsupported_dtypes: frozenset[torch.dtype]
"""dtypes this backend will reject. MPS includes float64, complex128,
float8_e4m3fn, float8_e5m2."""

silently_downcast_dtypes: frozenset[torch.dtype]
"""dtypes this backend SILENTLY accepts via downcast. MPS includes
float64 and complex128 IFF used as 0-d scalars (per
aten/src/ATen/native/mps/OperationUtils.mm:124-127, 144-147).
gpucheck.assertions.assert_close MUST consult this set when picking
the reference precision; otherwise the reference will be silently
downcast and the assertion becomes vacuous."""
```

Implementation sketch for `MPSBackend` (`src/gpucheck/backends/mps.py`):
```python
unsupported_dtypes: ClassVar[frozenset[torch.dtype]] = frozenset({
    torch.float64,        # tensor allocation rejects (EmptyTensor.cpp:45)
    torch.complex128,     # same
    torch.float8_e4m3fn,  # falls through default branch in getMPSDataType
    torch.float8_e5m2,    # same
})
silently_downcast_dtypes: ClassVar[frozenset[torch.dtype]] = frozenset({
    torch.float64,        # 0-d scalar -> fp32 (OperationUtils.mm:124-127)
    torch.complex128,     # 0-d scalar -> complex64 (OperationUtils.mm:144-147)
})
```
This makes the dtype-asymmetry of §2.4 a first-class fact in the type system.

### 5.2 Tighten `EventTimer` to expose a dimension token

The unit confusion from §2.3 means callers who consume `elapsed_ms`
generically across CUDA and MPS can be off by 1000x. Add:
```python
class EventTimer(Protocol):
    elapsed_ms: float  # already present
    units: ClassVar[Literal["ms"]] = "ms"
```
A `Literal["ms"]` type declaration is load-bearing documentation and lets
mypy reject any backend that returns `elapsed_us` or `elapsed_ns`.

### 5.3 Strict-typing the `compile_shader` / `load_metallib` re-exports

If gpucheck ever wraps these for shader-level kernel testing, the
gpucheck-side wrappers should NOT inherit PyTorch's bare `compile_shader(source: str)`
no-return-type signature. Provide:
```python
class MPSCompiledLibrary(Protocol):
    def __getattr__(self, kernel_name: str) -> Callable[..., None]: ...

def compile_shader(source: str) -> MPSCompiledLibrary: ...
def load_metallib(source: bytes | bytearray | str | os.PathLike[str]) -> MPSCompiledLibrary: ...
```
This gives mypy strict mode something to bite on instead of `Any`.

### 5.4 Mark fp64-on-MPS as a load-bearing xfail in the test matrix

Per Round 1 §1.3 ("MPS-correct" is FP32-vs-FP32 not FP32-vs-FP64), and
Round 2 §3 (fp64 tensor allocation raises but fp64 scalar silently
downcasts), gpucheck's MPS test matrix **must** xfail any test that
parametrizes `dtype=torch.float64` on MPS. Decorator suggestion:
```python
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.skipif(
    backend.name == "mps" and dtype in backend.unsupported_dtypes,
    reason=f"{dtype} not supported on MPS — see EvidenceLog linguist-v2.md §3",
)
```

## §6. Confidence

**High** for the dtype matrix, type-stub gaps, and asymmetry findings —
all are quoted directly from PyTorch source on `main` as of 2026-05-01.
**Medium** for the Protocol override recommendations — they are
type-system claims (will mypy catch X?) that depend on user code shapes;
they are sound under standard gpucheck call patterns but should be
validated by writing a small mypy harness in CI before being declared
load-bearing.

## Citations new in Round 2 (≥3 required, 9 actual)

1. `torch/mps/__init__.py:72,160` — untyped `fraction` and `source` parameters
2. `torch/mps/event.py:41,45` — `end_event` parameter and name-mangled `__eventId` access
3. `torch/_C/__init__.pyi.in:2072–2096` — full MPS native binding stub block
4. `aten/src/ATen/mps/EmptyTensor.cpp:45` — `TORCH_CHECK_TYPE(dtype != kDouble && dtype != kComplexDouble, ...)` runtime gate
5. `aten/src/ATen/native/mps/OperationUtils.mm:49–88` — `getMPSDataType` switch (tensor path)
6. `aten/src/ATen/native/mps/OperationUtils.mm:120–158` — `getMPSScalarType` switch (scalar downcast path)
7. `aten/src/ATen/native/mps/OperationUtils.h:612–625` — `supportedFloatingType` and `supportedFloatingOrComplexType` predicates
8. `aten/src/ATen/mps/MPSEvent.mm:221–238` — `elapsedTime` returning `(end - start) * 1e-6` ms
9. `torch/mps/profiler.py:9,34` — `ProfilerMode = Literal["interval", "event", "interval,event"]` and the `.lower().replace(" ", "")` normalisation that lets variants through

(Round 1 only had: `linguist.md` cited 3 PyTorch issue threads (#181936,
#170837, #142836, #137001) and the Apple MSL spec PDF — none of the above
9 source-level citations appear in Round 1.)
