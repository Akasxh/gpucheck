# gpucheck

**pytest for GPU kernels.**

[![PyPI](https://img.shields.io/pypi/v/gpucheck)](https://pypi.org/project/gpucheck/)
[![Python](https://img.shields.io/pypi/pyversions/gpucheck)](https://pypi.org/project/gpucheck/)
[![License](https://img.shields.io/github/license/Akasxh/gpucheck)](https://github.com/Akasxh/gpucheck/blob/main/LICENSE)
[![CI](https://github.com/Akasxh/gpucheck/actions/workflows/ci.yml/badge.svg)](https://github.com/Akasxh/gpucheck/actions/workflows/ci.yml)

GPU kernel testing is painful. You write a CUDA kernel, eyeball `torch.allclose` with magic tolerances, and pray it works on a different GPU architecture. gpucheck is a pytest plugin that gives you dtype-aware assertions, parametric testing across dtypes/shapes/devices, CUDA-event benchmarking, shape fuzzing, and memory leak detection -- all from decorators and fixtures you already know how to use.

We tested gpucheck against Triton tutorials and PyTorch CUDA ops with **511 test configurations** and found **8 real bugs**, including a **83% error in Triton's layer norm** for non-power-of-2 dimensions ([triton#9838](https://github.com/triton-lang/triton/issues/9838)) and **FP16 accumulation drift in the tutorial matmul** ([triton#9839](https://github.com/triton-lang/triton/issues/9839)).

```python
import torch
import pytest
from gpucheck import assert_close, dtypes, shapes, devices

@pytest.mark.gpu
@dtypes("float16", "bfloat16", "float32")
@shapes((128, 128), (512, 512), (1024, 1024))
@devices("cuda:0")
def test_relu_kernel(dtype, shape, device):
    dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    x = torch.randn(shape, dtype=dtype, device=device)
    result = torch.relu(x)
    expected = torch.clamp(x, min=0)
    assert_close(result, expected)  # tolerances auto-selected by dtype
```

## Installation

```bash
pip install gpucheck
```

Optional dependencies for specific backends:

```bash
pip install gpucheck[torch]       # PyTorch + CUDA
pip install gpucheck[hypothesis]  # Property-based shape fuzzing
pip install gpucheck[all]         # Everything
```

For development:

```bash
pip install gpucheck[dev]
```

## Step by step usage guide

This section walks through everything from installation to running your first GPU kernel test, writing benchmarks, detecting memory leaks, and setting up shape fuzzing. Every code block here has been tested on a real NVIDIA GeForce GTX 1650 (Turing, SM75, 4GB VRAM) running PyTorch 2.11.0 with CUDA 13.0.

### 1. Install and verify

Start by installing gpucheck with PyTorch support and verifying your GPU is detected:

```bash
pip install gpucheck[torch]
```

```python
python -c "
from gpucheck import detect_gpu, gpu_available
print(f'GPU available: {gpu_available()}')
gpu = detect_gpu()
if gpu:
    print(f'Device: {gpu.name}')
    print(f'Compute capability: {gpu.compute_capability}')
    print(f'Memory: {gpu.memory_total_mb}MB')
"
```

On our test machine this prints:

```
GPU available: True
Device: NVIDIA GeForce GTX 1650
Compute capability: (7, 5)
Memory: 3715MB
```

### 2. Write your first test

Create a file called `test_my_kernel.py`:

```python
import torch
import pytest
from gpucheck import assert_close, dtypes

@pytest.mark.gpu
@dtypes("float16", "float32")
def test_relu(dtype):
    dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    x = torch.randn(256, 256, dtype=dtype, device="cuda")
    result = torch.relu(x)
    expected = torch.clamp(x, min=0)
    assert_close(result, expected)
```

Run it:

```bash
pytest test_my_kernel.py -v
```

This generates two test variants automatically, one for float16 and one for float32. Each variant uses the correct tolerance for its dtype without you having to look anything up.

Note that the `@dtypes` decorator passes dtype names as strings to keep torch from being imported at collection time. Convert them with `getattr(torch, dtype)` inside the test body.

### 3. Parametric testing across dtypes, shapes, and devices

Stack decorators to test across the full matrix:

```python
import torch
import pytest
from gpucheck import assert_close, dtypes, shapes, devices

@pytest.mark.gpu
@dtypes("float16", "bfloat16", "float32")
@shapes((64, 64), (128, 128), (7, 13))
@devices("cuda:0")
def test_softmax(dtype, shape, device):
    dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    x = torch.randn(shape, dtype=dtype, device=device)
    result = torch.softmax(x, dim=-1)
    expected = torch.softmax(x.float(), dim=-1).to(dtype)
    assert_close(result, expected)
```

This generates 3 dtypes x 3 shapes x 1 device = 9 test variants from a single function. The `(7, 13)` shape is important because non-tile-aligned dimensions catch bugs that power-of-2 shapes miss entirely.

You can also use the all-in-one decorator:

```python
from gpucheck.decorators import parametrize_gpu

@parametrize_gpu(
    dtypes=("float16", "bfloat16"),
    shapes=((128, 128), (512, 512)),
    devices=("cuda:0",),
)
def test_kernel(dtype, shape, device):
    ...
```

Predefined groups are available for common combinations: `FLOAT_DTYPES`, `HALF_DTYPES`, `FP8_DTYPES`, `ALL_DTYPES`, `SMALL_SHAPES`, `MEDIUM_SHAPES`, `LARGE_SHAPES`, `EDGE_SHAPES`.

### 4. Matmul tolerance scaling with k_dim

When testing matrix multiplication or any operation that accumulates over a reduction dimension, floating point errors grow proportionally to `sqrt(k)`. Pass `k_dim` to scale the tolerance automatically:

```python
import torch
from gpucheck import assert_close

a = torch.randn(128, 4096, device="cuda", dtype=torch.float16)
b = torch.randn(4096, 128, device="cuda", dtype=torch.float16)
result = torch.mm(a, b)
expected = torch.mm(a.float(), b.float()).half()
assert_close(result, expected, k_dim=4096)
```

Without `k_dim`, this test would fail because the default float16 tolerance (atol=1e-2) does not account for the accumulation over 4096 elements. With `k_dim=4096`, the tolerance scales by `sqrt(4096/128) = 5.66x`, which matches the CUTLASS error accumulation model.

For FlashAttention-style testing, use `baseline_2x=True` to double the default tolerances:

```python
assert_close(result, expected, baseline_2x=True)
```

### 5. GPU benchmarking

The `gpu_benchmark` fixture measures kernel performance using CUDA events, which are more accurate than wall-clock timing. It handles warmup iterations, L2 cache flushing between runs, and statistical outlier removal automatically:

```python
def test_matmul_perf(gpu_benchmark):
    a = torch.randn(256, 256, device="cuda", dtype=torch.float32)
    b = torch.randn(256, 256, device="cuda", dtype=torch.float32)

    result = gpu_benchmark(torch.mm, a, b)

    print(f"median={result.median:.3f}ms, std={result.std:.4f}ms")
    assert result.median < 1.0  # ms
```

On our GTX 1650, a 256x256 float32 matmul takes about 0.055ms median with 0.005ms standard deviation. The `BenchmarkResult` provides `median`, `mean`, `std`, `min`, `max`, `p5`, `p25`, `p75`, `p95`, and `raw_times`.

You can configure warmup and rounds via CLI flags:

```bash
pytest --gpu-benchmark-warmup=20 --gpu-benchmark-rounds=200
```

### 6. Shape fuzzing

The shape fuzzer generates tensor dimensions specifically designed to trigger GPU kernel bugs. These are not random shapes -- they are adversarial, prioritized by how likely they are to expose real issues:

```python
from gpucheck.fuzzing.shapes import fuzz_shapes

shapes = fuzz_shapes(ndim=2, max_size=4096, n=50, seed=42)
for shape in shapes:
    x = torch.randn(shape, device="cuda")
    result = my_kernel(x)
    reference = reference_impl(x)
    assert_close(result, reference)
```

The shapes come in six categories ranked by bug-finding probability: degenerate shapes (zeros, ones), non-tile-aligned (not divisible by 32/64/128), prime dimensions (7, 13, 31, 127, 257), power-of-2 boundaries (127, 128, 129), large (2048, 4096, 8192), and mixed asymmetric combinations. Most of the bugs we found in Triton tutorials were caught by non-power-of-2 shapes -- dimensions like 17 and 127 that hit tile boundary edge cases.

For property-based testing with Hypothesis:

```python
from hypothesis import given
from gpucheck.fuzzing.shapes import ShapeStrategy

@given(shape=ShapeStrategy(ndim=2, max_size=512))
def test_kernel_any_shape(shape):
    x = torch.randn(shape, device="cuda")
    result = my_kernel(x)
    assert result.shape == shape
```

### 7. Memory leak detection

Track GPU memory across a test to catch allocations that are never freed:

```python
# Using the pytest fixture
def test_no_leak(memory_tracker):
    x = torch.randn(1024, 1024, device="cuda")
    result = my_kernel(x)
    del x, result
    torch.cuda.empty_cache()
    report = memory_tracker.stop()
    assert not report.has_leak

# Using the context manager
from gpucheck.sanitizers import memory_guard

def test_memory_bounded():
    with memory_guard(threshold_bytes=10 * 1024 * 1024) as report:
        run_kernel()
    assert report.leaked_mb < 1.0
```

The memory tracker uses `torch.cuda.memory_stats()` when available and falls back to `pynvml` for process-level tracking. The leak threshold is 1MB by default, which filters out allocator fragmentation noise.

### 8. Architecture gating

Skip tests that require specific GPU features:

```python
from gpucheck.arch.compatibility import require_arch, require_capability

@require_arch("Ampere", "Hopper")
def test_bf16_matmul():
    ...

@require_capability(8, 9)  # Ada Lovelace+
def test_fp8_kernel():
    ...
```

gpucheck detects your GPU architecture automatically. On our GTX 1650 (Turing, SM75), tests marked `@require_arch("Hopper")` are skipped with a clear message explaining why. Tests marked `@require_arch("Turing")` run normally.

gpucheck correctly handles the GTX 16xx edge case where the GPU shares SM75 (Turing architecture) with RTX 20xx cards but lacks tensor cores. If your kernel requires tensor cores, use `require_capability(7, 5)` along with a tensor core check rather than `require_arch("Turing")` alone.

Supported architectures: Volta (SM70), Turing (SM75), Ampere (SM80/86), Ada (SM89), Hopper (SM90), Blackwell (SM100 datacenter / SM120 consumer). You can use `@require_arch("Blackwell")` to match any Blackwell GPU, or `@require_arch("Blackwell-DC")` to match only datacenter Blackwell.

### 9. Performance regression detection

The analysis module includes statistical regression detection using the Mann-Whitney U test:

```python
from gpucheck.analysis.regression import detect_regression

baseline = [0.85, 0.87, 0.84, 0.86, 0.85, 0.88, 0.84, 0.86, 0.85, 0.87]
current  = [0.95, 0.96, 0.94, 0.97, 0.95, 0.98, 0.94, 0.96, 0.95, 0.97]

result = detect_regression(current, baseline, threshold=0.05, min_effect=1.1)
print(result.description)
# REGRESSION DETECTED: +12.0% (p=0.0001, Cohen's d=4.21)
```

The roofline analysis module classifies kernels as memory-bound, compute-bound, or balanced:

```python
from gpucheck.analysis.roofline import compute_roofline, classify_bottleneck, GPUSpecs

specs = GPUSpecs(peak_flops=3.5e12, peak_bandwidth=128e9)  # GTX 1650
point = compute_roofline(timing_results, flops=2*M*N*K, bytes_accessed=bytes, gpu_specs=specs)
print(classify_bottleneck(point))  # "memory_bound" or "compute_bound"
```

## Tolerance table

Default tolerances used by `assert_close` when no explicit `atol`/`rtol` is provided:

| dtype | atol | rtol |
|---|---|---|
| `float64` | `1e-10` | `1e-7` |
| `float32` | `1e-4` | `1e-4` |
| `tf32` | `5e-4` | `5e-4` |
| `float16` | `1e-2` | `1e-2` |
| `bfloat16` | `5e-2` | `5e-2` |
| `float8_e4m3fn` | `0.125` | `0.125` |
| `float8_e5m2` | `0.25` | `0.25` |

For matmul-like operations, pass `k_dim` to scale `atol` by `sqrt(k_dim / 128)`. Override per-project in `pyproject.toml`:

```toml
[tool.gpucheck.tolerances]
float16 = {atol = 2e-3, rtol = 2e-3}
bfloat16 = {atol = 3e-2, rtol = 3e-2}
```

When comparing tensors of different precisions (for example a float16 kernel output against a float32 reference), gpucheck automatically uses the tolerance of the lower-precision dtype, since that is the precision-limiting factor. This means the order of arguments does not matter.

## Bugs found

gpucheck's shape fuzzing and dtype-aware testing found these real bugs in widely-used GPU kernels:

| Bug | Severity | Error | Root Cause |
|-----|----------|-------|-----------|
| [Triton layer norm variance padding](https://github.com/triton-lang/triton/issues/9838) | HIGH | **83% relative error** at n_cols=17 | Zero-padded positions inject mean^2 into variance |
| [Triton matmul FP16 index wrapping](https://github.com/triton-lang/triton/issues/9839) | MEDIUM-HIGH | **0.125 abs error** at K=8192 | Modular `% M` wrapping reads wrong data into accumulator |
| cuFFT precision at N>=4096 | MEDIUM | **1.26% relative error** | Error scales O(sqrt(N)) instead of O(log(N)) |
| `torch.baddbmm` FP16 silent overflow | HIGH | **NaN output** with no warning | `alpha=1000` causes intermediate overflow to Inf |
| `torch.bmm` FP32 large-K | MEDIUM | **2.1e-3 relative error** | Accumulation path less careful than FP16/BF16 |

Most of these bugs were caught by non-power-of-2 shapes -- dimensions like 17, 127, 255 that hit tile boundary edge cases. This is exactly what `fuzz_shapes()` generates.

See [`examples/triton_layernorm_bug.py`](examples/triton_layernorm_bug.py) and [`examples/triton_matmul_bug.py`](examples/triton_matmul_bug.py) for standalone reproducers.

## Tested hardware and software

gpucheck has been validated on the following hardware and software stack:

**Hardware tested:**
- NVIDIA GeForce GTX 1650 (Turing, SM75, 4GB GDDR6, no tensor cores)

**Software stack:**
- Ubuntu 22.04+ with NVIDIA Driver 580.126.09
- CUDA 13.0 / CUDA Toolkit 13.1
- PyTorch 2.11.0+cu130
- Triton 3.6.0
- Python 3.10, 3.11, 3.12

**Test coverage:**
- 120 unit tests (CPU, no GPU required)
- 53 example tests (GPU)
- 235 GPU integration tests
- All 408 tests passing with zero failures

**Architecture detection verified for:**
- Turing (SM75) detection, including the GTX 16xx no-tensor-core edge case
- FP16 supported, BF16/FP8/TF32 correctly reported as unsupported on SM75
- Shared memory limits, compute capability, and driver version correctly detected via both pynvml and torch backends

**What is not yet tested on physical hardware:**
- Ampere (A100, RTX 30xx), Ada (L40, RTX 40xx), Hopper (H100), and Blackwell GPUs are supported in the architecture detection and gating code but have only been tested via mocked GPU info, not on actual hardware. The tolerance model for these architectures is calibrated against published CUTLASS and cuBLAS error models.
- AMD ROCm and Intel XPU are not supported yet. The architecture detection module is NVIDIA-only for now. ROCm support is planned and would involve adding HIP detection via `torch.version.hip` and AMD GPU enumeration via `amdsmi` or `rocm_smi`. Intel XPU support would use `torch.xpu`.
- Google TPU is not in scope for this project since TPUs use a fundamentally different programming model (XLA) that does not map to the kernel-level testing gpucheck provides.

## Comparison

| Feature | Manual `torch.allclose` | gpucheck |
|---|---|---|
| Dtype-aware tolerances | Hard-coded per test | Automatic from dtype |
| Mixed-precision comparison | Order-dependent, wrong tolerance | Uses lower-precision dtype automatically |
| Parametric dtypes/shapes/devices | Manual `@pytest.mark.parametrize` loops | `@dtypes`, `@shapes`, `@devices` decorators |
| GPU benchmarking | `time.time()` around kernel | CUDA events, warmup, L2 flush, outlier removal |
| Shape fuzzing | Random shapes, hope for the best | Adversarial shapes targeting tile boundaries, primes, edge cases |
| Memory leak detection | Not tested | `memory_tracker` fixture, `memory_guard` context manager |
| Architecture gating | `if` checks scattered through tests | `@require_arch`, `@require_capability` decorators |
| Failure diagnostics | "Tensors not close" | Rich error report with histogram, stats, worst-element location |
| Multi-GPU | Manual device loops | `@devices("all")` auto-detects and parametrizes |

## Project structure

```
gpucheck/
├── src/gpucheck/
│   ├── __init__.py              # Public API (lazy imports, no torch at collection time)
│   ├── plugin.py                # pytest plugin (hooks, fixtures, markers, CLI options)
│   ├── assertions/
│   │   ├── close.py             # assert_close() with GPU fast-path and mixed-precision
│   │   ├── tolerances.py        # Dtype-aware tolerance computation and k_dim scaling
│   │   └── reporting.py         # Rich-formatted mismatch reports with error histograms
│   ├── decorators/
│   │   ├── dtypes.py            # @dtypes decorator + dtype groups (FLOAT_DTYPES, etc.)
│   │   ├── shapes.py            # @shapes decorator + shape groups (EDGE_SHAPES, etc.)
│   │   ├── devices.py           # @devices decorator + auto-detection
│   │   └── parametrize.py       # @parametrize_gpu (cartesian product of dtypes x shapes x devices)
│   ├── fixtures/
│   │   ├── gpu.py               # GPUDevice dataclass and gpu_device fixture
│   │   ├── benchmark.py         # gpu_benchmark fixture with CUDA events and IQR outlier removal
│   │   └── profiler.py          # memory_tracker fixture with pynvml/torch dual backend
│   ├── fuzzing/
│   │   ├── shapes.py            # fuzz_shapes() deterministic corpus + ShapeStrategy for Hypothesis
│   │   ├── inputs.py            # random_inputs, edge_inputs, mixed_inputs generators
│   │   └── strategies.py        # gpu_shapes() and gpu_tensors() Hypothesis strategies
│   ├── sanitizers/
│   │   ├── memory.py            # check_memory_leaks, memory_guard context manager
│   │   └── race.py              # compute-sanitizer subprocess wrapper (memcheck, racecheck)
│   ├── arch/
│   │   ├── detection.py         # GPU detection via pynvml and torch (Pascal through Blackwell)
│   │   ├── compatibility.py     # @require_arch, @require_capability, SM compatibility checks
│   │   └── tensor_cores.py      # Tensor core support checks and architecture-aware tolerances
│   ├── analysis/
│   │   ├── roofline.py          # Roofline model, bottleneck classification, ASCII charts
│   │   ├── regression.py        # Mann-Whitney U test, Cohen's d, change-point detection
│   │   └── bottleneck.py        # Auto-classification via throughput scaling analysis
│   └── reporting/
│       ├── console.py           # Rich terminal reporter for test results and benchmarks
│       ├── json.py              # JSON reporter with run comparison for CI
│       └── ci.py                # GitHub Actions annotations, JUnit XML, PR comment generation
├── tests/                       # 120 unit tests + 235 GPU integration tests
├── examples/                    # 6 runnable examples including Triton bug reproducers
├── pyproject.toml
└── LICENSE
```

## Contributing

```bash
git clone https://github.com/Akasxh/gpucheck.git
cd gpucheck
pip install -e ".[dev]"

# Run tests (CPU-only, no GPU required)
pytest

# Lint and type check
ruff check src/ tests/
mypy src/
```

GPU tests live in `tests/gpu_integration/` and are skipped automatically when no GPU is available. To run them:

```bash
pytest tests/gpu_integration/ -v
```

The examples can be run individually:

```bash
pytest examples/basic_kernel_test.py -v
pytest examples/shape_fuzzing_example.py -v
pytest examples/benchmark_example.py -v
```

## License

Apache-2.0
