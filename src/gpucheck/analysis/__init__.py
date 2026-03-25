"""Performance analysis: roofline modelling, bottleneck detection, regression testing."""

from __future__ import annotations

import importlib
from typing import Any

_LAZY_MAP: dict[str, tuple[str, str]] = {
    # roofline
    "GPUSpecs": ("gpucheck.analysis.roofline", "GPUSpecs"),
    "RooflinePoint": ("gpucheck.analysis.roofline", "RooflinePoint"),
    "classify_bottleneck": ("gpucheck.analysis.roofline", "classify_bottleneck"),
    "compute_roofline": ("gpucheck.analysis.roofline", "compute_roofline"),
    "compute_roofline_point": ("gpucheck.analysis.roofline", "compute_roofline_point"),
    "lookup_gpu_specs": ("gpucheck.analysis.roofline", "lookup_gpu_specs"),
    "render_roofline_ascii": ("gpucheck.analysis.roofline", "render_roofline_ascii"),
    # bottleneck
    "BottleneckAnalysis": ("gpucheck.analysis.bottleneck", "BottleneckAnalysis"),
    "auto_classify_bottleneck": ("gpucheck.analysis.bottleneck", "auto_classify_bottleneck"),
    # regression
    "RegressionReport": ("gpucheck.analysis.regression", "RegressionReport"),
    "RegressionResult": ("gpucheck.analysis.regression", "RegressionResult"),
    "detect_regression": ("gpucheck.analysis.regression", "detect_regression"),
    "e_divisive_single": ("gpucheck.analysis.regression", "e_divisive_single"),
    "format_regression_table": ("gpucheck.analysis.regression", "format_regression_table"),
    "load_baseline": ("gpucheck.analysis.regression", "load_baseline"),
    "mann_whitney_u": ("gpucheck.analysis.regression", "mann_whitney_u"),
    "save_baseline": ("gpucheck.analysis.regression", "save_baseline"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_MAP:
        module_path, attr = _LAZY_MAP[name]
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    raise AttributeError(f"module 'gpucheck.analysis' has no attribute {name!r}")


__all__ = [
    # roofline
    "GPUSpecs",
    "RooflinePoint",
    "classify_bottleneck",
    "compute_roofline",
    "compute_roofline_point",
    "lookup_gpu_specs",
    "render_roofline_ascii",
    # bottleneck
    "BottleneckAnalysis",
    "auto_classify_bottleneck",
    # regression
    "RegressionReport",
    "RegressionResult",
    "detect_regression",
    "e_divisive_single",
    "format_regression_table",
    "load_baseline",
    "mann_whitney_u",
    "save_baseline",
]
