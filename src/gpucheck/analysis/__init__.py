"""Performance analysis: roofline modelling, bottleneck detection, regression testing."""

from __future__ import annotations

from gpucheck.analysis.bottleneck import BottleneckAnalysis
from gpucheck.analysis.bottleneck import classify_bottleneck as classify_bottleneck_auto
from gpucheck.analysis.regression import (
    RegressionReport,
    RegressionResult,
    detect_regression,
    e_divisive_single,
    format_regression_table,
    load_baseline,
    mann_whitney_u,
    save_baseline,
)
from gpucheck.analysis.roofline import (
    GPUSpecs,
    RooflinePoint,
    classify_bottleneck,
    compute_roofline,
    compute_roofline_point,
    lookup_gpu_specs,
    render_roofline_ascii,
)

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
    "classify_bottleneck_auto",
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
