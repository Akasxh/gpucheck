"""GPU architecture detection and capability querying."""

from __future__ import annotations

from gpucheck.arch.compatibility import require_arch, require_capability
from gpucheck.arch.detection import GPUInfo, detect_gpus

__all__ = ["GPUInfo", "detect_gpus", "require_arch", "require_capability"]
