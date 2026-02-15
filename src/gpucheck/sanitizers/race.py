"""Wrapper around NVIDIA compute-sanitizer for race/memory checks."""

from __future__ import annotations

import inspect
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

SanitizerTool = Literal["memcheck", "racecheck", "initcheck", "synccheck"]
_VALID_TOOLS: frozenset[str] = frozenset({"memcheck", "racecheck", "initcheck", "synccheck"})


@dataclass(frozen=True, slots=True)
class SanitizerError:
    """A single error reported by compute-sanitizer."""

    description: str
    address: str = ""
    size: str = ""
    location: str = ""


@dataclass(frozen=True, slots=True)
class SanitizerReport:
    """Parsed output from compute-sanitizer."""

    tool: SanitizerTool
    errors: tuple[SanitizerError, ...] = ()
    warnings: tuple[str, ...] = ()
    raw_output: str = ""
    return_code: int = 0

    @property
    def clean(self) -> bool:
        return len(self.errors) == 0

    @property
    def error_count(self) -> int:
        return len(self.errors)


def _find_compute_sanitizer() -> str | None:
    """Locate compute-sanitizer binary on PATH or in CUDA_HOME."""
    path = shutil.which("compute-sanitizer")
    if path:
        return path

    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH", "")
    if cuda_home:
        candidate = os.path.join(cuda_home, "bin", "compute-sanitizer")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    return None


def _parse_sanitizer_output(raw: str, tool: SanitizerTool) -> tuple[list[SanitizerError], list[str]]:
    """Parse compute-sanitizer text output into errors and warnings."""
    errors: list[SanitizerError] = []
    warnings: list[str] = []

    for line in raw.splitlines():
        stripped = line.strip()

        # compute-sanitizer prefixes lines with ========
        if not stripped.startswith("========"):
            continue

        content = stripped.lstrip("=").strip()

        if not content:
            continue

        lowered = content.lower()

        if "error" in lowered and "0 errors" not in lowered:
            # Try to extract structured info
            errors.append(SanitizerError(description=content))
        elif "warning" in lowered:
            warnings.append(content)

    return errors, warnings


def _build_wrapper_script(fn: Callable[..., Any]) -> str:
    """Generate a temporary Python script that imports and calls *fn*."""
    module = inspect.getmodule(fn)
    if module is None or module.__name__ == "__main__":
        raise ValueError(
            "run_with_sanitizer requires a function defined in an importable module, "
            f"got {fn!r} from __main__"
        )

    return (
        f"import sys; sys.path[:0] = {sys.path!r}\n"
        f"from {module.__name__} import {fn.__name__}\n"
        f"{fn.__name__}()\n"
    )


def run_with_sanitizer(
    fn: Callable[..., Any],
    *,
    tool: SanitizerTool = "memcheck",
    timeout: float = 120.0,
    extra_args: list[str] | None = None,
) -> SanitizerReport:
    """Run *fn* under NVIDIA compute-sanitizer and return parsed results.

    The function must be importable (defined in a proper module, not ``__main__``).
    It is invoked in a subprocess via ``compute-sanitizer --tool <tool> python script.py``.

    Parameters
    ----------
    fn:
        Zero-argument callable to execute under the sanitizer.
    tool:
        One of ``memcheck``, ``racecheck``, ``initcheck``, ``synccheck``.
    timeout:
        Maximum seconds to wait for the subprocess.
    extra_args:
        Additional CLI flags forwarded to compute-sanitizer.
    """
    if tool not in _VALID_TOOLS:
        raise ValueError(f"Invalid tool {tool!r}, must be one of {sorted(_VALID_TOOLS)}")

    sanitizer_bin = _find_compute_sanitizer()
    if sanitizer_bin is None:
        raise FileNotFoundError(
            "compute-sanitizer not found. Install the CUDA toolkit or set CUDA_HOME."
        )

    script_content = _build_wrapper_script(fn)

    fd, script_path = tempfile.mkstemp(suffix=".py", prefix="gpucheck_sanitizer_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(script_content)

        cmd: list[str] = [
            sanitizer_bin,
            "--tool",
            tool,
            "--print-level",
            "warn",
        ]
        if extra_args:
            cmd.extend(extra_args)
        cmd.extend([sys.executable, script_path])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        raw = result.stdout + "\n" + result.stderr
        errors, warnings = _parse_sanitizer_output(raw, tool)

        return SanitizerReport(
            tool=tool,
            errors=tuple(errors),
            warnings=tuple(warnings),
            raw_output=raw,
            return_code=result.returncode,
        )

    except subprocess.TimeoutExpired:
        return SanitizerReport(
            tool=tool,
            errors=(SanitizerError(description=f"Timed out after {timeout}s"),),
            warnings=(),
            raw_output="",
            return_code=-1,
        )
    finally:
        os.unlink(script_path)
