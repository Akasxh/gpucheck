"""Static HTML dashboard generator for gpucheck JSON run records.

Reads a ``results.json`` produced by :class:`gpucheck.reporting.json.JSONReporter`
and writes a single self-contained HTML file: zero external CSS, zero
external JavaScript, no fetches at view time. Inline SVG renders the
benchmark bar chart so the file works on a flight without WiFi.

Sections (in order):

1. **Summary** — total tests, pass count, fail count, skip count, GPU info.
2. **Test results table** — one row per test with status pill and
   collapsed message via ``<details>``.
3. **Benchmark table** — one row per kernel; inline SVG bar chart of
   median timings for at-a-glance regression spotting.
4. **Memory table** — peak / leaked MB per test.
5. **Comparison band** — when a comparison diff is supplied, surfaces
   regression / ok / new / removed rows in red / green / blue / gray.

The renderer is deliberately small (no Jinja, no D3) so it vendorizes
cleanly. Callers needing richer charts can post-process the JSON in any
external dashboard.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_STATUS_PILL_BG: dict[str, str] = {
    "passed": "#1f7a4d",
    "failed": "#a8231f",
    "error": "#a8231f",
    "skipped": "#a87a1f",
    "ok": "#1f7a4d",
    "regression": "#a8231f",
    "new": "#1f5fa8",
    "removed": "#666666",
}


@dataclass
class HTMLReporter:
    """Render a JSON run record into a self-contained HTML file."""

    json_path: str | Path
    comparison: dict[str, Any] | None = None
    title: str = "gpucheck dashboard"
    _data: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def _load(self) -> dict[str, Any]:
        if not self._data:
            self._data = json.loads(Path(self.json_path).read_text(encoding="utf-8"))
        return self._data

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self, out_path: str | Path) -> Path:
        """Write the dashboard to *out_path* and return its :class:`Path`."""
        data = self._load()
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        sections = [
            _render_head(self.title),
            _render_summary(data),
            _render_test_results(data),
            _render_benchmarks(data),
            _render_memory(data),
            _render_comparison(self.comparison) if self.comparison else "",
            _render_foot(),
        ]
        out.write_text("\n".join(s for s in sections if s) + "\n", encoding="utf-8")
        return out


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _pill(status: str) -> str:
    bg = _STATUS_PILL_BG.get(status, "#666666")
    return (
        f'<span class="pill" style="background:{bg};color:#fff;'
        f'padding:2px 8px;border-radius:9px;font-size:12px;">'
        f"{_esc(status.upper())}</span>"
    )


def _render_head(title: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><title>{_esc(title)}</title>
<style>
body {{ font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif;
        background:#fafafa; color:#222; max-width:1100px; margin:0 auto; padding:24px; }}
h1 {{ font-size:22px; margin:0 0 16px; }}
h2 {{ font-size:16px; margin:24px 0 8px; color:#444;
      border-bottom:1px solid #ddd; padding-bottom:4px; }}
table {{ border-collapse:collapse; width:100%; margin:8px 0; background:#fff;
         box-shadow:0 1px 2px rgba(0,0,0,0.06); }}
th, td {{ text-align:left; padding:6px 10px; border-bottom:1px solid #eee; font-size:13px; }}
th {{ background:#f4f4f4; font-weight:600; }}
tr.regression {{ background:#fdebec; }}
tr.passed-row {{ background:#fff; }}
tr.new {{ background:#eaf3fc; }}
tr.removed {{ background:#f0f0f0; }}
details summary {{ cursor:pointer; color:#06c; }}
.summary-cards {{ display:flex; gap:12px; margin:8px 0; }}
.card {{ flex:1; background:#fff; padding:12px; border-radius:6px;
         box-shadow:0 1px 2px rgba(0,0,0,0.06); }}
.card .num {{ font-size:24px; font-weight:600; }}
.card.passed .num {{ color:#1f7a4d; }}
.card.failed .num {{ color:#a8231f; }}
.card.skipped .num {{ color:#a87a1f; }}
.card.gpu {{ font-size:12px; color:#444; }}
.bar {{ fill:#1f5fa8; }}
</style>
</head><body>
<h1>{_esc(title)}</h1>"""


def _render_foot() -> str:
    return "</body></html>"


def _render_summary(data: dict[str, Any]) -> str:
    results = data.get("test_results", [])
    passed = sum(1 for r in results if r.get("status") == "passed")
    failed = sum(1 for r in results if r.get("status") == "failed")
    skipped = sum(1 for r in results if r.get("status") == "skipped")

    gpu_info = data.get("gpu_info", {}) or {}
    gpu_summary = " | ".join(
        f"{_esc(k)}: {_esc(v)}" for k, v in list(gpu_info.items())[:5]
    ) or "no GPU info recorded"

    return f"""<h2>Summary</h2>
<div class="summary-cards">
  <div class="card passed"><div class="num">{passed}</div>passed</div>
  <div class="card failed"><div class="num">{failed}</div>failed</div>
  <div class="card skipped"><div class="num">{skipped}</div>skipped</div>
  <div class="card gpu">{gpu_summary}</div>
</div>
<p style="font-size:12px;color:#666;">timestamp: {_esc(data.get("timestamp", "unknown"))}</p>"""


def _render_test_results(data: dict[str, Any]) -> str:
    results = data.get("test_results", [])
    if not results:
        return ""
    rows = []
    for r in results:
        status = r.get("status", "unknown")
        klass = "regression" if status in {"failed", "error"} else "passed-row"
        msg = r.get("message", "")
        msg_cell = (
            f'<details><summary>view</summary><pre style="white-space:pre-wrap;'
            f'margin:4px 0;">{_esc(msg)}</pre></details>' if msg else ""
        )
        rows.append(
            f'<tr class="{klass}"><td>{_esc(r.get("name", ""))}</td>'
            f'<td>{_pill(status)}</td>'
            f'<td>{r.get("duration", 0.0):.4f}s</td>'
            f'<td>{msg_cell}</td></tr>',
        )
    body = "\n".join(rows)
    return f"""<h2>Test Results</h2>
<table><tr><th>Test</th><th>Status</th><th>Duration</th><th>Message</th></tr>
{body}
</table>"""


def _render_benchmarks(data: dict[str, Any]) -> str:
    benches = data.get("benchmarks", [])
    if not benches:
        return ""
    max_med = max((b.get("median_ms", 0.0) or 0.0) for b in benches) or 1.0

    rows = []
    for b in benches:
        med = float(b.get("median_ms", 0.0) or 0.0)
        std = float(b.get("std_ms", 0.0) or 0.0)
        bar_w = max(2, int(180 * (med / max_med)))
        rows.append(
            f'<tr><td>{_esc(b.get("name", ""))}</td>'
            f'<td>{med:.3f} ms</td>'
            f'<td>{std:.3f} ms</td>'
            f'<td>{b.get("samples", 0)}</td>'
            f'<td><svg width="200" height="14"><rect class="bar" x="0" y="2" '
            f'width="{bar_w}" height="10"/></svg></td></tr>',
        )
    return f"""<h2>Benchmarks</h2>
<table><tr><th>Kernel</th><th>Median</th><th>Std</th><th>Samples</th>
<th>Distribution (relative)</th></tr>
{"".join(rows)}
</table>"""


def _render_memory(data: dict[str, Any]) -> str:
    mem = data.get("memory", [])
    if not mem:
        return ""
    rows = []
    for m in mem:
        leaked = float(m.get("leaked_mb", 0.0) or 0.0)
        klass = "regression" if leaked > 0 else "passed-row"
        rows.append(
            f'<tr class="{klass}"><td>{_esc(m.get("name", ""))}</td>'
            f'<td>{m.get("peak_mb", 0):.2f} MB</td>'
            f'<td>{leaked:.2f} MB</td>'
            f'<td>{m.get("allocations", 0)}</td></tr>',
        )
    return f"""<h2>Memory</h2>
<table><tr><th>Test</th><th>Peak</th><th>Leaked</th><th>Allocations</th></tr>
{"".join(rows)}
</table>"""


def _render_comparison(diff: dict[str, Any]) -> str:
    benches = diff.get("benchmarks", [])
    if not benches:
        return ""
    rows = []
    for b in benches:
        status = b.get("status", "ok")
        klass = status if status in {"regression", "new", "removed"} else "passed-row"
        base = b.get("baseline_median_ms", "-")
        curr = b.get("current_median_ms", "-")
        delta = b.get("delta_pct", 0)
        base_str = f"{base:.3f} ms" if isinstance(base, (int, float)) else _esc(base)
        curr_str = f"{curr:.3f} ms" if isinstance(curr, (int, float)) else _esc(curr)
        rows.append(
            f'<tr class="{klass}"><td>{_esc(b.get("name", ""))}</td>'
            f'<td>{base_str}</td><td>{curr_str}</td>'
            f'<td>{delta:+.1f}%</td>'
            f'<td>{_pill(status)}</td></tr>',
        )
    return f"""<h2>Comparison vs Baseline</h2>
<table><tr><th>Kernel</th><th>Baseline</th><th>Current</th><th>Delta</th>
<th>Status</th></tr>
{"".join(rows)}
</table>"""


__all__ = ["HTMLReporter"]
