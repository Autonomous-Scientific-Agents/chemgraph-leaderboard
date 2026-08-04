"""Server-rendered (inline-SVG) visualizations for a completed community task.

The submission-status board is a single ``gr.HTML`` string, and ``gr.HTML`` does
NOT execute embedded ``<script>`` — so Plotly/JS charts can't be injected there.
Instead we render every chart as **static inline SVG** (plus the existing
``.cg-kpi-*`` cards as HTML), which survives Gradio's sanitizer and paints with
no client JS. This matches the repo's "charts as server-rendered markup" idiom.

Data source (Phase-1 / front-end only): a committed snapshot
``sample_task_metrics.json`` distilled from a real prior single-agent eval run,
keyed by the 12 benchmark categories (which are exactly the community-submission
category choices). When per-task eval data lands (Phase 2), swap the loader for
the real per-task payload — the renderers are payload-shaped and stay unchanged.

Payload shape (per category / overall)::

    {"n_models": int, "n_passed": int, "eval_time_s": float,
     "per_model": [{"model", "prompt_fresh", "cached", "completion"}, ...],
     "per_tool":  {tool_name: time_seconds, ...}}
"""

from __future__ import annotations

import html
import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

_SAMPLE_PATH = Path(__file__).resolve().parent / "sample_task_metrics.json"

# Token-segment colours: blue = input/prompt, green = output/completion. Cache is
# drawn in the prompt colour but hatched (it is a subset of the prompt tokens).
_PROMPT_COLOR = "#4285f4"
_COMPLETION_COLOR = "#10a37f"
# Donut palette (family palette + a neutral tail), reused in listed order.
_PIE_COLORS = ["#4285f4", "#10a37f", "#d97757", "#ca8a04", "#7c3aed",
               "#0891b2", "#db2777", "#94a3b8"]

# Friendlier labels for the 5 agent-level tools (backends collapse into run_ase).
_TOOL_LABELS = {
    "run_ase": "run_ase (QC compute)",
    "molecule_name_to_smiles": "name → SMILES",
    "smiles_to_coordinate_file": "SMILES → coords",
    "calculator": "calculator",
    "extract_output_json": "extract output",
}


@lru_cache(maxsize=1)
def load_sample_metrics() -> Dict[str, Any]:
    """Parse the committed sample metrics once (returns {} if missing/broken)."""
    try:
        return json.loads(_SAMPLE_PATH.read_text())
    except Exception:
        return {}


def _payload_for(task: Dict[str, Any]) -> tuple[Dict[str, Any], str]:
    """Pick the metrics payload for a task by its category (else overall)."""
    data = load_sample_metrics()
    cats = data.get("categories") or {}
    cat = (task.get("category") or "").strip()
    payload = cats.get(cat) or data.get("overall") or {}
    return payload, (cat if cat in cats else "overall")


# --------------------------------------------------------------------------- #
# formatting helpers
# --------------------------------------------------------------------------- #
def _fmt_tokens(v: float) -> str:
    """24189 -> '24.2k', 1_570_000 -> '1.57M', 0 -> '0'."""
    try:
        v = float(v)
    except Exception:
        return "—"
    if v >= 1_000_000:
        return f"{v / 1_000_000:.2f}M"
    if v >= 1_000:
        return f"{v / 1_000:.1f}k"
    return f"{int(v)}"


def _fmt_dur(secs: float) -> str:
    """438.8 -> '7m 19s', 45 -> '45s', 8901 -> '2h 28m'."""
    try:
        s = int(round(float(secs)))
    except Exception:
        return "—"
    if s < 60:
        return f"{s}s"
    m, sec = divmod(s, 60)
    if m < 60:
        return f"{m}m {sec}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m"


def _esc(s: Any) -> str:
    return html.escape(str(s))


# --------------------------------------------------------------------------- #
# Part 1 — KPI cards (reuse the leaderboard's .cg-kpi-* look)
# --------------------------------------------------------------------------- #
def _kpi_card(label: str, value: str, sub: str = "") -> str:
    sub_html = f"<div class='cg-kpi-sub'>{_esc(sub)}</div>" if sub else ""
    return (
        "<div class='cg-kpi-card'>"
        f"<div class='cg-kpi-label'>{_esc(label)}</div>"
        f"<div class='cg-kpi-value'>{_esc(value)}</div>"
        "<div class='cg-kpi-model'></div>"
        f"{sub_html}</div>"
    )


def _kpi_html(payload: Dict[str, Any]) -> str:
    n = int(payload.get("n_models") or 0)
    x = int(payload.get("n_passed") or 0)
    success = f"{x}/{n}" if n else "—"
    pct_sub = f"{round(x / n * 100)}% of models solved it" if n else "no model data"
    cards = (
        _kpi_card("Success rate", success, pct_sub)
        + _kpi_card("Evaluation time", _fmt_dur(payload.get("eval_time_s")),
                    "total wall clock, all models")
    )
    return f"<div class='cg-kpi-strip'>{cards}</div>"


# --------------------------------------------------------------------------- #
# Part 2 — token consumption: horizontal stacked bars, one row per model
# --------------------------------------------------------------------------- #
def _token_stack_svg(rows: List[Dict[str, Any]], uid: str) -> str:
    rows = [r for r in (rows or []) if (r.get("prompt_fresh", 0) + r.get("cached", 0)
                                        + r.get("completion", 0)) > 0]
    if not rows:
        return "<div class='cg-chart-empty'>No token data.</div>"

    # geometry (viewBox units; the SVG scales to its container width)
    LABEL_W, BAR_W, RIGHT_W = 132, 486, 78
    TOP, ROW_H, GAP, BOT = 30, 22, 9, 26
    W = LABEL_W + BAR_W + RIGHT_W
    H = TOP + len(rows) * ROW_H + (len(rows) - 1) * GAP + BOT

    totals = [r["prompt_fresh"] + r["cached"] + r["completion"] for r in rows]
    vmax = max(totals) or 1

    def sx(v: float) -> float:
        return v / vmax * BAR_W

    hatch = f"cg-hatch-{uid}"
    parts: List[str] = [
        f"<svg class='cg-token-svg' viewBox='0 0 {W} {H}' width='100%' "
        f"preserveAspectRatio='xMinYMin meet' role='img' "
        f"xmlns='http://www.w3.org/2000/svg'>",
        # hatch pattern for cached tokens: prompt colour + white diagonal lines
        f"<defs><pattern id='{hatch}' width='6' height='6' "
        "patternUnits='userSpaceOnUse' patternTransform='rotate(45)'>"
        f"<rect width='6' height='6' fill='{_PROMPT_COLOR}'/>"
        "<line x1='0' y1='0' x2='0' y2='6' stroke='rgba(255,255,255,0.55)' "
        "stroke-width='2'/></pattern></defs>",
    ]

    # faint vertical gridlines + axis scale (0 .. vmax)
    for frac in (0.25, 0.5, 0.75, 1.0):
        gx = LABEL_W + BAR_W * frac
        parts.append(
            f"<line x1='{gx:.1f}' y1='{TOP - 6}' x2='{gx:.1f}' y2='{H - BOT + 4}' "
            "stroke='rgba(148,163,184,0.25)' stroke-width='1'/>"
        )
        parts.append(
            f"<text x='{gx:.1f}' y='{H - BOT + 16}' font-size='10' "
            "fill='rgba(100,116,139,0.85)' text-anchor='middle'>"
            f"{_fmt_tokens(vmax * frac)}</text>"
        )

    for i, r in enumerate(rows):
        y = TOP + i * (ROW_H + GAP)
        cy = y + ROW_H / 2
        model = _esc(r.get("model", ""))
        fresh, cached, comp = r["prompt_fresh"], r["cached"], r["completion"]
        tot = fresh + cached + comp
        # model label (right-aligned into the left gutter)
        parts.append(
            f"<text x='{LABEL_W - 8}' y='{cy:.1f}' font-size='11' "
            "fill='var(--cg-text-secondary, #475569)' text-anchor='end' "
            f"dominant-baseline='central'>{model}</text>"
        )
        x0 = float(LABEL_W)
        # Cached first (drawn from the left edge), then fresh prompt, then completion.
        for value, fill, seg in (
            (cached, f"url(#{hatch})", "Prompt (cached)"),
            (fresh, _PROMPT_COLOR, "Prompt (fresh)"),
            (comp, _COMPLETION_COLOR, "Completion"),
        ):
            if value <= 0:
                continue
            w = sx(value)
            parts.append(
                f"<rect x='{x0:.2f}' y='{y}' width='{w:.2f}' height='{ROW_H}' "
                f"fill='{fill}'><title>{model} · {seg}: {int(value):,} tok</title></rect>"
            )
            x0 += w
        # total label at the end of the bar
        parts.append(
            f"<text x='{x0 + 6:.2f}' y='{cy:.1f}' font-size='10.5' "
            "fill='var(--cg-text-muted, #94a3b8)' dominant-baseline='central'>"
            f"{_fmt_tokens(tot)}</text>"
        )

    parts.append("</svg>")
    legend = (
        "<div class='cg-chart-legend'>"
        f"<span class='cg-lg'><span class='cg-sw cg-sw-hatch' style='background:{_PROMPT_COLOR}'></span>Prompt (cached)</span>"
        f"<span class='cg-lg'><span class='cg-sw' style='background:{_PROMPT_COLOR}'></span>Prompt (fresh)</span>"
        f"<span class='cg-lg'><span class='cg-sw' style='background:{_COMPLETION_COLOR}'></span>Completion</span>"
        "</div>"
    )
    return legend + "".join(parts)


# --------------------------------------------------------------------------- #
# Part 3 — per-tool execution time: donut + legend
# --------------------------------------------------------------------------- #
def _polar(cx: float, cy: float, r: float, deg: float) -> tuple[float, float]:
    a = math.radians(deg - 90)  # 0deg at 12 o'clock, clockwise
    return cx + r * math.cos(a), cy + r * math.sin(a)


def _annular_sector(cx: float, cy: float, ro: float, ri: float,
                    a0: float, a1: float) -> str:
    large = 1 if (a1 - a0) > 180 else 0
    xo0, yo0 = _polar(cx, cy, ro, a0)
    xo1, yo1 = _polar(cx, cy, ro, a1)
    xi1, yi1 = _polar(cx, cy, ri, a1)
    xi0, yi0 = _polar(cx, cy, ri, a0)
    return (
        f"M{xo0:.2f},{yo0:.2f} "
        f"A{ro:.2f},{ro:.2f} 0 {large} 1 {xo1:.2f},{yo1:.2f} "
        f"L{xi1:.2f},{yi1:.2f} "
        f"A{ri:.2f},{ri:.2f} 0 {large} 0 {xi0:.2f},{yi0:.2f} Z"
    )


# Time-composition donut palette — deliberately distinct from _PIE_COLORS so a
# blue "LLM calls" slice is never confused with the blue "run_ase" slice next to it.
_TIME_COLORS = ["#6366f1", "#0891b2", "#f59e0b", "#94a3b8"]


def _fmt_secs(v: float) -> str:
    """Short duration for legends: '4.8s' when small, '16m 18s' / '4h 11m' when big."""
    v = float(v)
    return f"{v:.1f}s" if v < 100 else _fmt_dur(v)


def _donut_svg(items_dict: Dict[str, Any], center_sub: str, *,
               colors: Optional[List[str]] = None,
               label_map: Optional[Dict[str, str]] = None,
               size: int = 150) -> str:
    """Generic donut + legend. ``items_dict`` maps label -> value (seconds)."""
    colors = colors or _PIE_COLORS
    label_map = label_map or {}
    items = [(k, float(v)) for k, v in (items_dict or {}).items() if float(v) > 0]
    items.sort(key=lambda kv: -kv[1])
    total = sum(v for _, v in items)
    if not items or total <= 0:
        return "<div class='cg-chart-empty'>No timing data.</div>"

    R_OUT, R_IN = size * 0.46, size * 0.28
    cx = cy = size / 2
    svg = [
        f"<svg class='cg-pie-svg' viewBox='0 0 {size} {size}' width='{size}' "
        f"height='{size}' role='img' xmlns='http://www.w3.org/2000/svg'>"
    ]
    acc = 0.0
    for idx, (key, val) in enumerate(items):
        color = colors[idx % len(colors)]
        frac = val / total
        a0, a1 = acc * 360, (acc + frac) * 360
        acc += frac
        label = label_map.get(key, key)
        title = f"<title>{_esc(label)}: {_fmt_secs(val)} ({frac * 100:.0f}%)</title>"
        if frac >= 0.999:  # single slice → full ring
            rmid = (R_OUT + R_IN) / 2
            svg.append(
                f"<circle cx='{cx}' cy='{cy}' r='{rmid:.2f}' fill='none' "
                f"stroke='{color}' stroke-width='{R_OUT - R_IN:.2f}'>{title}</circle>"
            )
        else:
            svg.append(
                f"<path d='{_annular_sector(cx, cy, R_OUT, R_IN, a0, a1)}' "
                f"fill='{color}'>{title}</path>"
            )
    svg.append(
        f"<text x='{cx}' y='{cy - 2}' font-size='15' font-weight='700' "
        "fill='var(--cg-text-primary, #0f172a)' text-anchor='middle'>"
        f"{_fmt_dur(total)}</text>"
    )
    svg.append(
        f"<text x='{cx}' y='{cy + 12}' font-size='9' "
        "fill='var(--cg-text-muted, #94a3b8)' text-anchor='middle'>"
        f"{_esc(center_sub)}</text>"
    )
    svg.append("</svg>")

    rows = []
    for idx, (key, val) in enumerate(items):
        color = colors[idx % len(colors)]
        label = label_map.get(key, key)
        pct = val / total * 100
        rows.append(
            "<li class='cg-pie-li'>"
            f"<span class='cg-sw' style='background:{color}'></span>"
            f"<span class='cg-pie-name'>{_esc(label)}</span>"
            f"<span class='cg-pie-val'>{_fmt_secs(val)} · {pct:.0f}%</span>"
            "</li>"
        )
    legend = "<ul class='cg-pie-legend'>" + "".join(rows) + "</ul>"
    return f"<div class='cg-pie-wrap'>{''.join(svg)}{legend}</div>"


def _tool_pie_svg(per_tool: Dict[str, Any]) -> str:
    """Donut of execution time per agent tool (run_ase, name→SMILES, ...)."""
    return _donut_svg(per_tool, "tool time", label_map=_TOOL_LABELS)


def _time_pie_svg(time_breakdown: Dict[str, Any]) -> str:
    """Donut of agent wall time split into LLM calls vs tool execution vs other."""
    return _donut_svg(time_breakdown, "agent time", colors=_TIME_COLORS)


# --------------------------------------------------------------------------- #
# assembly — the expandable <details> panel appended to a completed card
# --------------------------------------------------------------------------- #
def render_task_charts(task: Dict[str, Any]) -> str:
    """Return the collapsible results panel for a completed task ('' if no data)."""
    payload, cat = _payload_for(task)
    if not payload:
        return ""
    uid = re.sub(r"[^a-z0-9]+", "", (task.get("name") or cat or "x").lower())[:24] or "x"

    wf = (load_sample_metrics().get("workflow") or "").replace("_", "-") or "prior"
    note = (
        f"<div class='cg-charts-note'>Demo data from a prior {_esc(wf)} run · "
        f"category <b>{_esc(cat)}</b></div>"
    )
    body = (
        note
        + _kpi_html(payload)
        + "<div class='cg-charts-row'>"
        "<div class='cg-chart-block cg-chart-bar'>"
        "<div class='cg-chart-title'>Token consumption by model</div>"
        + _token_stack_svg(payload.get("per_model") or [], uid)
        + "</div>"
        # Right column: two donuts stacked — time split (LLM vs tools) on top,
        # per-tool breakdown below — so the column fills the token bar's height.
        "<div class='cg-chart-pie-col'>"
        "<div class='cg-chart-block'>"
        "<div class='cg-chart-title'>Time split: LLM vs tools</div>"
        + _time_pie_svg(payload.get("time_breakdown") or {})
        + "</div>"
        "<div class='cg-chart-block'>"
        "<div class='cg-chart-title'>Execution time by tool</div>"
        + _tool_pie_svg(payload.get("per_tool") or {})
        + "</div>"
        "</div>"
        "</div>"
    )
    return (
        "<details class='cg-task-charts'>"
        "<summary>View evaluation results</summary>"
        f"<div class='cg-charts-body'>{body}</div>"
        "</details>"
    )
