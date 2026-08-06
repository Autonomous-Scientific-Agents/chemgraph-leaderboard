"""Render a :class:`~src.leaderboard.trace_model.Trace` as HTML.

Two views per query, both fed from the same :attr:`Trace.timeline`:

* :func:`render_minimap` — a compact inline-SVG topology diagram, one node per
  *phase*, so the shape of the run (chain vs fan-out) is visible at a glance.
* :func:`render_tree` — a vertical trace tree where every step is a ``<details>``
  carrying its role label and full log content.

Hard constraints this module works under (the panel lands in a ``gr.HTML``, which
inserts via Svelte ``{@html}``):

* **``<script>`` is never executed.** Anything interactive lives in the
  ``group_columns_head`` IIFE in ``css_html_js.py``; this module emits only markup.
* **Everything model-generated must be escaped** — see :func:`esc`.
* **No document-global ids.** A panel renders 2-10 of these side by side, so an
  SVG ``<defs>`` id would collide across cards. Arrowheads are inline ``<path>``
  triangles instead, and nothing here emits an ``id`` attribute.
* **No ``<table>``** — the capture-phase cell handler in ``wireCellClicks()``
  keys off ``<td>``, and a table in the panel would let it re-trigger.

Inline-SVG-as-chart is an established idiom here; see ``src/submission/task_charts.py``.
"""

from __future__ import annotations

import html
import json

from src.leaderboard.trace_model import (
    DANGLING,
    ERR,
    K_AGENT,
    K_ANSWER,
    K_FORMATTER,
    K_PLANNER,
    K_TOOL,
    K_USER,
    WARN,
    Fan,
    Trace,
)

# Role -> theme token. Kinds come straight from trace_model, so the two stay in
# step; the fallback hex keeps the SVG readable if the token is ever missing.
_ROLE_VAR = {
    K_USER: "var(--cg-role-user, #2563eb)",
    K_PLANNER: "var(--cg-role-planner, #7c3aed)",
    K_AGENT: "var(--cg-role-agent, #0d9488)",
    K_ANSWER: "var(--cg-role-answer, #16a34a)",
    K_TOOL: "var(--cg-role-tool, #d97706)",
    K_FORMATTER: "var(--cg-role-format, #65a30d)",
}
_EXEC_VAR = "var(--cg-role-exec, #0891b2)"
_ERR_VAR = "var(--cg-role-err, #dc2626)"
_WARN_VAR = "var(--cg-role-tool, #d97706)"

_ROLE_CLASS = {
    K_USER: "cg-r-user",
    K_PLANNER: "cg-r-planner",
    K_AGENT: "cg-r-agent",
    K_ANSWER: "cg-r-answer",
    K_TOOL: "cg-r-tool",
    K_FORMATTER: "cg-r-format",
}

_ROLE_TEXT = {
    K_USER: "query",
    K_PLANNER: "planner",
    K_AGENT: "agent",
    K_ANSWER: "agent",
    K_TOOL: "tool",
    K_FORMATTER: "formatter",
}


def esc(v) -> str:
    """Escape any value for HTML. dict/list are pretty-printed JSON first.

    ``quote=True`` (the default) also escapes ``"``, so the result is safe in a
    double-quoted attribute. It does **not** escape ``'`` — which is why SVG
    attributes here are never built from model-generated text.
    """
    if isinstance(v, (dict, list)):
        v = json.dumps(v, indent=2, ensure_ascii=False, default=str)
    elif not isinstance(v, str):
        v = str(v)
    return html.escape(v)


def _clip(s: str, n: int) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _role_class(step) -> str:
    """Which `--role` class a node carries. Status wins over kind."""
    if step.status == ERR:
        return "cg-r-err"
    if step.status in (WARN, DANGLING):
        return "cg-r-warn"
    return _ROLE_CLASS.get(step.kind, "cg-r-muted")


def _fmt_int(n) -> str:
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return str(n)


# --------------------------------------------------------------------------- #
# Minimap
# --------------------------------------------------------------------------- #

_NODE_W, _NODE_H = 112, 40
_COL_GAP, _LANE_GAP = 24, 10
_PAD_X, _PAD_TOP, _PAD_BOT = 12, 16, 12
_MM_MAX_COLS = 11

# The tool vocabulary is a fixed set of 5; shorten the long ones to fit the ~83px
# of usable label width inside a node (about 12 chars at 10.5px semibold).
_TOOL_SHORT = {
    "molecule_name_to_smiles": "name→SMILES",
    "smiles_to_coordinate_file": "SMILES→xyz",  # 10 ch, fits
    "run_ase": "run_ase",
    "calculator": "calc",
    "extract_output_json": "extract",
}

_EDGE_COLOR = "rgba(148,163,184,0.55)"  # theme-agnostic slate, as used by the other charts


class _MMNode:
    __slots__ = ("kind", "label", "sub", "title", "stack", "status")

    def __init__(self, kind, label, sub="", title="", stack=1, status="ok"):
        self.kind = kind
        self.label = label
        self.sub = sub
        self.title = title or label
        self.stack = stack
        self.status = status


def _mm_step_node(step) -> _MMNode:
    if step.kind == K_AGENT and step.children:
        names = [c.label for c in step.children]
        uniq = set(names)
        label = _TOOL_SHORT.get(names[0], names[0]) if len(uniq) == 1 else f"{len(names)} tools"
        sub = f"{len(names)} call{'s' if len(names) != 1 else ''}"
        title = "Tool calls: " + ", ".join(names)
        return _MMNode(K_TOOL, label, sub, title, stack=len(names), status=step.status)
    if step.kind == K_USER:
        return _MMNode(K_USER, "Query", f"{len(step.content):,} chars", _clip(step.content, 300))
    if step.kind == K_PLANNER:
        out = step.meta.get("output_tokens")
        return _MMNode(K_PLANNER, step.label, f"{out:,} tok" if out else "", _clip(step.content, 300), status=step.status)
    if step.kind == K_FORMATTER:
        fields = step.meta.get("fields") or []
        sub = str(fields[0][0]) if fields else "empty"
        return _MMNode(K_FORMATTER, "Formatter", sub, _clip(step.content, 300), status=step.status)
    out = step.meta.get("output_tokens")
    return _MMNode(K_ANSWER, "Answer", f"{out:,} tok" if out else "", _clip(step.content, 300), status=step.status)


def _mm_columns(trace: Trace) -> list:
    cols: list = []
    for item in trace.timeline:
        if isinstance(item, Fan):
            lanes = [
                _MMNode(
                    "exec",
                    b.label.replace("Executor", "Exec"),
                    f"{len(b.steps)} steps · {b.n_tools} tools",
                    f"{b.label}: {_clip(b.task, 240)}",
                    status=b.status,
                )
                for b in item.branches
            ]
            cols.append(lanes or [_MMNode("gap", "no executors", "", "Planner answered directly")])
        else:
            cols.append([_mm_step_node(item)])
    return _elide(cols)


def _elide(cols: list) -> list:
    """Long single-agent chains only. Fan-outs are never elided."""
    if len(cols) <= _MM_MAX_COLS:
        return cols
    hidden = len(cols) - 9
    gap = [_MMNode("gap", f"+{hidden} steps", "collapsed", "Expand the trace below to see every step")]
    return cols[:4] + [gap] + cols[-5:]


def _mm_color(nd: _MMNode) -> str:
    if nd.status == ERR:
        return _ERR_VAR
    if nd.status in (WARN, DANGLING):
        return _WARN_VAR
    if nd.kind == "exec":
        return _EXEC_VAR
    if nd.kind == "gap":
        return "var(--cg-text-muted, #94a3b8)"
    return _ROLE_VAR.get(nd.kind, "var(--cg-text-muted, #94a3b8)")


def _edge(x0: float, y0: float, x1: float, y1: float, color: str) -> str:
    if abs(y1 - y0) < 0.5:
        d = f"M{x0:.1f},{y0:.1f} L{x1 - 6:.1f},{y1:.1f}"
    else:
        dx = (x1 - x0) * 0.5
        d = f"M{x0:.1f},{y0:.1f} C{x0 + dx:.1f},{y0:.1f} {x1 - dx:.1f},{y1:.1f} {x1 - 6:.1f},{y1:.1f}"
    return (
        f"<path d='{d}' fill='none' stroke='{color}' stroke-width='1.6' "
        f"stroke-linecap='round' opacity='0.8'/>"
        f"<path d='M{x1:.1f},{y1:.1f} l-6,-3.4 l0,6.8 z' fill='{color}' opacity='0.8'/>"
    )


def _node_svg(x: float, y: float, nd: _MMNode) -> str:
    """One minimap node.

    Model-generated text goes ONLY into <text>/<title> element content — never an
    attribute — because html.escape() leaves ``'`` alone and the attributes here
    are single-quoted.
    """
    color = _mm_color(nd)
    p = ["<g>"]
    for k in range(min(nd.stack - 1, 2), 0, -1):  # stacked-card silhouette for parallel calls
        p.append(
            f"<rect x='{x + 3 * k:.0f}' y='{y - 3 * k:.0f}' width='{_NODE_W}' height='{_NODE_H}' rx='9' "
            f"fill='var(--cg-surface, #ffffff)' stroke='{color}' stroke-width='1' opacity='0.4'/>"
        )
    p.append(
        f"<rect x='{x:.0f}' y='{y:.0f}' width='{_NODE_W}' height='{_NODE_H}' rx='9' "
        f"fill='var(--cg-surface, #ffffff)' stroke='{color}' stroke-width='1.4'/>"
    )
    if nd.status == ERR:
        p.append(f"<rect x='{x:.0f}' y='{y:.0f}' width='{_NODE_W}' height='{_NODE_H}' rx='9' fill='rgba(220,38,38,0.10)'/>")
    elif nd.status in (WARN, DANGLING):
        p.append(f"<rect x='{x:.0f}' y='{y:.0f}' width='{_NODE_W}' height='{_NODE_H}' rx='9' fill='rgba(217,119,6,0.10)'/>")
    p.append(f"<circle cx='{x + 13:.0f}' cy='{y + 14:.0f}' r='3.6' fill='{color}'/>")
    # SVG text can't ellipsize, so the budget is the only overflow guard. A stack
    # badge eats the right ~28px, so the label has to give that width back.
    p.append(
        f"<text x='{x + 23:.0f}' y='{y + 18:.0f}' font-size='10.5' font-weight='600' "
        f"fill='var(--cg-text-primary, #0f172a)'>{esc(_clip(nd.label, 8 if nd.stack > 1 else 12))}</text>"
    )
    if nd.sub:
        p.append(
            f"<text x='{x + 13:.0f}' y='{y + 31.5:.0f}' font-size='9.5' "
            f"fill='var(--cg-text-muted, #94a3b8)'>{esc(_clip(nd.sub, 18))}</text>"
        )
    if nd.stack > 1:
        p.append(
            f"<rect x='{x + _NODE_W - 28:.0f}' y='{y + 6:.0f}' width='22' height='13' rx='6.5' "
            f"fill='{color}' opacity='0.16'/>"
            f"<text x='{x + _NODE_W - 17:.0f}' y='{y + 15.6:.0f}' font-size='8.5' font-weight='700' "
            f"text-anchor='middle' fill='{color}'>&#215;{nd.stack}</text>"
        )
    p.append(f"<title>{esc(nd.title)}</title></g>")
    return "".join(p)


_LEGEND = (
    (K_USER, "Query"),
    (K_PLANNER, "Planner"),
    ("exec", "Executor"),
    (K_TOOL, "Tool call"),
    (K_ANSWER, "Answer"),
    (K_FORMATTER, "Formatter"),
)


def _legend_html(kinds: set) -> str:
    items = []
    for kind, label in _LEGEND:
        if kind not in kinds:
            continue
        color = _EXEC_VAR if kind == "exec" else _ROLE_VAR.get(kind, "")
        items.append(f'<span class="cg-lg"><span class="cg-sw" style="background:{color}"></span>{label}</span>')
    if not items:
        return ""
    return f'<div class="cg-chart-legend">{"".join(items)}</div>'


def render_minimap(trace: Trace) -> str:
    """Compact topology diagram: one node per phase, executor branches as lanes.

    Parallel *tool* calls are deliberately not lanes — a 10-wide `calculator`
    batch would blow the height up. They render as a stacked card plus an
    ``xN`` pill, which bounds the diagram at the 4-lane widest dispatch.
    """
    cols = _mm_columns(trace)
    if not cols:
        return ""
    n = len(cols)
    lanes_max = max(len(c) for c in cols)
    w = _PAD_X * 2 + n * _NODE_W + (n - 1) * _COL_GAP
    h = _PAD_TOP + lanes_max * _NODE_H + (lanes_max - 1) * _LANE_GAP + _PAD_BOT
    midy = _PAD_TOP + (h - _PAD_TOP - _PAD_BOT) / 2

    def pos(ci: int, li: int, k: int):
        x = _PAD_X + ci * (_NODE_W + _COL_GAP)
        y = midy + (li - (k - 1) / 2) * (_NODE_H + _LANE_GAP) - _NODE_H / 2
        return x, y

    parts = []
    for ci in range(n - 1):  # edges first so nodes paint over them
        a, b = cols[ci], cols[ci + 1]
        for ai in range(len(a)):
            for bi in range(len(b)):
                # 1->k and k->1 bundle; k->k (never occurs) pairs by index.
                if len(a) > 1 and len(b) > 1 and ai != bi:
                    continue
                x0, y0 = pos(ci, ai, len(a))
                x1, y1 = pos(ci + 1, bi, len(b))
                color = _EXEC_VAR if (len(a) > 1 or len(b) > 1) else _EDGE_COLOR
                parts.append(_edge(x0 + _NODE_W, y0 + _NODE_H / 2, x1, y1 + _NODE_H / 2, color))
    kinds = set()
    for ci, lanes in enumerate(cols):
        for li, nd in enumerate(lanes):
            x, y = pos(ci, li, len(lanes))
            parts.append(_node_svg(x, y, nd))
            kinds.add(nd.kind)

    svg = (
        f"<svg class='cg-mm-svg' viewBox='0 0 {w} {h:.0f}' width='100%' style='max-width:{w}px' "
        f"preserveAspectRatio='xMidYMid meet' role='img' xmlns='http://www.w3.org/2000/svg'>"
        f"{''.join(parts)}</svg>"
    )
    return f'<div class="cg-mm-wrap">{svg}{_legend_html(kinds)}</div>'


# --------------------------------------------------------------------------- #
# Trace tree
# --------------------------------------------------------------------------- #


def _step_meta(step) -> str:
    """Small right-aligned facts for a summary row."""
    bits = []
    if step.kind == K_TOOL:
        if step.content:
            bits.append(f"{_fmt_int(len(step.content))} B")
    else:
        out = step.meta.get("output_tokens")
        if out is not None:
            bits.append(f"{_fmt_int(out)} tok out")
        elif step.content:
            bits.append(f"{_fmt_int(len(step.content))} B")
    if not bits:
        return ""
    return f'<span class="cg-tr-meta">{esc(" · ".join(bits))}</span>'


def _tag_html(step) -> str:
    if not step.tag:
        return ""
    cls = "cg-tr-tag-err" if step.status == ERR else "cg-tr-tag-warn"
    return f'<span class="cg-tr-tag {cls}">{esc(step.tag)}</span>'


def _formatter_body(step) -> str:
    fields = step.meta.get("fields")
    parts = []
    if fields:
        rows = "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in fields)
        parts.append(f'<dl class="cg-tr-kv">{rows}</dl>')
        parts.append(
            f'<details class="cg-tr-sub"><summary>raw JSON</summary>'
            f'<pre class="cg-tr-pre">{esc(step.content)}</pre></details>'
        )
    else:
        parts.append(f'<pre class="cg-tr-pre">{esc(step.content)}</pre>')
    return "".join(parts)


def _step_html(step, open_default: bool = False) -> str:
    """One node: a `<details>` whose summary is the role row."""
    rcls = _role_class(step)
    role = _ROLE_TEXT.get(step.kind, step.kind)

    body_parts = []
    if step.kind == K_FORMATTER:
        body_parts.append(_formatter_body(step))
    else:
        args = step.meta.get("args")
        if args and args.strip() not in ("{}", ""):
            body_parts.append(
                f'<div class="cg-tr-field"><b>arguments</b>'
                f'<pre class="cg-tr-pre">{esc(args)}</pre></div>'
            )
        if step.content.strip():
            label = "result" if step.kind == K_TOOL else "output"
            body_parts.append(
                f'<div class="cg-tr-field"><b>{label}</b>'
                f'<pre class="cg-tr-pre">{esc(step.content)}</pre></div>'
            )
        elif step.status == DANGLING:
            body_parts.append(
                '<div class="cg-tr-none">The model requested this tool but no result '
                "was ever returned — the run ended first.</div>"
            )
        elif not step.children:
            body_parts.append('<div class="cg-tr-none">(no content)</div>')

    n_kids = len(step.children)
    kid_note = f'<span class="cg-tr-count">{n_kids} tool{"s" if n_kids != 1 else ""}</span>' if n_kids else ""

    # Tool calls render OUTSIDE the parent's <details>, as their own nested rail.
    # Keeping them inside would hide the graph's structure behind a collapsed
    # node — the shape of the run has to be visible without clicking anything.
    # A batch of parallel calls therefore shows as N siblings on one rail.
    kids = ""
    if step.children:
        kids = f'<div class="cg-tr-tools">{"".join(_step_html(c) for c in step.children)}</div>'

    row = (
        f'<span class="cg-tr-role">{esc(role)}</span>'
        f'<span class="cg-tr-name">{esc(step.label)}</span>'
        f"{_tag_html(step)}{kid_note}{_step_meta(step)}"
    )
    body = "".join(body_parts)
    if not body:
        # 78% of tool-dispatch turns carry no prose at all. Rendering those as a
        # <details> gives a disclosure arrow that opens onto nothing, so emit a
        # plain row instead — the node still marks the turn, it just doesn't lie
        # about having something inside.
        inner = f'<div class="cg-tr-sum cg-tr-flat">{row}</div>'
    else:
        inner = (
            f'<details class="cg-tr-node"{" open" if open_default else ""}>'
            f'<summary class="cg-tr-sum">{row}</summary>'
            f'<div class="cg-tr-body">{body}</div></details>'
        )
    return f'<div class="cg-tr-item {rcls}">{inner}{kids}</div>'


def _branch_html(branch) -> str:
    """One executor lane — a <details> so a 10-branch fan-out stays navigable.

    Collapsed by default: the minimap already shows the topology, and ten
    expanded branches bury the rest of the trace. The summary therefore has to
    carry enough to triage without opening — step/tool counts and a status mark.
    """
    lcls = "cg-r-err" if branch.status == ERR else "cg-r-exec"
    steps = "".join(_step_html(s) for s in branch.steps)
    if not steps:
        steps = '<div class="cg-tr-none">This executor produced no steps.</div>'
    task = ""
    if branch.task.strip():
        task = (
            f'<details class="cg-tr-task"><summary>task dispatch</summary>'
            f'<pre class="cg-tr-pre">{esc(branch.task)}</pre></details>'
        )
    meta = f'{len(branch.steps)} step{"s" if len(branch.steps) != 1 else ""} · {branch.n_tools} tools'
    if branch.status == ERR:
        mark = '<span class="cg-tr-lanemark cg-tr-lanemark-bad">&#10007;</span>'
    elif branch.status == WARN:
        mark = '<span class="cg-tr-lanemark cg-tr-lanemark-warn">!</span>'
    else:
        mark = '<span class="cg-tr-lanemark cg-tr-lanemark-ok">&#10003;</span>'
    # First line of the dispatch prompt, so a collapsed lane still says what it did.
    gist = f'<span class="cg-tr-lanegist">{esc(_clip(branch.task, 72))}</span>' if branch.task.strip() else ""
    return (
        f'<details class="cg-tr-lane {lcls}">'
        f'<summary class="cg-tr-lanehead">{mark}'
        f'<span class="cg-tr-lanetag">{esc(branch.label)}</span>'
        f"{gist}"
        f'<span class="cg-tr-lanemeta">{esc(meta)}</span></summary>'
        f"{task}{steps}</details>"
    )


def _fan_html(fan) -> str:
    if not fan.branches:
        return ""
    lanes = "".join(_branch_html(b) for b in fan.branches)
    return (
        f'<div class="cg-tr-fan cg-r-exec">'
        f'<div class="cg-tr-fanhead">&#9219; {esc(fan.label)}</div>'
        f"{lanes}</div>"
    )


def render_tree(trace: Trace) -> str:
    parts = []
    for item in trace.timeline:
        if isinstance(item, Fan):
            parts.append(_fan_html(item))
        else:
            parts.append(_step_html(item, open_default=item.kind == K_FORMATTER))
    return f'<div class="cg-tr">{"".join(parts)}</div>'


# --------------------------------------------------------------------------- #
# Crash card + summary line
# --------------------------------------------------------------------------- #


def render_crash(trace: Trace, parse_error: str = "") -> str:
    """For the 56 queries whose run died before writing a transcript."""
    tail = ""
    if parse_error:
        tail = f'<div class="cg-tr-crash-s">Judge could not parse an answer: {esc(parse_error)}</div>'
    return (
        '<div class="cg-tr-crash">'
        '<div class="cg-tr-crash-h">&#9888; Run crashed &mdash; no execution trace was recorded</div>'
        f'<pre class="cg-tr-pre">{esc(trace.crash_reason)}</pre>{tail}</div>'
    )


def render_summary(trace: Trace) -> str:
    """The one-line caption above the minimap."""
    wf = "multi-agent" if trace.workflow == "multi_agent" else "single-agent"
    bits = [wf]
    if trace.workflow == "multi_agent":
        rounds = trace.planner_iterations or 0
        bits.append(f'{rounds} planner round{"s" if rounds != 1 else ""}')
        bits.append(f"{trace.n_branches} executor{'s' if trace.n_branches != 1 else ''}")
    bits.append(f"{trace.n_messages} steps")
    bits.append(f"{trace.n_tools} tool call{'s' if trace.n_tools != 1 else ''}")
    if trace.n_errors:
        bits.append(f"{trace.n_errors} error{'s' if trace.n_errors != 1 else ''}")
    note = ""
    if trace.role_conflict or trace.orphans:
        note = (
            '<span class="cg-tr-warnnote" title="Agent roles are inferred from the '
            'transcript structure; this run did not match the usual shape. Use the raw '
            'transcript below to double-check.">roles inferred</span>'
        )
    return f'<div class="cg-mm-cap">{esc(" · ".join(bits))}{note}</div>'


# --------------------------------------------------------------------------- #
# Raw transcript (the collapsed fallback)
# --------------------------------------------------------------------------- #

_RAW_LABEL = {"human": "User", "ai": "Assistant", "tool": "Tool", "system": "System"}


def _raw_message(m: dict, cap: int) -> str:
    from src.leaderboard.trace_model import content_text

    role = m.get("type") or m.get("role") or "?"
    label = _RAW_LABEL.get(role, role)
    if role == "tool" and m.get("name"):
        label = f"Tool · {m['name']}"

    def _cap(t: str) -> str:
        if cap and len(t) > cap:
            return t[:cap] + f"\n… +{len(t) - cap:,} chars (truncated — see the graph above)"
        return t

    body = []
    text = content_text(m.get("content"))
    if text.strip():
        body.append(f'<pre class="cg-msg-text">{esc(_cap(text))}</pre>')
    for tc in m.get("tool_calls") or []:
        args = json.dumps(tc.get("args") or {}, indent=2, ensure_ascii=False, default=str)
        body.append(
            f'<div class="cg-tool-call">&rarr; <b>{esc(tc.get("name", "?"))}</b>'
            f'<pre class="cg-msg-text">{esc(_cap(args))}</pre></div>'
        )
    if not body:
        body.append('<span class="cg-msg-empty">(no content)</span>')
    return (
        f'<div class="cg-msg cg-msg-{esc(role)}">'
        f'<span class="cg-msg-role">{esc(label)}</span>'
        f'<div class="cg-msg-body">{"".join(body)}</div></div>'
    )


def render_raw_transcript(thread: dict | None, cap: int = 0) -> str:
    """The original flat chronological view, kept as a collapsed safety net.

    Roles in the tree are *derived*; if that grouping is ever wrong for a new
    transcript shape, this is the unmodified record to fall back on.
    """
    msgs = ((thread or {}).get("state") or {}).get("messages") or []
    if not msgs:
        return ""
    body = "".join(_raw_message(m, cap) for m in msgs)
    note = " (long messages truncated)" if cap else ""
    return (
        f'<details class="cg-tr-raw"><summary>Raw transcript &mdash; {len(msgs)} messages{note}</summary>'
        f'<div class="cg-transcript">{body}</div></details>'
    )
