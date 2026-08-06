"""Lazy per-query log loader for the Full-table log-detail drawer.

When a user clicks a per-task accuracy cell, the browser sends
``"{workflow}|||{full_model}|||{col_name}"`` into a hidden textbox; the
``.input()`` handler calls :func:`render_log_panel`, which:

1. maps ``col_name`` -> benchmark category via the ``Tasks`` enum,
2. lazily fetches that (workflow, model)'s ``detail.json`` (judge scores) and the
   per-query ``state_thread_<N>.json`` transcripts from the logs dataset
   (``src/envs.py:LOGS_REPO``) — local ``hub_logs/`` staging dir first, else
   ``hf_hub_download`` (private dataset -> needs ``HF_TOKEN``),
3. renders an expandable ``<details>`` list (one per query in the category),
   each opening to the full transcript.

The files are the RAW eval logs (uploaded unchanged by
``scripts/upload_eval_logs.py``); all transformation/rendering happens here at
runtime. Every model-generated string is ``html.escape()``d before it enters the
returned HTML, because it lands in a ``gr.HTML`` component.
"""

from __future__ import annotations

import functools
import html
import json
from pathlib import Path

from src.about import _LOGO_URI, Tasks
from src.envs import LOGS_REPO, TOKEN
from src.leaderboard.trace_model import build_trace, content_text
from src.leaderboard.trace_render import (
    render_crash,
    render_minimap,
    render_raw_transcript,
    render_summary,
    render_tree,
)

# Above this much raw transcript text in one panel, the collapsed raw-transcript
# duplicate is what makes the payload heavy — cap that, not the graph.
_BIG_PANEL_CHARS = 150_000

# The panel header. Static, so it lives here next to the renderer rather than as
# a literal in app.py.
LOG_PANEL_HEAD_HTML = (
    '<div class="cg-drawer-head">'
    + (f'<img class="cg-drawer-mark" src="{_LOGO_URI}" alt="">' if _LOGO_URI else "")
    + '<div class="cg-drawer-titles">'
    '<div class="cg-drawer-title">Execution trace</div>'
    '<div class="cg-drawer-sub">per-query agent run for the selected model &amp; task</div>'
    "</div>"
    '<button id="cg-logpanel-close" class="cg-drawer-close" aria-label="Close">&#10005;</button>'
    "</div>"
)

# Repo root = three levels up (src/leaderboard/logs.py).
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
# Where scripts/upload_eval_logs.py stages files locally; checked before HF so a
# local `python app.py --local` needs no network / token.
_LOCAL_LOGS_DIR = _REPO_ROOT / "hub_logs"

# Full-table column header (e.g. "Reaction Energy") -> benchmark key
# (e.g. "reaction_energy"), the category stamped in each judge result.
_COLNAME_TO_BENCH = {t.value.col_name: t.value.benchmark for t in Tasks}


def _safe_name(full_model: str) -> str:
    """``org/model`` (or raw ``argo:...``) -> the dataset dir key.

    MUST match ``scripts/upload_eval_logs.py:safe_name`` exactly.
    """
    return full_model.replace("/", "__").replace(" ", "_").replace(":", "_")


@functools.lru_cache(maxsize=512)
def _fetch(workflow: str, key: str, fname: str) -> dict | None:
    """Load one JSON file (``detail.json`` or ``state_thread_<N>.json``).

    Local staging dir first, then the HF dataset. Returns the parsed object, or
    ``None`` on any miss/error (callers degrade gracefully).
    """
    rel = f"{workflow}/{key}/{fname}"
    local = _LOCAL_LOGS_DIR / workflow / key / fname
    if local.exists():
        try:
            return json.loads(local.read_text())
        except (OSError, json.JSONDecodeError):
            return None
    try:
        from huggingface_hub import hf_hub_download

        p = hf_hub_download(
            repo_id=LOGS_REPO, filename=rel, repo_type="dataset", token=TOKEN
        )
        return json.loads(Path(p).read_text())
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# HTML rendering
# --------------------------------------------------------------------------- #

_EMPTY_HTML = (
    '<div class="cg-log-empty">Click a task accuracy cell in the Full table to '
    "see that model's per-query logs.</div>"
)


def _err(msg: str) -> str:
    # msg may carry intentional <b> markup; callers escape their own interpolations.
    return f'<div class="cg-log-empty">{msg}</div>'


def _render_query(
    judge: dict,
    tokens: dict | None,
    thread: dict | None,
    workflow: str,
    crash: str,
    raw_cap: int,
    *,
    open_card: bool = False,
) -> str:
    """One query -> a collapsible card holding its execution graph."""
    qid = str(judge.get("query_id", "?"))
    query = content_text(judge.get("query", ""))
    parse_error = judge.get("parse_error")
    passed = bool(judge.get("score")) and not parse_error

    badge = (
        '<span class="cg-qbadge cg-qbadge-pass">&#10003;</span>'
        if passed
        else '<span class="cg-qbadge cg-qbadge-fail">&#10007;</span>'
    )
    q_short = query if len(query) <= 110 else query[:110] + "\u2026"
    summary = (
        f'<summary class="cg-q-summary">{badge}'
        f'<span class="cg-q-id">Q{html.escape(qid)}</span>'
        f'<span class="cg-q-text">{html.escape(q_short)}</span></summary>'
    )

    rows = [f'<div class="cg-q-field"><b>Query</b><pre>{html.escape(query)}</pre></div>']

    # A judge score of 0 on a healthy trace is a *semantic* miss (wrong number,
    # missing field) — it belongs here on the query, never on a graph node.
    rationale = judge.get("rationale")
    if rationale:
        rows.append(f'<div class="cg-q-field"><b>Judge rationale</b><pre>{html.escape(str(rationale))}</pre></div>')
    field_scores = judge.get("field_scores")
    if field_scores:
        chips = "".join(
            f'<span class="cg-chip cg-chip-{"ok" if v else "no"}">'
            f'{html.escape(str(k))}: {"&#10003;" if v else "&#10007;"}</span>'
            for k, v in field_scores.items()
        )
        rows.append(f'<div class="cg-q-field"><b>Field scores</b><div>{chips}</div></div>')

    if tokens:
        tu = tokens.get("token_usage") or {}
        ti = tokens.get("timing") or {}
        bits = []
        if tu.get("total_tokens") is not None:
            bits.append(f'{tu["total_tokens"]:,} tok')
        if tu.get("llm_calls") is not None:
            bits.append(f'{tu["llm_calls"]} LLM calls')
        if ti.get("agent_wall_s") is not None:
            bits.append(f'{ti["agent_wall_s"]:.1f}s')
        if bits:
            rows.append('<div class="cg-q-meta">' + " &middot; ".join(html.escape(b) for b in bits) + "</div>")

    trace = build_trace(thread, workflow, crash_reason=crash)
    if not trace.ok:
        rows.append(render_crash(trace, str(parse_error or "")))
    else:
        rows.append(render_summary(trace))
        rows.append(render_minimap(trace))
        rows.append(
            '<div class="cg-tr-bar">'
            '<button type="button" class="cg-tr-btn" data-cg-tr-all="open">Expand all</button>'
            '<button type="button" class="cg-tr-btn" data-cg-tr-all="close">Collapse all</button></div>'
        )
        rows.append(render_tree(trace))
        rows.append(render_raw_transcript(thread, cap=raw_cap))

    return (
        f'<details class="cg-q"{" open" if open_card else ""}>{summary}'
        f'<div class="cg-q-body">{"".join(rows)}</div></details>'
    )


def _thread_for(workflow: str, key: str, qid: str) -> dict | None:
    """``state_thread_<N>.json`` where N = query_id - 1 (query_id is 1-based)."""
    try:
        n = int(qid) - 1
    except (TypeError, ValueError):
        return None
    return _fetch(workflow, key, f"state_thread_{n}.json")


def _crash_reason(detail: dict, qid: str) -> str:
    """Transport-level error for a query whose run never wrote a transcript.

    ``raw_tool_calls`` is index-aligned with ``query_id - 1`` and its ``result``
    holds the raw provider failure (timeout, 403, connection error).
    """
    try:
        raw = detail.get("raw_tool_calls") or []
        return str(raw[int(qid) - 1].get("result") or "").strip()
    except (TypeError, ValueError, IndexError, AttributeError):
        return ""


def render_log_panel(payload: str) -> str:
    """Entry point wired to the hidden textbox's ``.input()`` event."""
    if not payload:
        return _EMPTY_HTML
    try:
        workflow, full_model, col_name = payload.split("|||")
    except ValueError:
        return _err("Malformed request.")

    bench = _COLNAME_TO_BENCH.get(col_name)
    if bench is None:
        return _err(f"Unknown task column: {html.escape(col_name)}.")

    key = _safe_name(full_model)
    detail = _fetch(workflow, key, "detail.json")
    if detail is None:
        return _err(f"No logs found for <b>{html.escape(full_model)}</b> ({html.escape(workflow)}).")

    judges = [j for j in detail.get("structured_judge_results", []) if j.get("category") == bench]
    if not judges:
        return _err(f"No <b>{html.escape(col_name)}</b> queries logged for <b>{html.escape(full_model)}</b>.")

    # per_query_results carries tokens/timing; key by query_id for a safe join.
    per_query = {}
    for pq in detail.get("per_query_results", []):
        qid = (pq.get("token_usage") or {}).get("query_id") or (pq.get("timing") or {}).get("query_id")
        if qid is not None:
            per_query[str(qid)] = pq

    # Fetch every transcript up front so the panel can size itself before
    # rendering (see _BIG_PANEL_CHARS). _fetch is cached, so this is not extra I/O.
    threads = {str(j.get("query_id", "")): _thread_for(workflow, key, str(j.get("query_id", ""))) for j in judges}
    bulk = sum(
        len(m.get("content") or "")
        for t in threads.values()
        if t
        for m in ((t.get("state") or {}).get("messages") or [])
        if isinstance(m.get("content"), str)
    )
    # The tree already shows every message; past this size the *duplicate* raw
    # transcript is what blows the payload up, so cap that rather than the graph.
    raw_cap = 800 if bulk > _BIG_PANEL_CHARS else 0

    n_pass = sum(1 for j in judges if bool(j.get("score")) and not j.get("parse_error"))
    n_crash = sum(1 for j in judges if j.get("parse_error"))
    pills = [
        f'<span class="cg-lp-pill">{html.escape(col_name)}</span>',
        f'<span class="cg-lp-pill">{html.escape(workflow.replace("_", "-"))}</span>',
        f'<span class="cg-lp-pill cg-lp-pill-ok">{n_pass}/{len(judges)} passed</span>',
    ]
    if n_crash:
        pills.append(f'<span class="cg-lp-pill cg-lp-pill-bad">{n_crash} crashed</span>')
    header = (
        f'<div class="cg-log-head">'
        f'<div class="cg-log-model">{html.escape(full_model)}</div>'
        f'<div class="cg-lp-pills">{"".join(pills)}</div></div>'
    )

    cards = []
    for i, j in enumerate(judges):
        qid = str(j.get("query_id", ""))
        cards.append(
            _render_query(
                j,
                per_query.get(qid),
                threads.get(qid),
                workflow,
                _crash_reason(detail, qid),
                raw_cap,
                open_card=(i == 0),
            )
        )

    return header + '<div class="cg-log-list">' + "".join(cards) + "</div>"
