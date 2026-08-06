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

from src.about import Tasks
from src.envs import LOGS_REPO, TOKEN

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
    return f'<div class="cg-log-empty">{msg}</div>'


def _esc(v) -> str:
    """Escape any value for HTML text; dicts/lists are pretty-printed JSON."""
    if isinstance(v, (dict, list)):
        v = json.dumps(v, indent=2, ensure_ascii=False)
    elif not isinstance(v, str):
        v = str(v)
    return html.escape(v)


def _content_text(content) -> str:
    """LangChain message content may be a string or a list of blocks."""
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", block.get("type", ""))))
            else:
                parts.append(str(block))
        return "\n".join(p for p in parts if p)
    return content if isinstance(content, str) else ("" if content is None else str(content))


def _render_message(m: dict) -> str:
    """One transcript message -> an HTML block."""
    role = m.get("type") or m.get("role") or "?"
    label = {"human": "User", "ai": "Assistant", "tool": "Tool", "system": "System"}.get(
        role, role
    )
    tool_name = m.get("name")
    if role == "tool" and tool_name:
        label = f"Tool · {html.escape(str(tool_name))}"

    body_parts = []
    text = _content_text(m.get("content"))
    if text.strip():
        body_parts.append(f'<pre class="cg-msg-text">{html.escape(text)}</pre>')

    # Assistant tool calls (the model deciding to call a tool).
    for tc in m.get("tool_calls") or []:
        name = html.escape(str(tc.get("name", "?")))
        args = _esc(tc.get("args", {}))
        body_parts.append(
            f'<div class="cg-tool-call">&rarr; <b>{name}</b>'
            f'<pre class="cg-msg-text">{args}</pre></div>'
        )

    if not body_parts:
        body_parts.append('<span class="cg-msg-empty">(no content)</span>')

    return (
        f'<div class="cg-msg cg-msg-{html.escape(role)}">'
        f'<span class="cg-msg-role">{html.escape(label)}</span>'
        f'<div class="cg-msg-body">{"".join(body_parts)}</div></div>'
    )


def _render_query(judge: dict, tokens: dict | None, state: dict | None) -> str:
    """One query -> a collapsible <details> card."""
    qid = str(judge.get("query_id", "?"))
    query = _content_text(judge.get("query", ""))
    parse_error = judge.get("parse_error")
    passed = bool(judge.get("score")) and not parse_error
    badge = (
        '<span class="cg-qbadge cg-qbadge-pass">&#10003;</span>'
        if passed
        else '<span class="cg-qbadge cg-qbadge-fail">&#10007;</span>'
    )
    q_short = query if len(query) <= 90 else query[:90] + "…"
    summary = (
        f'<summary class="cg-q-summary">{badge}'
        f'<span class="cg-q-id">Q{html.escape(qid)}</span>'
        f'<span class="cg-q-text">{html.escape(q_short)}</span></summary>'
    )

    rows = []
    rows.append(f'<div class="cg-q-field"><b>Query</b><pre>{html.escape(query)}</pre></div>')

    rationale = judge.get("rationale")
    if rationale:
        rows.append(
            f'<div class="cg-q-field"><b>Judge rationale</b>'
            f'<pre>{html.escape(str(rationale))}</pre></div>'
        )
    if parse_error:
        rows.append(
            f'<div class="cg-q-field"><b>Parse error</b>'
            f'<pre>{html.escape(str(parse_error))}</pre></div>'
        )
    field_scores = judge.get("field_scores")
    if field_scores:
        chips = "".join(
            f'<span class="cg-chip cg-chip-{"ok" if v else "no"}">'
            f'{html.escape(str(k))}: {"✓" if v else "✗"}</span>'
            for k, v in field_scores.items()
        )
        rows.append(f'<div class="cg-q-field"><b>Field scores</b><div>{chips}</div></div>')

    # Token / timing chips (from detail.per_query_results), best-effort.
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
            rows.append(
                '<div class="cg-q-meta">'
                + " · ".join(html.escape(b) for b in bits)
                + "</div>"
            )

    # Transcript.
    if state is None:
        rows.append('<div class="cg-q-field"><b>Transcript</b>'
                    '<div class="cg-msg-empty">(transcript unavailable)</div></div>')
    else:
        msgs = (state.get("state") or {}).get("messages") or []
        # Skip the leading human turn — it repeats the query shown above.
        body = "".join(
            _render_message(m)
            for i, m in enumerate(msgs)
            if not (i == 0 and (m.get("type") or m.get("role")) == "human")
        )
        rows.append(f'<div class="cg-q-field"><b>Transcript</b>'
                    f'<div class="cg-transcript">{body or "(empty)"}</div></div>')

    return f'<details class="cg-q">{summary}<div class="cg-q-body">{"".join(rows)}</div></details>'


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
        return _err(
            f"No logs found for <b>{html.escape(full_model)}</b> "
            f"({html.escape(workflow)})."
        )

    judges = [
        j for j in detail.get("structured_judge_results", [])
        if j.get("category") == bench
    ]
    if not judges:
        return _err(
            f"No <b>{html.escape(col_name)}</b> queries logged for "
            f"<b>{html.escape(full_model)}</b>."
        )

    # per_query_results carries tokens/timing; key by query_id for a safe join.
    per_query = {}
    for pq in detail.get("per_query_results", []):
        qid = (pq.get("token_usage") or {}).get("query_id") or (
            pq.get("timing") or {}
        ).get("query_id")
        if qid is not None:
            per_query[str(qid)] = pq

    n_pass = sum(1 for j in judges if bool(j.get("score")) and not j.get("parse_error"))
    header = (
        f'<div class="cg-log-head">'
        f'<div class="cg-log-model">{html.escape(full_model)}</div>'
        f'<div class="cg-log-sub">{html.escape(col_name)} · '
        f'{html.escape(workflow.replace("_", "-"))} · '
        f'{n_pass}/{len(judges)} passed</div></div>'
    )

    cards = []
    for j in judges:
        qid = str(j.get("query_id", ""))
        # N = query_id - 1 (query_id is 1-based, state_thread index is 0-based).
        state = None
        try:
            n = int(qid) - 1
            state = _fetch(workflow, key, f"state_thread_{n}.json")
        except (TypeError, ValueError):
            pass
        cards.append(_render_query(j, per_query.get(qid), state))

    return header + '<div class="cg-log-list">' + "".join(cards) + "</div>"
