"""Turn a raw ``state_thread_<N>.json`` into a normalized execution graph.

This module is deliberately **pure and HTML-free** — it only reshapes data, so it
can be swept over every trace on disk and asserted on without rendering anything
(see ``scripts/check_trace_model.py``). All markup lives in ``trace_render.py``.

The shape it produces is a flat :attr:`Trace.timeline` of :class:`Step` items with
:class:`Fan` items marking fan-out. One structure feeds both renderers: the tree
walks it directly, the minimap turns each entry into one column.

Roles are DERIVED, not stored
-----------------------------
The logs carry no agent name anywhere — ``ai.name`` is ``null`` on every AI message,
there is no handoff tool, and no LangGraph node metadata survives. Two independent
signals recover the role and agree on all 1600 (workflow, model, query) triples:

* **structural** — a planner turn is an ``ai`` at index 1, or an ``ai`` immediately
  preceded by another ``ai``. The resulting count equals ``state.planner_iterations``
  exactly, with no violations across all 766 multi-agent traces.
* **metadata** — planner turns are invoked with structured output, so their LLM
  metadata is stripped: ``usage_metadata is None`` and ``response_metadata == {}``.

We decide with the structural signal and cross-check against the metadata one,
setting :attr:`Trace.role_conflict` on any disagreement. That flag is the honest
signal to the UI that the grouping is inferred; the raw transcript stays available
as the escape hatch.

Grammar (verified over all 1544 traces, zero violations):

* single-agent — ``H A (T+ A)* H``; one agent, no branches.
* multi-agent  — ``USER, PLANNER, {dispatch, EXECUTOR…}*, PLANNER(synthesis), FORMATTER``
  where a ``human`` at ``0 < i < len-1`` is an executor task dispatch.
* in both, the **trailing** ``human`` is the Formatter's structured answer, not a
  user turn.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

# Step.status values. WARN means "ran, but something is off" (truncated generation,
# a tool call that never returned); DANGLING is specifically a tool_call with no
# matching tool message — 6 such cases exist, all single-agent Gemini.
OK = "ok"
ERR = "err"
WARN = "warn"
DANGLING = "dangling"

# Kinds double as the CSS role key (``--cg-role-<kind>``), so keep them in sync
# with the tokens in css_html_js.py.
K_USER = "user"
K_PLANNER = "planner"
K_AGENT = "agent"
K_ANSWER = "answer"
K_TOOL = "tool"
K_FORMATTER = "formatter"

# Tool errors surface as ``Error: SomeError("...")`` — pull the exception name out
# for the node tag. Anchored, so it can't match the word "error" inside a payload.
_ERR_RE = re.compile(r"^\s*Error:\s*([A-Za-z][A-Za-z0-9_]*(?:Error|Exception))")


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass
class Step:
    """One node in the graph. ``content`` is RAW — the renderer escapes it."""

    kind: str
    label: str
    content: str = ""
    status: str = OK
    tag: str = ""  # short badge text, e.g. "ValueError" / "truncated"
    idx: int = -1  # source message index; -1 for synthetic nodes
    meta: dict = field(default_factory=dict)  # args, tokens, formatter fields
    children: list["Step"] = field(default_factory=list)  # resolved tool calls


@dataclass
class Branch:
    """One executor sub-agent's run (multi-agent only)."""

    label: str = ""
    task: str = ""  # verbatim dispatch prompt
    steps: list[Step] = field(default_factory=list)
    status: str = OK
    n_tools: int = 0
    idx: int = -1


@dataclass
class Fan:
    """A parallel dispatch: one planner turn spawning N executor branches."""

    label: str = ""
    branches: list[Branch] = field(default_factory=list)


@dataclass
class Trace:
    workflow: str
    ok: bool = True
    crash_reason: str = ""
    timeline: list = field(default_factory=list)  # list[Step | Fan]
    n_messages: int = 0
    n_tools: int = 0
    n_errors: int = 0
    n_branches: int = 0
    planner_iterations: int = 0
    role_conflict: bool = False  # the two planner signals disagreed
    orphans: int = 0  # messages that didn't fit the grammar (still rendered)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def content_text(content) -> str:
    """LangChain message content -> plain text.

    Every message in the current corpus carries a plain ``str``, but the list-of-
    blocks form is part of the LangChain contract, so handle it rather than crash
    on a future provider that uses it.
    """
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", block.get("type", ""))))
            else:
                parts.append(str(block))
        return "\n".join(p for p in parts if p)
    if isinstance(content, str):
        return content
    return "" if content is None else str(content)


def _is_planner(msgs: list, i: int) -> bool:
    """Structural planner signal — the decision we act on."""
    return msgs[i].get("type") == "ai" and (i == 1 or msgs[i - 1].get("type") == "ai")


def _looks_planner(m: dict) -> bool:
    """Independent metadata signal — used only to cross-check the structural one."""
    return m.get("usage_metadata") is None and not (m.get("response_metadata") or {})


def _ai_meta(m: dict) -> dict:
    """Token counts + finish reason for an AI turn, when the provider reported them."""
    out = {}
    um = m.get("usage_metadata") or {}
    for k in ("input_tokens", "output_tokens", "total_tokens"):
        if um.get(k) is not None:
            out[k] = um[k]
    fr = (m.get("response_metadata") or {}).get("finish_reason")
    if fr:
        out["finish_reason"] = fr
    return out


def _tool_status(tm: dict) -> tuple[str, str]:
    """Classify a tool result.

    Two independent failure channels, and only checking the first misses 141 of
    391 real failures:

    1. ``status == "error"`` — the LangChain-level tool exception (250 cases).
    2. ``status == "success"`` but the JSON payload says ``{"status": "failure"}``
       — a *domain* failure the tool reported normally (141 cases).

    Substring-matching "error" is wrong here: it matches 934 messages, because
    healthy ``extract_output_json`` payloads legitimately contain an ``error`` key.
    """
    c = tm.get("content")
    c = c if isinstance(c, str) else ""
    if tm.get("status") == "error":
        hit = _ERR_RE.match(c)
        return ERR, (hit.group(1) if hit else "error")
    s = c.lstrip()
    if s[:1] == "{":  # cheap guard so we don't json-parse every tool payload twice
        try:
            obj = json.loads(s)
        except (ValueError, TypeError):
            return OK, ""
        if isinstance(obj, dict) and obj.get("status") == "failure":
            return ERR, str(obj.get("error_type") or "failure")
    return OK, ""


def _tool_windows(msgs: list) -> dict:
    """Map each AI message index -> the contiguous run of tool messages after it.

    Tool results always immediately follow the turn that requested them (the
    ``A T+ A`` grammar holds with zero violations), so this window *is* the set of
    candidates for that turn's calls.

    Scoping the join to a window is not a nicety — it is required for correctness.
    Gemini reuses literal ids like ``call_1``/``call_2`` across different turns, so
    a single global ``{tool_call_id: message}`` map silently collapses them and
    loses earlier results (22 traces in this corpus). Ids are unique *within* a
    window; they are not unique across the trace.
    """
    windows: dict = {}
    for i, m in enumerate(msgs):
        if m.get("type") != "ai":
            continue
        run = []
        j = i + 1
        while j < len(msgs) and msgs[j].get("type") == "tool":
            run.append((j, msgs[j]))
            j += 1
        if run:
            windows[i] = run
    return windows


def _tool_result_step(name: str, args: str, j: int, tm: dict) -> Step:
    status, tag = _tool_status(tm)
    return Step(
        kind=K_TOOL,
        label=name,
        idx=j,
        status=status,
        tag=tag,
        content=content_text(tm.get("content")),
        meta={"args": args},
    )


def _agent_step(m: dict, i: int, windows: dict) -> Step:
    """An AI turn plus the tool calls it issued, resolved to their results.

    A turn with tool calls is a "reasoning" step; one without is the branch's
    answer. ``invalid_tool_calls`` is always empty in this corpus, so it is ignored.
    """
    tcs = m.get("tool_calls") or []
    st = Step(
        kind=K_AGENT if tcs else K_ANSWER,
        label="Reasoning" if tcs else "Answer",
        idx=i,
        content=content_text(m.get("content")),
        meta=_ai_meta(m),
    )
    if st.meta.get("finish_reason") == "length":
        st.status, st.tag = WARN, "truncated"

    window = windows.get(i, [])
    by_id: dict = {}
    for j, tm in window:
        by_id.setdefault(tm.get("tool_call_id"), []).append((j, tm))
    consumed: set = set()

    for tc in tcs:
        name = str(tc.get("name") or "?")
        # json.dumps builds a new string; the source dict is never mutated (it is
        # shared out of logs._fetch's lru_cache).
        args = json.dumps(tc.get("args") or {}, indent=2, ensure_ascii=False, default=str)
        queue = by_id.get(tc.get("id")) or []
        hit = None
        while queue:
            j, tm = queue.pop(0)
            if j not in consumed:
                hit = (j, tm)
                break
        if hit is None:
            # The model asked for a tool and no result came back (6 cases, all
            # single-agent Gemini). Show the call with an unresolved marker.
            st.children.append(Step(kind=K_TOOL, label=name, status=DANGLING, tag="no result", meta={"args": args}))
            continue
        consumed.add(hit[0])
        st.children.append(_tool_result_step(name, args, hit[0], hit[1]))

    # Any result in the window no call claimed. Shouldn't happen, but attach it
    # rather than drop it — a silently swallowed message is the one failure mode
    # the sweep in scripts/check_trace_model.py exists to catch.
    for j, tm in window:
        if j not in consumed:
            orphan = _tool_result_step(str(tm.get("name") or "?"), "", j, tm)
            orphan.tag = orphan.tag or "unlinked"
            st.children.append(orphan)

    if any(c.status == ERR for c in st.children):
        st.status = ERR
    elif st.status == OK and any(c.status == DANGLING for c in st.children):
        st.status, st.tag = WARN, "no tool result"
    return st


def _formatter_step(m: dict, i: int) -> Step:
    """The trailing ``human`` message: the Formatter's structured answer.

    Always valid JSON in the current corpus, with the fixed key set
    ``{smiles, scalar_answer, dipole, vibrational_answer, ir_spectrum, atoms_data}``.
    Degrade to raw text rather than raising if that ever stops being true.
    """
    raw = content_text(m.get("content"))
    st = Step(kind=K_FORMATTER, label="Structured answer", idx=i, content=raw)
    try:
        obj = json.loads(raw)
    except (ValueError, TypeError):
        obj = None
    if isinstance(obj, dict):
        st.meta["fields"] = [(k, v) for k, v in obj.items() if v is not None]
        if not st.meta["fields"]:
            st.status, st.tag = WARN, "all fields null"
    else:
        st.status, st.tag = WARN, "not JSON"
    return st


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def _build_single(tr: Trace, msgs: list, lo: int, hi: int, windows: dict) -> None:
    """Single-agent: a flat chain. Tool messages hang off their calling AI turn."""
    for i in range(lo, hi):
        t = msgs[i].get("type")
        if t == "tool":
            continue  # already attached to its parent via tool_call_id
        if t != "ai":
            # Doesn't fit `H A (T+ A)* H`. Never happens today; surface it rather
            # than dropping the message on the floor.
            tr.orphans += 1
        tr.timeline.append(_agent_step(msgs[i], i, windows))


def _build_multi(tr: Trace, msgs: list, lo: int, hi: int, windows: dict) -> None:
    """Multi-agent: planner turns punctuating groups of parallel executor branches."""
    fan: Fan | None = None
    branch: Branch | None = None
    n_planner = 0

    for i in range(lo, hi):
        m = msgs[i]
        t = m.get("type")

        if _is_planner(msgs, i):
            if not _looks_planner(m):
                tr.role_conflict = True
            n_planner += 1
            fan = branch = None  # a planner turn closes the current dispatch group
            tr.timeline.append(
                Step(
                    kind=K_PLANNER,
                    label=f"Planner {n_planner}",
                    idx=i,
                    content=content_text(m.get("content")),
                    meta=_ai_meta(m),
                )
            )
        elif t == "human":
            # An executor task dispatch. Consecutive dispatches without an
            # intervening planner turn are one parallel batch.
            if fan is None:
                fan = Fan()
                tr.timeline.append(fan)
            branch = Branch(task=content_text(m.get("content")), idx=i)
            fan.branches.append(branch)
        elif t == "tool":
            continue
        elif t == "ai" and branch is not None:
            if _looks_planner(m):
                tr.role_conflict = True
            branch.steps.append(_agent_step(m, i, windows))
        else:
            # An AI turn outside any branch and not flagged as a planner.
            tr.orphans += 1
            tr.timeline.append(_agent_step(m, i, windows))

    _label_fans(tr)


def _label_fans(tr: Trace) -> None:
    for it in tr.timeline:
        if not isinstance(it, Fan):
            continue
        n = len(it.branches)
        it.label = "1 executor" if n == 1 else f"{n} executors in parallel"
        for k, b in enumerate(it.branches, 1):
            b.label = "Executor" if n == 1 else f"Executor {k}/{n}"
            b.n_tools = sum(len(s.children) for s in b.steps)
            if any(s.status == ERR for s in b.steps):
                b.status = ERR
            elif any(s.status == WARN for s in b.steps):
                b.status = WARN
    # A trailing planner turn with no dispatch after it is the synthesis/FINISH turn.
    if tr.timeline and isinstance(tr.timeline[-1], Step) and tr.timeline[-1].kind == K_PLANNER:
        tr.timeline[-1].label = "Synthesis"


def _finalize(tr: Trace) -> None:
    """Roll per-node status up into the panel-level counters."""

    def visit(s: Step) -> None:
        if s.kind == K_TOOL:
            tr.n_tools += 1
        # Count where the failure actually happened. A parent AI turn inherits ERR
        # from its tool child, so counting every ERR node would roughly double the
        # tally the header shows the user.
        if s.status == ERR and not any(c.status == ERR for c in s.children):
            tr.n_errors += 1
        for c in s.children:
            visit(c)

    for it in tr.timeline:
        if isinstance(it, Fan):
            tr.n_branches += len(it.branches)
            for b in it.branches:
                for s in b.steps:
                    visit(s)
        else:
            visit(it)


def build_trace(thread: dict | None, workflow: str, *, crash_reason: str = "") -> Trace:
    """``state_thread_<N>.json`` -> :class:`Trace`. Never raises.

    ``thread`` is ``None`` for the 56 queries whose run crashed before writing a
    transcript (``parse_error`` in detail.json is a perfect biconditional with the
    file being absent); ``crash_reason`` then carries the transport-level error.
    """
    if not isinstance(thread, dict):
        return Trace(
            workflow=workflow,
            ok=False,
            crash_reason=crash_reason or "No transcript was recorded for this query.",
        )

    state = thread.get("state") or {}
    msgs = state.get("messages") or []
    if not msgs:
        return Trace(
            workflow=workflow,
            ok=False,
            crash_reason=crash_reason or "The recorded transcript is empty.",
        )

    tr = Trace(
        workflow=workflow,
        n_messages=len(msgs),
        planner_iterations=int(state.get("planner_iterations") or 0),
    )

    windows = _tool_windows(msgs)

    # The trailing human message is the Formatter's answer, not a user turn.
    fmt_i = len(msgs) - 1 if len(msgs) > 1 and msgs[-1].get("type") == "human" else None
    end = fmt_i if fmt_i is not None else len(msgs)

    tr.timeline.append(Step(kind=K_USER, label="Query", idx=0, content=content_text(msgs[0].get("content"))))

    build = _build_multi if workflow == "multi_agent" else _build_single
    build(tr, msgs, 1, end, windows)

    if fmt_i is not None:
        tr.timeline.append(_formatter_step(msgs[fmt_i], fmt_i))

    _finalize(tr)
    return tr
