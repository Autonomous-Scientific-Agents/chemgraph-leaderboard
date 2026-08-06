#!/usr/bin/env python
"""Sweep every staged trace through ``build_trace`` and assert it holds up.

There is no unit-test suite for the leaderboard, so this is the test for the
trace model. It is cheap (a few seconds over ~1544 files) and checks the four
properties that actually matter:

1. **No exceptions.** ``build_trace`` must never raise, whatever the transcript.
2. **Nothing dropped.** The set of source message indices reachable from the
   timeline equals the message count — a grammar bug that silently swallowed
   messages would otherwise be invisible.
3. **Roles agree.** ``role_conflict`` stays False, i.e. the structural planner
   signal and the metadata one never disagree.
4. **Planner count matches** ``state.planner_iterations`` for multi-agent runs.

Run offline against the local staging dir::

    HF_HUB_OFFLINE=1 python scripts/check_trace_model.py

Exits non-zero if anything fails, so it can gate a commit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.leaderboard.trace_model import Fan, build_trace  # noqa: E402

LOGS_DIR = Path(__file__).resolve().parent.parent / "hub_logs"


def _indices(trace) -> set:
    """Every source message index the timeline actually reaches."""
    seen: set = set()

    def visit(step) -> None:
        if step.idx >= 0:
            seen.add(step.idx)
        for child in step.children:
            visit(child)

    for item in trace.timeline:
        if isinstance(item, Fan):
            for branch in item.branches:
                if branch.idx >= 0:
                    seen.add(branch.idx)
                for step in branch.steps:
                    visit(step)
        else:
            visit(item)
    return seen


def main() -> int:
    if not LOGS_DIR.is_dir():
        print(f"No staging dir at {LOGS_DIR} — run scripts/upload_eval_logs.py first.")
        return 1

    n_files = n_bad = 0
    stats = {"messages": 0, "tools": 0, "errors": 0, "branches": 0, "crashes": 0}
    widest_fan = deepest = 0

    for workflow in ("single_agent", "multi_agent"):
        wf_dir = LOGS_DIR / workflow
        if not wf_dir.is_dir():
            continue
        for model_dir in sorted(wf_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            for path in sorted(model_dir.glob("state_thread_*.json")):
                n_files += 1
                rel = f"{workflow}/{model_dir.name}/{path.name}"
                try:
                    thread = json.loads(path.read_text())
                except (OSError, json.JSONDecodeError) as exc:
                    print(f"UNREADABLE {rel}: {exc}")
                    n_bad += 1
                    continue

                try:
                    trace = build_trace(thread, workflow)
                except Exception as exc:  # noqa: BLE001 - the whole point is to catch anything
                    print(f"RAISED     {rel}: {type(exc).__name__}: {exc}")
                    n_bad += 1
                    continue

                if not trace.ok:
                    stats["crashes"] += 1
                    continue

                seen = _indices(trace)
                if len(seen) != trace.n_messages:
                    missing = sorted(set(range(trace.n_messages)) - seen)
                    print(f"DROPPED    {rel}: reached {len(seen)}/{trace.n_messages}, missing {missing[:8]}")
                    n_bad += 1
                if trace.orphans:
                    print(f"ORPHANS    {rel}: {trace.orphans} message(s) outside the grammar")
                    n_bad += 1
                if trace.role_conflict:
                    print(f"ROLE       {rel}: structural and metadata planner signals disagree")
                    n_bad += 1

                if workflow == "multi_agent":
                    n_planner = sum(
                        1 for it in trace.timeline if not isinstance(it, Fan) and it.kind == "planner"
                    )
                    if n_planner != trace.planner_iterations:
                        print(f"PLANNER    {rel}: found {n_planner}, state says {trace.planner_iterations}")
                        n_bad += 1

                stats["messages"] += trace.n_messages
                stats["tools"] += trace.n_tools
                stats["errors"] += trace.n_errors
                stats["branches"] += trace.n_branches
                for item in trace.timeline:
                    if isinstance(item, Fan):
                        widest_fan = max(widest_fan, len(item.branches))
                deepest = max(deepest, trace.n_messages)

    print(
        f"\n{n_files} traces | {stats['messages']:,} messages | {stats['tools']:,} tool calls | "
        f"{stats['errors']:,} errors | {stats['branches']:,} executor branches | "
        f"{stats['crashes']} crashed runs"
    )
    print(f"widest parallel dispatch: {widest_fan} | longest trace: {deepest} messages")
    print(f"bad: {n_bad}")
    return 1 if n_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
