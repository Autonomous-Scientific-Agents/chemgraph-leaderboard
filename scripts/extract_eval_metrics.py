#!/usr/bin/env python3
"""Extract token-usage and time-breakdown metrics from ChemGraph eval output.

Reads ``benchmark_*.json`` files produced by the (instrumented) ChemGraph
evaluation harness and writes a clean, compact set of metrics files to a local
directory.  This is **local-only** — it never touches the HF Hub.  The daily
pipeline pushes only per-category accuracy to HF (see
``chemgraph_to_leaderboard.py``); these performance metrics stay on disk for
local analysis / trend tracking.

Each ChemGraph benchmark file holds ``results[model][workflow]`` with the
instrumentation fields added by ``chemgraph.eval`` :

    timing_aggregate, token_usage_aggregate,
    per_query_timing, per_query_token_usage

Because the daily pipeline runs each workflow as a separate ``chemgraph-eval``
invocation, the workflows are typically spread across multiple benchmark files;
this script merges them by ``(model, workflow)``.

Outputs (under ``--out-dir``):
    metrics_<date>.json   full: metadata + per (model, workflow) aggregates
                          + per-query detail
    metrics_<date>.csv    one row per (model, workflow): tokens + time buckets
                          + accuracy, for quick spreadsheet/trend analysis

Usage::

    python scripts/extract_eval_metrics.py --eval-dir /path/to/eval_results \
        --out-dir eval_metrics
    python scripts/extract_eval_metrics.py \
        --benchmark-file /path/to/benchmark_2026-06-22.json --out-dir eval_metrics
"""

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract token/time metrics from ChemGraph eval output (local only).")
    p.add_argument("--eval-dir", type=Path, default=None, help="Directory containing benchmark_*.json files.")
    p.add_argument(
        "--benchmark-file",
        type=str,
        default=None,
        help="Specific benchmark_*.json to read. Overrides --eval-dir discovery.",
    )
    p.add_argument("--out-dir", type=Path, required=True, help="Local output directory for metrics files.")
    p.add_argument(
        "--no-per-query",
        action="store_true",
        help="Write only aggregates (omit per-query detail from the JSON).",
    )
    args = p.parse_args()
    if not args.benchmark_file and not args.eval_dir:
        p.error("one of --eval-dir or --benchmark-file is required")
    return args


def find_benchmarks(eval_dir: Path) -> List[Path]:
    """Return all benchmark_*.json files in *eval_dir*, sorted by name."""
    files = sorted(eval_dir.glob("benchmark_*.json"))
    # Skip the transient running report if present.
    return [f for f in files if f.name != "benchmark_running.json"]


def load_json(path: Path) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"  Warning: could not read {path}: {e}", file=sys.stderr)
        return None


def merge_benchmarks(paths: List[Path]) -> tuple[Dict[str, Dict[str, dict]], dict]:
    """Merge ``results[model][workflow]`` across benchmark files.

    Later files (sorted by name, i.e. timestamp) win on conflict.  Returns the
    merged results plus the metadata from the most recent file.
    """
    merged: Dict[str, Dict[str, dict]] = {}
    metadata: dict = {}
    for path in paths:
        data = load_json(path)
        if not data:
            continue
        metadata = data.get("metadata", metadata) or metadata
        for model, model_data in (data.get("results") or {}).items():
            if not isinstance(model_data, dict):
                continue
            dst = merged.setdefault(model, {})
            for workflow, wf_data in model_data.items():
                if isinstance(wf_data, dict):
                    dst[workflow] = wf_data
    return merged, metadata


def resolve_date(metadata: dict) -> str:
    """YYYY-MM-DD from benchmark metadata timestamp, else today (UTC)."""
    ts = metadata.get("timestamp")
    if ts:
        try:
            return datetime.fromisoformat(ts).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            pass
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# CSV columns: identity, accuracy, tokens, then time buckets.
CSV_FIELDS = [
    "model", "workflow", "n_queries",
    "accuracy", "n_correct",
    "total_tokens", "prompt_tokens", "completion_tokens", "cached_tokens",
    "avg_total_tokens_per_query", "llm_calls",
    "agent_wall_s", "llm_s", "tool_s", "tool_compute_s",
    "calc_load_s", "calc_load_count", "judge_s", "other_s", "agent_init_s",
]


def build_csv_row(model: str, workflow: str, wf_data: dict) -> Dict[str, Any]:
    t = wf_data.get("timing_aggregate", {}) or {}
    u = wf_data.get("token_usage_aggregate", {}) or {}
    # Accuracy: prefer structured judge, fall back to llm judge.
    acc = wf_data.get("structured_judge_aggregate") or wf_data.get("judge_aggregate") or {}
    row = {f: "" for f in CSV_FIELDS}
    row.update(
        model=model,
        workflow=workflow,
        n_queries=t.get("n_queries", u.get("n_queries", "")),
        accuracy=round(acc.get("accuracy", 0.0), 4) if acc else "",
        n_correct=acc.get("n_correct", ""),
        total_tokens=u.get("total_tokens", ""),
        prompt_tokens=u.get("prompt_tokens", ""),
        completion_tokens=u.get("completion_tokens", ""),
        cached_tokens=u.get("cached_tokens", ""),
        avg_total_tokens_per_query=u.get("avg_total_tokens_per_query", ""),
        llm_calls=u.get("llm_calls", t.get("llm_calls", "")),
        agent_wall_s=t.get("agent_wall_s", ""),
        llm_s=t.get("llm_s", ""),
        tool_s=t.get("tool_s", ""),
        tool_compute_s=t.get("tool_compute_s", ""),
        calc_load_s=t.get("calc_load_s", ""),
        calc_load_count=t.get("calc_load_count", ""),
        judge_s=t.get("judge_s", ""),
        other_s=t.get("other_s", ""),
        agent_init_s=t.get("agent_init_s", ""),
    )
    return row


def main() -> None:
    args = parse_args()

    if args.benchmark_file:
        paths = [Path(args.benchmark_file)]
    else:
        paths = find_benchmarks(args.eval_dir)
        if not paths:
            print(f"Error: no benchmark_*.json found in {args.eval_dir}", file=sys.stderr)
            sys.exit(1)

    print(f"Reading {len(paths)} benchmark file(s):")
    for p in paths:
        print(f"  - {p}")

    results, metadata = merge_benchmarks(paths)
    if not results:
        print("Error: no results found in benchmark file(s).", file=sys.stderr)
        sys.exit(1)

    date = resolve_date(metadata)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Build the metrics structure ----
    metrics: Dict[str, Any] = {
        "metadata": {
            "date": date,
            "timestamp": metadata.get("timestamp"),
            "judge_type": metadata.get("judge_type"),
            "source_benchmarks": [str(p) for p in paths],
        },
        "models": {},
    }
    csv_rows: List[Dict[str, Any]] = []

    for model in sorted(results):
        for workflow, wf_data in sorted(results[model].items()):
            entry: Dict[str, Any] = {
                "timing_aggregate": wf_data.get("timing_aggregate", {}),
                "token_usage_aggregate": wf_data.get("token_usage_aggregate", {}),
            }
            acc = wf_data.get("structured_judge_aggregate") or wf_data.get("judge_aggregate")
            if acc:
                entry["accuracy_aggregate"] = {
                    "accuracy": acc.get("accuracy"),
                    "n_correct": acc.get("n_correct"),
                    "n_queries": acc.get("n_queries"),
                }
            if not args.no_per_query:
                entry["per_query_timing"] = wf_data.get("per_query_timing", [])
                entry["per_query_token_usage"] = wf_data.get("per_query_token_usage", [])
            metrics["models"].setdefault(model, {})[workflow] = entry
            csv_rows.append(build_csv_row(model, workflow, wf_data))

    # ---- Write JSON ----
    json_path = args.out_dir / f"metrics_{date}.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\nWrote {json_path}")

    # ---- Write CSV ----
    csv_path = args.out_dir / f"metrics_{date}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    print(f"Wrote {csv_path}  ({len(csv_rows)} model/workflow rows)")


if __name__ == "__main__":
    main()
