#!/usr/bin/env python3
"""Merge ALCF open-weight token metrics into the closed-model metrics CSV.

The ALCF eval did not store per-run token usage in its benchmark JSONs, but the
per-model totals were recorded in ``dataset/alcf_tokens.json`` (sourced from
``alcf/IE_Eval/plot_tokens.py``): single-agent totals are **measured** (the ALCF
vLLM endpoint returns a ``usage`` field), multi-agent totals are **estimated**.

The leaderboard's metrics loader (``src/leaderboard/metrics.py``) reads a single
newest ``metrics_*.csv`` and keys it by ``full_model`` — it does NOT merge across
files. So ALCF rows must live in the *same* CSV as the closed models. This script
takes a base metrics CSV (the closed-model one that has real tokens, e.g. the
prod ``metrics/metrics_2026-06-23.csv``), appends one row per (ALCF model,
workflow), and writes a combined, newer-dated CSV.

    tokens_per_query = <workflow>_total / n_queries   (n_queries = 40)

Usage::

    python scripts/alcf_token_metrics.py \
        --base-csv /path/to/metrics_2026-06-23.csv \
        --alcf-eval-dir /home/zhye/alcf/IE_Eval \
        --out dataset/metrics/metrics_$(date -u +%F).csv
"""

import argparse
import csv
import glob
import json
import os
from datetime import datetime, timezone
from pathlib import Path

# Must match scripts/extract_eval_metrics.py::CSV_FIELDS so metrics.py reads it.
CSV_FIELDS = [
    "model", "workflow", "n_queries",
    "accuracy", "n_correct",
    "total_tokens", "prompt_tokens", "completion_tokens", "cached_tokens",
    "avg_total_tokens_per_query", "llm_calls",
    "agent_wall_s", "llm_s", "tool_s", "tool_compute_s",
    "calc_load_s", "calc_load_count", "judge_s", "other_s", "agent_init_s",
]

_WF_TO_TOTAL = {"single_agent": "single_total", "multi_agent": "multi_total"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-csv", type=Path, required=True,
                   help="Closed-model metrics CSV to merge into (must have real tokens).")
    p.add_argument("--alcf-tokens", type=Path,
                   default=Path(__file__).resolve().parent.parent / "dataset" / "alcf_tokens.json",
                   help="alcf_tokens.json with per-model single/multi totals.")
    p.add_argument("--alcf-eval-dir", type=Path, default=Path("/home/zhye/alcf/IE_Eval"),
                   help="Dir with ALCF *_detail.json (for accuracy).")
    p.add_argument("--out", type=Path, default=None,
                   help="Output CSV path. Default dataset/metrics/metrics_<today>.csv.")
    return p.parse_args()


def load_alcf_totals(path: Path) -> dict:
    with open(path) as fp:
        return json.load(fp).get("models", {})


def alcf_accuracy(eval_dir: Path) -> dict:
    """{(full_model, workflow): (accuracy_fraction, n_queries)} from *_detail.json."""
    out: dict = {}
    for f in glob.glob(str(eval_dir / "*_detail.json")):
        base = os.path.basename(f)
        workflow = None
        for wf in ("single_agent", "multi_agent"):
            suffix = f"_{wf}_detail.json"
            if base.endswith(suffix):
                # "nvidia_nemotron-3-super-120b" -> "nvidia/nemotron-3-super-120b"
                model = base[: -len(suffix)].replace("_", "/", 1)
                workflow = wf
                break
        if workflow is None:
            continue
        try:
            data = json.load(open(f))
        except (OSError, json.JSONDecodeError):
            continue
        scores = [q.get("score") for q in data.get("structured_judge_results", [])
                  if q.get("score") is not None]
        if scores:
            out[(model, workflow)] = (sum(scores) / len(scores), len(scores))
    return out


def build_alcf_rows(totals: dict, acc: dict) -> list:
    rows = []
    for model, tok in totals.items():
        for workflow, total_key in _WF_TO_TOTAL.items():
            total = tok.get(total_key)
            if total is None:
                continue  # e.g. gemma-4-26B has no multi run
            accuracy, n = acc.get((model, workflow), (None, 40))
            n = n or 40
            row = {f: "" for f in CSV_FIELDS}
            row.update(
                model=model,
                workflow=workflow,
                n_queries=n,
                accuracy=round(accuracy, 4) if accuracy is not None else "",
                n_correct=round(accuracy * n) if accuracy is not None else "",
                total_tokens=int(total),
                avg_total_tokens_per_query=round(total / n),
            )
            rows.append(row)
    return rows


def main() -> None:
    args = parse_args()

    with open(args.base_csv, newline="") as f:
        base_rows = list(csv.DictReader(f))
    # Keep only closed-model rows (drop any pre-existing ALCF rows so re-runs are idempotent).
    alcf_models = set(load_alcf_totals(args.alcf_tokens).keys())
    base_rows = [r for r in base_rows if r.get("model") not in alcf_models]

    totals = load_alcf_totals(args.alcf_tokens)
    acc = alcf_accuracy(args.alcf_eval_dir)
    alcf_rows = build_alcf_rows(totals, acc)

    out = args.out
    if out is None:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        out = Path(__file__).resolve().parent.parent / "dataset" / "metrics" / f"metrics_{date}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in base_rows:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})
        for r in alcf_rows:
            w.writerow(r)

    print(f"Wrote {out}")
    print(f"  closed-model rows: {len(base_rows)}")
    print(f"  ALCF rows added:   {len(alcf_rows)}")
    for r in alcf_rows:
        print(f"    {r['model']:42} {r['workflow']:13} "
              f"tpq={r['avg_total_tokens_per_query']:>7} acc={r['accuracy']}")


if __name__ == "__main__":
    main()
