#!/usr/bin/env python3
"""Transform ChemGraph benchmark results into HF leaderboard-compatible JSON files.

Reads the latest ``benchmark_*.json`` from a ChemGraph eval output directory,
extracts per-model single_agent judge scores, groups queries by the
``category`` field in each judge detail entry, and writes per-model results +
request JSON files that the leaderboard app can consume.

Optionally pushes the generated files to the HF Hub datasets.

Usage::

    # Generate files locally (auto-detect latest benchmark)
    python scripts/chemgraph_to_leaderboard.py \
        --eval-dir /path/to/ChemGraph/eval_results \
        --model-map dataset/model_map.json \
        --results-outdir hub_results \
        --requests-outdir hub_requests

    # Generate from a specific benchmark file and push to HF Hub
    python scripts/chemgraph_to_leaderboard.py \
        --eval-dir /path/to/ChemGraph/eval_results \
        --benchmark-file /path/to/benchmark_2026-04-13.json \
        --model-map dataset/model_map.json \
        --push-to-hub

    # Auto-detect latest benchmark and push to HF Hub
    python scripts/chemgraph_to_leaderboard.py \
        --eval-dir /path/to/ChemGraph/eval_results \
        --model-map dataset/model_map.json \
        --push-to-hub

    # Process ALL benchmark files in a directory (batch / backfill)
    python scripts/chemgraph_to_leaderboard.py \
        --eval-dir /path/to/ChemGraph/eval_results \
        --model-map dataset/model_map.json \
        --all --push-to-hub
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# ---- HF Hub repo IDs ----------------------------------------------------
# Import from src/envs.py when running inside the repo.  Fall back to
# hardcoded defaults so the script can also be used standalone.
try:
    # Add the repo root to sys.path so ``src`` is importable.
    _SCRIPT_DIR = Path(__file__).resolve().parent
    _REPO_ROOT = _SCRIPT_DIR.parent
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from src.envs import RESULTS_REPO, QUEUE_REPO as REQUESTS_REPO  # noqa: F401
    from src.about import Tasks as _Tasks

    KNOWN_CATEGORIES: Set[str] = {t.value.benchmark for t in _Tasks}
except Exception:
    RESULTS_REPO = "Autonomous-Scientific-Agents/results"
    REQUESTS_REPO = "Autonomous-Scientific-Agents/requests"
    KNOWN_CATEGORIES = set()  # validation disabled when running standalone


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transform ChemGraph eval results to HF leaderboard format.")
    p.add_argument(
        "--eval-dir",
        type=Path,
        default=None,
        help="Path to ChemGraph eval_results directory. Required when --benchmark-file is not given.",
    )
    p.add_argument(
        "--model-map",
        type=Path,
        default=None,
        help="JSON file mapping ChemGraph short model names to org/model display names.",
    )
    p.add_argument(
        "--results-outdir",
        type=Path,
        default=Path("hub_results"),
        help="Output directory for leaderboard results JSON files.",
    )
    p.add_argument(
        "--requests-outdir",
        type=Path,
        default=Path("hub_requests"),
        help="Output directory for leaderboard request JSON files.",
    )
    p.add_argument(
        "--benchmark-file",
        type=str,
        default=None,
        help="Specific benchmark_*.json file to process. If not given, the latest file by timestamp is used.",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Process all benchmark_*.json files in --eval-dir (not just the latest).",
    )
    p.add_argument(
        "--workflow",
        type=str,
        default="single_agent",
        help="Workflow type to extract (default: single_agent).",
    )
    p.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push generated files to the HF Hub datasets.",
    )
    p.add_argument(
        "--model-dtype",
        default="torch.float16",
        help="Value for config.model_dtype in results JSON.",
    )
    p.add_argument(
        "--default-sha",
        default="main",
        help="Value for config.model_sha in results JSON.",
    )
    args = p.parse_args()
    if args.all and args.benchmark_file:
        p.error("--all and --benchmark-file are mutually exclusive")
    if not args.benchmark_file and not args.eval_dir:
        p.error("--eval-dir is required when --benchmark-file is not specified")
    if args.all and not args.eval_dir:
        p.error("--eval-dir is required when using --all")
    return args


def find_latest_benchmark(eval_dir: Path, workflow: str = "single_agent") -> Path:
    """Find the most recent benchmark_*.json that contains the given workflow.

    Falls back to the latest file if none contain the workflow.
    """
    candidates = sorted(eval_dir.glob("benchmark_*.json"))
    if not candidates:
        print(f"Error: No benchmark_*.json files found in {eval_dir}", file=sys.stderr)
        sys.exit(1)

    # Search from newest to oldest for a file containing the workflow
    for candidate in reversed(candidates):
        try:
            with open(candidate) as f:
                data = json.load(f)
            results = data.get("results", {})
            for model_data in results.values():
                if workflow in model_data:
                    return candidate
        except (json.JSONDecodeError, IOError):
            continue

    # Fallback: return the latest file regardless
    print(f"Warning: No benchmark file contains workflow '{workflow}', using latest file")
    return candidates[-1]


def load_model_map(path: Optional[Path]) -> Dict[str, str]:
    """Load the model name mapping file."""
    if path is None or not path.exists():
        return {}
    with open(path) as f:
        raw = json.load(f)
    return {str(k): str(v) for k, v in raw.items()}


def resolve_model_name(short_name: str, model_map: Dict[str, str]) -> str:
    """Map a ChemGraph short model name to an org/model display name."""
    if short_name in model_map:
        return model_map[short_name]
    # Fallback: use as-is
    return short_name


def sanitize_filename(model_id: str) -> str:
    """Convert org/model to a safe filename component."""
    return model_id.replace("/", "__").replace(" ", "_")


def extract_category_scores(
    judge_details: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Compute per-category accuracy from per-query judge details.

    Categories are read directly from the ``category`` field in each
    judge detail entry — no hardcoded mapping is needed.

    Parameters
    ----------
    judge_details : list of dict
        Each dict has at minimum ``category`` (str) and ``score`` (0 or 1).
        Entries with ``parse_error`` set are treated as score=0.

    Returns
    -------
    dict
        ``{category_key: accuracy}`` where accuracy is in [0.0, 1.0].
        Keys are sorted alphabetically for deterministic output.
    """
    cat_correct: Dict[str, int] = {}
    cat_total: Dict[str, int] = {}

    for detail in judge_details:
        category = detail.get("category")
        if category is None:
            qid = detail.get("query_id", "?")
            print(f"  Warning: query_id={qid!r} has no category field, skipping")
            continue

        score = detail.get("score", 0)
        # Treat parse errors as incorrect
        if detail.get("parse_error"):
            score = 0

        cat_total[category] = cat_total.get(category, 0) + 1
        cat_correct[category] = cat_correct.get(category, 0) + int(score)

    # Compute accuracy per category (sorted for deterministic output)
    results: Dict[str, float] = {}
    for cat in sorted(cat_total):
        results[cat] = cat_correct[cat] / cat_total[cat]

    # Validate category keys against the Tasks enum when available
    if KNOWN_CATEGORIES:
        unknown = set(results.keys()) - KNOWN_CATEGORIES
        missing = KNOWN_CATEGORIES - set(results.keys())
        if unknown:
            print(f"  WARNING: Category keys not in Tasks enum (will be ignored by leaderboard): {sorted(unknown)}")
        if missing:
            print(f"  WARNING: Expected categories missing from eval data: {sorted(missing)}")

    return results


def build_results_json(
    model_id: str,
    category_scores: Dict[str, float],
    model_dtype: str,
    model_sha: str,
    eval_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a leaderboard-compatible results JSON structure.

    Parameters
    ----------
    eval_date : str, optional
        ISO date string (YYYY-MM-DD) for this evaluation run.
        If not provided, defaults to today's UTC date.
    """
    if eval_date is None:
        eval_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return {
        "config": {
            "model_dtype": model_dtype,
            "model_name": model_id,
            "model_sha": model_sha,
            "eval_date": eval_date,
        },
        "results": {cat: {"accuracy": score} for cat, score in category_scores.items()},
    }


def build_request_json(
    model_id: str,
    model_sha: str,
    precision: str = "float16",
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a leaderboard-compatible request JSON structure."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "model": model_id,
        "base_model": model_id,
        "revision": model_sha,
        "precision": precision,
        "params": {},
        "architectures": "",
        "weight_type": "",
        "status": "FINISHED",
        "submitted_time": timestamp,
    }


def push_to_hub(results_dir: Path, requests_dir: Path) -> None:
    """Upload generated results and requests to HF Hub datasets.

    Uses per-file uploads so that existing files on the Hub are not
    deleted — each daily run *adds* new date-indexed result files
    alongside any previously uploaded results.
    """
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        print(
            "Error: HF_TOKEN environment variable not set. Cannot push to Hub.",
            file=sys.stderr,
        )
        sys.exit(1)

    api = HfApi(token=token)

    # Upload result files individually (additive — won't delete old files)
    print(f"\nPushing results to {RESULTS_REPO} ...")
    for result_file in results_dir.rglob("*.json"):
        path_in_repo = str(result_file.relative_to(results_dir))
        print(f"  Uploading {path_in_repo}")
        api.upload_file(
            path_or_fileobj=str(result_file),
            path_in_repo=path_in_repo,
            repo_id=RESULTS_REPO,
            repo_type="dataset",
            commit_message=f"Add eval result: {path_in_repo}",
        )

    # Upload request files individually
    print(f"Pushing requests to {REQUESTS_REPO} ...")
    for request_file in requests_dir.rglob("*.json"):
        path_in_repo = str(request_file.relative_to(requests_dir))
        print(f"  Uploading {path_in_repo}")
        api.upload_file(
            path_or_fileobj=str(request_file),
            path_in_repo=path_in_repo,
            repo_id=REQUESTS_REPO,
            repo_type="dataset",
            commit_message=f"Update request: {path_in_repo}",
        )

    print("Done pushing to Hub.")


def process_benchmark_file(
    benchmark_path: Path,
    workflow: str,
    model_map: Dict[str, str],
    results_outdir: Path,
    requests_outdir: Path,
    model_dtype: str,
    default_sha: str,
) -> int:
    """Process a single benchmark JSON file and write leaderboard-format output.

    Returns the number of models processed.
    """
    print(f"Using benchmark file: {benchmark_path}")

    with open(benchmark_path) as f:
        benchmark = json.load(f)

    metadata = benchmark.get("metadata", {})
    results = benchmark.get("results", {})

    print(f"Timestamp: {metadata.get('timestamp', 'unknown')}")
    print(f"Judge model: {metadata.get('judge_model', 'unknown')}")
    print(f"Workflow: {workflow}")
    print(f"Models: {list(results.keys())}")

    timestamp_str = metadata.get("timestamp")
    eval_date = None  # YYYY-MM-DD string for date-indexed filenames
    if timestamp_str:
        # Normalize to ISO format
        try:
            dt = datetime.fromisoformat(timestamp_str)
            timestamp_str = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            eval_date = dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            timestamp_str = None

    # Fallback: use current UTC date
    if eval_date is None:
        eval_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    processed = 0
    for short_name, model_data in results.items():
        if workflow not in model_data:
            print(f"  Skipping {short_name}: no {workflow} results")
            continue

        wf_data = model_data[workflow]
        # Support both old ("judge_details") and new ("structured_judge_details") key names
        judge_details = wf_data.get("structured_judge_details") or wf_data.get("judge_details", [])
        if not judge_details:
            print(f"  Skipping {short_name}: no judge_details")
            continue

        model_id = resolve_model_name(short_name, model_map)
        category_scores = extract_category_scores(judge_details)

        # Print summary
        avg = sum(category_scores.values()) / len(category_scores) if category_scores else 0
        print(f"\n  {short_name} -> {model_id}")
        for cat, score in category_scores.items():
            print(f"    {cat}: {score * 100:.1f}%")
        print(f"    Average: {avg * 100:.1f}%")

        # Write results JSON (date-indexed filename for historical tracking)
        # Files are placed under a workflow subfolder so that
        # single_agent and multi_agent results live side-by-side.
        results_obj = build_results_json(model_id, category_scores, model_dtype, default_sha, eval_date=eval_date)
        safe_name = sanitize_filename(model_id)
        # Put each model in its own subdirectory under the workflow folder
        model_results_dir = results_outdir / workflow / safe_name
        model_results_dir.mkdir(parents=True, exist_ok=True)
        results_path = model_results_dir / f"results_{eval_date}.json"
        with open(results_path, "w") as f:
            json.dump(results_obj, f, indent=2)
        print(f"    Results: {results_path}")

        # Write request JSON (also under workflow subfolder)
        requests_workflow_dir = requests_outdir / workflow
        requests_workflow_dir.mkdir(parents=True, exist_ok=True)
        request_obj = build_request_json(model_id, default_sha, timestamp=timestamp_str)
        requests_path = requests_workflow_dir / f"{safe_name}.request.json"
        with open(requests_path, "w") as f:
            json.dump(request_obj, f, indent=2)
        print(f"    Request: {requests_path}")

        processed += 1

    return processed


def find_all_benchmarks(eval_dir: Path, workflow: str = "single_agent") -> List[Path]:
    """Find all benchmark_*.json files in *eval_dir* that contain the given workflow."""
    candidates = sorted(eval_dir.glob("benchmark_*.json"))
    if not candidates:
        print(f"Error: No benchmark_*.json files found in {eval_dir}", file=sys.stderr)
        sys.exit(1)

    matched = []
    for candidate in candidates:
        try:
            with open(candidate) as f:
                data = json.load(f)
            results = data.get("results", {})
            for model_data in results.values():
                if workflow in model_data:
                    matched.append(candidate)
                    break
        except (json.JSONDecodeError, IOError):
            continue

    if not matched:
        print(f"Warning: No benchmark files contain workflow '{workflow}', using all files")
        return candidates

    return matched


def main() -> None:
    args = parse_args()

    # Determine which benchmark files to process
    if args.benchmark_file:
        benchmark_paths = [Path(args.benchmark_file)]
    elif args.all:
        benchmark_paths = find_all_benchmarks(args.eval_dir, workflow=args.workflow)
        print(f"Found {len(benchmark_paths)} benchmark file(s) to process")
    else:
        benchmark_paths = [find_latest_benchmark(args.eval_dir, workflow=args.workflow)]

    model_map = load_model_map(args.model_map)

    # Create output directories
    args.results_outdir.mkdir(parents=True, exist_ok=True)
    args.requests_outdir.mkdir(parents=True, exist_ok=True)

    total_processed = 0
    for i, benchmark_path in enumerate(benchmark_paths, 1):
        if len(benchmark_paths) > 1:
            print(f"\n{'=' * 60}")
            print(f"File {i}/{len(benchmark_paths)}: {benchmark_path.name}")
            print(f"{'=' * 60}")

        processed = process_benchmark_file(
            benchmark_path=benchmark_path,
            workflow=args.workflow,
            model_map=model_map,
            results_outdir=args.results_outdir,
            requests_outdir=args.requests_outdir,
            model_dtype=args.model_dtype,
            default_sha=args.default_sha,
        )
        total_processed += processed

    print(f"\nProcessed {total_processed} model result(s) from {len(benchmark_paths)} file(s).")

    # Push to HF Hub if requested
    if args.push_to_hub:
        push_to_hub(args.results_outdir, args.requests_outdir)


if __name__ == "__main__":
    main()
