#!/usr/bin/env python3
"""Transform ChemGraph benchmark results into HF leaderboard-compatible JSON files.

Reads the latest ``benchmark_*.json`` from a ChemGraph eval output directory,
extracts per-model single_agent judge scores, groups the 14 queries into 8
task categories, and writes per-model results + request JSON files that the
leaderboard app can consume.

Optionally pushes the generated files to the HF Hub datasets.

Usage::

    # Generate files locally
    python scripts/chemgraph_to_leaderboard.py \
        --eval-dir /path/to/ChemGraph/eval_results \
        --model-map dataset/model_map.json \
        --results-outdir hub_results \
        --requests-outdir hub_requests

    # Generate and push to HF Hub
    python scripts/chemgraph_to_leaderboard.py \
        --eval-dir /path/to/ChemGraph/eval_results \
        --model-map dataset/model_map.json \
        --push-to-hub
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---- Query-to-category mapping -------------------------------------------
# The 14 ChemGraph ground-truth queries grouped into 8 leaderboard tasks.
# Keys are query IDs (str), values are category benchmark keys.

QUERY_TO_CATEGORY: Dict[str, str] = {
    "1": "smi_lookup",  # Name -> SMILES (single molecule)
    "2": "smi_lookup",  # Name -> SMILES (multiple molecules)
    "3": "coord_gen",  # SMILES -> 3D coordinates (single)
    "4": "coord_gen",  # SMILES -> 3D coordinates (multiple)
    "5": "geom_opt",  # Geometry optimization
    "6": "vib_freq",  # Vibrational frequency analysis
    "7": "thermo",  # Thermochemical properties
    "8": "dipole",  # Dipole moment
    "9": "energy",  # Single-point energy + JSON extraction
    "10": "energy",  # Single-point energy (from SMILES)
    "11": "energy",  # Geometry opt + JSON extraction
    "12": "react_gibbs",  # Reaction Gibbs free energy (methane combustion)
    "13": "react_gibbs",  # Reaction Gibbs free energy (ammonia synthesis)
    "14": "react_gibbs",  # Reaction Gibbs free energy (water-gas shift)
}

# All category keys in display order
CATEGORIES = [
    "smi_lookup",
    "coord_gen",
    "geom_opt",
    "vib_freq",
    "thermo",
    "dipole",
    "energy",
    "react_gibbs",
]

# ---- HF Hub repo IDs (must match src/envs.py) ----------------------------
RESULTS_REPO = "Autonomous-Scientific-Agents/results"
REQUESTS_REPO = "Autonomous-Scientific-Agents/requests"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transform ChemGraph eval results to HF leaderboard format.")
    p.add_argument(
        "--eval-dir",
        type=Path,
        required=True,
        help="Path to ChemGraph eval_results directory.",
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
        help=("Specific benchmark_*.json file to process. If not given, the latest file by timestamp is used."),
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
    return p.parse_args()


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

    Parameters
    ----------
    judge_details : list of dict
        Each dict has at minimum ``query_id`` (str) and ``score`` (0 or 1).
        Entries with ``parse_error`` set are treated as score=0.

    Returns
    -------
    dict
        ``{category_key: accuracy}`` where accuracy is in [0.0, 1.0].
    """
    # Accumulate correct/total per category
    cat_correct: Dict[str, int] = {c: 0 for c in CATEGORIES}
    cat_total: Dict[str, int] = {c: 0 for c in CATEGORIES}

    for detail in judge_details:
        qid = str(detail.get("query_id", ""))
        category = QUERY_TO_CATEGORY.get(qid)
        if category is None:
            print(f"  Warning: query_id={qid!r} not in QUERY_TO_CATEGORY, skipping")
            continue

        score = detail.get("score", 0)
        # Treat parse errors as incorrect
        if detail.get("parse_error"):
            score = 0

        cat_total[category] += 1
        cat_correct[category] += int(score)

    # Compute accuracy per category
    results: Dict[str, float] = {}
    for cat in CATEGORIES:
        total = cat_total[cat]
        if total > 0:
            results[cat] = cat_correct[cat] / total
        else:
            # No queries for this category — should not happen with the
            # standard 14-query dataset, but handle gracefully
            results[cat] = 0.0

    return results


def build_results_json(
    model_id: str,
    category_scores: Dict[str, float],
    model_dtype: str,
    model_sha: str,
) -> Dict[str, Any]:
    """Build a leaderboard-compatible results JSON structure."""
    return {
        "config": {
            "model_dtype": model_dtype,
            "model_name": model_id,
            "model_sha": model_sha,
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
    """Upload generated results and requests to HF Hub datasets."""
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        print(
            "Error: HF_TOKEN environment variable not set. Cannot push to Hub.",
            file=sys.stderr,
        )
        sys.exit(1)

    api = HfApi(token=token)

    print(f"\nPushing results to {RESULTS_REPO} ...")
    api.upload_folder(
        repo_id=RESULTS_REPO,
        folder_path=str(results_dir),
        repo_type="dataset",
        commit_message="Update leaderboard results from ChemGraph eval",
    )

    print(f"Pushing requests to {REQUESTS_REPO} ...")
    api.upload_folder(
        repo_id=REQUESTS_REPO,
        folder_path=str(requests_dir),
        repo_type="dataset",
        commit_message="Update leaderboard requests from ChemGraph eval",
    )

    print("Done pushing to Hub.")


def main() -> None:
    args = parse_args()

    # 1. Find and load the benchmark file
    if args.benchmark_file:
        benchmark_path = Path(args.benchmark_file)
    else:
        benchmark_path = find_latest_benchmark(args.eval_dir, workflow=args.workflow)
    print(f"Using benchmark file: {benchmark_path}")

    with open(benchmark_path) as f:
        benchmark = json.load(f)

    metadata = benchmark.get("metadata", {})
    results = benchmark.get("results", {})
    workflow = args.workflow

    print(f"Timestamp: {metadata.get('timestamp', 'unknown')}")
    print(f"Judge model: {metadata.get('judge_model', 'unknown')}")
    print(f"Workflow: {workflow}")
    print(f"Models: {list(results.keys())}")

    # 2. Load model map
    model_map = load_model_map(args.model_map)

    # 3. Create output directories
    args.results_outdir.mkdir(parents=True, exist_ok=True)
    args.requests_outdir.mkdir(parents=True, exist_ok=True)

    timestamp_str = metadata.get("timestamp")
    if timestamp_str:
        # Normalize to ISO format
        try:
            dt = datetime.fromisoformat(timestamp_str)
            timestamp_str = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except (ValueError, TypeError):
            timestamp_str = None

    # 4. Process each model
    processed = 0
    for short_name, model_data in results.items():
        if workflow not in model_data:
            print(f"  Skipping {short_name}: no {workflow} results")
            continue

        wf_data = model_data[workflow]
        judge_details = wf_data.get("judge_details", [])
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

        # Write results JSON
        results_obj = build_results_json(model_id, category_scores, args.model_dtype, args.default_sha)
        safe_name = sanitize_filename(model_id)
        # Put each model in its own subdirectory (leaderboard expects this)
        model_results_dir = args.results_outdir / safe_name
        model_results_dir.mkdir(parents=True, exist_ok=True)
        results_path = model_results_dir / f"{safe_name}.results.json"
        with open(results_path, "w") as f:
            json.dump(results_obj, f, indent=2)
        print(f"    Results: {results_path}")

        # Write request JSON
        request_obj = build_request_json(model_id, args.default_sha, timestamp=timestamp_str)
        requests_path = args.requests_outdir / f"{safe_name}.request.json"
        with open(requests_path, "w") as f:
            json.dump(request_obj, f, indent=2)
        print(f"    Request: {requests_path}")

        processed += 1

    print(f"\nProcessed {processed} model(s).")

    # 5. Push to HF Hub if requested
    if args.push_to_hub:
        push_to_hub(args.results_outdir, args.requests_outdir)


if __name__ == "__main__":
    main()
