#!/usr/bin/env python3
"""Stage + upload per-query ChemGraph eval logs to an HF dataset.

The leaderboard's Full table shows a per-task accuracy number but not the
underlying per-query logs. This script mirrors the raw logs from a ChemGraph
eval output directory into a clean, lookup-stable layout and (optionally)
uploads them to a private HF dataset the leaderboard app fetches lazily when a
user clicks a task cell.

IMPORTANT: file *contents* are copied byte-for-byte — nothing inside the JSON is
trimmed or transformed. Only the ``state_thread`` filename is normalized from
``state_thread_<N>_<hash>_<date>_<time>.json`` to ``state_thread_<N>.json`` so
the runtime loader can address a query by its index without listing the repo.

Layout produced (per model x workflow)::

    <outdir>/<workflow>/<safe_name>/detail.json          # copy of *_detail.json
    <outdir>/<workflow>/<safe_name>/state_thread_<N>.json # copy of the transcript

where ``safe_name = sanitize_filename(resolve_model_name(detail.model_name))`` —
identical to the leaderboard's ``full_model`` keying, so the browser-derived key
matches.

Usage::

    # Stage locally only (inspect before pushing)
    python scripts/upload_eval_logs.py \
        --eval-dir /home/zhye/ChemGraph/eval_2026-08-04 \
        --model-map dataset/model_map.json \
        --outdir hub_logs

    # Stage + push to the private dataset
    python scripts/upload_eval_logs.py \
        --eval-dir /home/zhye/ChemGraph/eval_2026-08-04 \
        --model-map dataset/model_map.json \
        --push-to-hub
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

# Reuse the leaderboard's exact model-name keying so the dataset paths match the
# Full table's ``full_model`` (mapped and unmapped models alike).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.chemgraph_to_leaderboard import (  # noqa: E402
    load_model_map,
    resolve_model_name,
)
from src.envs import LOGS_REPO  # noqa: E402  (owner/name; owner from CG_OWNER)


def safe_name(full_model: str) -> str:
    """``org/model`` (or a raw ``argo:...`` string) -> a filesystem/HF-safe key.

    Same as the ETL's ``sanitize_filename`` but ALSO maps ``:`` -> ``_`` so the
    11 currently-unmapped ``argo:...`` models don't produce colon paths (colons
    are awkward on the HF Hub and un-checkout-able on Windows). Mapped models
    contain no colon, so this is identical to ``sanitize_filename`` for them.

    The runtime loader (``src/leaderboard/logs.py``) MUST use this exact
    transform on the browser-derived ``full_model`` for the keys to match.
    """
    return full_model.replace("/", "__").replace(" ", "_").replace(":", "_")

# state_thread_<N>_<hash>_<date>_<time>.json  ->  capture N (0-based query index)
_STATE_THREAD_RE = re.compile(r"^state_thread_(\d+)_.*\.json$")
# argo_<model>_<workflow>_detail.json — we don't parse the model out of the
# filename (model names contain '.' / '-'); the detail file is self-describing.


def _load_detail_meta(detail_path: Path) -> tuple[str, str] | None:
    """Return (model_name, workflow_type) from a *_detail.json, or None."""
    import json

    try:
        with open(detail_path) as fp:
            d = json.load(fp)
    except (OSError, json.JSONDecodeError):
        return None
    model_name = d.get("model_name")
    workflow = d.get("workflow_type")
    if not model_name or not workflow:
        return None
    return str(model_name), str(workflow)


def stage(eval_dir: Path, model_map: dict[str, str], outdir: Path) -> dict:
    """Copy detail + state_thread files into the clean layout. Returns stats."""
    logs_root = eval_dir / "logs"
    detail_files = sorted(eval_dir.glob("*_detail.json"))
    if not detail_files:
        print(f"No *_detail.json found in {eval_dir}", file=sys.stderr)
        return {"models": 0, "transcripts": 0, "unmapped": []}

    n_models = 0
    n_transcripts = 0
    unmapped: list[str] = []

    for detail_path in detail_files:
        meta = _load_detail_meta(detail_path)
        if meta is None:
            print(f"  skip (unreadable/no meta): {detail_path.name}", file=sys.stderr)
            continue
        model_name, workflow = meta  # model_name is colon form, e.g. argo:gpt-4.1

        full_model = resolve_model_name(model_name, model_map)
        if full_model == model_name and "/" not in full_model:
            unmapped.append(model_name)
        key = safe_name(full_model)

        dest_dir = outdir / workflow / key
        dest_dir.mkdir(parents=True, exist_ok=True)

        # 1) detail file — raw copy
        shutil.copyfile(detail_path, dest_dir / "detail.json")

        # 2) transcripts — raw copy, filename normalized to the query index N.
        #    Logs dir name uses underscores in place of the colon.
        model_dir = logs_root / model_name.replace(":", "_") / workflow
        copied = 0
        if model_dir.is_dir():
            # If a run somehow produced two files for the same N, keep the
            # newest (last write wins) so the panel shows the final attempt.
            by_index: dict[int, Path] = {}
            for f in model_dir.glob("state_thread_*.json"):
                m = _STATE_THREAD_RE.match(f.name)
                if not m:
                    continue
                n = int(m.group(1))
                prev = by_index.get(n)
                if prev is None or f.stat().st_mtime >= prev.stat().st_mtime:
                    by_index[n] = f
            for n, src in sorted(by_index.items()):
                shutil.copyfile(src, dest_dir / f"state_thread_{n}.json")
                copied += 1
        else:
            print(f"  warn: no transcripts dir for {model_name} [{workflow}] "
                  f"(expected {model_dir})", file=sys.stderr)

        n_models += 1
        n_transcripts += copied
        print(f"  {full_model:40s} [{workflow:12s}] detail + {copied:2d} transcripts")

    return {"models": n_models, "transcripts": n_transcripts, "unmapped": unmapped}


def push(outdir: Path, repo_id: str, private: bool) -> None:
    from huggingface_hub import HfApi, create_repo

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN not set — cannot push.", file=sys.stderr)
        sys.exit(1)

    print(f"\nEnsuring dataset repo {repo_id} exists (private={private}) ...")
    create_repo(repo_id, repo_type="dataset", private=private,
                exist_ok=True, token=token)

    api = HfApi(token=token)
    print(f"Uploading {outdir} -> {repo_id} (this may take a minute) ...")
    api.upload_folder(
        folder_path=str(outdir),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload eval logs (state_thread + detail)",
    )
    print("Done pushing to Hub.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", required=True, type=Path,
                    help="ChemGraph eval output dir (contains *_detail.json + logs/)")
    ap.add_argument("--model-map", type=Path, default=Path("dataset/model_map.json"),
                    help="ChemGraph short-name -> org/model map")
    ap.add_argument("--outdir", type=Path, default=Path("hub_logs"),
                    help="Local staging dir for the clean layout (default: hub_logs)")
    ap.add_argument("--repo-id", default=LOGS_REPO,
                    help=f"HF dataset repo id (default: {LOGS_REPO}, "
                         "owner from CG_OWNER / name from CG_LOGS_DATASET)")
    ap.add_argument("--public", action="store_true",
                    help="Create the dataset public (default: private)")
    ap.add_argument("--push-to-hub", action="store_true",
                    help="Upload the staged dir to the HF dataset")
    args = ap.parse_args()

    if not args.eval_dir.is_dir():
        print(f"Error: --eval-dir not found: {args.eval_dir}", file=sys.stderr)
        sys.exit(1)

    model_map = load_model_map(args.model_map)
    if not model_map:
        print(f"warn: empty/absent model map at {args.model_map}; "
              "all models will use raw argo names.", file=sys.stderr)

    print(f"Staging logs from {args.eval_dir} -> {args.outdir}")
    stats = stage(args.eval_dir, model_map, args.outdir)
    print(f"\nStaged {stats['models']} model/workflow dirs, "
          f"{stats['transcripts']} transcripts.")
    if stats["unmapped"]:
        uniq = sorted(set(stats["unmapped"]))
        print(f"warn: {len(uniq)} unmapped model(s) (kept raw argo name): "
              f"{', '.join(uniq)}", file=sys.stderr)

    if args.push_to_hub:
        push(args.outdir, args.repo_id, private=not args.public)
    else:
        print("\n(dry run — not pushed; pass --push-to-hub to upload)")


if __name__ == "__main__":
    main()
