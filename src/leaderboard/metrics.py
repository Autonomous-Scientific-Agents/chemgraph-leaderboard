"""Token-consumption / efficiency metrics loader.

Side-channel data path: the leaderboard's results JSON carries only per-task
accuracy. Token and timing data comes from a metrics CSV produced by the
chemgraph eval run (one row per model x workflow). We read the newest
``dataset/metrics/metrics_*.csv``, resolve the CSV's ``argo:<name>`` model
strings to the leaderboard's ``org/model`` key via the existing
``dataset/model_map.json``, and expose a per-(model, workflow) DataFrame the
Highlights view joins against ``LEADERBOARD_DF`` on ``full_model``.

Accuracy is NOT taken from this CSV — it stays sourced from the accuracy JSON
pipeline (LEADERBOARD_DF). This module contributes only the token columns.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pandas as pd

# Repo root = two levels up from this file (src/leaderboard/metrics.py).
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_LOCAL_METRICS_DIR = _REPO_ROOT / "dataset" / "metrics"
_MODEL_MAP_PATH = _REPO_ROOT / "dataset" / "model_map.json"

_DATE_RE = re.compile(r"metrics_(\d{4}-\d{2}-\d{2})\.csv$")

# Columns the redesigned Highlights view consumes.
_OUT_COLS = [
    "full_model",
    "tokens_per_query",
    "accuracy",
    "accuracy_per_1k_tokens",
    "n_queries",
    "llm_calls",
    "cached_tokens",
    "low_conf",
]


def _empty_metrics_df() -> pd.DataFrame:
    return pd.DataFrame(columns=_OUT_COLS)


def _load_model_map() -> dict[str, str]:
    """ChemGraph short/argo model names -> ``org/model`` display names."""
    if not _MODEL_MAP_PATH.exists():
        return {}
    try:
        with open(_MODEL_MAP_PATH) as fp:
            raw = json.load(fp)
        return {str(k): str(v) for k, v in raw.items()}
    except (json.JSONDecodeError, OSError):
        return {}


def _candidate_metrics_dirs() -> list[Path]:
    """Directories to scan for ``metrics_*.csv``, in priority order.

    On a deployed Space the metrics CSV ships inside the results dataset
    snapshot, which ``snapshot_download`` writes to ``EVAL_RESULTS_PATH``
    (``./eval-results``) — we look there first (under a ``metrics/`` subfolder,
    then the root). Locally we also keep a committed copy at
    ``dataset/metrics/`` as a fallback. We scan all and pick the newest by the
    date in the filename, so whichever source has the freshest file wins.
    """
    dirs: list[Path] = []
    try:
        from src.envs import EVAL_RESULTS_PATH
        dirs.append(Path(EVAL_RESULTS_PATH) / "metrics")
        dirs.append(Path(EVAL_RESULTS_PATH))
    except Exception:
        pass
    dirs.append(_LOCAL_METRICS_DIR)
    return dirs


def _newest_metrics_csv() -> Path | None:
    """Return the metrics CSV with the latest YYYY-MM-DD in its filename,
    searching every candidate directory."""
    dated: list[tuple[str, Path]] = []
    for d in _candidate_metrics_dirs():
        if not d.is_dir():
            continue
        for p in d.glob("metrics_*.csv"):
            m = _DATE_RE.search(p.name)
            if m:
                dated.append((m.group(1), p))
    if not dated:
        return None
    dated.sort(key=lambda t: t[0], reverse=True)
    return dated[0][1]


def get_metrics_df(workflow: str) -> pd.DataFrame:
    """Per-model token metrics for one workflow, keyed by ``full_model``.

    Returns an empty (but correctly-columned) DataFrame when no CSV is present
    or the workflow has no rows, so callers can join unconditionally.
    """
    csv_path = _newest_metrics_csv()
    if csv_path is None:
        return _empty_metrics_df()

    try:
        df = pd.read_csv(csv_path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return _empty_metrics_df()

    required = {"model", "workflow", "avg_total_tokens_per_query", "n_queries", "accuracy"}
    if not required.issubset(df.columns):
        return _empty_metrics_df()

    df = df[df["workflow"] == workflow].copy()
    if df.empty:
        return _empty_metrics_df()

    model_map = _load_model_map()
    # CSV model strings (e.g. "argo:gpt-4o") are direct keys in model_map.json
    # (it carries both stripped and argo-prefixed forms). Fall back to the raw
    # string if unmapped so nothing silently vanishes.
    df["full_model"] = df["model"].map(lambda m: model_map.get(str(m), str(m)))

    df["tokens_per_query"] = pd.to_numeric(
        df["avg_total_tokens_per_query"], errors="coerce"
    )
    df["n_queries"] = pd.to_numeric(df["n_queries"], errors="coerce").fillna(0).astype(int)
    df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")
    df["llm_calls"] = pd.to_numeric(df.get("llm_calls"), errors="coerce")
    df["cached_tokens"] = pd.to_numeric(df.get("cached_tokens"), errors="coerce")

    # Degenerate rows: zero queries or zero tokens (e.g. a model that never
    # ran, like gpt-4o-latest) carry no usable cost — blank the token cell so
    # the matrix leaves it empty rather than drawing a misleading 0.
    degenerate = (df["n_queries"] <= 0) | (df["tokens_per_query"] <= 0)
    df.loc[degenerate, "tokens_per_query"] = pd.NA

    # accuracy-per-1k-tokens efficiency metric (KPI card / frontier only).
    tpq = df["tokens_per_query"]
    df["accuracy_per_1k_tokens"] = df["accuracy"] / (tpq / 1000.0)
    df.loc[tpq.isna() | (tpq <= 0), "accuracy_per_1k_tokens"] = pd.NA

    # Low-confidence: ran fewer than the full 40-query benchmark.
    df["low_conf"] = df["n_queries"] < 40

    # One row per model; if a model somehow appears twice, keep the one with
    # the most queries.
    df = df.sort_values("n_queries", ascending=False).drop_duplicates("full_model")
    return df[_OUT_COLS].reset_index(drop=True)
