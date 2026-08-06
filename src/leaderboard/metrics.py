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
_PRICING_PATH = _REPO_ROOT / "dataset" / "pricing.json"

_DATE_RE = re.compile(r"metrics_(\d{4}-\d{2}-\d{2})\.csv$")

# Share of prompt tokens assumed cache-hit when a run recorded no cached tokens
# (the 40-query benchmark repeats a large system prompt, so most input is
# cacheable). Used only to fill in the missing cached split for the $ estimate.
_ASSUMED_CACHE_HIT_RATE = 0.60

# Columns the redesigned Highlights view consumes.
_OUT_COLS = [
    "full_model",
    "tokens_per_query",
    "accuracy",
    "accuracy_per_1k_tokens",
    "n_queries",
    "llm_calls",
    "cached_tokens",
    "usd_per_query",
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


def _load_pricing() -> dict[str, dict]:
    """``org/model`` -> {"input", "cached", "output"} $ per 1M tokens.

    Only PROPRIETARY (API-billed) models are listed in ``dataset/pricing.json``;
    open-weight models run on local hardware and are intentionally absent, so
    they get no dollar cost. ``cached`` defaults to the full input price when a
    model omits it (i.e. no cache discount). Returns {} if the file is
    missing/unreadable, in which case no model gets a dollar cost.
    """
    if not _PRICING_PATH.exists():
        return {}
    try:
        with open(_PRICING_PATH) as fp:
            raw = json.load(fp)
    except (json.JSONDecodeError, OSError):
        return {}
    models = raw.get("models", raw) if isinstance(raw, dict) else {}
    out: dict[str, dict] = {}
    for k, v in models.items():
        if not isinstance(v, dict) or "input" not in v or "output" not in v:
            continue
        try:
            inp = float(v["input"])
            out[str(k)] = {
                "input": inp,
                "cached": float(v.get("cached", inp)),
                "output": float(v["output"]),
            }
        except (TypeError, ValueError):
            continue
    return out


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

    # Real-dollar cost per query, proprietary models only. We price the measured
    # prompt/completion token split at each vendor's list price (dataset/
    # pricing.json). Prompt tokens split into cache-hit vs full-price input: use
    # the recorded cached_tokens when a run reports any, otherwise assume 60% of
    # prompt tokens were cache hits (the benchmark repeats a big system prompt).
    # Models absent from pricing.json (all open-weight, plus any unpriced closed
    # model) get NaN and never win the "cheapest $" pick.
    def _numcol(name: str) -> pd.Series:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
        return pd.Series([float("nan")] * len(df), index=df.index)

    prompt_tok = _numcol("prompt_tokens")
    completion_tok = _numcol("completion_tokens")
    cached_tok_raw = df["cached_tokens"]
    pricing = _load_pricing()

    def _usd_per_query(model, p_tok, c_tok, cached, nq) -> float:
        price = pricing.get(str(model))
        if price is None or nq <= 0 or pd.isna(p_tok) or pd.isna(c_tok):
            return float("nan")
        cached_in = cached if (pd.notna(cached) and cached > 0) else _ASSUMED_CACHE_HIT_RATE * p_tok
        cached_in = min(cached_in, p_tok)             # never exceed total prompt
        full_in = p_tok - cached_in
        usd_total = (full_in * price["input"]
                     + cached_in * price["cached"]
                     + c_tok * price["output"]) / 1_000_000.0
        return usd_total / nq

    df["usd_per_query"] = [
        _usd_per_query(m, p, c, ca, n)
        for m, p, c, ca, n in zip(
            df["full_model"], prompt_tok, completion_tok, cached_tok_raw, df["n_queries"]
        )
    ]

    # Degenerate rows: zero queries or zero tokens (e.g. a model that never
    # ran, like gpt-4o-latest) carry no usable cost — blank the token cell so
    # the matrix leaves it empty rather than drawing a misleading 0.
    degenerate = (df["n_queries"] <= 0) | (df["tokens_per_query"] <= 0)
    df.loc[degenerate, "tokens_per_query"] = pd.NA
    # A never-run model has ~0 tokens; don't let it read as "$0.00, cheapest".
    df.loc[degenerate, "usd_per_query"] = pd.NA

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
