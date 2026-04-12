"""Aggregation utilities for time-based evaluation metrics.

Provides functions to compute 1-day, 3-day, and 7-day average scores
from historical evaluation results, as well as to build DataFrames
suitable for trend visualisation.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from src.about import Tasks
from src.leaderboard.read_evals import EvalResult


def _eval_results_to_history_df(eval_results: list[EvalResult]) -> pd.DataFrame:
    """Convert a list of EvalResult objects into a flat history DataFrame.

    Returns a DataFrame with columns:
        model, eval_date, average, <task_col_1>, <task_col_2>, ...

    Rows without a valid ``eval_date`` are excluded.
    """
    num_tasks = len(Tasks)
    rows: list[dict] = []
    for r in eval_results:
        if not r.eval_date:
            continue
        avg = sum(v for v in r.results.values() if v is not None) / num_tasks
        row = {
            "model": r.full_model,
            "eval_date": r.eval_date,
            "average": round(avg, 2),
        }
        for task in Tasks:
            col = task.value.col_name
            row[col] = r.results.get(task.value.benchmark)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["eval_date"] = pd.to_datetime(df["eval_date"])
    df = df.sort_values(["model", "eval_date"])
    return df


def get_history_df(eval_results: list[EvalResult]) -> pd.DataFrame:
    """Return the full history DataFrame (all models x all dates).

    Suitable for charting score trajectories over time.
    """
    return _eval_results_to_history_df(eval_results)


def compute_n_day_average(
    history_df: pd.DataFrame,
    n: int,
    reference_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """Compute the N-day average of the overall 'average' score per model.

    Parameters
    ----------
    history_df : pd.DataFrame
        Output of ``get_history_df`` / ``_eval_results_to_history_df``.
    n : int
        Number of days to look back (inclusive of *reference_date*).
    reference_date : datetime, optional
        The anchor date.  Defaults to today (UTC).

    Returns
    -------
    pd.DataFrame
        Columns: ``model``, ``n_day_avg``, ``days_available``
        One row per model.
    """
    if history_df.empty:
        return pd.DataFrame(columns=["model", "n_day_avg", "days_available"])

    if reference_date is None:
        reference_date = datetime.now(timezone.utc)

    # Normalize to end-of-day so that dates are compared at day granularity
    ref = pd.Timestamp(reference_date).tz_localize(None).normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    cutoff = ref.normalize() - pd.Timedelta(days=n - 1)

    window = history_df[(history_df["eval_date"] >= cutoff) & (history_df["eval_date"] <= ref)]

    if window.empty:
        return pd.DataFrame(columns=["model", "n_day_avg", "days_available"])

    grouped = window.groupby("model")["average"].agg(["mean", "count"]).reset_index()
    grouped.columns = ["model", "n_day_avg", "days_available"]
    grouped["n_day_avg"] = grouped["n_day_avg"].round(2)
    grouped["days_available"] = grouped["days_available"].astype(int)
    return grouped


def build_trend_summary(
    eval_results: list[EvalResult],
    reference_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """Build a summary DataFrame with 1-day, 3-day, and 7-day average scores.

    Parameters
    ----------
    eval_results : list[EvalResult]
        Full evaluation history (all models, all dates).
    reference_date : datetime, optional
        The anchor date.  Defaults to today (UTC).

    Returns
    -------
    pd.DataFrame
        Columns: ``Model``, ``1-Day``, ``3-Day Avg``, ``3-Day (N)``,
        ``7-Day Avg``, ``7-Day (N)``
    """
    history = _eval_results_to_history_df(eval_results)
    if history.empty:
        return pd.DataFrame(columns=["Model", "1-Day", "3-Day Avg", "3-Day (N)", "7-Day Avg", "7-Day (N)"])

    day1 = compute_n_day_average(history, 1, reference_date)
    day3 = compute_n_day_average(history, 3, reference_date)
    day7 = compute_n_day_average(history, 7, reference_date)

    # Start from the list of all known models
    models = sorted(history["model"].unique())

    rows = []
    for m in models:
        r1 = day1[day1["model"] == m]
        r3 = day3[day3["model"] == m]
        r7 = day7[day7["model"] == m]

        val_1 = r1["n_day_avg"].iloc[0] if not r1.empty else None
        val_3 = r3["n_day_avg"].iloc[0] if not r3.empty else None
        cnt_3 = int(r3["days_available"].iloc[0]) if not r3.empty else 0
        val_7 = r7["n_day_avg"].iloc[0] if not r7.empty else None
        cnt_7 = int(r7["days_available"].iloc[0]) if not r7.empty else 0

        rows.append(
            {
                "Model": m,
                "1-Day": val_1,
                "3-Day Avg": val_3,
                "3-Day (N)": f"({cnt_3}/3)" if val_3 is not None else "",
                "7-Day Avg": val_7,
                "7-Day (N)": f"({cnt_7}/7)" if val_7 is not None else "",
            }
        )

    return pd.DataFrame(rows)


def build_leaderboard_trend_columns(
    eval_results: list[EvalResult],
    reference_date: Optional[datetime] = None,
) -> dict[str, dict[str, float | None]]:
    """Compute trend columns keyed by model name for merging into the leaderboard DF.

    Returns
    -------
    dict
        ``{full_model: {"1-Day": val, "3-Day Avg": val, "7-Day Avg": val}}``
    """
    summary = build_trend_summary(eval_results, reference_date)
    if summary.empty:
        return {}

    result = {}
    for _, row in summary.iterrows():
        result[row["Model"]] = {
            "1-Day": row["1-Day"],
            "3-Day Avg": row["3-Day Avg"],
            "7-Day Avg": row["7-Day Avg"],
        }
    return result
