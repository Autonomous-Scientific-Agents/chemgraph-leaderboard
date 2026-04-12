import json
import os

import pandas as pd

from src.display.formatting import has_no_nan_values, make_clickable_model
from src.display.utils import AutoEvalColumn, EvalQueueColumn
from src.leaderboard.aggregate import (
    build_leaderboard_trend_columns,
    build_trend_summary,
    get_history_df,
)
from src.leaderboard.read_evals import get_all_eval_results, get_raw_eval_results


def get_leaderboard_df(results_path: str, requests_path: str, cols: list, benchmark_cols: list) -> pd.DataFrame:
    """Creates a dataframe from all the individual experiment results.

    Includes 1-Day, 3-Day Avg, and 7-Day Avg trend columns when
    historical (date-indexed) evaluation data is available.
    """
    raw_data = get_raw_eval_results(results_path, requests_path)
    all_data_json = [v.to_dict() for v in raw_data]

    if not all_data_json:
        # Return an empty DataFrame with the expected columns so the app
        # can still render without crashing.
        print("WARNING: No valid evaluation results found. Leaderboard will be empty.")
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame.from_records(all_data_json)

    # --- Merge trend columns from full history ---
    all_history = get_all_eval_results(results_path, requests_path)
    trend_map = build_leaderboard_trend_columns(all_history)

    if trend_map:
        # Map from full_model stored in raw_data to trend values.
        # The df uses the "eval_name" hidden column; we need to match
        # via the full_model that was used to build trend_map.
        model_lookup = {v.eval_name: v.full_model for v in raw_data}
        for col_name in ("1-Day", "3-Day Avg", "7-Day Avg"):
            df[col_name] = df["eval_name"].map(
                lambda en, c=col_name: trend_map.get(model_lookup.get(en, ""), {}).get(c)
            )
    else:
        for col_name in ("1-Day", "3-Day Avg", "7-Day Avg"):
            df[col_name] = None

    df = df.sort_values(by=[AutoEvalColumn.average.name], ascending=False)
    df = df[cols].round(decimals=2)

    # filter out if any of the benchmarks have not been produced
    # df = df[has_no_nan_values(df, benchmark_cols)]
    return df


def get_trend_summary_df(results_path: str, requests_path: str) -> pd.DataFrame:
    """Return a summary DataFrame with 1-day, 3-day, and 7-day averages per model."""
    all_results = get_all_eval_results(results_path, requests_path)
    return build_trend_summary(all_results)


def get_trend_history_df(results_path: str, requests_path: str) -> pd.DataFrame:
    """Return the full history DataFrame for trend charting."""
    all_results = get_all_eval_results(results_path, requests_path)
    return get_history_df(all_results)


def get_evaluation_queue_df(save_path: str, cols: list) -> list[pd.DataFrame]:
    """Creates the different dataframes for the evaluation queue requests."""
    all_evals = []

    if not os.path.isdir(save_path):
        print(f"WARNING: Eval queue path does not exist: {save_path}")
        empty = pd.DataFrame(columns=cols)
        return empty, empty.copy(), empty.copy()

    entries = [entry for entry in os.listdir(save_path) if not entry.startswith(".")]

    for entry in entries:
        entry_path = os.path.join(save_path, entry)

        if entry.endswith(".json") and os.path.isfile(entry_path):
            _load_queue_entry(entry_path, all_evals)

        elif os.path.isdir(entry_path):
            # Recurse into subdirectories (e.g. paper_requests/)
            for sub_entry in os.listdir(entry_path):
                if sub_entry.startswith(".") or not sub_entry.endswith(".json"):
                    continue
                sub_path = os.path.join(entry_path, sub_entry)
                if os.path.isfile(sub_path):
                    _load_queue_entry(sub_path, all_evals)

    pending_list = [e for e in all_evals if e.get("status", "") in ["PENDING", "RERUN"]]
    running_list = [e for e in all_evals if e.get("status", "") in ["RUNNING", "running"]]
    finished_list = [
        e
        for e in all_evals
        if e.get("status", "").startswith("FINISHED") or e.get("status", "") in ["completed", "PENDING_NEW_EVAL"]
    ]
    df_pending = pd.DataFrame.from_records(pending_list, columns=cols)
    df_running = pd.DataFrame.from_records(running_list, columns=cols)
    df_finished = pd.DataFrame.from_records(finished_list, columns=cols)
    return df_finished, df_running, df_pending


def _load_queue_entry(file_path: str, all_evals: list) -> None:
    """Load a single queue entry JSON file into all_evals, with error handling."""
    try:
        with open(file_path) as fp:
            data = json.load(fp)

        if "model" not in data:
            print(f"WARNING: Skipping {file_path}: missing 'model' key")
            return

        data[EvalQueueColumn.model.name] = make_clickable_model(data["model"])
        data[EvalQueueColumn.revision.name] = data.get("revision", "main")
        # Ensure 'private' key exists (Gradio expects it for the bool column)
        if "private" not in data:
            data["private"] = False

        all_evals.append(data)
    except (json.JSONDecodeError, IOError, KeyError) as e:
        print(f"WARNING: Skipping {file_path}: {e}")
