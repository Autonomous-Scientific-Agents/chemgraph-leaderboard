import json
import os

import pandas as pd

from src.display.formatting import has_no_nan_values, make_clickable_model
from src.display.utils import AutoEvalColumn, EvalQueueColumn
from src.envs import WORKFLOWS, get_eval_results_path, get_eval_requests_path
from src.leaderboard.aggregate import build_trend_summary, get_history_df
from src.leaderboard.model_registry import current_models
from src.leaderboard.read_evals import get_all_eval_results, get_raw_eval_results


def _apply_current_model_filter(raw_data: list, results_path: str) -> list:
    """Keep only models on the curated roster (``dataset/current_models.json``).

    Applied to the raw ``EvalResult`` list — BEFORE ``to_dict()``, which wraps the
    name in a markdown anchor, and before the rank renumbering below, so ranks
    come out contiguous 1..N over the visible rows.

    Fails open: no usable roster -> ``raw_data`` unchanged. Deliberately NOT
    applied to the Trends tab, which shows the full history including retired
    models (see ``get_combined_trend_history_df``).
    """
    roster = current_models()
    if roster is None:
        return raw_data

    label = os.path.basename(os.path.normpath(results_path)) or results_path
    present = {str(r.full_model).strip().lower() for r in raw_data}
    kept = [r for r in raw_data if str(r.full_model).strip().lower() in roster]
    print(f"Current-model filter [{label}]: kept {len(kept)}/{len(raw_data)} models")

    missing = sorted(roster[k] for k in set(roster) - present)
    if missing:
        print(f"  NOTE [{label}]: {len(missing)} listed model(s) have no results yet: " + ", ".join(missing))
    return kept


def get_leaderboard_df(results_path: str, requests_path: str, cols: list, benchmark_cols: list) -> pd.DataFrame:
    """Creates a dataframe from all the individual experiment results.

    Restricted to the curated current-model roster — see
    ``_apply_current_model_filter``.
    """
    raw_data = get_raw_eval_results(results_path, requests_path)
    raw_data = _apply_current_model_filter(raw_data, results_path)
    all_data_json = [v.to_dict() for v in raw_data]

    if not all_data_json:
        # Return an empty DataFrame with the expected columns so the app
        # can still render without crashing.
        print("WARNING: No valid evaluation results found. Leaderboard will be empty.")
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame.from_records(all_data_json)
    df = df.sort_values(by=[AutoEvalColumn.average.name], ascending=False)
    df[AutoEvalColumn.rank.name] = range(1, len(df) + 1)
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


def get_combined_trend_history_df(
    base_results_path: str, base_requests_path: str
) -> pd.DataFrame:
    """Return a combined trend history DataFrame across all workflows.

    Each row is annotated with a ``workflow`` column so the chart can
    distinguish between single_agent and multi_agent results.
    """
    frames: list[pd.DataFrame] = []
    for wf in WORKFLOWS:
        results_path = get_eval_results_path(wf)
        requests_path = get_eval_requests_path(wf)
        all_results = get_all_eval_results(results_path, requests_path)
        if all_results:
            df = get_history_df(all_results)
            if not df.empty:
                df["workflow"] = wf
                frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def get_combined_trend_summary_df(
    base_results_path: str, base_requests_path: str
) -> pd.DataFrame:
    """Return a combined trend summary DataFrame across all workflows.

    Each row is annotated with a ``Workflow`` column.
    """
    frames: list[pd.DataFrame] = []
    for wf in WORKFLOWS:
        results_path = get_eval_results_path(wf)
        requests_path = get_eval_requests_path(wf)
        all_results = get_all_eval_results(results_path, requests_path)
        if all_results:
            df = build_trend_summary(all_results)
            if not df.empty:
                df["Workflow"] = wf
                frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


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
