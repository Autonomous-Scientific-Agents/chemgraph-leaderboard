import sys
import time
from datetime import datetime, timezone

import gradio as gr
from gradio_leaderboard import Leaderboard, ColumnFilter, SelectColumns
import pandas as pd
import plotly.express as px
from apscheduler.schedulers.background import BackgroundScheduler
from huggingface_hub import snapshot_download

from src.about import (
    CITATION_BUTTON_LABEL,
    CITATION_BUTTON_TEXT,
    EVALUATION_QUEUE_TEXT,
    INTRODUCTION_TEXT,
    LLM_BENCHMARKS_TEXT,
    TITLE,
)
from src.display.css_html_js import custom_css, group_columns_head
from src.display.utils import (
    BENCHMARK_COLS,
    COLS,
    EVAL_COLS,
    EVAL_TYPES,
    AutoEvalColumn,
    ModelType,
    fields,
    WeightType,
    Precision,
)
from src.envs import (
    API,
    EVAL_REQUESTS_PATH,
    EVAL_RESULTS_PATH,
    QUEUE_REPO,
    REPO_ID,
    RESULTS_REPO,
    TOKEN,
    WORKFLOWS,
    get_eval_results_path,
    get_eval_requests_path,
)
from src.leaderboard.read_evals import clear_eval_cache
from src.populate import (
    get_evaluation_queue_df,
    get_leaderboard_df,
    get_trend_summary_df,
    get_trend_history_df,
    get_combined_trend_history_df,
    get_combined_trend_summary_df,
)
from src.submission.submit import add_new_eval

# --local flag: skip HF Hub downloads and scheduler, use local data only.
LOCAL_MODE = "--local" in sys.argv


def restart_space():
    API.restart_space(repo_id=REPO_ID)


### Space initialisation

MAX_DOWNLOAD_RETRIES = 3


def download_with_retry(repo_id: str, local_dir: str, label: str) -> None:
    """Download a HF Hub dataset with retries. Restarts the Space only after all retries are exhausted."""
    for attempt in range(1, MAX_DOWNLOAD_RETRIES + 1):
        try:
            print(f"Downloading {label} ({repo_id}) — attempt {attempt}/{MAX_DOWNLOAD_RETRIES}")
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                repo_type="dataset",
                tqdm_class=None,
                etag_timeout=30,
                token=TOKEN,
            )
            return  # success
        except Exception as e:
            print(f"WARNING: Failed to download {label}: {e}")
            if attempt < MAX_DOWNLOAD_RETRIES:
                wait = 10 * attempt
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"ERROR: All {MAX_DOWNLOAD_RETRIES} download attempts failed for {label}. Restarting space.")
                restart_space()


if not LOCAL_MODE:
    download_with_retry(QUEUE_REPO, EVAL_REQUESTS_PATH, "eval requests")
    download_with_retry(RESULTS_REPO, EVAL_RESULTS_PATH, "eval results")
else:
    print("LOCAL MODE: skipping HF Hub downloads, using local eval-results/ and eval-queue/")

# Load leaderboard data for each workflow
SINGLE_AGENT_RESULTS = get_eval_results_path("single_agent")
SINGLE_AGENT_REQUESTS = get_eval_requests_path("single_agent")
MULTI_AGENT_RESULTS = get_eval_results_path("multi_agent")
MULTI_AGENT_REQUESTS = get_eval_requests_path("multi_agent")

LEADERBOARD_DF = get_leaderboard_df(SINGLE_AGENT_RESULTS, SINGLE_AGENT_REQUESTS, COLS, BENCHMARK_COLS)
LEADERBOARD_DF_MULTI = get_leaderboard_df(MULTI_AGENT_RESULTS, MULTI_AGENT_REQUESTS, COLS, BENCHMARK_COLS)

# Load combined trend data for the Trends tab
try:
    TREND_SUMMARY_DF = get_combined_trend_summary_df(EVAL_RESULTS_PATH, EVAL_REQUESTS_PATH)
    TREND_HISTORY_DF = get_combined_trend_history_df(EVAL_RESULTS_PATH, EVAL_REQUESTS_PATH)
except Exception as e:
    print(f"WARNING: Failed to load trend data: {e}")
    TREND_SUMMARY_DF = pd.DataFrame()
    TREND_HISTORY_DF = pd.DataFrame()

try:
    (
        finished_eval_queue_df,
        running_eval_queue_df,
        pending_eval_queue_df,
    ) = get_evaluation_queue_df(EVAL_REQUESTS_PATH, EVAL_COLS)
except Exception as e:
    print(f"WARNING: Failed to load evaluation queue: {e}")
    _empty_queue = pd.DataFrame(columns=EVAL_COLS)
    finished_eval_queue_df = _empty_queue
    running_eval_queue_df = _empty_queue.copy()
    pending_eval_queue_df = _empty_queue.copy()


def _all_families(history_df: pd.DataFrame) -> list[str]:
    """Sorted list of model families (the org slug part of full_name)."""
    if history_df is None or history_df.empty or "model" not in history_df.columns:
        return []
    fams = (
        history_df["model"]
        .dropna()
        .astype(str)
        .str.split("/", n=1).str[0]
        .unique()
        .tolist()
    )
    return sorted(fams)


# Parse a vaguely-versioned model name into a comparable release-order
# tuple. We don't know real release dates, so we synthesize an order from
# the version-like tokens in the name itself: bigger version → later.
# Examples:
#   "openai/gpt-4.1-nano"        -> (4, 1, 0, 0, 0, ...,  "openai/gpt-4.1-nano")
#   "openai/gpt-5.4"             -> (5, 4)
#   "anthropic/claude-opus-4.6"  -> (4, 6)
#   "anthropic/claude-3.5-haiku" -> (3, 5)
import re as _re
_VERSION_TOKEN = _re.compile(r"\d+(?:\.\d+)*")
def _release_sort_key(model_full: str):
    nums = []
    for tok in _VERSION_TOKEN.findall(model_full or ""):
        for part in tok.split("."):
            try:
                nums.append(int(part))
            except ValueError:
                pass
    # Pad so tuples sort consistently even when version depths differ.
    while len(nums) < 4:
        nums.append(0)
    return (*nums, model_full or "")


def _all_models(history_df: pd.DataFrame) -> list[str]:
    """Sorted list of every model that has at least one history row."""
    if history_df is None or history_df.empty or "model" not in history_df.columns:
        return []
    return sorted(history_df["model"].dropna().unique().tolist())


def _top_n_models(history_df: pd.DataFrame, n: int = 3) -> list[str]:
    """Top-N models by their most-recent average score (one row per model)."""
    if history_df is None or history_df.empty:
        return []
    df = history_df.dropna(subset=["model", "eval_date", "average"])
    if df.empty:
        return []
    latest = df.sort_values("eval_date").groupby("model").tail(1)
    return latest.sort_values("average", ascending=False)["model"].head(n).tolist()


# Date-range presets (radio shortcuts). "Past week" / "Past month"
# select a trailing-days window. From/To inputs (always visible) are
# the authoritative range; presets just snap the inputs to a window.
DATE_RANGE_PRESETS = {
    "Past week": 7,
    "Past month": 30,
}


def _parse_iso_date(text: str):
    if not text:
        return None
    try:
        return pd.to_datetime(text.strip()).normalize()
    except (ValueError, TypeError):
        return None


def _data_date_bounds(df: pd.DataFrame):
    """Return (min, max) eval_date as ISO strings, or ("", "") if empty."""
    if df is None or df.empty or "eval_date" not in df.columns:
        return "", ""
    dates = pd.to_datetime(df["eval_date"]).dropna()
    if dates.empty:
        return "", ""
    return dates.min().strftime("%Y-%m-%d"), dates.max().strftime("%Y-%m-%d")


# Tasks the user can pick in the Family chart. "Average" stays as a
# synthetic option pointing at the overall column; the rest are the
# per-category accuracy columns produced by aggregate.py.
def _family_task_choices() -> list[str]:
    from src.about import Tasks  # local import to avoid pulling at module top
    return ["Average"] + [t.value.col_name for t in Tasks]


def _family_window_stats(
    history_df: pd.DataFrame,
    family: str,
    workflow_filter: str,
    date_from: str,
    date_to: str,
    tasks: list[str] | None = None,
) -> pd.DataFrame:
    """Long-form stats: one row per (Model, Workflow, Task) in the window.

    Columns: Model, Workflow, Task, #Days, Score, Std.
    """
    cols = ["Model", "Workflow", "Task", "#Days", "Score", "Std"]
    if history_df is None or history_df.empty:
        return pd.DataFrame(columns=cols)
    if not tasks:
        tasks = ["Average"]

    df = history_df.copy()
    if workflow_filter != "All" and "workflow" in df.columns:
        df = df[df["workflow"] == workflow_filter]
    df = _apply_date_range(df, date_from, date_to)
    if df.empty:
        return pd.DataFrame(columns=cols)

    df = df[df["model"].astype(str).str.startswith(f"{family}/")]
    if df.empty:
        return pd.DataFrame(columns=cols)

    available = [t for t in tasks if t == "Average" or t in df.columns]
    if not available:
        return pd.DataFrame(columns=cols)

    group_cols = ["model"] + (["workflow"] if "workflow" in df.columns else [])
    rows: list[pd.DataFrame] = []
    for t in available:
        col = "average" if t == "Average" else t
        if col not in df.columns:
            continue
        agg = (
            df.dropna(subset=[col])
            .groupby(group_cols, dropna=False)[col]
            .agg(["count", "mean", "std"])
            .reset_index()
            .rename(columns={
                "model": "Model",
                "workflow": "Workflow",
                "count": "#Days",
                "mean": "Score",
                "std": "Std",
            })
        )
        if agg.empty:
            continue
        if "Workflow" not in agg.columns:
            agg["Workflow"] = ""
        agg["Task"] = t
        # Keep Std as NaN when N==1 — a 0-filled std reads visually
        # identical to a tight 7-day std and dishonestly suggests
        # "rock solid." We also blank Std at N<3 so the chart suppresses
        # whiskers below the threshold where std is meaningful.
        agg.loc[agg["#Days"] < 3, "Std"] = float("nan")
        rows.append(agg)

    if not rows:
        return pd.DataFrame(columns=cols)

    out = pd.concat(rows, ignore_index=True)
    for c in ("Score", "Std"):
        out[c] = out[c].round(2)
    out["__order"] = out["Model"].map(_release_sort_key)
    out = out.sort_values(["__order", "Workflow", "Task"]).drop(columns="__order")
    return out[cols].reset_index(drop=True)


def build_family_chart(
    history_df: pd.DataFrame,
    family: str,
    workflow_filter: str = "All",
    date_from: str = "",
    date_to: str = "",
    tasks: list[str] | None = None,
):
    """Bar chart: X = models (release order), one colored bar per task,
    error bar = std over the window."""
    if not tasks:
        tasks = ["Average"]
    stats = _family_window_stats(
        history_df, family, workflow_filter, date_from, date_to, tasks
    )
    if stats.empty:
        fig = px.bar(title=f"No data for family '{family}' in the selected range")
        fig.update_layout(xaxis_title="Model", yaxis_title="Score (%)")
        return fig

    stats = stats.copy()
    stats["Label"] = stats["Model"].str.replace(f"{family}/", "", regex=False)
    # Preserve release-order for the X axis categorical.
    label_order = (
        stats.drop_duplicates("Label")
        .assign(__o=lambda d: d["Model"].map(_release_sort_key))
        .sort_values("__o")["Label"]
        .tolist()
    )
    # Preserve task selection order for legend / cluster order.
    task_order = [t for t in tasks if t in stats["Task"].unique()]

    # When workflow=All, draw two bars per (model, task): solid for
    # single_agent, diagonal pattern for multi_agent.
    workflows_present = (
        stats["Workflow"].dropna().unique().tolist()
        if "Workflow" in stats.columns else []
    )
    use_pattern = workflow_filter == "All" and len(workflows_present) > 1
    chart_kwargs = dict(
        x="Label",
        y="Score",
        color="Task",
        error_y="Std",
        barmode="group",
        category_orders={"Label": label_order, "Task": task_order},
        title=f"{family} — score by task over selected range",
        labels={"Label": "Model (release order →)", "Score": "Score (%)"},
    )
    if use_pattern:
        chart_kwargs["pattern_shape"] = "Workflow"
        chart_kwargs["pattern_shape_sequence"] = ["", "/"]  # solid, diagonal
        chart_kwargs["category_orders"]["Workflow"] = ["single_agent", "multi_agent"]
    fig = px.bar(stats, **chart_kwargs)
    # Y range considers the top of the error bars (mean + std), not
    # just the mean, so the cap doesn't clip a sticking-up whisker.
    # Allow the upper limit to drift above 100 a few percent so chart
    # never feels claustrophobic when bars hug the ceiling.
    upper = (stats["Score"] + stats["Std"]).dropna()
    lower = (stats["Score"] - stats["Std"]).dropna()
    if not upper.empty:
        y_min = float(lower.min())
        y_max = float(upper.max())
        span = y_max - y_min
        pad = max(span * 0.15, 4.0)
        y_range = [max(0.0, y_min - pad), min(110.0, y_max + pad)]
    else:
        y_range = None
    fig.update_layout(
        autosize=True,
        xaxis_title="Model (release order →)",
        yaxis_title="Score (%)",
        yaxis=dict(range=y_range) if y_range else dict(),
        legend_title="Task",
        hovermode="x unified",
    )
    return fig


def filter_family_data(
    family: str,
    workflow_filter: str = "All",
    date_from: str = "",
    date_to: str = "",
    tasks: list[str] | None = None,
):
    """Re-render family chart + table from cached history."""
    chart = build_family_chart(
        TREND_HISTORY_DF, family, workflow_filter, date_from, date_to, tasks
    )
    table = _family_window_stats(
        TREND_HISTORY_DF, family, workflow_filter, date_from, date_to, tasks
    )
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return chart, table, f"<small>Last view: {stamp}</small>"


def refresh_family_data(
    family: str,
    workflow_filter: str = "All",
    date_from: str = "",
    date_to: str = "",
    tasks: list[str] | None = None,
):
    """Re-download eval results from HF Hub then re-filter the family view."""
    global TREND_HISTORY_DF, TREND_SUMMARY_DF
    try:
        if not LOCAL_MODE:
            download_with_retry(RESULTS_REPO, EVAL_RESULTS_PATH, "eval results")
        clear_eval_cache()
        TREND_SUMMARY_DF = get_combined_trend_summary_df(EVAL_RESULTS_PATH, EVAL_REQUESTS_PATH)
        TREND_HISTORY_DF = get_combined_trend_history_df(EVAL_RESULTS_PATH, EVAL_REQUESTS_PATH)
    except Exception as e:
        print(f"WARNING: Failed to refresh trend data: {e}")
        TREND_SUMMARY_DF = pd.DataFrame()
        TREND_HISTORY_DF = pd.DataFrame()
    chart, table, _ = filter_family_data(family, workflow_filter, date_from, date_to, tasks)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return chart, table, f"<small>Hub data: {stamp}</small>"


def _default_family_window(history_df: pd.DataFrame) -> tuple[str, str]:
    """Default 'past 7 days' anchored on dataset max date as ISO strings."""
    _, dmax = _data_date_bounds(history_df)
    if not dmax:
        return "", ""
    end = pd.to_datetime(dmax)
    start = end - pd.Timedelta(days=6)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _window_stats(
    history_df: pd.DataFrame,
    workflow_filter: str,
    models: list[str] | None,
    date_from: str,
    date_to: str,
) -> pd.DataFrame:
    """One row per (Model, Workflow) summarizing the visible window.

    Columns: Model, Workflow, N, Average, Min, Max, Std.
    """
    cols = ["Model", "Workflow", "#Days", "Average", "Min", "Max", "Std"]
    if history_df is None or history_df.empty:
        return pd.DataFrame(columns=cols)

    df = history_df.copy()
    if workflow_filter != "All" and "workflow" in df.columns:
        df = df[df["workflow"] == workflow_filter]
    if models:
        df = df[df["model"].isin(models)]
    df = _apply_date_range(df, date_from, date_to)
    if df.empty:
        return pd.DataFrame(columns=cols)

    group_cols = ["model"] + (["workflow"] if "workflow" in df.columns else [])
    grouped = df.groupby(group_cols, dropna=False)["average"].agg(
        ["count", "mean", "min", "max", "std"]
    ).reset_index()
    grouped["std"] = grouped["std"].fillna(0.0)
    grouped = grouped.rename(columns={
        "model": "Model",
        "workflow": "Workflow",
        "count": "#Days",
        "mean": "Average",
        "min": "Min",
        "max": "Max",
        "std": "Std",
    })
    if "Workflow" not in grouped.columns:
        grouped["Workflow"] = ""
    for c in ("Average", "Min", "Max", "Std"):
        grouped[c] = grouped[c].round(2)
    return grouped.sort_values(["Average"], ascending=False)[cols].reset_index(drop=True)


def _apply_date_range(
    df: pd.DataFrame,
    date_from: str | None = None,
    date_to: str | None = None,
) -> pd.DataFrame:
    """Filter by inclusive [from, to]. Empty/invalid bounds are ignored."""
    if df is None or df.empty or "eval_date" not in df.columns:
        return df
    start = _parse_iso_date(date_from)
    end = _parse_iso_date(date_to)
    if start is None and end is None:
        return df
    eval_dt = pd.to_datetime(df["eval_date"])
    mask = pd.Series(True, index=df.index)
    if start is not None:
        mask &= eval_dt >= start
    if end is not None:
        mask &= eval_dt <= end
    return df[mask]


def build_trend_chart(
    history_df: pd.DataFrame,
    workflow_filter: str = "All",
    models: list[str] | None = None,
    date_from: str = "",
    date_to: str = "",
):
    """Build a Plotly line chart showing model scores over time.

    Parameters
    ----------
    history_df : pd.DataFrame
        Combined history with an optional ``workflow`` column.
    workflow_filter : str
        One of "All", "single_agent", or "multi_agent".
    models : list[str], optional
        Restrict the chart to these models. None or empty list means
        "show nothing" (so the user can clear all models intentionally).
    date_range : str
        One of the keys in DATE_RANGE_PRESETS. Filters rows to the last
        N days of available data.
    """
    if history_df.empty:
        fig = px.line(title="No historical data available yet")
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Average Score (%)",
        )
        return fig

    df = history_df.copy()

    # Apply workflow filter
    if workflow_filter != "All" and "workflow" in df.columns:
        df = df[df["workflow"] == workflow_filter]

    # Apply model filter
    if not models:
        fig = px.line(title="Select one or more models to show their trend")
        fig.update_layout(xaxis_title="Date", yaxis_title="Average Score (%)")
        return fig
    df = df[df["model"].isin(models)]

    # Apply date-range filter
    df = _apply_date_range(df, date_from, date_to)

    if df.empty:
        fig = px.line(title=f"No data in {date_from or '…'} → {date_to or '…'} for workflow: {workflow_filter}")
        fig.update_layout(xaxis_title="Date", yaxis_title="Average Score (%)")
        return fig

    # When workflow=All, distinguish single vs multi with line dash.
    if "workflow" in df.columns and workflow_filter == "All":
        fig = px.line(
            df,
            x="eval_date",
            y="average",
            color="model",
            line_dash="workflow",
            markers=True,
            title="Model Performance Over Time",
            labels={
                "eval_date": "Evaluation Date",
                "average": "Average Score (%)",
                "model": "Model",
                "workflow": "Workflow",
            },
        )
    else:
        fig = px.line(
            df,
            x="eval_date",
            y="average",
            color="model",
            markers=True,
            title="Model Performance Over Time",
            labels={
                "eval_date": "Evaluation Date",
                "average": "Average Score (%)",
                "model": "Model",
            },
        )

    # Y-axis: auto-scale with a small padding around the visible
    # min/max so points don't sit flush against the chart edges.
    # Upper limit drifts to 105 (matching the Family chart's 110 cap
    # in spirit) so a 99-score point keeps a bit of breathing room.
    visible = df["average"].dropna()
    if not visible.empty:
        y_min, y_max = float(visible.min()), float(visible.max())
        span = y_max - y_min
        pad = max(span * 0.1, 2.0)  # at least 2 percentage points of headroom
        y_range = [max(0.0, y_min - pad), min(105.0, y_max + pad)]
    else:
        y_range = None

    fig.update_layout(
        autosize=True,
        xaxis_title="Date",
        yaxis_title="Average Score (%)",
        legend_title="Model",
        hovermode="x unified",
        yaxis=dict(range=y_range) if y_range else dict(),
    )
    return fig


def filter_trend_data(
    workflow_filter: str = "All",
    models: list[str] | None = None,
    date_from: str = "",
    date_to: str = "",
):
    """Filter the already-loaded trend data in memory (no re-download)."""
    history_df = TREND_HISTORY_DF
    summary_df = _window_stats(history_df, workflow_filter, models, date_from, date_to)
    chart = build_trend_chart(history_df, workflow_filter, models, date_from, date_to)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return chart, summary_df, f"<small>Last view: {stamp}</small>"


def refresh_trend_data(
    workflow_filter: str = "All",
    models: list[str] | None = None,
    date_from: str = "",
    date_to: str = "",
):
    """Re-download eval results from HF Hub and recompute trend data.

    Returns updated values for the trend chart, summary table, and a
    last-updated timestamp string.  Errors during download or computation
    are caught so the UI never crashes.
    """
    global TREND_HISTORY_DF, TREND_SUMMARY_DF

    try:
        if not LOCAL_MODE:
            download_with_retry(RESULTS_REPO, EVAL_RESULTS_PATH, "eval results")

        # Clear the evaluation cache so fresh data is loaded from disk.
        clear_eval_cache()

        TREND_SUMMARY_DF = get_combined_trend_summary_df(EVAL_RESULTS_PATH, EVAL_REQUESTS_PATH)
        TREND_HISTORY_DF = get_combined_trend_history_df(EVAL_RESULTS_PATH, EVAL_REQUESTS_PATH)
    except Exception as e:
        print(f"WARNING: Failed to refresh trend data: {e}")
        TREND_SUMMARY_DF = pd.DataFrame()
        TREND_HISTORY_DF = pd.DataFrame()

    # Delegate to the lightweight filter function for the current view,
    # then override the timestamp prefix so the caption flags this as a
    # Hub pull (not just a re-filter).
    chart, summary_df, _ = filter_trend_data(workflow_filter, models, date_from, date_to)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return chart, summary_df, f"<small>Hub data: {stamp}</small>"


# =========================================================================
# Highlights view — top-10 hero + per-task top-5 mini bars
# =========================================================================

_MODEL_LINK_RE = _re.compile(r"<a[^>]*>(.*?)</a>", _re.IGNORECASE | _re.DOTALL)


def _strip_model_markdown(s: str) -> str:
    """Pull the visible model name out of the markdown anchor stored in the
    leaderboard's Model column."""
    if not isinstance(s, str):
        return str(s)
    m = _MODEL_LINK_RE.search(s)
    return (m.group(1) if m else s).strip()


def _family_logo_url(family: str) -> str:
    """Hugging Face org avatar URL. Returns a placeholder for unknowns."""
    if not family:
        return ""
    return f"https://huggingface.co/api/organizations/{family}/avatar"


_TASK_COL_NAMES = None
def _task_col_names() -> list[str]:
    global _TASK_COL_NAMES
    if _TASK_COL_NAMES is None:
        from src.about import Tasks
        _TASK_COL_NAMES = [t.value.col_name for t in Tasks]
    return _TASK_COL_NAMES


def _topn_for_column(df: pd.DataFrame, score_col: str, n: int) -> pd.DataFrame:
    """Return the top-N rows of `df` by `score_col`, with the columns
    needed to draw a Highlights bar: (display_name, family, score)."""
    cols_needed = ["Model", "Model Family", score_col]
    missing = [c for c in cols_needed if c not in df.columns]
    if missing or df.empty:
        return pd.DataFrame(columns=["display", "family", "score"])
    out = df[cols_needed].dropna(subset=[score_col]).copy()
    if out.empty:
        return pd.DataFrame(columns=["display", "family", "score"])
    out["display"] = out["Model"].map(_strip_model_markdown)
    out["family"] = out["Model Family"].fillna("unknown")
    # Strip "org/" prefix from the model display name — the org is
    # already encoded by the logo to the left of each tick, so
    # repeating it in the text just makes the y-axis labels long
    # enough to overlap the logo column.
    out["display"] = out["display"].str.replace(r"^[^/]+/", "", regex=True)
    out["score"] = out[score_col]
    out = out.sort_values("score", ascending=False).head(n).reset_index(drop=True)
    return out[["display", "family", "score"]]


def _build_highlight_bar(
    df: pd.DataFrame, score_col: str, n: int, title: str, height: int
):
    """Horizontal Plotly bar with logo images next to each model tick."""
    top = _topn_for_column(df, score_col, n)
    fig = px.bar(orientation="h", title=title)
    if top.empty:
        fig.update_layout(
            height=height,
            xaxis_title="Score (%)",
            yaxis_title="",
            margin=dict(l=140, r=20, t=40, b=30),
            annotations=[dict(
                text="No data", showarrow=False,
                xref="paper", yref="paper", x=0.5, y=0.5,
            )],
        )
        return fig

    # Force the y-axis category order to follow our descending-score
    # ordering. Without this, color="family" makes Plotly draw one
    # trace per family and the categorical y axis ends up grouped by
    # family ("anthropic block" then "openai block") instead of by
    # rank. With autorange="reversed", the first entry in this list
    # lands at the top of the chart.
    order = top["display"].tolist()
    fig = px.bar(
        top,
        x="score",
        y="display",
        orientation="h",
        color="family",
        text=top["score"].round(1).astype(str) + "%",
        title=title,
        labels={"score": "Score (%)", "display": "", "family": "Family"},
        category_orders={"display": order},
    )
    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
    )
    # Put logos in their own column to the LEFT of the model-name
    # ticks (x=-0.18 paper coord, well into the left margin) so the
    # logo image and the text don't sit on top of each other. The
    # left margin is widened to 230px to host both.
    images = []
    for _, row in top.iterrows():
        url = _family_logo_url(row["family"])
        if not url:
            continue
        images.append(dict(
            source=url,
            xref="paper", yref="y",
            x=-0.18, y=row["display"],
            sizex=0.05, sizey=0.7,
            xanchor="center", yanchor="middle",
            layer="above",
        ))
    fig.update_layout(
        autosize=True,
        height=height,
        margin=dict(l=230, r=60, t=50, b=30),
        xaxis=dict(range=[0, 110], title="Score (%)"),
        # category_orders above already pins the order. Don't ALSO
        # set autorange="reversed" — that double-flips and puts the
        # worst model at the top.
        yaxis=dict(title="", automargin=True),
        showlegend=False,
        images=images,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def build_highlights_hero(df: pd.DataFrame, workflow_label: str):
    """Top-10 by overall Average for the given benchmark."""
    return _build_highlight_bar(
        df,
        score_col="Average ⬆️",
        n=10,
        title=f"{workflow_label} — Top 10 by Average",
        height=520,
    )


def build_highlights_task_grid(df: pd.DataFrame) -> list[tuple[str, "object"]]:
    """List of (task_name, figure) for the 12 per-task top-5 mini bars."""
    out = []
    for task in _task_col_names():
        if task not in df.columns:
            continue
        fig = _build_highlight_bar(df, score_col=task, n=5, title=task, height=300)
        out.append((task, fig))
    return out


def init_leaderboard(dataframe):
    if dataframe is None or dataframe.empty:
        # Show an empty leaderboard instead of crashing.
        dataframe = pd.DataFrame(columns=COLS)
    return Leaderboard(
        value=dataframe,
        datatype=[c.type for c in fields(AutoEvalColumn)],
        select_columns=SelectColumns(
            default_selection=[c.name for c in fields(AutoEvalColumn) if c.displayed_by_default],
            cant_deselect=[c.name for c in fields(AutoEvalColumn) if c.never_hidden],
            label="Columns to display",
        ),
        search_columns=[AutoEvalColumn.model.name, AutoEvalColumn.model_family.name],
        hide_columns=[c.name for c in fields(AutoEvalColumn) if c.hidden],
        filter_columns=[
            ColumnFilter(AutoEvalColumn.model_family.name, type="checkboxgroup", label="Model family"),
        ],
        interactive=False,
    )


demo = gr.Blocks(css=custom_css, head=group_columns_head)
with demo:
    gr.HTML(TITLE)
    gr.Markdown(INTRODUCTION_TEXT, elem_classes="markdown-text", elem_id="cg-intro-block")

    with gr.Tabs(elem_classes="tab-buttons") as tabs:
        def _benchmark_subtabs(label: str, df: pd.DataFrame, base_elem_id: str):
            """Render Highlights + Full table sub-tabs for one benchmark."""
            full_tab_idx = 1  # the View all button switches to this sub-tab
            n_models = 0 if df is None or df.empty else len(df)
            with gr.Tabs(elem_classes="tab-buttons", elem_id=f"{base_elem_id}-subtabs") as subtabs:
                with gr.TabItem("🏆 Highlights", id=0):
                    gr.Plot(
                        value=build_highlights_hero(df, label),
                        elem_id=f"{base_elem_id}-hero",
                        elem_classes="cg-highlight-plot",
                    )
                    gr.HTML('<div class="cg-task-grid-label">By task — top 5</div>')
                    # Use plain gr.Column (single-column stack) and let CSS
                    # turn its inner .cg-task-card children into a responsive
                    # grid. Using gr.Row here caused Gradio to wedge the
                    # whole Highlights view into half the page width.
                    with gr.Column(elem_classes="cg-task-grid"):
                        for task_name, fig in build_highlights_task_grid(df):
                            with gr.Column(elem_classes="cg-task-card"):
                                gr.Plot(
                                    value=fig,
                                    elem_classes="cg-task-plot",
                                )
                    view_all_btn = gr.Button(
                        f"Open full table ({n_models} models) →",
                        size="sm",
                        elem_id=f"{base_elem_id}-viewall",
                        elem_classes="cg-viewall-btn",
                    )
                with gr.TabItem("📊 Full table", id=full_tab_idx) as full_tab:
                    init_leaderboard(df)
            view_all_btn.click(
                fn=lambda: gr.Tabs(selected=full_tab_idx), outputs=subtabs
            )

        with gr.TabItem("🏅 Single-Agent Benchmark", elem_id="llm-benchmark-tab-table", id=0):
            _benchmark_subtabs("Single-Agent", LEADERBOARD_DF, "cg-single")

        with gr.TabItem("🤝 Multi-Agent Benchmark", elem_id="multi-agent-benchmark-tab-table", id=1):
            _benchmark_subtabs("Multi-Agent", LEADERBOARD_DF_MULTI, "cg-multi")

        with gr.TabItem("📈 Trends", elem_id="llm-benchmark-tab-trends", id=2):
            # Default to single_agent + its top-3 models by latest score.
            _default_workflow = "single_agent"
            _initial_history = (
                TREND_HISTORY_DF[TREND_HISTORY_DF["workflow"] == _default_workflow]
                if (not TREND_HISTORY_DF.empty and "workflow" in TREND_HISTORY_DF.columns)
                else TREND_HISTORY_DF
            )
            _initial_models = _top_n_models(_initial_history, n=3)
            _data_min, _data_max = _data_date_bounds(TREND_HISTORY_DF)
            # Family-evolution defaults: full-range window (was 7d, which
            # hid models that weren't re-evaluated last week), Average
            # task only (was [Average, Reaction Energy] — biased toward
            # the hardest task), and single_agent (was All — turned on
            # the pattern-stripe encoding before the user opted in).
            _fam_from_default, _fam_to_default = _data_min, _data_max
            _all_fams = _all_families(TREND_HISTORY_DF)
            _default_family = "openai" if "openai" in _all_fams else (_all_fams[0] if _all_fams else "")

            with gr.Tabs(elem_classes="tab-buttons", elem_id="cg-trend-subtabs"):
                # ---------- Sub-tab 1: Over time ----------
                with gr.TabItem("Over time", id=0):
                    with gr.Row(elem_id="cg-trend-controls", elem_classes="cg-controls-row", equal_height=True):
                        with gr.Column(scale=10, min_width=360, elem_classes="cg-zone cg-zone-data"):
                            workflow_filter = gr.Dropdown(
                                choices=["All", "single_agent", "multi_agent"],
                                value="single_agent",
                                label="Workflow",
                                interactive=True,
                            )
                            model_filter = gr.Dropdown(
                                choices=_all_models(TREND_HISTORY_DF),
                                value=_initial_models,
                                label="Models",
                                multiselect=True,
                                interactive=True,
                            )
                        with gr.Column(scale=5, min_width=260, elem_classes="cg-zone cg-zone-view"):
                            with gr.Row(elem_id="cg-trend-date-range"):
                                date_from = gr.Textbox(
                                    value=_data_min,
                                    label="From",
                                    placeholder="YYYY-MM-DD",
                                    interactive=True,
                                    lines=1,
                                    max_lines=1,
                                    elem_id="cg-trend-from",
                                    elem_classes="cg-date-input",
                                )
                                date_to = gr.Textbox(
                                    value=_data_max,
                                    label="To",
                                    placeholder="YYYY-MM-DD",
                                    interactive=True,
                                    lines=1,
                                    max_lines=1,
                                    elem_id="cg-trend-to",
                                    elem_classes="cg-date-input",
                                )
                            gr.HTML(
                                f'<span id="cg-trend-date-bounds" '
                                f'data-min="{_data_min}" data-max="{_data_max}" '
                                f'style="display:none"></span>'
                            )
                        with gr.Column(scale=3, min_width=180, elem_classes="cg-zone cg-zone-actions"):
                            refresh_btn = gr.Button(
                                "🔄 Re-pull from Hub",
                                size="sm",
                                elem_id="cg-refresh-btn",
                            )
                            last_updated_box = gr.Markdown(
                                value=f"<small>Last view: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</small>",
                                elem_classes="cg-last-updated",
                            )
                    trend_chart = gr.Plot(
                        value=build_trend_chart(
                            TREND_HISTORY_DF,
                            workflow_filter=_default_workflow,
                            models=_initial_models,
                            date_from=_data_min,
                            date_to=_data_max,
                        ),
                        elem_id="cg-trend-chart",
                    )
                    gr.Markdown(
                        "### Window stats (Average / Min / Max / Std over the selected range)",
                        elem_id="cg-trend-summary-label",
                    )
                    _initial_summary = _window_stats(
                        TREND_HISTORY_DF,
                        _default_workflow,
                        _initial_models,
                        _data_min,
                        _data_max,
                    )
                    trend_table = gr.Dataframe(
                        value=_initial_summary,
                        interactive=False,
                        elem_id="cg-trend-summary",
                    )

                    _trend_inputs = [workflow_filter, model_filter, date_from, date_to]
                    for ctl in (workflow_filter, model_filter, date_from, date_to):
                        ctl.change(
                            fn=filter_trend_data,
                            inputs=_trend_inputs,
                            outputs=[trend_chart, trend_table, last_updated_box],
                        )
                    refresh_btn.click(
                        fn=refresh_trend_data,
                        inputs=_trend_inputs,
                        outputs=[trend_chart, trend_table, last_updated_box],
                        show_progress="full",
                    )

                # ---------- Sub-tab 2: Family evolution ----------
                with gr.TabItem("Family evolution", id=1):
                    _fam_task_choices = _family_task_choices()
                    _fam_default_tasks = (
                        ["Average"] if "Average" in _fam_task_choices else []
                    )
                    _fam_default_workflow = "single_agent"
                    with gr.Row(elem_id="cg-family-controls", elem_classes="cg-controls-row", equal_height=True):
                        with gr.Column(scale=10, min_width=360, elem_classes="cg-zone cg-zone-data"):
                            with gr.Row():
                                family_filter = gr.Dropdown(
                                    choices=_all_fams,
                                    value=_default_family,
                                    label="Family",
                                    interactive=True,
                                )
                                fam_workflow_filter = gr.Dropdown(
                                    choices=["All", "single_agent", "multi_agent"],
                                    value=_fam_default_workflow,
                                    label="Workflow",
                                    interactive=True,
                                )
                            fam_task_filter = gr.Dropdown(
                                choices=_fam_task_choices,
                                value=_fam_default_tasks,
                                label="Tasks",
                                multiselect=True,
                                interactive=True,
                            )
                        with gr.Column(scale=5, min_width=260, elem_classes="cg-zone cg-zone-view"):
                            with gr.Row(elem_id="cg-family-date-range"):
                                fam_date_from = gr.Textbox(
                                    value=_fam_from_default,
                                    label="From",
                                    placeholder="YYYY-MM-DD",
                                    interactive=True,
                                    lines=1,
                                    max_lines=1,
                                    elem_id="cg-fam-from",
                                    elem_classes="cg-date-input",
                                )
                                fam_date_to = gr.Textbox(
                                    value=_fam_to_default,
                                    label="To",
                                    placeholder="YYYY-MM-DD",
                                    interactive=True,
                                    lines=1,
                                    max_lines=1,
                                    elem_id="cg-fam-to",
                                    elem_classes="cg-date-input",
                                )
                            gr.HTML(
                                f'<span id="cg-fam-date-bounds" '
                                f'data-min="{_data_min}" data-max="{_data_max}" '
                                f'style="display:none"></span>'
                            )
                        with gr.Column(scale=3, min_width=180, elem_classes="cg-zone cg-zone-actions"):
                            fam_refresh_btn = gr.Button(
                                "🔄 Re-pull from Hub",
                                size="sm",
                                elem_id="cg-fam-refresh-btn",
                            )
                            fam_last_updated = gr.Markdown(
                                value=f"<small>Last view: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</small>",
                                elem_classes="cg-last-updated",
                            )
                    family_chart = gr.Plot(
                        value=build_family_chart(
                            TREND_HISTORY_DF,
                            family=_default_family,
                            workflow_filter=_fam_default_workflow,
                            date_from=_fam_from_default,
                            date_to=_fam_to_default,
                            tasks=_fam_default_tasks,
                        ),
                        elem_id="cg-family-chart",
                    )
                    gr.Markdown(
                        "### Family stats (Score ± Std per task, models in release order)",
                        elem_id="cg-family-summary-label",
                    )
                    family_table = gr.Dataframe(
                        value=_family_window_stats(
                            TREND_HISTORY_DF,
                            _default_family,
                            _fam_default_workflow,
                            _fam_from_default,
                            _fam_to_default,
                            _fam_default_tasks,
                        ),
                        interactive=False,
                        elem_id="cg-family-summary",
                    )

                    _family_inputs = [family_filter, fam_workflow_filter,
                                      fam_date_from, fam_date_to, fam_task_filter]
                    for ctl in _family_inputs:
                        ctl.change(
                            fn=filter_family_data,
                            inputs=_family_inputs,
                            outputs=[family_chart, family_table, fam_last_updated],
                        )
                    fam_refresh_btn.click(
                        fn=refresh_family_data,
                        inputs=_family_inputs,
                        outputs=[family_chart, family_table, fam_last_updated],
                        show_progress="full",
                    )

        with gr.TabItem("📝 About", elem_id="llm-benchmark-tab-table", id=3):
            gr.Markdown(LLM_BENCHMARKS_TEXT, elem_classes="markdown-text", elem_id="cg-about-content")

        with gr.TabItem("🚀 Submit here! ", elem_id="llm-benchmark-tab-table", id=4):
            with gr.Column():
                with gr.Row():
                    gr.Markdown(EVALUATION_QUEUE_TEXT, elem_classes="markdown-text", elem_id="cg-submit-guide")

                with gr.Column():
                    with gr.Accordion(
                        f"✅ Finished Evaluations ({len(finished_eval_queue_df)})",
                        open=False,
                        elem_classes="cg-queue-accordion",
                    ):
                        with gr.Row():
                            finished_eval_table = gr.components.Dataframe(
                                value=finished_eval_queue_df,
                                headers=EVAL_COLS,
                                datatype=EVAL_TYPES,
                                row_count=5,
                            )
                    with gr.Accordion(
                        f"🔄 Running Evaluation Queue ({len(running_eval_queue_df)})",
                        open=False,
                        elem_classes="cg-queue-accordion",
                    ):
                        with gr.Row():
                            running_eval_table = gr.components.Dataframe(
                                value=running_eval_queue_df,
                                headers=EVAL_COLS,
                                datatype=EVAL_TYPES,
                                row_count=5,
                            )

                    with gr.Accordion(
                        f"⏳ Pending Evaluation Queue ({len(pending_eval_queue_df)})",
                        open=False,
                        elem_classes="cg-queue-accordion",
                    ):
                        with gr.Row():
                            pending_eval_table = gr.components.Dataframe(
                                value=pending_eval_queue_df,
                                headers=EVAL_COLS,
                                datatype=EVAL_TYPES,
                                row_count=5,
                            )
            with gr.Row():
                gr.Markdown("# ✉️✨ Submit your model here!", elem_classes="markdown-text", elem_id="cg-submit-heading")

            with gr.Row():
                with gr.Column():
                    model_name_textbox = gr.Textbox(label="Model name")
                    revision_name_textbox = gr.Textbox(label="Revision commit", placeholder="main")
                    model_type = gr.Dropdown(
                        choices=[t.to_str(" : ") for t in ModelType if t != ModelType.Unknown],
                        label="Model type",
                        multiselect=False,
                        value=None,
                        interactive=True,
                    )

                with gr.Column():
                    precision = gr.Dropdown(
                        choices=[i.value.name for i in Precision if i != Precision.Unknown],
                        label="Precision",
                        multiselect=False,
                        value="float16",
                        interactive=True,
                    )
                    weight_type = gr.Dropdown(
                        choices=[i.value.name for i in WeightType],
                        label="Weights type",
                        multiselect=False,
                        value="Original",
                        interactive=True,
                    )
                    base_model_name_textbox = gr.Textbox(label="Base model (for delta or adapter weights)")
                    workflow_type = gr.Dropdown(
                        choices=["Both", "single_agent", "multi_agent"],
                        label="Workflow",
                        multiselect=False,
                        value="Both",
                        interactive=True,
                    )

            submit_button = gr.Button("Submit Eval", elem_id="cg-submit-btn")
            submission_result = gr.Markdown()
            submit_button.click(
                add_new_eval,
                [
                    model_name_textbox,
                    base_model_name_textbox,
                    revision_name_textbox,
                    precision,
                    weight_type,
                    model_type,
                    workflow_type,
                ],
                submission_result,
            )

    with gr.Row(elem_id="cg-citation-section"):
        with gr.Accordion("📙 Citation", open=False):
            citation_button = gr.Textbox(
                value=CITATION_BUTTON_TEXT,
                label=CITATION_BUTTON_LABEL,
                lines=20,
                elem_id="citation-button",
                show_copy_button=True,
            )

if not LOCAL_MODE:
    scheduler = BackgroundScheduler()
    scheduler.add_job(restart_space, "interval", seconds=21600)  # every 6 hours
    scheduler.start()

demo.queue(default_concurrency_limit=4).launch(ssr_mode=False)
