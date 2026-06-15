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


# Date-range presets shown to the user (radio labels) and the number of
# trailing days each one keeps. "All time" means no filter.
DATE_RANGE_PRESETS = {
    "Past week": 7,
    "Past month": 30,
    "All time": None,
}


def _apply_date_range(df: pd.DataFrame, preset: str) -> pd.DataFrame:
    """Filter a history DataFrame to the last N days based on the preset.

    Anchored on the dataset's max eval_date, not wall-clock today, so the
    window stays useful even if the most recent eval was a few days ago.
    """
    if df is None or df.empty or "eval_date" not in df.columns:
        return df
    days = DATE_RANGE_PRESETS.get(preset)
    if days is None:
        return df
    max_date = pd.to_datetime(df["eval_date"]).max()
    if pd.isna(max_date):
        return df
    cutoff = max_date - pd.Timedelta(days=days - 1)
    return df[pd.to_datetime(df["eval_date"]) >= cutoff]


def build_trend_chart(
    history_df: pd.DataFrame,
    workflow_filter: str = "All",
    models: list[str] | None = None,
    date_range: str = "All time",
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
    df = _apply_date_range(df, date_range)

    if df.empty:
        fig = px.line(title=f"No data in range '{date_range}' for workflow: {workflow_filter}")
        fig.update_layout(xaxis_title="Date", yaxis_title="Average Score (%)")
        return fig

    # Build a display label combining model name and workflow
    if "workflow" in df.columns and workflow_filter == "All":
        df["label"] = df["model"] + " (" + df["workflow"] + ")"
        # Use dash style to distinguish workflows
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
    visible = df["average"].dropna()
    if not visible.empty:
        y_min, y_max = float(visible.min()), float(visible.max())
        span = y_max - y_min
        pad = max(span * 0.1, 2.0)  # at least 2 percentage points of headroom
        y_range = [max(0.0, y_min - pad), min(100.0, y_max + pad)]
    else:
        y_range = None

    fig.update_layout(
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
    date_range: str = "All time",
):
    """Filter the already-loaded trend data in memory (no re-download).

    Used by the workflow dropdown, model multiselect, and date-range
    radio so any control change is instant.
    """
    global TREND_HISTORY_DF, TREND_SUMMARY_DF

    summary_df = TREND_SUMMARY_DF.copy() if not TREND_SUMMARY_DF.empty else pd.DataFrame()
    history_df = TREND_HISTORY_DF

    if workflow_filter != "All" and not summary_df.empty and "Workflow" in summary_df.columns:
        summary_df = summary_df[summary_df["Workflow"] == workflow_filter]

    if models and not summary_df.empty and "Model" in summary_df.columns:
        summary_df = summary_df[summary_df["Model"].isin(models)]

    chart = build_trend_chart(history_df, workflow_filter, models, date_range)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return chart, summary_df, timestamp


def refresh_trend_data(
    workflow_filter: str = "All",
    models: list[str] | None = None,
    date_range: str = "All time",
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

    # Delegate to the lightweight filter function for the current view.
    return filter_trend_data(workflow_filter, models, date_range)


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
        with gr.TabItem("🏅 Single-Agent Benchmark", elem_id="llm-benchmark-tab-table", id=0):
            leaderboard = init_leaderboard(LEADERBOARD_DF)

        with gr.TabItem("🤝 Multi-Agent Benchmark", elem_id="multi-agent-benchmark-tab-table", id=1):
            leaderboard_multi = init_leaderboard(LEADERBOARD_DF_MULTI)

        with gr.TabItem("📈 Trends", elem_id="llm-benchmark-tab-trends", id=2):
            gr.Markdown(
                "### Performance Trends\n"
                "Track how model scores change over time across workflows. "
                "Averages are computed over available evaluation days within each window. "
                "The *(N/M)* annotation shows how many days of data were available.\n\n"
                "When viewing **All** workflows, single-agent results are shown with solid lines "
                "and multi-agent results with dashed lines.",
                elem_classes="markdown-text",
                elem_id="cg-trends-header",
            )
            # Default to single_agent + its top-3 models by latest score.
            _default_workflow = "single_agent"
            _initial_history = (
                TREND_HISTORY_DF[TREND_HISTORY_DF["workflow"] == _default_workflow]
                if (not TREND_HISTORY_DF.empty and "workflow" in TREND_HISTORY_DF.columns)
                else TREND_HISTORY_DF
            )
            _initial_models = _top_n_models(_initial_history, n=3)
            _default_date_range = "All time"
            with gr.Row(elem_id="cg-trend-controls", equal_height=True):
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
                with gr.Column(scale=4, min_width=200, elem_classes="cg-zone cg-zone-view"):
                    date_range_filter = gr.Radio(
                        choices=list(DATE_RANGE_PRESETS.keys()),
                        value=_default_date_range,
                        label="Date range",
                        interactive=True,
                    )
                with gr.Column(scale=3, min_width=180, elem_classes="cg-zone cg-zone-actions"):
                    last_updated_box = gr.Textbox(
                        label="Last updated",
                        value=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                        interactive=False,
                    )
                    refresh_btn = gr.Button(
                        "🔄 Refresh", size="sm", elem_id="cg-refresh-btn"
                    )
            trend_chart = gr.Plot(
                value=build_trend_chart(
                    TREND_HISTORY_DF,
                    workflow_filter=_default_workflow,
                    models=_initial_models,
                    date_range=_default_date_range,
                ),
                elem_id="cg-trend-chart",
            )
            gr.Markdown("### Summary: 1-Day / 3-Day / 7-Day Averages", elem_id="cg-trend-summary-label")
            _initial_summary = TREND_SUMMARY_DF
            if not _initial_summary.empty and "Workflow" in _initial_summary.columns:
                _initial_summary = _initial_summary[_initial_summary["Workflow"] == _default_workflow]
            if not _initial_summary.empty and "Model" in _initial_summary.columns and _initial_models:
                _initial_summary = _initial_summary[_initial_summary["Model"].isin(_initial_models)]
            trend_table = gr.Dataframe(
                value=_initial_summary,
                interactive=False,
                elem_id="cg-trend-summary",
            )

            # Any control change just re-filters in memory (no re-download)
            for ctl in (workflow_filter, model_filter, date_range_filter):
                ctl.change(
                    fn=filter_trend_data,
                    inputs=[workflow_filter, model_filter, date_range_filter],
                    outputs=[trend_chart, trend_table, last_updated_box],
                )

            # Manual refresh on button click
            refresh_btn.click(
                fn=refresh_trend_data,
                inputs=[workflow_filter, model_filter, date_range_filter],
                outputs=[trend_chart, trend_table, last_updated_box],
            )

            # Auto-refresh removed: the BackgroundScheduler restarts the Space
            # every 6 hours, which reloads all data fresh.  The hourly
            # gr.Timer was calling snapshot_download() which blocks a worker
            # and can freeze the UI on resource-constrained Spaces.  Users
            # can still click the Refresh button for on-demand updates.

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
