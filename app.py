import sys

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
from src.display.css_html_js import custom_css
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
)
from src.populate import get_evaluation_queue_df, get_leaderboard_df, get_trend_summary_df, get_trend_history_df
from src.submission.submit import add_new_eval

# --local flag: skip HF Hub downloads and scheduler, use local data only.
LOCAL_MODE = "--local" in sys.argv


def restart_space():
    API.restart_space(repo_id=REPO_ID)


### Space initialisation
if not LOCAL_MODE:
    try:
        print(EVAL_REQUESTS_PATH)
        snapshot_download(
            repo_id=QUEUE_REPO,
            local_dir=EVAL_REQUESTS_PATH,
            repo_type="dataset",
            tqdm_class=None,
            etag_timeout=30,
            token=TOKEN,
        )
    except Exception:
        restart_space()
    try:
        print(EVAL_RESULTS_PATH)
        snapshot_download(
            repo_id=RESULTS_REPO,
            local_dir=EVAL_RESULTS_PATH,
            repo_type="dataset",
            tqdm_class=None,
            etag_timeout=30,
            token=TOKEN,
        )
    except Exception:
        restart_space()
else:
    print("LOCAL MODE: skipping HF Hub downloads, using local eval-results/ and eval-queue/")

LEADERBOARD_DF = get_leaderboard_df(EVAL_RESULTS_PATH, EVAL_REQUESTS_PATH, COLS, BENCHMARK_COLS)
if not LEADERBOARD_DF.empty:
    LEADERBOARD_DF["T"] = range(1, len(LEADERBOARD_DF) + 1)

# Load trend data for the Trends tab
try:
    TREND_SUMMARY_DF = get_trend_summary_df(EVAL_RESULTS_PATH, EVAL_REQUESTS_PATH)
    TREND_HISTORY_DF = get_trend_history_df(EVAL_RESULTS_PATH, EVAL_REQUESTS_PATH)
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


def build_trend_chart(history_df: pd.DataFrame):
    """Build a Plotly line chart showing model scores over time."""
    if history_df.empty:
        fig = px.line(title="No historical data available yet")
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Average Score (%)",
        )
        return fig

    fig = px.line(
        history_df,
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
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Average Score (%)",
        legend_title="Model",
        hovermode="x unified",
        yaxis=dict(range=[0, 105]),
    )
    return fig


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
            label="Select Columns to Display:",
        ),
        search_columns=[AutoEvalColumn.model.name, AutoEvalColumn.license.name],
        hide_columns=[c.name for c in fields(AutoEvalColumn) if c.hidden],
        filter_columns=[
            ColumnFilter(AutoEvalColumn.model_type.name, type="checkboxgroup", label="Model types"),
            ColumnFilter(AutoEvalColumn.precision.name, type="checkboxgroup", label="Precision"),
            ColumnFilter(
                AutoEvalColumn.params.name,
                type="slider",
                min=0.01,
                max=150,
                label="Select the number of parameters (B)",
            ),
            ColumnFilter(
                AutoEvalColumn.still_on_hub.name,
                type="boolean",
                label="Deleted/incomplete",
                default=False,
            ),
        ],
        bool_checkboxgroup_label="Hide models",
        interactive=False,
    )


demo = gr.Blocks(css=custom_css)
with demo:
    gr.HTML(TITLE)
    gr.Markdown(INTRODUCTION_TEXT, elem_classes="markdown-text")

    with gr.Tabs(elem_classes="tab-buttons") as tabs:
        with gr.TabItem("🏅 LLM Benchmark", elem_id="llm-benchmark-tab-table", id=0):
            leaderboard = init_leaderboard(LEADERBOARD_DF)

        with gr.TabItem("📈 Trends", elem_id="llm-benchmark-tab-trends", id=1):
            gr.Markdown(
                "### Performance Trends\n"
                "Track how model scores change over time. "
                "Averages are computed over available evaluation days within each window. "
                "The *(N/M)* annotation shows how many days of data were available.",
                elem_classes="markdown-text",
            )
            trend_chart = gr.Plot(value=build_trend_chart(TREND_HISTORY_DF))
            gr.Markdown("### Summary: 1-Day / 3-Day / 7-Day Averages")
            if not TREND_SUMMARY_DF.empty:
                trend_table = gr.Dataframe(
                    value=TREND_SUMMARY_DF,
                    headers=list(TREND_SUMMARY_DF.columns),
                    interactive=False,
                )
            else:
                gr.Markdown(
                    "*No historical data available yet. "
                    "Trend data will appear once the daily evaluation pipeline "
                    "has run for multiple days.*"
                )

        with gr.TabItem("📝 About", elem_id="llm-benchmark-tab-table", id=2):
            gr.Markdown(LLM_BENCHMARKS_TEXT, elem_classes="markdown-text")

        with gr.TabItem("🚀 Submit here! ", elem_id="llm-benchmark-tab-table", id=3):
            with gr.Column():
                with gr.Row():
                    gr.Markdown(EVALUATION_QUEUE_TEXT, elem_classes="markdown-text")

                with gr.Column():
                    with gr.Accordion(
                        f"✅ Finished Evaluations ({len(finished_eval_queue_df)})",
                        open=False,
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
                    ):
                        with gr.Row():
                            pending_eval_table = gr.components.Dataframe(
                                value=pending_eval_queue_df,
                                headers=EVAL_COLS,
                                datatype=EVAL_TYPES,
                                row_count=5,
                            )
            with gr.Row():
                gr.Markdown("# ✉️✨ Submit your model here!", elem_classes="markdown-text")

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

            submit_button = gr.Button("Submit Eval")
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
                ],
                submission_result,
            )

    with gr.Row():
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
    scheduler.add_job(restart_space, "interval", seconds=1800)
    scheduler.start()

demo.queue(default_concurrency_limit=40).launch()
