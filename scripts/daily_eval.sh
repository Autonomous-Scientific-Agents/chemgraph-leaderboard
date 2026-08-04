#!/bin/bash
# daily_eval.sh — Run ChemGraph evaluations and update the HF leaderboard.
#
# This script is intended to be run via cron. Example crontab entry:
#
#   MAILTO=example@anl.gov
#   0 2 * * * /home/zhye/leaderboard_dev/scripts/daily_eval.sh >> /home/zhye/leaderboard_dev/eval.log 2>&1
#
# Modes:
#   Full pipeline (default):
#     ./scripts/daily_eval.sh
#
#   Convert-only (skip eval, use an existing benchmark file):
#     SKIP_EVAL=true BENCHMARK_FILE=/path/to/benchmark_2026-04-13.json ./scripts/daily_eval.sh
#
#   Convert-only (skip eval, auto-detect latest benchmark in EVAL_OUTPUT_DIR):
#     SKIP_EVAL=true ./scripts/daily_eval.sh
#
# Prerequisites:
#   - the chemgraph-eval-env venv (in the ChemGraph repo) with the chemgraph-eval CLI installed
#   - HF_TOKEN environment variable set (via scripts/dev_env.sh or ~/.bashrc)
#   - config.toml in CHEMGRAPH_DIR with API keys for LLM providers
#
# Configuration — edit these variables or override via environment:

set -euo pipefail

# ---------- Environment Setup ----------
# Source per-machine dev/prod overrides if present. This (untracked) file sets
# CG_OWNER / CG_*_DATASET / HF_TOKEN for dev-vs-prod routing, plus machine
# paths: CHEMGRAPH_DIR, CG_VENV, ARGO_USER. Without it, the
# hardcoded production defaults below apply.
_SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
[ -f "$_SCRIPT_DIR/dev_env.sh" ] && source "$_SCRIPT_DIR/dev_env.sh"

# ---------- Push target (dev vs prod) ----------
# PUSH_TARGET=dev (default) keeps the dev_env.sh routing (InkedWings/*-dev).
# PUSH_TARGET=prod overrides the CG_* routing vars back to the production
# defaults from src/envs.py (Autonomous-Scientific-Agents/{chemgraph-leaderboard,
# results, requests}) while keeping HF_TOKEN + machine paths from dev_env.sh.
PUSH_TARGET="${PUSH_TARGET:-dev}"
if [ "$PUSH_TARGET" = "prod" ]; then
    export CG_OWNER="Autonomous-Scientific-Agents"
    export CG_SPACE_NAME="chemgraph-leaderboard"
    export CG_RESULTS_DATASET="results"
    export CG_QUEUE_DATASET="requests"
    echo "[daily_eval] PUSH_TARGET=prod -> pushing to PRODUCTION (Autonomous-Scientific-Agents)"
fi

# Activate the ChemGraph eval venv (has the chemgraph-eval CLI). Override the
# path via CG_VENV; defaults to a chemgraph-eval-env venv in the ChemGraph repo.
CG_VENV="${CG_VENV:-${CHEMGRAPH_DIR:-/home/zhye/ChemGraph}/chemgraph-eval-env}"
if [ ! -f "$CG_VENV/bin/activate" ]; then
    echo "[daily_eval] ERROR: venv not found at $CG_VENV (set CG_VENV or CHEMGRAPH_DIR)" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$CG_VENV/bin/activate"

# Source bashrc for HF_TOKEN only if dev_env.sh didn't already provide it.
# shellcheck disable=SC1090
if [ -z "${HF_TOKEN:-}" ]; then
    set +u  # bashrc may reference unset variables
    source "$HOME/.bashrc" 2>/dev/null || true
    set -u
fi

export ARGO_USER="${ARGO_USER:-<ARGO_USERNAME>}"

# ---------- Configuration ----------
# Path to the chemgraph-leaderboard repo
LEADERBOARD_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Path to the ChemGraph repo (where config.toml lives)
CHEMGRAPH_DIR="${CHEMGRAPH_DIR:-/home/zhye/ChemGraph}"

# Skip the chemgraph eval step and only convert + push existing results.
SKIP_EVAL="${SKIP_EVAL:-false}"

# Skip pushing to HF Hub (Step 4). Eval, archive and local token/time metrics
# still run; the converter runs locally but does NOT upload anything. Use for a
# full eval with no data pushed:  SKIP_PUSH=true ./scripts/daily_eval.sh
SKIP_PUSH="${SKIP_PUSH:-false}"

# Path to a specific benchmark_*.json to convert. When set, --benchmark-file
# is passed to the converter and --eval-dir is ignored for file discovery.
# Leave empty to auto-detect the latest file in EVAL_OUTPUT_DIR.
BENCHMARK_FILE="${BENCHMARK_FILE:-}"

# Path to the ChemGraph config file with API keys
CHEMGRAPH_CONFIG="${CHEMGRAPH_CONFIG:-$CHEMGRAPH_DIR/config.toml}"

# Models to evaluate (space-separated)
MODELS="${EVAL_MODELS:-argo:claude-opus-4.8 argo:claude-opus-4.7 argo:claude-sonnet-5 argo:claude-sonnet-4.6 argo:claude-sonnet-4.5 argo:claude-haiku-4.5 argo:gpt-5.6-sol argo:gpt-5.6-terra argo:gpt-5.6-luna argo:gpt-5.5 argo:gpt-5.4 argo:gpt-4.1 argo:gpt-4.1-mini argo:gpt-4o argo:o3 argo:o3-mini argo:o4-mini argo:gemini-3.5-flash argo:gemini-2.5-pro argo:gemini-2.5-flash}"

#MODELS="${EVAL_MODELS:-argo:gpt-4o argo:gpt-4.1-mini}"

# Judge type: structured (deterministic) or llm
JUDGE_TYPE="${EVAL_JUDGE_TYPE:-structured}"

# Workflow types (space-separated)
WORKFLOWS="${WORKFLOWS:-single_agent multi_agent}"

# Output directory for eval results (chemgraph-eval default)
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$CHEMGRAPH_DIR/eval_results}"

# Ground-truth dataset passed to chemgraph-eval via --dataset.
DATASET="${DATASET:-}"
DATASET_ARG=""
[ -n "$DATASET" ] && DATASET_ARG="--dataset $DATASET"

# Number of days to keep archived eval runs (0 = keep forever)
EVAL_RETENTION_DAYS="${EVAL_RETENTION_DAYS:-0}"

# Retry settings for core-dump / crash resilience.
# Each model is evaluated in its own chemgraph-eval invocation so that a
# crash (e.g. SIGSEGV from MACE/PyTorch) only kills one model, not the
# entire batch.  Failed models are retried up to MAX_RETRIES times,
# benefiting from --resume (checkpoint-based) on each attempt.
MAX_RETRIES="${MAX_RETRIES:-3}"
RETRY_DELAY="${RETRY_DELAY:-30}"   # seconds between retry attempts

# Leaderboard output directories
RESULTS_OUTDIR="$LEADERBOARD_DIR/hub_results"
REQUESTS_OUTDIR="$LEADERBOARD_DIR/hub_requests"

# Model map
MODEL_MAP="$LEADERBOARD_DIR/dataset/model_map.json"

# Local-only metrics archive (token usage + time breakdown). NEVER pushed to HF.
METRICS_DIR="${METRICS_DIR:-$LEADERBOARD_DIR/eval_metrics}"

# ---------- End Configuration ----------

echo "========================================"
echo "ChemGraph Daily Evaluation"
echo "Date: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Mode: $([ "$SKIP_EVAL" = "true" ] && echo "convert-only" || echo "full pipeline")"
echo "Workflows: $WORKFLOWS"
echo "Push: $([ "$SKIP_PUSH" = "true" ] && echo "DISABLED (local only, nothing uploaded)" || echo "$PUSH_TARGET")"
echo "========================================"

# Track models that failed across all workflows for the final summary.
ALL_FAILED_MODELS=()

# Step 1: Run ChemGraph evaluation (unless SKIP_EVAL=true)
if [ "$SKIP_EVAL" = "true" ]; then
    echo ""
    echo "[Step 1/5] Skipping ChemGraph evaluation (SKIP_EVAL=true)"
    if [ -n "$BENCHMARK_FILE" ]; then
        echo "  Benchmark file: $BENCHMARK_FILE"
    else
        echo "  Will auto-detect latest benchmark in: $EVAL_OUTPUT_DIR"
    fi
else
    # Run eval from the ChemGraph directory so config.toml is found
    pushd "$CHEMGRAPH_DIR" > /dev/null

    # Run each workflow as a separate invocation.
    # Within each workflow, each model is evaluated in its own
    # chemgraph-eval process so that a crash (core dump / SIGSEGV)
    # in one model cannot affect the others.
    for WF in $WORKFLOWS; do
        echo ""
        echo "[Step 1/5] Running ChemGraph evaluation (workflow: $WF)..."
        echo "  Models:      $MODELS"
        echo "  Judge type:  $JUDGE_TYPE"
        echo "  Workflow:    $WF"
        echo "  Max retries: $MAX_RETRIES"
        echo "  Config:      $CHEMGRAPH_CONFIG"
        echo "  Output:      $EVAL_OUTPUT_DIR"

        FAILED_MODELS=()

        # --- Phase 1: Evaluate each model individually with retries ---
        # shellcheck disable=SC2086
        for MODEL in $MODELS; do
            MODEL_OK=false
            for ATTEMPT in $(seq 1 "$MAX_RETRIES"); do
                echo ""
                echo "  [$WF] Evaluating $MODEL (attempt $ATTEMPT/$MAX_RETRIES)..."
                # Run in a subshell to capture the real exit code
                # (including signal-based codes like 139 for SIGSEGV).
                set +e
                # shellcheck disable=SC2086  # $DATASET_ARG is intentionally word-split (empty or "--dataset PATH")
                chemgraph-eval \
                    --models "$MODEL" \
                    --judge-type "$JUDGE_TYPE" \
                    --workflows "$WF" \
                    --output-dir "$EVAL_OUTPUT_DIR" \
                    $DATASET_ARG \
                    --config "$CHEMGRAPH_CONFIG" \
                    --resume \
                    --report json
                EVAL_EXIT=$?
                set -e

                if [ "$EVAL_EXIT" -eq 0 ]; then
                    MODEL_OK=true
                    echo "  [$WF] $MODEL succeeded on attempt $ATTEMPT."
                    break
                else
                    # Exit codes > 128 indicate a signal (e.g. 139 = SIGSEGV).
                    if [ "$EVAL_EXIT" -gt 128 ]; then
                        SIGNAL=$((EVAL_EXIT - 128))
                        echo "  [$WF] WARNING: $MODEL killed by signal $SIGNAL on attempt $ATTEMPT (exit code $EVAL_EXIT)."
                    else
                        echo "  [$WF] WARNING: $MODEL failed on attempt $ATTEMPT (exit code $EVAL_EXIT)."
                    fi
                    if [ "$ATTEMPT" -lt "$MAX_RETRIES" ]; then
                        echo "  [$WF] Retrying in ${RETRY_DELAY}s (--resume will skip completed queries)..."
                        sleep "$RETRY_DELAY"
                    fi
                fi
            done

            if [ "$MODEL_OK" = false ]; then
                echo "  [$WF] ERROR: $MODEL failed after $MAX_RETRIES attempts — skipping."
                FAILED_MODELS+=("$MODEL")
                ALL_FAILED_MODELS+=("$MODEL/$WF")
            fi
        done

        # --- Phase 2: Generate combined report from checkpoints ---
        # Build the list of models that succeeded.
        SUCCESSFUL_MODELS=()
        # shellcheck disable=SC2086
        for MODEL in $MODELS; do
            SKIP=false
            for FM in "${FAILED_MODELS[@]+"${FAILED_MODELS[@]}"}"; do
                if [ "$MODEL" = "$FM" ]; then
                    SKIP=true
                    break
                fi
            done
            if [ "$SKIP" = false ]; then
                SUCCESSFUL_MODELS+=("$MODEL")
            fi
        done

        if [ "${#SUCCESSFUL_MODELS[@]}" -gt 0 ]; then
            echo ""
            echo "  [$WF] Generating combined report for ${#SUCCESSFUL_MODELS[@]} models..."
            # Run with all successful models + --resume.  Every query is
            # already checkpointed so this just loads checkpoints and
            # writes the aggregate benchmark JSON/Markdown.
            # shellcheck disable=SC2086
            if ! chemgraph-eval \
                --models ${SUCCESSFUL_MODELS[*]} \
                --judge-type "$JUDGE_TYPE" \
                --workflows "$WF" \
                --output-dir "$EVAL_OUTPUT_DIR" \
                $DATASET_ARG \
                --config "$CHEMGRAPH_CONFIG" \
                --resume \
                --report all; then
                echo "  [$WF] WARNING: Combined report generation failed."
            fi
        else
            echo ""
            echo "  [$WF] ERROR: All models failed — no report to generate."
        fi

        if [ "${#FAILED_MODELS[@]}" -gt 0 ]; then
            echo ""
            echo "  [$WF] Failed models: ${FAILED_MODELS[*]}"
        fi
    done

    popd > /dev/null
fi

# Step 2: Archive eval results (move eval_results/ -> eval_YYYY-MM-DD/)
echo ""
echo "[Step 2/5] Archiving eval results..."
ARCHIVE_NAME="eval_$(date -u -d 'yesterday' '+%Y-%m-%d')"
ARCHIVE_DIR="$CHEMGRAPH_DIR/$ARCHIVE_NAME"

if [ "$SKIP_EVAL" = "false" ] && [ -d "$EVAL_OUTPUT_DIR" ]; then
    if [ "$EVAL_OUTPUT_DIR" != "$ARCHIVE_DIR" ]; then
        # If archive target already exists (e.g., re-run same day), append timestamp
        if [ -d "$ARCHIVE_DIR" ]; then
            ARCHIVE_NAME="eval_$(date -u '+%Y-%m-%d_%H%M%S')"
            ARCHIVE_DIR="$CHEMGRAPH_DIR/$ARCHIVE_NAME"
        fi
        echo "  Moving $EVAL_OUTPUT_DIR -> $ARCHIVE_DIR"
        mv "$EVAL_OUTPUT_DIR" "$ARCHIVE_DIR"
        EVAL_OUTPUT_DIR="$ARCHIVE_DIR"
    else
        echo "  Output dir is already date-stamped, skipping archive."
    fi
else
    echo "  Skipping archive (SKIP_EVAL=true or output dir not found)."
fi

# Step 3: Extract local token/time metrics. LOCAL ONLY — never pushed to HF.
# Reads the instrumented benchmark_*.json and writes clean metrics_<date>.json
# + metrics_<date>.csv to METRICS_DIR (outside the eval archive, so it is not
# removed by the Step 5 retention cleanup).
echo ""
echo "[Step 3/5] Extracting local token/time metrics (not pushed to HF)..."
METRICS_CMD=(
    python "$LEADERBOARD_DIR/scripts/extract_eval_metrics.py"
    --eval-dir "$EVAL_OUTPUT_DIR"
    --out-dir "$METRICS_DIR"
)
if [ -n "$BENCHMARK_FILE" ]; then
    METRICS_CMD+=(--benchmark-file "$BENCHMARK_FILE")
fi
if "${METRICS_CMD[@]}"; then
    echo "  Metrics written to $METRICS_DIR"
else
    echo "  WARNING: metrics extraction failed (non-fatal; continuing)."
fi

# Step 4: Transform results and (optionally) push to HF Hub
echo ""
if [ "$SKIP_PUSH" = "true" ]; then
    echo "[Step 4/5] Transforming results locally (SKIP_PUSH=true — NOT pushing to HF)..."
else
    echo "[Step 4/5] Transforming results and pushing to HF Hub..."
fi

# Clean staging directories so only this run's files are uploaded.
# The ETL uses date-indexed filenames (results_YYYY-MM-DD.json) and
# per-file additive uploads, so old files on the Hub are never deleted.
echo "  Cleaning staging directories..."
rm -rf "$RESULTS_OUTDIR" "$REQUESTS_OUTDIR"

# Run the converter once per workflow
for WF in $WORKFLOWS; do
    echo ""
    echo "  --- Processing workflow: $WF ---"

    # Build the converter command
    CONVERT_CMD=(
        python "$LEADERBOARD_DIR/scripts/chemgraph_to_leaderboard.py"
        --eval-dir "$EVAL_OUTPUT_DIR"
        --model-map "$MODEL_MAP"
        --results-outdir "$RESULTS_OUTDIR"
        --requests-outdir "$REQUESTS_OUTDIR"
        --workflow "$WF"
    )

    # Only upload when pushing is enabled; otherwise transform locally only.
    if [ "$SKIP_PUSH" != "true" ]; then
        CONVERT_CMD+=(--push-to-hub)
    fi

    # Add --benchmark-file if a specific file was provided
    if [ -n "$BENCHMARK_FILE" ]; then
        CONVERT_CMD+=(--benchmark-file "$BENCHMARK_FILE")
    fi

    "${CONVERT_CMD[@]}"

    PUSH_EXIT=$?
    if [ $PUSH_EXIT -ne 0 ]; then
        echo "ERROR: push to hub failed for workflow '$WF' with exit code $PUSH_EXIT"
        exit $PUSH_EXIT
    fi
done

# Step 5: Clean up old eval runs
echo ""
echo "[Step 5/5] Cleaning up old eval runs..."
if [ "$EVAL_RETENTION_DAYS" -gt 0 ]; then
    OLD_DIRS=$(find "$CHEMGRAPH_DIR" -maxdepth 1 -name "eval_20*" -type d -mtime +"$EVAL_RETENTION_DAYS" 2>/dev/null || true)
    if [ -n "$OLD_DIRS" ]; then
        echo "$OLD_DIRS" | while read -r dir; do
            echo "  Removing old eval run: $dir"
            rm -rf "$dir"
        done
    else
        echo "  No eval runs older than $EVAL_RETENTION_DAYS days."
    fi
else
    echo "  Retention disabled (EVAL_RETENTION_DAYS=0), skipping cleanup."
fi

echo ""
echo "========================================"
if [ "${#ALL_FAILED_MODELS[@]}" -gt 0 ]; then
    echo "Daily evaluation completed with FAILURES."
    echo "Failed model/workflow pairs:"
    for PAIR in "${ALL_FAILED_MODELS[@]}"; do
        echo "  - $PAIR"
    done
else
    echo "Daily evaluation completed successfully."
fi
echo "Date: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "========================================"

# Exit non-zero if any models failed so cron/CI notices.
if [ "${#ALL_FAILED_MODELS[@]}" -gt 0 ]; then
    exit 1
fi
