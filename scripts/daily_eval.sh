#!/bin/bash
# daily_eval.sh — Run ChemGraph evaluations and update the HF leaderboard.
#
# This script is intended to be run via cron. Example crontab entry:
#
#   0 2 * * * /path/to/chemgraph-leaderboard/scripts/daily_eval.sh >> /path/to/eval.log 2>&1
#
# Prerequisites:
#   - chemgraph package installed (pip install chemgraph)
#   - HF_TOKEN environment variable set (read/write token for HF Hub)
#   - config.toml with API keys for LLM providers
#
# Configuration — edit these variables to match your setup:

set -euo pipefail

# ---------- Configuration ----------
# Path to the chemgraph-leaderboard repo
LEADERBOARD_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Path to the ChemGraph config file with API keys
CHEMGRAPH_CONFIG="${CHEMGRAPH_CONFIG:-$HOME/.chemgraph/config.toml}"

# Models to evaluate (space-separated)
MODELS="${EVAL_MODELS:-gpt4o gpt52 claudeopus46 gpt41}"

# Judge model for scoring
JUDGE_MODEL="${EVAL_JUDGE_MODEL:-claudeopus46}"

# Workflow type
WORKFLOW="single_agent"

# Output directory for eval results (timestamped subdirectory created automatically)
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$LEADERBOARD_DIR/eval_runs}"

# Leaderboard output directories
RESULTS_OUTDIR="$LEADERBOARD_DIR/hub_results"
REQUESTS_OUTDIR="$LEADERBOARD_DIR/hub_requests"

# Model map
MODEL_MAP="$LEADERBOARD_DIR/dataset/model_map.json"

# ---------- End Configuration ----------

echo "========================================"
echo "ChemGraph Daily Evaluation"
echo "Date: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "========================================"

# Create eval output directory
mkdir -p "$EVAL_OUTPUT_DIR"

# Step 1: Run ChemGraph evaluation
echo ""
echo "[Step 1/2] Running ChemGraph evaluation..."
echo "  Models:      $MODELS"
echo "  Judge:       $JUDGE_MODEL"
echo "  Workflow:    $WORKFLOW"
echo "  Config:      $CHEMGRAPH_CONFIG"
echo "  Output:      $EVAL_OUTPUT_DIR"

# shellcheck disable=SC2086
#chemgraph eval \
#    --models $MODELS \
#    --judge-model "$JUDGE_MODEL" \
#    --workflows "$WORKFLOW" \
#    --output-dir "$EVAL_OUTPUT_DIR" \
#    --config "$CHEMGRAPH_CONFIG" \
#    --report all

#EVAL_EXIT=$?
#if [ $EVAL_EXIT -ne 0 ]; then
#    echo "ERROR: chemgraph eval failed with exit code $EVAL_EXIT"
#    exit $EVAL_EXIT
#fi

# Step 2: Transform results and push to HF Hub
echo ""
echo "[Step 2/2] Transforming results and pushing to HF Hub..."

# Clean staging directories so only this run's files are uploaded.
# The ETL now uses date-indexed filenames (results_YYYY-MM-DD.json) and
# per-file additive uploads, so old files on the Hub are never deleted.
echo "  Cleaning staging directories..."
rm -rf "$RESULTS_OUTDIR" "$REQUESTS_OUTDIR"

python "$LEADERBOARD_DIR/scripts/chemgraph_to_leaderboard.py" \
    --eval-dir "$EVAL_OUTPUT_DIR" \
    --model-map "$MODEL_MAP" \
    --results-outdir "$RESULTS_OUTDIR" \
    --requests-outdir "$REQUESTS_OUTDIR" \
    --workflow "$WORKFLOW" \
    --push-to-hub

PUSH_EXIT=$?
if [ $PUSH_EXIT -ne 0 ]; then
    echo "ERROR: push to hub failed with exit code $PUSH_EXIT"
    exit $PUSH_EXIT
fi

echo ""
echo "========================================"
echo "Daily evaluation completed successfully."
echo "Date: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "========================================"
