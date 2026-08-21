#!/bin/bash
# alcf_Eval.sh — Run ChemGraph evaluations against the ALCF inference endpoints.
#
# Handles the two things that have to happen by hand today:
#   1. Validate the Globus/ALCF access token, refreshing (or re-authenticating)
#      when it is expired or about to expire, and export ALCF_ACCESS_TOKEN.
#   2. Run chemgraph-eval over the ALCF model list, one model per process so a
#      crash or a dead endpoint only kills that model, then emit a combined
#      report for whatever succeeded.
#
# Usage:
#   ./scripts/alcf_Eval.sh                          # all models, multi_agent
#   MODELS="alcf:openai/gpt-oss-20b" ./scripts/alcf_Eval.sh
#   WORKFLOWS="single_agent multi_agent" ./scripts/alcf_Eval.sh
#   SKIP_COMBINED_REPORT=true ./scripts/alcf_Eval.sh
#
# The token step needs a Python with globus_sdk installed (the globus_env conda
# env by default) — it is deliberately kept separate from the eval venv.

set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---------- Configuration ----------

# Python interpreter that has globus_sdk installed (used only for the token).
GLOBUS_PY="${GLOBUS_PY:-$HOME/miniforge3/envs/globus_env/bin/python}"

# Globus token helper (copied from the ALCF inference-endpoints docs).
AUTH_SCRIPT="${AUTH_SCRIPT:-$_SCRIPT_DIR/inference_auth_token.py}"

# Refresh the token when fewer than this many seconds of life remain.
# Also re-checked before every model so long runs never die mid-way.
TOKEN_MIN_SECONDS="${TOKEN_MIN_SECONDS:-1800}"

# Allow the interactive Globus login flow (opens a browser) when the refresh
# token is dead. Forced off when stdin is not a TTY so cron never hangs.
AUTO_AUTH="${AUTO_AUTH:-true}"
[ -t 0 ] || AUTO_AUTH=false

# Eval venv with the chemgraph-eval CLI.
CG_VENV="${CG_VENV:-$HOME/ChemGraph/chemgraph-eval-env}"

# ChemGraph config with the ALCF base_url / API settings.
CHEMGRAPH_CONFIG="${CHEMGRAPH_CONFIG:-$HOME/ChemGraph/config.toml}"

# Ground-truth dataset.
DATASET="${DATASET:-$HOME/chemgraph_eval_data/eval_data.json}"

# Where benchmark JSON/MD + checkpoints land.
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/alcf/eval_$(date -u '+%Y%m')}"

# Judge type: structured (deterministic) or llm.
JUDGE_TYPE="${JUDGE_TYPE:-structured}"

# Workflows to evaluate (space-separated).
WORKFLOWS="${WORKFLOWS:-single_agent multi_agent}"

# ALCF models to evaluate (space-separated). Names must match
# supported_alcf_models in ChemGraph's src/chemgraph/models/supported_models.py,
# including the "alcf:" prefix -- ChemGraph strips it before the request goes
# out and uses it to pick the cluster endpoint. Bare names are not recognized.
#
# Metis-hosted models are deliberately absent: that cluster does not offer tool
# calling, so ChemGraph's workflows cannot run there.
MODELS="${MODELS:-\
alcf:google/gemma-4-31B-it \
alcf:google/gemma-4-E4B-it \
alcf:openai/gpt-oss-120b \
alcf:openai/gpt-oss-20b \
alcf:meta-llama/Llama-4-Scout-17B-16E-Instruct \
alcf:meta-llama/Llama-3.3-70B-Instruct \
alcf:nvidia/nemotron-3-super-120b \
alcf:nemotron-3-ultra \
alcf:inkling-bf16}"

# Per-model retry policy. --resume means a retry skips completed queries.
MAX_RETRIES="${MAX_RETRIES:-3}"
RETRY_DELAY="${RETRY_DELAY:-30}"

# Pass --resume to chemgraph-eval (reuses checkpoints in OUTPUT_DIR).
# USE_RESUME=false makes chemgraph-eval clear the checkpoints and re-evaluate
# every query from scratch.
USE_RESUME="${USE_RESUME:-true}"

# Skip the final combined --report all pass.
SKIP_COMBINED_REPORT="${SKIP_COMBINED_REPORT:-false}"

# ---------- End Configuration ----------

# ---------- Token handling ----------

# Seconds of remaining token life, or empty if it cannot be determined
# (no token stored yet, expired refresh token, globus error).
token_seconds_left() {
    local out
    if ! out=$("$GLOBUS_PY" "$AUTH_SCRIPT" get_time_until_token_expiration \
                   --units seconds 2>/dev/null); then
        return 1
    fi
    # Guard against the helper printing anything other than a number
    # (e.g. a traceback, or the "units must be ..." error string).
    case "$out" in
        ""|*[!0-9.-]*) return 1 ;;
    esac
    printf '%s' "$out"
}

# Fetch (auto-refreshing) the access token into ALCF_ACCESS_TOKEN.
fetch_token() {
    local tok
    if ! tok=$("$GLOBUS_PY" "$AUTH_SCRIPT" get_access_token 2>/dev/null); then
        return 1
    fi
    [ -n "$tok" ] || return 1
    export ALCF_ACCESS_TOKEN="$tok"
}

# Ensure ALCF_ACCESS_TOKEN is set and good for at least TOKEN_MIN_SECONDS.
# $1: "quiet" to suppress the per-model chatter.
ensure_token() {
    local quiet="${1:-}"
    local left needs_refresh=false

    if left=$(token_seconds_left); then
        # awk, because bash cannot compare the float the helper prints.
        if awk -v l="$left" -v m="$TOKEN_MIN_SECONDS" 'BEGIN{exit !(l < m)}'; then
            needs_refresh=true
            if [ "$quiet" != "quiet" ]; then
                if awk -v l="$left" 'BEGIN{exit !(l <= 0)}'; then
                    echo "  Token is EXPIRED (${left}s) — refreshing..."
                else
                    echo "  Token expires in ${left}s (< ${TOKEN_MIN_SECONDS}s) — refreshing..."
                fi
            fi
        elif [ "$quiet" != "quiet" ]; then
            printf '  Token valid for %.0f more minutes.\n' "$(awk -v l="$left" 'BEGIN{print l/60}')"
        fi
    else
        needs_refresh=true
        [ "$quiet" != "quiet" ] && echo "  No usable token found — authenticating..."
    fi

    # Always fetch: get_access_token() refreshes via the refresh token when the
    # access token is stale, and is a cheap no-op when it is still valid.
    if ! fetch_token; then
        if [ "$AUTO_AUTH" != "true" ]; then
            echo "[alcf_eval] ERROR: could not obtain an ALCF access token and the" >&2
            echo "  interactive login flow is disabled (not a TTY, or AUTO_AUTH=false)." >&2
            echo "  Run this by hand, then retry:" >&2
            echo "    $GLOBUS_PY $AUTH_SCRIPT authenticate" >&2
            exit 1
        fi
        echo "  Refresh failed — starting interactive Globus login (a browser will open)..."
        "$GLOBUS_PY" "$AUTH_SCRIPT" authenticate
        if ! fetch_token; then
            echo "[alcf_eval] ERROR: still no access token after authenticating." >&2
            exit 1
        fi
    fi

    if [ "$needs_refresh" = true ] && [ "$quiet" != "quiet" ]; then
        if left=$(token_seconds_left); then
            printf '  Token refreshed — valid for %.0f more minutes.\n' \
                "$(awk -v l="$left" 'BEGIN{print l/60}')"
        else
            echo "  Token refreshed."
        fi
    fi
}

# ---------- Preflight ----------

[ -x "$GLOBUS_PY" ] || {
    echo "[alcf_eval] ERROR: globus python not found at $GLOBUS_PY (set GLOBUS_PY)" >&2
    exit 1
}
[ -f "$AUTH_SCRIPT" ] || {
    echo "[alcf_eval] ERROR: auth helper not found at $AUTH_SCRIPT (set AUTH_SCRIPT)" >&2
    exit 1
}
[ -f "$CG_VENV/bin/activate" ] || {
    echo "[alcf_eval] ERROR: eval venv not found at $CG_VENV (set CG_VENV)" >&2
    exit 1
}
[ -f "$CHEMGRAPH_CONFIG" ] || {
    echo "[alcf_eval] ERROR: config not found at $CHEMGRAPH_CONFIG (set CHEMGRAPH_CONFIG)" >&2
    exit 1
}
[ -f "$DATASET" ] || {
    echo "[alcf_eval] ERROR: dataset not found at $DATASET (set DATASET)" >&2
    exit 1
}

echo "========================================"
echo "ChemGraph ALCF Evaluation"
echo "Date:      $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Workflows: $WORKFLOWS"
echo "Judge:     $JUDGE_TYPE"
echo "Dataset:   $DATASET"
echo "Output:    $OUTPUT_DIR"
echo "========================================"

# Step 1: token
echo ""
echo "[Step 1/3] Checking ALCF access token..."
ensure_token

# Step 2: eval
# shellcheck disable=SC1091
source "$CG_VENV/bin/activate"

mkdir -p "$OUTPUT_DIR"

RESUME_ARG=()
[ "$USE_RESUME" = "true" ] && RESUME_ARG=(--resume)

ALL_FAILED=()

echo ""
echo "[Step 2/3] Running chemgraph-eval..."

for WF in $WORKFLOWS; do
    echo ""
    echo "--- Workflow: $WF ---"

    SUCCESSFUL=()
    FAILED=()

    # shellcheck disable=SC2086  # MODELS is intentionally word-split
    for MODEL in $MODELS; do
        MODEL_OK=false
        for ATTEMPT in $(seq 1 "$MAX_RETRIES"); do
            # Long runs outlive a single token; top it up before each attempt.
            ensure_token quiet

            echo ""
            echo "  [$WF] $MODEL (attempt $ATTEMPT/$MAX_RETRIES)..."
            set +e
            chemgraph-eval \
                --models "$MODEL" \
                --judge-type "$JUDGE_TYPE" \
                --workflows "$WF" \
                --config "$CHEMGRAPH_CONFIG" \
                --dataset "$DATASET" \
                --output-dir "$OUTPUT_DIR" \
                "${RESUME_ARG[@]}" \
                --report json
            EVAL_EXIT=$?
            set -e

            if [ "$EVAL_EXIT" -eq 0 ]; then
                MODEL_OK=true
                echo "  [$WF] $MODEL OK (attempt $ATTEMPT)."
                break
            fi

            # Exit codes > 128 mean a signal (139 = SIGSEGV, common with MACE).
            if [ "$EVAL_EXIT" -gt 128 ]; then
                echo "  [$WF] WARNING: $MODEL killed by signal $((EVAL_EXIT - 128)) (exit $EVAL_EXIT)."
            else
                echo "  [$WF] WARNING: $MODEL failed (exit $EVAL_EXIT)."
            fi
            if [ "$ATTEMPT" -lt "$MAX_RETRIES" ]; then
                echo "  [$WF] Retrying in ${RETRY_DELAY}s..."
                sleep "$RETRY_DELAY"
            fi
        done

        if [ "$MODEL_OK" = true ]; then
            SUCCESSFUL+=("$MODEL")
        else
            echo "  [$WF] ERROR: $MODEL failed after $MAX_RETRIES attempts — skipping."
            FAILED+=("$MODEL")
            ALL_FAILED+=("$MODEL/$WF")
        fi
    done

    # Combined report across every model that made it through. Each query is
    # already checkpointed, so this just aggregates.
    if [ "$SKIP_COMBINED_REPORT" = "true" ]; then
        echo ""
        echo "  [$WF] Skipping combined report (SKIP_COMBINED_REPORT=true)."
    elif [ "${#SUCCESSFUL[@]}" -gt 0 ]; then
        echo ""
        echo "  [$WF] Generating combined report for ${#SUCCESSFUL[@]} model(s)..."
        ensure_token quiet
        if ! chemgraph-eval \
            --models "${SUCCESSFUL[@]}" \
            --judge-type "$JUDGE_TYPE" \
            --workflows "$WF" \
            --config "$CHEMGRAPH_CONFIG" \
            --dataset "$DATASET" \
            --output-dir "$OUTPUT_DIR" \
            --resume \
            --report all; then
            echo "  [$WF] WARNING: combined report generation failed."
        fi
    else
        echo ""
        echo "  [$WF] ERROR: all models failed — no report to generate."
    fi

    if [ "${#FAILED[@]}" -gt 0 ]; then
        echo ""
        echo "  [$WF] Failed models: ${FAILED[*]}"
    fi
done

# Step 3: summary
echo ""
echo "[Step 3/3] Summary"
echo "========================================"
if [ "${#ALL_FAILED[@]}" -gt 0 ]; then
    echo "ALCF evaluation completed with FAILURES."
    echo "Failed model/workflow pairs:"
    for PAIR in "${ALL_FAILED[@]}"; do
        echo "  - $PAIR"
    done
else
    echo "ALCF evaluation completed successfully."
fi
echo "Results: $OUTPUT_DIR"
echo "Date:    $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "========================================"

[ "${#ALL_FAILED[@]}" -eq 0 ]
