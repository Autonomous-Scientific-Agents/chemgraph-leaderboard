#!/bin/bash
# openrouter_Eval.sh — Run ChemGraph evaluations against OpenRouter, using the
# OpenRouter equivalents of the models we have already evaluated on ALCF.
#
# Mirrors alcf_Eval.sh: one model per process so a crash or a dead upstream only
# kills that model, then a combined report for whatever succeeded. Keeping the
# two harnesses the same shape is what makes the two runs comparable.
#
# Usage:
#   ./scripts/openrouter_Eval.sh
#   MODELS="openrouter:openai/gpt-oss-20b" ./scripts/openrouter_Eval.sh
#   WORKFLOWS="single_agent" ./scripts/openrouter_Eval.sh
#   SKIP_COMBINED_REPORT=true ./scripts/openrouter_Eval.sh

set -euo pipefail

# ---------- Configuration ----------

# OpenRouter API key. Leave blank here and export it in your shell, or fill it
# in. Get one at https://openrouter.ai/keys
OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}"

# Eval venv with the chemgraph-eval CLI.
CG_VENV="${CG_VENV:-$HOME/ChemGraph/chemgraph-eval-env}"

# ChemGraph config.
CHEMGRAPH_CONFIG="${CHEMGRAPH_CONFIG:-$HOME/ChemGraph/config.toml}"

# Ground-truth dataset -- the same one the ALCF run used.
DATASET="${DATASET:-$HOME/chemgraph_eval_data/eval_data.json}"

# Where benchmark JSON/MD + checkpoints land. Deliberately separate from the
# ALCF output dir so the two runs do not share checkpoints.
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/ChemGraph/eval_openrouter_alcf_models}"

# Judge type: structured (deterministic) or llm.
JUDGE_TYPE="${JUDGE_TYPE:-structured}"

# Workflows to evaluate (space-separated).
WORKFLOWS="${WORKFLOWS:-single_agent multi_agent}"

# OpenRouter slugs for the models we have ALCF numbers for. The comment on each
# line is the ALCF model it corresponds to -- the naming is not mechanical
# (OpenRouter lowercases, drops the "Meta-" prefix and the parameter-count
# suffix on Llama 4, but *adds* one on Nemotron), so do not derive these.
#
# Dispatch is by the "openrouter:" prefix, not list membership, so these do not
# need to be in supported_openrouter_models to work.
#
# No OpenRouter equivalent, deliberately absent:
#   google/gemma-4-E4B-it                  only gemma-3n-e4b-it exists (different
#                                          generation, no tool calling)
#   meta-llama/Meta-Llama-3.1-405B-Instruct  only NousResearch finetunes
#   mgoin/Nemotron-4-340B-Instruct-hf      delisted from ALCF as well
MODELS="${MODELS:-\
openrouter:openai/gpt-oss-20b \
openrouter:openai/gpt-oss-120b \
openrouter:google/gemma-3-27b-it \
openrouter:google/gemma-4-26b-a4b-it \
openrouter:google/gemma-4-31b-it \
openrouter:meta-llama/llama-3.1-8b-instruct \
openrouter:meta-llama/llama-3.1-70b-instruct \
openrouter:meta-llama/llama-3.3-70b-instruct \
openrouter:meta-llama/llama-4-scout \
openrouter:nvidia/nemotron-3-super-120b-a12b \
openrouter:nvidia/nemotron-3-ultra-550b-a55b \
openrouter:thinkingmachines/inkling}"
#   ^ gpt-oss-20b            <- openai/gpt-oss-20b
#     gpt-oss-120b           <- openai/gpt-oss-120b
#     gemma-3-27b-it         <- google/gemma-3-27b-it
#     gemma-4-26b-a4b-it     <- google/gemma-4-26B-A4B-it
#     gemma-4-31b-it         <- google/gemma-4-31B-it
#     llama-3.1-8b-instruct  <- meta-llama/Meta-Llama-3.1-8B-Instruct
#     llama-3.1-70b-instruct <- meta-llama/Meta-Llama-3.1-70B-Instruct
#     llama-3.3-70b-instruct <- meta-llama/Llama-3.3-70B-Instruct
#     llama-4-scout          <- meta-llama/Llama-4-Scout-17B-16E-Instruct
#     nemotron-3-super-120b-a12b   <- nvidia/nemotron-3-super-120b
#     nemotron-3-ultra-550b-a55b   <- nemotron-3-ultra          (ALCF Minerva)
#     thinkingmachines/inkling     <- inkling-bf16              (ALCF Minerva)

# Per-model retry policy. --resume means a retry skips completed queries.
MAX_RETRIES="${MAX_RETRIES:-3}"
RETRY_DELAY="${RETRY_DELAY:-30}"

# Pass --resume to chemgraph-eval (reuses checkpoints in OUTPUT_DIR).
USE_RESUME="${USE_RESUME:-true}"

# Skip the final combined --report all pass.
SKIP_COMBINED_REPORT="${SKIP_COMBINED_REPORT:-false}"

# ---------- End Configuration ----------

# ---------- Preflight ----------

[ -n "$OPENROUTER_API_KEY" ] || {
    echo "[openrouter_eval] ERROR: OPENROUTER_API_KEY is empty." >&2
    echo "  Set it in this script, or: export OPENROUTER_API_KEY='sk-or-v1-...'" >&2
    echo "  Get a key at https://openrouter.ai/keys" >&2
    exit 1
}
[ -f "$CG_VENV/bin/activate" ] || {
    echo "[openrouter_eval] ERROR: eval venv not found at $CG_VENV (set CG_VENV)" >&2
    exit 1
}
[ -f "$CHEMGRAPH_CONFIG" ] || {
    echo "[openrouter_eval] ERROR: config not found at $CHEMGRAPH_CONFIG (set CHEMGRAPH_CONFIG)" >&2
    exit 1
}
[ -f "$DATASET" ] || {
    echo "[openrouter_eval] ERROR: dataset not found at $DATASET (set DATASET)" >&2
    exit 1
}

export OPENROUTER_API_KEY

echo "========================================"
echo "ChemGraph OpenRouter Evaluation"
echo "Date:      $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Workflows: $WORKFLOWS"
echo "Judge:     $JUDGE_TYPE"
echo "Dataset:   $DATASET"
echo "Output:    $OUTPUT_DIR"
echo "Models:    $(echo $MODELS | wc -w)"
echo "========================================"

# shellcheck disable=SC1091
source "$CG_VENV/bin/activate"

mkdir -p "$OUTPUT_DIR"

RESUME_ARG=()
[ "$USE_RESUME" = "true" ] && RESUME_ARG=(--resume)

ALL_FAILED=()

echo ""
echo "[Step 1/2] Running chemgraph-eval..."

for WF in $WORKFLOWS; do
    echo ""
    echo "--- Workflow: $WF ---"

    SUCCESSFUL=()
    FAILED=()

    # shellcheck disable=SC2086  # MODELS is intentionally word-split
    for MODEL in $MODELS; do
        MODEL_OK=false
        for ATTEMPT in $(seq 1 "$MAX_RETRIES"); do
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

    if [ "$SKIP_COMBINED_REPORT" = "true" ]; then
        echo ""
        echo "  [$WF] Skipping combined report (SKIP_COMBINED_REPORT=true)."
    elif [ "${#SUCCESSFUL[@]}" -gt 0 ]; then
        echo ""
        echo "  [$WF] Generating combined report for ${#SUCCESSFUL[@]} model(s)..."
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

echo ""
echo "[Step 2/2] Summary"
echo "========================================"
if [ "${#ALL_FAILED[@]}" -gt 0 ]; then
    echo "OpenRouter evaluation completed with FAILURES."
    echo "Failed model/workflow pairs:"
    for PAIR in "${ALL_FAILED[@]}"; do
        echo "  - $PAIR"
    done
else
    echo "OpenRouter evaluation completed successfully."
fi
echo "Results: $OUTPUT_DIR"
echo "Date:    $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "========================================"

[ "${#ALL_FAILED[@]}" -eq 0 ]
