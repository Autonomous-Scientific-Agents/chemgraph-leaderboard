# How we run evals

A short map of the eval pipeline: what the entry points are, what they do, which
environment and codebase each one needs, and where the results end up.

## Two codebases

| Repo | Role |
| --- | --- |
| `~/ChemGraph` (`argonne-lcf/ChemGraph`, fork `InkedWings/ChemGraph`) | The eval **engine**. Provides the `chemgraph-eval` CLI (`src/chemgraph/eval/`, entry point in `pyproject.toml`). It runs the agent workflows, judges the answers, and writes `benchmark_*.json`. |
| `~/leaderboard_dev` (this repo) | The **ETL + the leaderboard app**. Converts ChemGraph benchmark JSON into leaderboard result files, pushes them to the HF datasets, and serves the Gradio Space (`app.py`). |

Common to every run:

- **Ground truth**: `~/chemgraph_eval_data/eval_data.json` — 40 queries, each with a
  `category` (`smiles_lookup`, dipole, vibrational, …) used as the leaderboard task columns.
- **Workflows**: `single_agent` and `multi_agent`.
- **Judge**: `structured` (deterministic; `llm` judge exists but is not what we run).
- **API config**: `~/ChemGraph/config.toml` — base URLs + keys for Argo / OpenRouter / ALCF.
- **Eval venv**: `~/ChemGraph/chemgraph-eval-env` (has the `chemgraph-eval` CLI). All three
  entry points `source` it themselves.

Every entry point runs **one model per `chemgraph-eval` process** (so a SIGSEGV from
MACE/PyTorch kills only that model), retries up to `MAX_RETRIES=3` with `--resume`
(checkpointed queries are skipped), then does one final `--report all` pass to write the
combined benchmark JSON/MD.

## The three entry points

### 1. `scripts/daily_eval.sh` — closed models via Argo, full pipeline

The only script that goes all the way to the leaderboard.

- **Models**: ~20 `argo:*` models (Claude / GPT / Gemini) — the `MODELS` list at the top.
- **Env**: sources `scripts/dev_env.sh` (untracked, gitignored — sets `HF_TOKEN`,
  `CHEMGRAPH_DIR`, `CG_VENV`, `ARGO_USER`, `DATASET`, and the dev-vs-prod HF routing),
  then activates `chemgraph-eval-env`.
- **Steps**:
  1. Run `chemgraph-eval` per model per workflow → `~/ChemGraph/eval_results/`
  2. Archive that dir → `~/ChemGraph/eval_<YYYY-MM-DD>/`
  3. `extract_eval_metrics.py` → token/time metrics, **local only**
  4. `chemgraph_to_leaderboard.py` → leaderboard JSON + push to HF
  5. Retention cleanup (disabled by default, `EVAL_RETENTION_DAYS=0`)
- **Useful flags**:
  - `SKIP_EVAL=true` — convert/push an existing benchmark only (`BENCHMARK_FILE=…` to pick one)
  - `SKIP_PUSH=true` — full eval, convert locally, upload nothing
  - `PUSH_TARGET=dev|prod` — dev pushes to `InkedWings/chemgraph-*-dev`, prod to
    `Autonomous-Scientific-Agents/*`. **`dev_env.sh` currently sets `PUSH_TARGET=prod`**,
    so a bare run pushes to production — pass `PUSH_TARGET=dev` explicitly for a dev run.
- **Cron**: the crontab entry exists but is commented out, so today this is run by hand.

### 2. `scripts/alcf_Eval.sh` — open-weight models on the ALCF inference endpoints

- **Models**: 14 `alcf:*` models (gemma / gpt-oss / llama / nemotron / inkling). Names must
  match `supported_alcf_models` in ChemGraph, prefix included. Metis-hosted models are
  excluded on purpose — that cluster has no tool calling.
- **Env**: additionally needs a **Globus token**. `inference_auth_token.py` is run with a
  *separate* interpreter (`~/miniforge3/envs/globus_env/bin/python`, has `globus_sdk`) to
  refresh/mint `ALCF_ACCESS_TOKEN`; the token is re-checked before every model so long runs
  don't die mid-way. Non-TTY (cron) disables the interactive browser login.
- **Does not** convert or push. Eval + report only.

### 3. `scripts/openrouter_Eval.sh` — the same open-weight models via OpenRouter

- Deliberately the same shape as `alcf_Eval.sh` so the two runs are comparable: 12
  `openrouter:*` slugs hand-mapped to their ALCF counterparts (the naming is not
  mechanical — see the comment block in the script). Three ALCF models have no OpenRouter
  equivalent and are listed as excluded.
- **Env**: `OPENROUTER_API_KEY` + the same venv/config/dataset. No Globus token.
- **Does not** convert or push. Eval + report only.

## Where results land

**Raw benchmark output** (written by `chemgraph-eval`, one dir per entry point):

| Run | Output dir |
| --- | --- |
| daily/Argo | `~/ChemGraph/eval_results/` → archived to `~/ChemGraph/eval_<date>/` |
| ALCF | `~/alcf/eval_<YYYYMM>/` |
| OpenRouter | `~/ChemGraph/eval_openrouter_alcf_models/` |

Each dir holds `benchmark_<date>_<time>.json` (+ `.md` for the combined report),
`<model>_<workflow>_detail.json` per-query judge detail, `checkpoints/`, and `logs/`
(the `state_thread_*.json` agent transcripts).

**Derived, in this repo** (all gitignored):

- `eval_metrics/metrics_<date>.{json,csv}` — token + time breakdown, **never pushed to HF**.
- `hub_results/<workflow>/<org>__<model>/results_<date>.json` — staging for the results dataset.
- `hub_requests/<workflow>/<org>__<model>.request.json` — staging for the requests dataset.
- `hub_logs/<workflow>/<org>__<model>/{detail,state_thread_N}.json` — staging for the logs dataset.

Step 4 wipes `hub_results/` and `hub_requests/` at the start of every run, so they only ever
hold the current run.

**On HF Hub** (ids resolved in `src/envs.py` from the `CG_*` env vars):

| | prod | dev |
| --- | --- | --- |
| results | `Autonomous-Scientific-Agents/results` | `InkedWings/chemgraph-results-dev` |
| requests | `Autonomous-Scientific-Agents/requests` | `InkedWings/chemgraph-requests-dev` |
| logs | `Autonomous-Scientific-Agents/logs` | — |
| Space | `Autonomous-Scientific-Agents/chemgraph-leaderboard` | `InkedWings/chemgraph-leaderboard-test` |

Uploads are additive and date-indexed (`results_<date>.json`), so nothing on the Hub is
overwritten. The Space downloads the results dataset into `eval-results/` at startup.

## Helper scripts

| Script | What it does |
| --- | --- |
| `chemgraph_to_leaderboard.py` | benchmark JSON → per-model/per-category leaderboard results + request files; `--push-to-hub` to upload. Called by `daily_eval.sh`; run by hand for ALCF/OpenRouter results. |
| `extract_eval_metrics.py` | Pulls the token/time instrumentation out of the benchmark JSONs into `metrics_<date>.{json,csv}`. |
| `upload_eval_logs.py` | Mirrors per-query `detail.json` + `state_thread_<N>.json` into `hub_logs/` and uploads them to the logs dataset (what the Full table's log drawer fetches). |
| `alcf_token_metrics.py` | Appends ALCF open-weight token rows to a closed-model metrics CSV (the loader reads a single newest CSV, so they must share one file). |
| `inference_auth_token.py` | Globus auth helper from the ALCF docs; used only by `alcf_Eval.sh`. |
| `check_trace_model.py`, `preview_log_panel.py` | Local debugging aids. |

Outside this repo, `~/alcf/` has its own staging + push helpers for the open-weight bundle
(`aggregate_oss_results.py`, `push_to_hf.sh`, staged in `~/alcf/hf_oss_upload/`).

## Gotchas

- ALCF and OpenRouter runs stop at the benchmark JSON. Getting them onto the leaderboard is
  a manual `chemgraph_to_leaderboard.py --eval-dir …` (or the `~/alcf/push_to_hf.sh` route).
- `USE_RESUME=true` is the default everywhere: a re-run reuses checkpoints in the output dir.
  Set `USE_RESUME=false` to actually re-evaluate from scratch.
- `openrouter_Eval.sh` currently has an OpenRouter API key hardcoded as the default value.
  It is untracked today — clear it (or move it to `dev_env.sh`) before committing the file.
