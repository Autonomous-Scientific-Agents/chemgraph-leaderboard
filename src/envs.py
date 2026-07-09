import os

from huggingface_hub import HfApi

# Info to change for your repository
# ----------------------------------
TOKEN = os.environ.get("HF_TOKEN")  # A read/write token for your org

# Production defaults. Override via environment variables so a dev fork
# can point at its own org / dataset names WITHOUT editing this file.
# Set CG_OWNER, CG_SPACE_NAME, CG_QUEUE_DATASET, CG_RESULTS_DATASET on
# the dev HF Space (Settings -> Variables) and on the NWX cron env to
# keep dev evals out of the production datasets.
OWNER = os.environ.get("CG_OWNER", "Autonomous-Scientific-Agents")
SPACE_NAME = os.environ.get("CG_SPACE_NAME", "chemgraph-leaderboard")
QUEUE_DATASET = os.environ.get("CG_QUEUE_DATASET", "requests")
RESULTS_DATASET = os.environ.get("CG_RESULTS_DATASET", "results")
# ----------------------------------

REPO_ID = f"{OWNER}/{SPACE_NAME}"
QUEUE_REPO = f"{OWNER}/{QUEUE_DATASET}"
RESULTS_REPO = f"{OWNER}/{RESULTS_DATASET}"

# If you setup a cache later, just change HF_HOME
CACHE_PATH = os.getenv("HF_HOME", ".")

# Local caches (top-level — used for HF Hub downloads)
EVAL_REQUESTS_PATH = os.path.join(CACHE_PATH, "eval-queue")
EVAL_RESULTS_PATH = os.path.join(CACHE_PATH, "eval-results")
EVAL_REQUESTS_PATH_BACKEND = os.path.join(CACHE_PATH, "eval-queue-bk")
EVAL_RESULTS_PATH_BACKEND = os.path.join(CACHE_PATH, "eval-results-bk")

# Supported workflow types
WORKFLOWS = ["single_agent", "multi_agent"]


def get_eval_results_path(workflow: str) -> str:
    """Return the local results path for a given workflow."""
    return os.path.join(EVAL_RESULTS_PATH, workflow)


def get_eval_requests_path(workflow: str) -> str:
    """Return the local requests path for a given workflow."""
    return os.path.join(EVAL_REQUESTS_PATH, workflow)


API = HfApi(token=TOKEN)
