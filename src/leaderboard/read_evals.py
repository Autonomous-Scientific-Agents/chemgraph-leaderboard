import glob
import json
import math
import os
from dataclasses import dataclass

import dateutil
import numpy as np

from src.display.formatting import make_clickable_model
from src.display.utils import AutoEvalColumn, ModelType, Tasks, Precision, WeightType
from src.submission.check_validity import is_model_on_hub


@dataclass
class EvalResult:
    """Represents one full evaluation. Built from a combination of the result and request file for a given run."""

    eval_name: str  # org_model_precision (uid)
    full_model: str  # org/model (path on hub)
    org: str
    model: str
    revision: str  # commit hash, "" if main
    results: dict
    precision: Precision = Precision.Unknown
    model_type: ModelType = ModelType.Unknown  # Pretrained, fine tuned, ...
    weight_type: WeightType = WeightType.Original  # Original or Adapter
    architecture: str = "Unknown"
    license: str = "?"
    likes: int = 0
    num_params: int = 0
    date: str = ""  # submission date of request file
    still_on_hub: bool = False

    @classmethod
    def init_from_json_file(cls, json_filepath):
        """Inits the result from the specific model result file"""
        with open(json_filepath) as fp:
            data = json.load(fp)

        config = data.get("config")

        # Precision
        precision = Precision.from_str(config.get("model_dtype"))

        # Get model and org
        org_and_model = config.get("model_name", config.get("model_args", None))
        org_and_model = org_and_model.split("/", 1)

        if len(org_and_model) == 1:
            org = None
            model = org_and_model[0]
            result_key = f"{model}_{precision.value.name}"
        else:
            org = org_and_model[0]
            model = org_and_model[1]
            result_key = f"{org}_{model}_{precision.value.name}"
        full_model = "/".join(org_and_model)

        # Check if model is on HF Hub. API-only models (OpenAI, Anthropic,
        # Google, etc.) won't have HF Hub pages, so we skip the check for
        # known API providers and default to still_on_hub=True.
        API_ONLY_ORGS = {"openai", "anthropic", "google", "meta", "cohere", "mistral"}
        still_on_hub = True
        architecture = "?"
        model_org = (org or "").lower()
        if model_org not in API_ONLY_ORGS:
            try:
                on_hub, _, model_config = is_model_on_hub(
                    full_model,
                    config.get("model_sha", "main"),
                    trust_remote_code=True,
                    test_tokenizer=False,
                )
                still_on_hub = on_hub
                if model_config is not None:
                    architectures = getattr(model_config, "architectures", None)
                    if architectures:
                        architecture = ";".join(architectures)
            except Exception:
                pass

        # Extract results available in this file (some results are split in several files)
        results = {}
        for task in Tasks:
            task = task.value

            # We average all scores of a given metric (not all metrics are present in all files)
            accs = np.array([v.get(task.metric, None) for k, v in data["results"].items() if task.benchmark == k])
            if accs.size == 0 or any([acc is None for acc in accs]):
                continue

            mean_acc = np.mean(accs) * 100.0
            results[task.benchmark] = mean_acc

        return cls(
            eval_name=result_key,
            full_model=full_model,
            org=org,
            model=model,
            results=results,
            precision=precision,
            revision=config.get("model_sha", ""),
            still_on_hub=still_on_hub,
            architecture=architecture,
        )

    def update_with_request_file(self, requests_path):
        """Finds the relevant request file for the current model and updates info with it"""
        request_file = get_request_file_for_model(requests_path, self.full_model, self.precision.value.name)

        try:
            with open(request_file, "r") as f:
                request = json.load(f)
            self.model_type = ModelType.from_str(request.get("model_type", ""))
            wt = request.get("weight_type", "Original") or "Original"
            try:
                self.weight_type = WeightType[wt]
            except KeyError:
                self.weight_type = WeightType.Original
            self.license = request.get("license", "?")
            self.likes = request.get("likes", 0)
            params = request.get("params", 0)
            # params might be a dict (empty) or a number
            if isinstance(params, dict):
                params = 0
            self.num_params = params
            self.date = request.get("submitted_time", request.get("created_at", ""))
        except Exception:
            print(
                f"Could not find request file for {self.org}/{self.model} with precision {self.precision.value.name}"
            )

    def to_dict(self):
        """Converts the Eval Result to a dict compatible with our dataframe display"""
        average = sum([v for v in self.results.values() if v is not None]) / len(Tasks)
        data_dict = {
            "eval_name": self.eval_name,  # not a column, just a save name,
            AutoEvalColumn.precision.name: self.precision.value.name,
            AutoEvalColumn.model_type.name: self.model_type.value.name,
            AutoEvalColumn.model_type_symbol.name: self.model_type.value.symbol,
            AutoEvalColumn.weight_type.name: self.weight_type.value.name,
            AutoEvalColumn.architecture.name: self.architecture,
            AutoEvalColumn.model.name: make_clickable_model(self.full_model),
            AutoEvalColumn.revision.name: self.revision,
            AutoEvalColumn.average.name: average,
            AutoEvalColumn.license.name: self.license,
            AutoEvalColumn.likes.name: self.likes,
            AutoEvalColumn.params.name: self.num_params,
            AutoEvalColumn.still_on_hub.name: self.still_on_hub,
        }

        for task in Tasks:
            data_dict[task.value.col_name] = self.results[task.value.benchmark]

        return data_dict


def get_request_file_for_model(requests_path, model_name, precision):
    """Selects the correct request file for a given model. Only keeps runs tagged as FINISHED/completed."""
    # Try multiple naming patterns:
    # 1. New pattern: org__model.request.json (from chemgraph_to_leaderboard.py)
    # 2. Legacy pattern: model_name_eval_request_*.json
    sanitized = model_name.replace("/", "__").replace(" ", "_")

    candidates = []

    # Pattern 1: direct request file
    direct = os.path.join(requests_path, f"{sanitized}.request.json")
    if os.path.exists(direct):
        candidates.append(direct)

    # Pattern 2: legacy eval request pattern
    legacy_pattern = os.path.join(requests_path, f"{model_name}_eval_request_*.json")
    candidates.extend(glob.glob(legacy_pattern))

    # Pattern 3: look in subdirectories (paper_requests/, etc.)
    for subdir in ["paper_requests", "requests"]:
        subdir_path = os.path.join(requests_path, subdir)
        if os.path.isdir(subdir_path):
            sub_direct = os.path.join(subdir_path, f"{sanitized}.request.json")
            if os.path.exists(sub_direct):
                candidates.append(sub_direct)

    # Pattern 4: walk the entire requests_path for any matching file
    if not candidates:
        for root, _, files in os.walk(requests_path):
            for f in files:
                if f == f"{sanitized}.request.json":
                    candidates.append(os.path.join(root, f))

    # Select the best candidate (prefer FINISHED/completed status)
    request_file = ""
    for candidate in candidates:
        try:
            with open(candidate, "r") as f:
                req_content = json.load(f)
            status = req_content.get("status", "")
            if status in ["FINISHED", "completed", "PENDING_NEW_EVAL"]:
                request_file = candidate
                break
        except (json.JSONDecodeError, IOError):
            continue

    # Fallback: use the first candidate regardless of status
    if not request_file and candidates:
        request_file = candidates[0]

    return request_file


def get_raw_eval_results(results_path: str, requests_path: str) -> list[EvalResult]:
    """From the path of the results folder root, extract all needed info for results"""
    model_result_filepaths = []

    for root, _, files in os.walk(results_path):
        # We should only have json files in model results
        if len(files) == 0 or any([not f.endswith(".json") for f in files]):
            continue

        # Sort the files by date
        try:
            files.sort(key=lambda x: x.removesuffix(".json").removeprefix("results_")[:-7])
        except dateutil.parser._parser.ParserError:
            files = [files[-1]]

        for file in files:
            model_result_filepaths.append(os.path.join(root, file))
    eval_results = {}
    print(f"MODEL FILE PATHS: {model_result_filepaths}")
    for model_result_filepath in model_result_filepaths:
        # Creation of result
        try:
            eval_result = EvalResult.init_from_json_file(model_result_filepath)
        except Exception as e:
            print(f"Error loading {model_result_filepath}: {e}")
            continue
        eval_result.update_with_request_file(requests_path)

        # Store results of same eval together
        eval_name = eval_result.eval_name
        if eval_name in eval_results.keys():
            eval_results[eval_name].results.update({k: v for k, v in eval_result.results.items() if v is not None})
        else:
            eval_results[eval_name] = eval_result

    results = []
    for v in eval_results.values():
        try:
            v.to_dict()  # we test if the dict version is complete
            results.append(v)
        except KeyError:  # not all eval values present
            continue

    return results
