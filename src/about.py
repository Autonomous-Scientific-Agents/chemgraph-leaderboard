from dataclasses import dataclass
from enum import Enum


@dataclass
class Task:
    benchmark: str
    metric: str
    col_name: str


# Select your tasks here
# ---------------------------------------------------
class Tasks(Enum):
    # task_key in the json file, metric_key in the json file, name to display in the leaderboard
    task0 = Task("exp1", "accuracy", "name2smi")
    task1 = Task("exp2", "accuracy", "name2coord")
    task2 = Task("exp3", "accuracy", "name2opt")
    task3 = Task("exp4", "accuracy", "name2vib")
    task4 = Task("exp5", "accuracy", "name2gibbs")
    task5 = Task("exp6", "accuracy", "name2file")
    task6 = Task("exp7", "accuracy", "smi2coord")
    task7 = Task("exp8", "accuracy", "smi2opt")
    task8 = Task("exp9", "accuracy", "smi2vib")
    task9 = Task("exp10", "accuracy", "smi2gibbs")
    task10 = Task("exp11", "accuracy", "smi2file")
    task11 = Task("exp12", "accuracy", "react2enthalpy")
    task12 = Task("exp13", "accuracy", "react2gibbs")
    task13 = Task("exp14", "accuracy", "react2enthalpy_multiagent")
    task14 = Task("exp15", "accuracy", "react2gibbs_multiagent")


NUM_FEWSHOT = 0  # Change with your few shot
# ---------------------------------------------------


# Your leaderboard name
TITLE = """<h1 align="center" id="space-title">ChemGraph Leaderboard</h1>"""

# What does your leaderboard evaluate?
INTRODUCTION_TEXT = """
ChemGraph Leaderboard provides a reproducible evaluation of **agentic AI frameworks and large language models (LLMs)** for computational chemistry and materials science.

This leaderboard benchmarks models on a diverse set of tasks, including:
- Molecular geometry optimization, vibration analysis, and thermochemistry estimation.
- Reaction thermodynamics prediction (enthalpy, Gibbs free energy)  .
- Tool-usage accuracy in multi-agent workflows.

Each model’s score reflects its ability to **follow structured tool protocols, generate physically meaningful results, and reason across chemistry-specific contexts**.  
The benchmark results are generated offline and uploaded as part of the [**ChemGraph paper**](https://arxiv.org/abs/2506.06363).

Use this leaderboard to explore how different models and agents perform across core chemistry tasks, from small-molecule modeling to multi-step reaction workflows.
"""

# Which evaluations are you running? how can people reproduce what you have?
LLM_BENCHMARKS_TEXT = f"""
## How it works

## Reproducibility
To reproduce our results, here is the commands you can run:

"""

EVALUATION_QUEUE_TEXT = """
## Some good practices before submitting a model

### 1) Make sure you can load your model and tokenizer using AutoClasses:
```python
from transformers import AutoConfig, AutoModel, AutoTokenizer
config = AutoConfig.from_pretrained("your model name", revision=revision)
model = AutoModel.from_pretrained("your model name", revision=revision)
tokenizer = AutoTokenizer.from_pretrained("your model name", revision=revision)
```
If this step fails, follow the error messages to debug your model before submitting it. It's likely your model has been improperly uploaded.

Note: make sure your model is public!
Note: if your model needs `use_remote_code=True`, we do not support this option yet but we are working on adding it, stay posted!

### 2) Convert your model weights to [safetensors](https://huggingface.co/docs/safetensors/index)
It's a new format for storing weights which is safer and faster to load and use. It will also allow us to add the number of parameters of your model to the `Extended Viewer`!

### 3) Make sure your model has an open license!
This is a leaderboard for Open LLMs, and we'd love for as many people as possible to know they can use your model 🤗

### 4) Fill up your model card
When we add extra information about models to the leaderboard, it will be automatically taken from the model card

## In case of model failure
If your model is displayed in the `FAILED` category, its execution stopped.
Make sure you have followed the above steps first.
If everything is done, check you can launch the EleutherAIHarness on your model locally, using the above command without modifications (you can add `--limit` to limit the number of examples per task).
"""

CITATION_BUTTON_LABEL = "Copy the following snippet to cite these results"
CITATION_BUTTON_TEXT = r"""
@article{pham2025chemgraph,
title={ChemGraph: An Agentic Framework for Computational Chemistry Workflows},
author={Pham, Thang D and Tanikanti, Aditya and Keçeli, Murat},
journal={arXiv preprint arXiv:2506.06363},
year={2025}
url={https://arxiv.org/abs/2506.06363}
}
"""
