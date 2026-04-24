from dataclasses import dataclass
from enum import Enum


@dataclass
class Task:
    benchmark: str
    metric: str
    col_name: str
    num_questions: int = 1  # number of individual questions in this category


# 12 task categories matching the ``category`` field in ChemGraph benchmark
# judge details.  Each category's accuracy is averaged within the group by
# the transform script (scripts/chemgraph_to_leaderboard.py).
# ---------------------------------------------------
class Tasks(Enum):
    # benchmark key in results JSON, metric key, display column name, num questions
    task0 = Task("smiles_lookup", "accuracy", "SMILES Lookup", 4)
    task1 = Task("optimization_from_name", "accuracy", "Opt (Name)", 4)
    task2 = Task("optimization_from_smiles", "accuracy", "Opt (SMILES)", 2)
    task3 = Task("vibrations_from_name", "accuracy", "Vib (Name)", 2)
    task4 = Task("vibrations_from_smiles", "accuracy", "Vib (SMILES)", 2)
    task5 = Task("thermochemistry_from_name", "accuracy", "Thermo (Name)", 4)
    task6 = Task("thermochemistry_from_smiles", "accuracy", "Thermo (SMILES)", 2)
    task7 = Task("dipole_from_name", "accuracy", "Dipole (Name)", 2)
    task8 = Task("dipole_from_smiles", "accuracy", "Dipole (SMILES)", 2)
    task9 = Task("energy_from_name", "accuracy", "Energy (Name)", 4)
    task10 = Task("energy_from_smiles", "accuracy", "Energy (SMILES)", 2)
    task11 = Task("reaction_energy", "accuracy", "Reaction Energy", 10)


NUM_FEWSHOT = 0  # Change with your few shot
# ---------------------------------------------------


# Your leaderboard name
TITLE = """<h1 align="center" id="space-title">ChemGraph Leaderboard</h1>"""

# What does your leaderboard evaluate?
INTRODUCTION_TEXT = """
ChemGraph Leaderboard provides a reproducible evaluation of **agentic AI frameworks and large language models (LLMs)** for computational chemistry and materials science.

Models are evaluated daily on **40 chemistry queries** grouped into **12 task categories**:

| Category | Queries | Description |
|----------|---------|-------------|
| **SMILES Lookup** | 4 | Convert molecule names to SMILES strings |
| **Opt (Name)** | 4 | Geometry optimization from molecule name |
| **Opt (SMILES)** | 2 | Geometry optimization from SMILES |
| **Vib (Name)** | 2 | Vibrational frequency from molecule name |
| **Vib (SMILES)** | 2 | Vibrational frequency from SMILES |
| **Thermo (Name)** | 4 | Thermochemistry from molecule name |
| **Thermo (SMILES)** | 2 | Thermochemistry from SMILES |
| **Dipole (Name)** | 2 | Dipole moment from molecule name |
| **Dipole (SMILES)** | 2 | Dipole moment from SMILES |
| **Energy (Name)** | 4 | Single-point energy from molecule name |
| **Energy (SMILES)** | 2 | Single-point energy from SMILES |
| **Reaction Energy** | 10 | Reaction Gibbs free energy calculation |

Each model's score reflects its ability to **follow structured tool protocols, generate physically meaningful results, and reason across chemistry-specific contexts**.
Results are scored by a structured judge via JSON output for evaluation with binary accuracy (correct/incorrect) and 5% relative tolerance for numerical values.

Models are evaluated under two workflow types:
- **Single-Agent** — one agent handles all tool calls and reasoning independently.
- **Multi-Agent** — multiple specialised agents collaborate to solve queries.

Use this leaderboard to explore how different models and agents perform across core chemistry tasks, from small-molecule modeling to multi-step reaction workflows.
"""

# Which evaluations are you running? how can people reproduce what you have?
LLM_BENCHMARKS_TEXT = f"""
## How it works

Models are evaluated using the [ChemGraph](https://github.com/Autonomous-Scientific-Agents/ChemGraph) evaluation framework
across two workflow types:

- **Single-Agent**: Each model operates as a single agent, invoking chemistry tools
  (SMILES lookup, coordinate generation, ASE simulations) to answer 40 ground-truth queries.
- **Multi-Agent**: Multiple specialised agents collaborate to solve the same 40 queries,
  coordinating tool calls and reasoning across agents.

Both workflows are evaluated on the same 12 task categories and scored identically:
a structured judge scores each answer as correct or incorrect (binary accuracy with 5%
relative tolerance for numerical values).

Results are updated daily via an automated pipeline that:
1. Runs `chemgraph-eval` against all configured models for both workflows
2. Transforms the benchmark results into leaderboard format
3. Pushes updated results to the HF Hub datasets

## Reproducibility

To reproduce the evaluation locally:

```bash
pip install chemgraph

# Run evaluation (single-agent)
chemgraph-eval --models gpt4o gpt54 --workflows single_agent --judge-type structured --config config.toml

# Run evaluation (multi-agent)
chemgraph-eval --models gpt4o gpt54 --workflows multi_agent --judge-type structured --config config.toml

# Transform results for the leaderboard
python scripts/chemgraph_to_leaderboard.py \\
    --eval-dir eval_results \\
    --model-map dataset/model_map.json \\
    --workflow single_agent --push-to-hub

python scripts/chemgraph_to_leaderboard.py \\
    --eval-dir eval_results \\
    --model-map dataset/model_map.json \\
    --workflow multi_agent --push-to-hub
```

See the [ChemGraph paper](https://arxiv.org/abs/2506.06363) for full details on the benchmark design and evaluation methodology.
"""

EVALUATION_QUEUE_TEXT = """
## Some good practices before submitting a model

### 1) Make sure your model is accessible via an API
ChemGraph evaluates models through their API endpoints. Ensure your model is available
and correctly configured in the evaluation config.

### 2) Verify tool-calling support
ChemGraph requires models that support function/tool calling. The evaluation uses
structured tool calls for chemistry operations (SMILES lookup, coordinate generation,
ASE simulations).

### 3) Check API rate limits
The evaluation runs 40 queries per model, each potentially requiring multiple tool calls.
Ensure your API key has sufficient quota for the evaluation run.

## In case of model failure
If your model appears in the `FAILED` category, check that:
- The API endpoint is accessible
- The model supports tool/function calling
- There are no rate limiting issues
"""

CITATION_BUTTON_LABEL = "Copy the following snippet to cite these results"
CITATION_BUTTON_TEXT = r"""
@article{pham_chemgraph_2026,
  title = {{ChemGraph} as an agentic framework for computational chemistry workflows},
  url = {https://doi.org/10.1038/s42004-025-01776-9},
  doi = {10.1038/s42004-025-01776-9},
  author = {Pham, Thang D. and Tanikanti, Aditya and Ke\c{c}eli, Murat},
  date = {2026-01-08},
  author={Pham, Thang D and Tanikanti, Aditya and Ke{\c{c}}eli, Murat},
  journal={Communications Chemistry},
  year={2026},
  publisher={Nature Publishing Group UK London}
}
"""
