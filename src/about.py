from dataclasses import dataclass
from enum import Enum


@dataclass
class Task:
    benchmark: str
    metric: str
    col_name: str


# 8 task categories derived from the 14 ChemGraph ground-truth queries.
# Each category groups related queries; accuracy is averaged within
# the group by the transform script (scripts/chemgraph_to_leaderboard.py).
# ---------------------------------------------------
class Tasks(Enum):
    # benchmark key in results JSON, metric key, display column name
    task0 = Task("smi_lookup", "accuracy", "SMILES Lookup")
    task1 = Task("coord_gen", "accuracy", "Coordinate Gen")
    task2 = Task("geom_opt", "accuracy", "Geometry Opt")
    task3 = Task("vib_freq", "accuracy", "Vib Frequency")
    task4 = Task("thermo", "accuracy", "Thermochem")
    task5 = Task("dipole", "accuracy", "Dipole")
    task6 = Task("energy", "accuracy", "Energy")
    task7 = Task("react_gibbs", "accuracy", "Reaction Gibbs")


NUM_FEWSHOT = 0  # Change with your few shot
# ---------------------------------------------------


# Your leaderboard name
TITLE = """<h1 align="center" id="space-title">ChemGraph Leaderboard</h1>"""

# What does your leaderboard evaluate?
INTRODUCTION_TEXT = """
ChemGraph Leaderboard provides a reproducible evaluation of **agentic AI frameworks and large language models (LLMs)** for computational chemistry and materials science.

Models are evaluated daily on **14 chemistry queries** grouped into **8 task categories**:

| Category | Queries | Description |
|----------|---------|-------------|
| **SMILES Lookup** | 2 | Convert molecule names to SMILES strings |
| **Coordinate Gen** | 2 | Generate 3D coordinates from SMILES |
| **Geometry Opt** | 1 | Geometry optimization with DFT/ML potentials |
| **Vib Frequency** | 1 | Vibrational frequency analysis |
| **Thermochem** | 1 | Thermochemical properties (enthalpy, entropy, Gibbs) |
| **Dipole** | 1 | Dipole moment calculation |
| **Energy** | 3 | Single-point energy and geometry opt with JSON extraction |
| **Reaction Gibbs** | 3 | Reaction Gibbs free energy for multi-step workflows |

Each model's score reflects its ability to **follow structured tool protocols, generate physically meaningful results, and reason across chemistry-specific contexts**.
Results are scored by an LLM judge with binary accuracy (correct/incorrect) and 5% relative tolerance for numerical values.

Use this leaderboard to explore how different models and agents perform across core chemistry tasks, from small-molecule modeling to multi-step reaction workflows.
"""

# Which evaluations are you running? how can people reproduce what you have?
LLM_BENCHMARKS_TEXT = f"""
## How it works

Models are evaluated using the [ChemGraph](https://github.com/Autonomous-Scientific-Agents/ChemGraph) evaluation framework.
Each model runs as a **single-agent** workflow, invoking chemistry tools (SMILES lookup, coordinate generation, ASE simulations)
to answer 14 ground-truth queries. An LLM judge then scores each answer as correct or incorrect.

Results are updated daily via an automated pipeline that:
1. Runs `chemgraph eval` against all configured models
2. Transforms the benchmark results into leaderboard format
3. Pushes updated results to the HF Hub datasets

## Reproducibility

To reproduce the evaluation locally:

```bash
pip install chemgraph

# Run evaluation
chemgraph eval \\
    --models gpt4o gpt52 claudeopus46 \\
    --judge-model claudeopus46 \\
    --workflows single_agent \\
    --config config.toml

# Transform results for the leaderboard
python scripts/chemgraph_to_leaderboard.py \\
    --eval-dir eval_results \\
    --model-map dataset/model_map.json
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
The evaluation runs 14 queries per model, each potentially requiring multiple tool calls.
Ensure your API key has sufficient quota for the evaluation run.

## In case of model failure
If your model appears in the `FAILED` category, check that:
- The API endpoint is accessible
- The model supports tool/function calling
- There are no rate limiting issues
"""

CITATION_BUTTON_LABEL = "Copy the following snippet to cite these results"
CITATION_BUTTON_TEXT = r"""
@article{pham2025chemgraph,
title={ChemGraph: An Agentic Framework for Computational Chemistry Workflows},
author={Pham, Thang D and Tanikanti, Aditya and Ke\c{c}eli, Murat},
journal={arXiv preprint arXiv:2506.06363},
year={2025}
url={https://arxiv.org/abs/2506.06363}
}
"""
