# Harbor task-pack templates

These are the boilerplate files used to assemble a community-submitted eval task into a
complete, directly-runnable **harbor-style task directory** (terminal-bench-science format).

A contributor only supplies the task-specific pieces (query, `solve.sh` + `solve.py`, ground
truth); the packer (`src/submission/submit_task.py`) fills everything here around them.

**Verbatim (copied as-is into every task):**
- `tests/test.sh`, `tests/test_outputs.py` — the pytest verifier harness (task-agnostic: it
  reads `/tests/ground_truth.json` and parametrizes over its ids, writing `reward.txt`).
- `tests/structured_output_judge.py` — deterministic ChemGraph judge (SMILES via RDKit
  canonical, scalars within 5% relative tolerance).
- `tests/llm_judge.py` — diagnostic-only LLM judge (does not affect reward).
- `tests/Dockerfile` — lean verifier image (ubuntu + pytest + rdkit + pydantic + langchain).

**Templated (`{{PLACEHOLDER}}` substituted at pack time):**
- `task.toml.tmpl`, `instruction.md.tmpl`, `environment/Dockerfile.tmpl`, `solution/solve.sh`.

These were synced from the pilot task in the integration repo:
`ChemGraph_TBSci_Integration/terminal-bench-science/tasks/physical-sciences/`
`chemistry-and-materials/chemgraph-eval-suite-lite/`. Re-sync if the canonical harness changes.
