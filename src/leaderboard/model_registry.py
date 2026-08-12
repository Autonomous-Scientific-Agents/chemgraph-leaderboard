"""Curated "current model list" for the Full table and the Highlights view.

The daily eval keeps results for every model ever run, but the Full table and
the Highlights KPI strip / efficiency frontier should only show the models we
currently maintain. That roster lives in ``dataset/current_models.json`` (a
hand-maintained mirror of ``scripts/daily_eval.sh`` MODELS +
``scripts/alcf_Eval.sh`` MODELS) and is applied in
``src.populate.get_leaderboard_df``.

Deliberately NOT applied to the Trends tab: trends show the full history,
retired models included.

FAILS OPEN. A missing, unparseable, or empty roster means "no filtering" — a bad
edit degrades to the pre-filter behaviour instead of blanking a live public
leaderboard. Every failure prints a WARNING. This matches the sibling loaders in
``metrics.py`` (``_load_pricing`` / ``_load_model_map``), which return empty and
let the app degrade rather than raise.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

# Repo root = two levels up from this file (src/leaderboard/model_registry.py),
# the same idiom as src/leaderboard/metrics.py.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_ROSTER_PATH = _REPO_ROOT / "dataset" / "current_models.json"


def _extract(raw) -> list[str]:
    """Model names out of any accepted shape.

    Accepts ``{"models": {group: [...]}}`` (what we ship), ``{"models": [...]}``,
    or a bare ``[...]`` — so hand-flattening the file later doesn't break it.
    Keys starting with ``_`` are treated as comments.
    """
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if not isinstance(raw, dict):
        return []
    models = raw.get("models", raw)
    if isinstance(models, list):
        return [str(x) for x in models]
    if isinstance(models, dict):
        out: list[str] = []
        for group, names in models.items():
            if str(group).startswith("_"):
                continue
            if isinstance(names, list):
                out.extend(str(x) for x in names)
        return out
    return []


@lru_cache(maxsize=1)
def current_models() -> dict[str, str] | None:
    """``{lowercased "org/model": name as written}`` for the curated roster.

    Returns ``None`` when no usable roster exists. Callers MUST treat ``None`` as
    "show everything" — see the module docstring.
    """
    if not _ROSTER_PATH.exists():
        print(f"WARNING: {_ROSTER_PATH.name} not found — showing ALL models.")
        return None
    try:
        with open(_ROSTER_PATH) as fp:
            raw = json.load(fp)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"WARNING: could not read {_ROSTER_PATH.name} ({exc}) — showing ALL models.")
        return None

    names = [n.strip() for n in _extract(raw) if str(n).strip()]
    if not names:
        # An explicitly empty roster is far more likely a mistake than an intent
        # to blank the board, so treat it as fail-open too.
        print(f"WARNING: {_ROSTER_PATH.name} lists no models — showing ALL models.")
        return None

    # Case-insensitive: the roster carries error-prone casing
    # (Meta-Llama-3.1-70B-Instruct, gemma-4-E4B-it). Keyed lowercase, valued with
    # the name as written so warnings can echo it back readably.
    roster = {n.lower(): n for n in names}
    if len(roster) != len(names):
        print(f"WARNING: {_ROSTER_PATH.name} has duplicate entries ({len(names)} listed, {len(roster)} unique).")
    return roster
