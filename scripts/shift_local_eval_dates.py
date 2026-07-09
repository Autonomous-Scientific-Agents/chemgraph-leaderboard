"""Shift every local eval-results/*/<model>/results_YYYY-MM-DD.json forward
so that the newest file lands on today's UTC date. Both the filename and
the `config.eval_date` field inside are updated.

Local-only convenience script: the production data on HF Hub is the source
of truth; this is just so the leaderboard's 1-Day / 3-Day / 7-Day columns
have data inside their look-back window when running the app locally.

Run from repo root:
    python scripts/shift_local_eval_dates.py
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "eval-results"
WORKFLOWS = ("single_agent", "multi_agent")
PATTERN = re.compile(r"^results_(\d{4}-\d{2}-\d{2})\.json$")


def collect_files() -> list[tuple[Path, date]]:
    files: list[tuple[Path, date]] = []
    for wf in WORKFLOWS:
        wf_root = ROOT / wf
        if not wf_root.is_dir():
            continue
        for path in wf_root.rglob("results_*.json"):
            if ".cache" in path.parts:
                continue
            m = PATTERN.match(path.name)
            if not m:
                continue
            files.append((path, date.fromisoformat(m.group(1))))
    return files


def shift_file(path: Path, old: date, new: date) -> Path:
    with path.open() as fp:
        data = json.load(fp)
    data.setdefault("config", {})["eval_date"] = new.isoformat()

    new_path = path.with_name(f"results_{new.isoformat()}.json")
    if new_path != path and new_path.exists():
        raise RuntimeError(f"Target already exists: {new_path}")

    with path.open("w") as fp:
        json.dump(data, fp, indent=2)

    if new_path != path:
        path.rename(new_path)
    return new_path


def main() -> None:
    files = collect_files()
    if not files:
        print("No results_YYYY-MM-DD.json files found under eval-results/.")
        return

    max_date = max(d for _, d in files)
    today = datetime.now(timezone.utc).date()
    offset = (today - max_date).days

    if offset == 0:
        print(f"Newest file is already today ({today.isoformat()}); nothing to do.")
        return
    if offset < 0:
        print(
            f"Newest local file ({max_date.isoformat()}) is AFTER today "
            f"({today.isoformat()}); refusing to shift backwards."
        )
        return

    print(
        f"Shifting {len(files)} file(s) by +{offset} day(s) "
        f"(max {max_date.isoformat()} -> {today.isoformat()})."
    )

    # Rename newest-first so we never collide with an existing future-dated file
    # within the same model directory.
    files.sort(key=lambda t: t[1], reverse=True)
    for path, old in files:
        new = old + timedelta(days=offset)
        shift_file(path, old, new)

    print("Done.")


if __name__ == "__main__":
    main()
