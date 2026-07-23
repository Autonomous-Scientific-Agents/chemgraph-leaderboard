"""Community eval-task submission: pack a user's task into a harbor-style bundle.

A contributor supplies only the task-specific pieces — a query, an oracle
(solve.sh + solve.py) proving the task is solvable, and the ground truth. This
module assembles those with the vendored boilerplate templates into a complete,
directly-runnable terminal-bench-science / harbor task directory and uploads it
to the community-tasks HF dataset with status PENDING. Nothing is executed here
(Phase 1); running happens later on NWX via `harbor run` after human review.
"""

from __future__ import annotations

import html
import json
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from huggingface_hub import hf_hub_download

from src.display.formatting import styled_error, styled_message
from src.envs import API, TASK_SUBMISSIONS_PATH, TASKS_REPO, TOKEN
from src.submission.check_task import (
    check_solve_py,
    check_solve_sh,
    parse_ground_truth,
    slugify,
    validate_slug,
)

_TEMPLATES = Path(__file__).resolve().parent / "task_pack" / "templates"
_DEFAULT_BASE_IMAGE = "chemgraph-arm:latest"
_CANARY_RE = re.compile(r"harbor-canary GUID [0-9a-fA-F-]+")

_VERBATIM_TESTS = (
    "test.sh",
    "test_outputs.py",
    "structured_output_judge.py",
    "llm_judge.py",
    "Dockerfile",
)


def _render(text: str, mapping: Dict[str, str]) -> str:
    for key, val in mapping.items():
        text = text.replace("{{" + key + "}}", val)
    return text


def _set_canary(text: str, guid: str) -> str:
    """Refresh any embedded harbor-canary GUID so every file in the bundle
    carries this task's unique contamination marker."""
    return _CANARY_RE.sub(f"harbor-canary GUID {guid}", text)


def build_task_bundle(
    *,
    slug: str,
    title: str,
    category: str,
    query: str,
    answer_obj: Dict[str, Any],
    solve_sh: str,
    solve_py: str,
    author_name: str,
    author_email: str,
    author_org: str,
    domain: str,
    field: str,
    subfield: str,
    tags: List[str],
    difficulty: str = "",
    base_image: Optional[str] = None,
    dest_root: Optional[str] = None,
) -> Path:
    """Assemble a complete harbor-style task directory; return its path."""
    guid = str(uuid.uuid4())
    base_image = base_image or _DEFAULT_BASE_IMAGE
    root = Path(dest_root or TASK_SUBMISSIONS_PATH) / slug
    if root.exists():
        shutil.rmtree(root)
    (root / "environment" / "data").mkdir(parents=True, exist_ok=True)
    (root / "solution").mkdir(parents=True, exist_ok=True)
    (root / "tests").mkdir(parents=True, exist_ok=True)

    # --- generated task-specific data (single query) ---
    queries = [{"id": "1", "category": category, "query": query}]
    (root / "environment" / "data" / "queries.json").write_text(
        json.dumps(queries, indent=2) + "\n"
    )

    gt = [{"id": "1", "category": category, "query": query, "answer": answer_obj}]
    gt_json = json.dumps(gt, indent=2) + "\n"
    (root / "tests" / "ground_truth.json").write_text(gt_json)          # verifier-side
    (root / "solution" / "oracle_answers.json").write_text(gt_json)     # oracle reference

    # --- user oracle (solve.sh + solve.py); default solve.sh if left blank ---
    if solve_sh and solve_sh.strip():
        sh = _set_canary(solve_sh, guid)  # user-provided: refresh any embedded canary
    else:
        # default template carries a {{CANARY_GUID}} placeholder → render it
        sh = _render((_TEMPLATES / "solution" / "solve.sh").read_text(), {"CANARY_GUID": guid})
    (root / "solution" / "solve.sh").write_text(sh)
    (root / "solution" / "solve.py").write_text(solve_py)

    # --- templated files ---
    def esc(s: Optional[str]) -> str:  # -> TOML basic string (json.dumps escaping is compatible)
        return json.dumps(s if s is not None else "")

    toml_map = {
        "CANARY_GUID": guid,
        "AUTHOR_NAME": esc(author_name),
        "AUTHOR_EMAIL": esc(author_email),
        "AUTHOR_ORG": esc(author_org),
        "DIFFICULTY": esc(difficulty or f"Community-contributed {category} task."),
        "SOLUTION_EXPL": esc(
            "Author-provided oracle (solve.sh + solve.py) that reproduces the ground truth."
        ),
        "VERIFICATION_EXPL": esc(
            "structured_output compared to ground truth: SMILES via RDKit canonical, "
            "scalars within 5% relative tolerance; reward = fraction of queries passing."
        ),
        "DOMAIN": esc(domain),
        "FIELD": esc(field),
        "SUBFIELD": esc(subfield),
        "TAGS": json.dumps(tags),
    }
    (root / "task.toml").write_text(
        _render((_TEMPLATES / "task.toml.tmpl").read_text(), toml_map)
    )
    (root / "instruction.md").write_text(
        _render(
            (_TEMPLATES / "instruction.md.tmpl").read_text(),
            {"CANARY_GUID": guid, "TITLE": title},
        )
    )
    (root / "environment" / "Dockerfile").write_text(
        _render(
            (_TEMPLATES / "environment" / "Dockerfile.tmpl").read_text(),
            {"CANARY_GUID": guid, "BASE_IMAGE": base_image},
        )
    )

    # --- vendored verbatim verifier harness (canary refreshed) ---
    for fn in _VERBATIM_TESTS:
        (root / "tests" / fn).write_text(
            _set_canary((_TEMPLATES / "tests" / fn).read_text(), guid)
        )

    return root


def add_new_task(
    task_name: str,
    category: str,
    query: str,
    ground_truth: str,
    solve_sh: str,
    solve_py: str,
    author_name: str,
    author_email: str,
    author_org: str = "",
    domain: str = "physical-sciences",
    field: str = "chemistry-and-materials",
    subfield: str = "computational-chemistry",
    notes: str = "",
    tools: Optional[List[str]] = None,
):
    """Validate a submission, assemble the harbor task bundle, upload to HF."""
    # --- required fields ---
    if not (author_name and author_name.strip()) or not (author_email and author_email.strip()):
        return styled_error("Please provide your name and email.")
    if not (query and query.strip()):
        return styled_error("Please provide a query.")
    if not (category and category.strip()):
        return styled_error("Please select a category.")

    slug = slugify(task_name)
    slug_err = validate_slug(slug)
    if slug_err:
        return styled_error(slug_err)

    answer_obj, gt_err = parse_ground_truth(ground_truth)
    if gt_err:
        return styled_error(f"Ground truth: {gt_err}")

    ok, py_err, warnings = check_solve_py(solve_py or "")
    if not ok:
        return styled_error(py_err)
    warnings = list(warnings) + check_solve_sh(solve_sh or "")

    domain = (domain or "physical-sciences").strip()
    field = (field or "chemistry-and-materials").strip()
    subfield = (subfield or "computational-chemistry").strip()
    # Flat layout: the task lives one level deep at the dataset root (<slug>/),
    # no tasks/<domain>/<field> nesting. domain/field are kept in task.toml metadata.
    prefix = slug

    # --- duplicate check against the dataset ---
    try:
        existing = set(API.list_repo_files(TASKS_REPO, repo_type="dataset"))
    except Exception:
        existing = set()  # dataset may not exist yet — first submission
    if any(f.startswith(prefix + "/") for f in existing):
        return styled_error(
            f'A task named "{slug}" already exists in {TASKS_REPO}. Choose a different name.'
        )

    # Selected compute tools become extra tags (deduped, order-preserving).
    tool_list = [t for t in (tools or []) if t]
    tag_list = ["chemistry", "community", category] + tool_list

    # --- assemble ---
    root = build_task_bundle(
        slug=slug,
        title=(task_name or slug).strip(),
        category=category,
        query=query,
        answer_obj=answer_obj,
        solve_sh=solve_sh or "",
        solve_py=solve_py,
        author_name=author_name.strip(),
        author_email=author_email.strip(),
        author_org=(author_org or "").strip(),
        domain=domain,
        field=field,
        subfield=subfield,
        tags=tag_list,
        difficulty=notes or "",
    )

    submission = {
        "slug": slug,
        "title": (task_name or slug).strip(),
        "category": category,
        "submitter": author_name.strip(),
        "email": author_email.strip(),
        "organization": (author_org or "").strip(),
        "tools": tool_list,
        # Lifecycle stage. The PR stays open through review -> validation ->
        # evaluation (the pipeline bumps this field on the PR branch); merging
        # the PR is the final "done" step. See STATUS_STAGES below.
        "status": "under_review",
        "submitted_time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "leaderboard",
        "path_in_repo": prefix,
    }
    (root / "submission.json").write_text(json.dumps(submission, indent=2) + "\n")

    # --- open a pull request with the whole bundle (not a direct commit to
    #     main) so a maintainer reviews it before it's added. ---
    try:
        commit_info = API.upload_folder(
            folder_path=str(root),
            path_in_repo=prefix,
            repo_id=TASKS_REPO,
            repo_type="dataset",
            commit_message=f"Add community task: {slug}",
            create_pr=True,
        )
    except Exception as exc:
        shutil.rmtree(root, ignore_errors=True)
        return styled_error(f"Submitting a pull request to {TASKS_REPO} failed: {exc}")
    shutil.rmtree(root, ignore_errors=True)

    # NB: styled_message wraps this in a raw <p>, so use HTML (not markdown).
    pr_url = getattr(commit_info, "pr_url", None)
    msg = (
        f"✅ Task <b>{slug}</b> submitted as a pull request to {TASKS_REPO} — "
        "it won't be added until a maintainer reviews and merges it."
    )
    if pr_url:
        msg += f'<br><br>🔗 <a href="{pr_url}" target="_blank">View your pull request</a>'
    if warnings:
        msg += "<br><br>⚠️ Reviewer notes:<br>" + "<br>".join(f"• {w}" for w in warnings)
    return styled_message(msg)


# --------------------------------------------------------------------------- #
# Submission status board
#
# A task's lifecycle runs entirely on its (open) pull request: it advances
# through review -> validation -> evaluation while the PR is open, and only gets
# merged once it has passed evaluation ("done"). So the status board reads:
#   - each OPEN PR   -> its current stage (submission.json.status on the PR ref)
#   - each MERGED task on main -> "done"
# The mechanism that bumps a PR's stage is Phase 2 (deferred); the frontend just
# reflects whatever stage each task is currently in.
# --------------------------------------------------------------------------- #

# Ordered pipeline stages (index == progress). Keep in sync with the stepper.
STATUS_STAGES = ["under_review", "under_validation", "under_evaluation", "done"]
_STATUS_STEP_LABELS = ["Review", "Validation", "Evaluation", "Done"]
_STATUS_ALIASES = {
    "pending": "under_review",
    "review": "under_review",
    "under_review": "under_review",
    "validate": "under_validation",
    "validating": "under_validation",
    "under_validate": "under_validation",
    "under_validation": "under_validation",
    "evaluate": "under_evaluation",
    "evaluating": "under_evaluation",
    "under_evaluate": "under_evaluation",
    "under_evaluation": "under_evaluation",
    "done": "done",
    "complete": "done",
    "completed": "done",
    "finished": "done",
}

# Per-tool tag colours (by name, so order-independent — matches the palette used
# by the contribute-form tool picker in css_html_js.py).
_TOOL_COLORS = {
    "RDKit": "#2563eb",
    "MACE": "#0d9488",
    "TBLite": "#d97706",
    "NWChem": "#16a34a",
    "ORCA": "#7c3aed",
    "UMA": "#db2777",
    "AIMNet2": "#0891b2",
    "gRASPA": "#dc2626",
    "XANES": "#65a30d",
}


def _normalize_status(status: Optional[str]) -> str:
    """Map any stored/legacy status string to one of STATUS_STAGES."""
    key = (status or "").strip().lower().replace(" ", "_").replace("-", "_")
    return _STATUS_ALIASES.get(key, "under_review")


def _read_submission_json(path_in_repo: str, revision: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Download and parse one task's submission.json (returns None on any error)."""
    try:
        local = hf_hub_download(
            repo_id=TASKS_REPO,
            filename=path_in_repo,
            repo_type="dataset",
            revision=revision,
            token=TOKEN,
        )
        return json.loads(Path(local).read_text())
    except Exception:
        return None


def get_submitted_tasks() -> List[Dict[str, Any]]:
    """Aggregate community task submissions and their pipeline stage.

    Returns a list of ``{"name", "tools", "status"}`` dicts. Never raises — on
    any network/repo error it returns whatever it managed to gather (possibly
    empty). Merged tasks (on ``main``) are reported as ``done``; open PRs report
    the stage recorded in their own ``submission.json``.
    """
    tasks: List[Dict[str, Any]] = []
    merged_slugs: set[str] = set()

    # --- merged tasks (live on main) -> done ---
    try:
        files = API.list_repo_files(repo_id=TASKS_REPO, repo_type="dataset")
    except Exception:
        files = []
    for f in files:
        # A task root holds a submission.json. Match at any depth so both the
        # current flat layout ("<slug>/submission.json") and legacy nested ones
        # ("tasks/<domain>/<field>/<slug>/submission.json") are picked up.
        if f.endswith("/submission.json"):
            slug = f.rsplit("/", 2)[-2]  # parent-dir name
            data = _read_submission_json(f) or {}
            merged_slugs.add(data.get("slug") or slug)
            tasks.append(
                {
                    "name": data.get("title") or slug,
                    "tools": data.get("tools") or [],
                    "status": "done",  # on main == merged == done
                    # Evaluation outcome (how it's produced is deferred — the
                    # renderer shows "pending" when absent).
                    "result": data.get("result"),
                }
            )

    # --- open PRs -> their current in-flight stage ---
    try:
        discussions = list(API.get_repo_discussions(repo_id=TASKS_REPO, repo_type="dataset"))
    except Exception:
        discussions = []
    for disc in discussions:
        if not (getattr(disc, "is_pull_request", False) and disc.status == "open"):
            continue
        # Submission PR titles are "Add community task: <slug>".
        title = disc.title or ""
        slug = title.split(":")[-1].strip() if ":" in title else title.strip()
        if not slug or slug in merged_slugs:
            continue
        merged_slugs.add(slug)  # de-dupe across multiple PRs for the same slug
        data = _read_submission_json(f"{slug}/submission.json", revision=f"refs/pr/{disc.num}") or {}
        tasks.append(
            {
                "name": data.get("title") or slug,
                "tools": data.get("tools") or [],
                "status": _normalize_status(data.get("status")),
                "result": None,
            }
        )

    return tasks


def _status_stepper_html(status: str) -> str:
    """Render the 4-step progress stepper for a normalized status."""
    try:
        current = STATUS_STAGES.index(status)
    except ValueError:
        current = 0
    parts: List[str] = []
    is_done = status == "done"
    for i, label in enumerate(_STATUS_STEP_LABELS):
        if i < current or (i == current and is_done):
            # "done" is terminal — the whole bar reads complete (green ✓).
            cls, glyph = "done", "✓"
        elif i == current:
            cls, glyph = "current", "●"
        else:
            cls, glyph = "todo", "○"
        parts.append(f"<span class='cg-step {cls}'>{glyph} {label}</span>")
        if i < len(_STATUS_STEP_LABELS) - 1:
            parts.append(f"<span class='cg-conn {'done' if i < current else ''}'></span>")
    return "<div class='cg-stepper'>" + "".join(parts) + "</div>"


def _format_result(result: Any) -> str:
    """Render the evaluation-result chip for a completed task.

    The result source is deferred; until a task carries one we show "pending".
    A number is treated as a reward fraction (0–1) and shown as a percentage; a
    dict is probed for reward/score/accuracy; a string is shown verbatim.
    """
    if result is None or result == "":
        return "<span class='cg-result cg-result-pending'>Result pending</span>"
    val = result
    if isinstance(result, dict):
        for k in ("reward", "score", "accuracy"):
            if result.get(k) is not None:
                val = result[k]
                break
        else:
            val = None
    if val is None or val == "":
        return "<span class='cg-result cg-result-pending'>Result pending</span>"
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        pct = val * 100 if 0 <= val <= 1 else val
        return f"<span class='cg-result'>Reward {pct:.0f}%</span>"
    return f"<span class='cg-result'>{html.escape(str(val))}</span>"


def _tags_html(tools: List[str]) -> str:
    tags = "".join(
        "<span class='cg-sub-tag' style='--tag:{c}'>{n}</span>".format(
            c=_TOOL_COLORS.get(tool, "#94a3b8"), n=html.escape(str(tool))
        )
        for tool in (tools or [])
    )
    return f"<div class='cg-sub-tags'>{tags}</div>" if tags else ""


def _task_card(task: Dict[str, Any], *, done: bool) -> str:
    name = html.escape(task["name"] or "")
    body = f"<div class='cg-sub-name'>{name}</div>{_tags_html(task.get('tools'))}"
    if done:
        # Completed: a done badge + the evaluation result (source TBD).
        body += (
            "<div class='cg-sub-result'>"
            "<span class='cg-done-badge'>✓ Completed</span>"
            f"{_format_result(task.get('result'))}"
            "</div>"
        )
    else:
        body += _status_stepper_html(task["status"])
    return f"<div class='cg-sub-card'>{body}</div>"


def _status_section(title: str, tasks: List[Dict[str, Any]], *, done: bool, empty: str) -> str:
    head = (
        f"<div class='cg-sub-section-head'>{title}"
        f"<span class='cg-sub-count'>{len(tasks)}</span></div>"
    )
    if not tasks:
        body = f"<div class='cg-sub-empty'>{empty}</div>"
    else:
        body = (
            "<div class='cg-sub-board'>"
            + "".join(_task_card(t, done=done) for t in tasks)
            + "</div>"
        )
    return f"<div class='cg-sub-section'>{head}{body}</div>"


def build_task_status_view() -> str:
    """Build the status board: a Pending section and a Completed section.

    Pending = tasks whose PR is still open (review / validation / evaluation);
    Completed = tasks merged into ``main`` (shown with their evaluation result).
    """
    tasks = get_submitted_tasks()
    order = {s: i for i, s in enumerate(STATUS_STAGES)}

    pending = [t for t in tasks if t["status"] != "done"]
    completed = [t for t in tasks if t["status"] == "done"]
    pending.sort(key=lambda t: (order.get(t["status"], 0), (t["name"] or "").lower()))
    completed.sort(key=lambda t: (t["name"] or "").lower())

    return (
        _status_section(
            "Pending", pending, done=False,
            empty="No tasks in the review pipeline right now.",
        )
        + _status_section(
            "Completed", completed, done=True,
            empty="No tasks have completed evaluation yet.",
        )
    )
