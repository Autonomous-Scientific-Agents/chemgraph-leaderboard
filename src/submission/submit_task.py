"""Community eval-task submission: pack a user's task into a harbor-style bundle.

A contributor supplies only the task-specific pieces — a query, an oracle
(solve.sh + solve.py) proving the task is solvable, and the ground truth. This
module assembles those with the vendored boilerplate templates into a complete,
directly-runnable terminal-bench-science / harbor task directory and uploads it
to the community-tasks HF dataset with status PENDING. Nothing is executed here
(Phase 1); running happens later on NWX via `harbor run` after human review.
"""

from __future__ import annotations

import json
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.display.formatting import styled_error, styled_message
from src.envs import API, TASK_SUBMISSIONS_PATH, TASKS_REPO
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
        "status": "PENDING",
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
