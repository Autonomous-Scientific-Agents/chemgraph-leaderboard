"""Basic validation for community-submitted eval tasks.

Phase 1 does NOT execute any user code — real isolation is Phase 2's harbor/Docker
sandbox. These checks only catch obvious mistakes at submit time (malformed slug,
invalid ground-truth JSON, un-parseable solve.py) and surface *warnings* for risky
constructs so a human reviewer knows what to look at before approving a run.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, List, Optional, Tuple

# The scored ResponseFormatter fields (see the task instruction / structured judge).
RESPONSE_FORMATTER_FIELDS = (
    "smiles",
    "scalar_answer",
    "dipole",
    "vibrational_answer",
    "ir_spectrum",
    "atoms_data",
)

# Task-name slug: lowercase, digits, dashes; 3–61 chars; must start alphanumeric.
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,60}$")

# Imports that reach the network / spawn processes — worth a reviewer's eye.
_RISKY_IMPORTS = {
    "subprocess", "socket", "urllib", "urllib2", "urllib3", "requests",
    "httpx", "http", "ftplib", "telnetlib", "smtplib", "paramiko", "pexpect",
}
# Call/name substrings that warrant a warning (belt-and-suspenders on top of AST).
_RISKY_SUBSTRINGS = (
    "os.system", "os.popen", "os.remove", "os.rmdir", "os.unlink",
    "shutil.rmtree", "eval(", "exec(", "__import__", "compile(",
    "pty.spawn", "ctypes",
)


def slugify(name: str) -> str:
    """Turn a free-text task name into a filesystem/URL-safe slug."""
    s = (name or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s


def validate_slug(slug: str) -> Optional[str]:
    """Return an error string if the slug is invalid, else None."""
    if not slug:
        return "Task name is empty or reduced to nothing after slugifying."
    if not _SLUG_RE.match(slug):
        return (
            f'Task name slug "{slug}" is invalid — use 3–61 chars of lowercase '
            "letters, digits and dashes (must start with a letter or digit)."
        )
    return None


def parse_ground_truth(gt_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Parse the submitted ground truth into a canonical ``answer`` object.

    Accepts either:
      - a full answer object ``{tool_calls?, result?, structured_output: {...}}``, or
      - a bare ``structured_output`` dict (with ResponseFormatter fields), which we
        wrap into ``{"structured_output": {...}}``.

    Returns ``(answer_obj, None)`` on success or ``(None, error)`` on failure. The
    returned ``answer_obj`` always has a dict ``structured_output`` with at least one
    non-null ResponseFormatter field.
    """
    if not gt_text or not gt_text.strip():
        return None, "Ground truth is empty."
    try:
        data = json.loads(gt_text)
    except json.JSONDecodeError as exc:
        return None, f"Ground truth is not valid JSON: {exc}"
    if not isinstance(data, dict):
        return None, "Ground truth must be a JSON object (a structured_output dict or an answer object)."

    if isinstance(data.get("structured_output"), dict):
        answer_obj = data
        so = data["structured_output"]
    elif any(k in data for k in RESPONSE_FORMATTER_FIELDS):
        so = data
        answer_obj = {"structured_output": so}
    else:
        return None, (
            "Ground truth must either be a structured_output dict containing at least "
            f"one of {', '.join(RESPONSE_FORMATTER_FIELDS)}, or an answer object with a "
            '"structured_output" key.'
        )

    if not any(so.get(k) is not None for k in RESPONSE_FORMATTER_FIELDS):
        return None, (
            "structured_output has no non-null answer field — populate exactly the one "
            f"relevant to your query ({', '.join(RESPONSE_FORMATTER_FIELDS)})."
        )
    return answer_obj, None


def check_solve_py(code: str) -> Tuple[bool, Optional[str], List[str]]:
    """Validate the oracle solve.py.

    Returns ``(ok, error, warnings)``. ``ok`` is False only on a hard error
    (empty / syntax error). Risky constructs produce warnings, not failures —
    the Phase-2 Docker sandbox is the real containment.
    """
    if not code or not code.strip():
        return False, "solve.py is empty.", []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"solve.py has a syntax error (line {exc.lineno}): {exc.msg}", []

    warnings: List[str] = []
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    risky = sorted(imported & _RISKY_IMPORTS)
    if risky:
        warnings.append(f"solve.py imports network/process modules: {', '.join(risky)}.")

    for pat in _RISKY_SUBSTRINGS:
        if pat in code:
            warnings.append(f"solve.py contains `{pat}` — review before running.")
    return True, None, warnings


# Shell constructs worth a reviewer's eye in the oracle entrypoint.
_RISKY_SH_SUBSTRINGS = (
    "rm -rf", "curl", "wget", "nc ", "ncat", "ssh ", "scp ", "sftp",
    "dd ", "mkfs", "chmod 777", "sudo", ":(){", "> /dev/", "/etc/passwd",
    "base64 -d", "history -c", "/dev/tcp/",
)


def check_solve_sh(code: str) -> List[str]:
    """Basic danger-scan of the oracle entrypoint solve.sh.

    Empty is allowed (the packer substitutes a default ``python3 solve.py``
    wrapper). Returns a list of warnings for risky shell constructs — never a
    hard error, mirroring check_solve_py; the Phase-2 Docker sandbox is the
    real containment.
    """
    warnings: List[str] = []
    if not code or not code.strip():
        return warnings
    for pat in _RISKY_SH_SUBSTRINGS:
        if pat in code:
            warnings.append(f"solve.sh contains `{pat.strip()}` — review before running.")
    return warnings
