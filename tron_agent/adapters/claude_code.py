from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

from ..agent import SecurityReviewer
from .feedback import format_remediation_prompt


def get_diff(cwd: Optional[Path] = None) -> str:
    """Return git diff of working-tree vs HEAD. Returns empty string on any git error."""
    result = subprocess.run(
        ["git", "diff", "HEAD"],
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    if result.returncode != 0:
        return ""
    return result.stdout or ""


def handle_claude_code_stop(
    reviewer: Optional[SecurityReviewer] = None,
    cwd: Optional[Path] = None,
) -> None:
    """
    Claude Code Stop hook entry point.
    Exit 0 → PERMIT (session ends).
    Exit 1 with remediation prompt on stdout → DENY (session continues).
    """
    try:
        json.load(sys.stdin)
    except Exception:
        pass

    diff = get_diff(cwd=cwd)
    if reviewer is None:
        reviewer = SecurityReviewer()

    verdict = reviewer.run_review(diff)
    reviewer.write_audit_log(verdict, diff)

    if verdict["decision"] == "PERMIT":
        sys.exit(0)
    else:
        print(format_remediation_prompt(verdict))
        sys.exit(1)
