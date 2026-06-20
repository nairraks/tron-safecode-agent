from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from ..agent import SecurityReviewer
from ..config import load_config
from .feedback import format_remediation_prompt

_API_KEY_VARS = ("GOOGLE_API_KEY", "GEMINI_API_KEY")

_MISSING_KEY_MESSAGE = """\
tron-agent: GOOGLE_API_KEY is not set.

Security reviews cannot run without a Gemini API key.

Quick fix:
  export GOOGLE_API_KEY=your-key-here

Get a free key at: https://ai.google.dev/gemini-api/docs/api-key

To skip reviews when no key is available (not recommended for production):
  export TRON_FAIL_OPEN=1\
"""


def _api_key_available() -> bool:
    return any(os.environ.get(v) for v in _API_KEY_VARS)


def get_diff(cwd: Optional[Path] = None) -> str:
    """Return a combined diff of tracked changes (vs HEAD) plus untracked new files.

    Untracked files are synthesized into unified-diff format so the reviewer
    sees every line of code the agent produced, not just the staged delta.
    Returns an empty string on any git error (e.g. empty repo, not a git repo).
    """
    work_dir = Path(cwd) if cwd else Path.cwd()

    # Tracked changes: staged + unstaged vs HEAD
    tracked = subprocess.run(
        ["git", "diff", "HEAD"],
        capture_output=True, text=True, cwd=cwd,
    )
    if tracked.returncode != 0:
        return ""
    tracked_diff = tracked.stdout or ""

    # Untracked files: new files not yet staged
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        capture_output=True, text=True, cwd=cwd,
    )
    if untracked.returncode != 0:
        return tracked_diff

    untracked_files = [f for f in untracked.stdout.splitlines() if f]
    if not untracked_files:
        return tracked_diff

    synthetic: list[str] = []
    for filename in untracked_files:
        file_path = work_dir / filename
        try:
            content = file_path.read_text(errors="replace")
            lines = content.splitlines()
            header = [
                f"diff --git a/{filename} b/{filename}",
                "new file mode 100644",
                "--- /dev/null",
                f"+++ b/{filename}",
                f"@@ -0,0 +1,{len(lines)} @@",
            ]
            synthetic.append("\n".join(header + [f"+{line}" for line in lines]))
        except OSError:
            synthetic.append(
                f"diff --git a/{filename} b/{filename}\nnew file mode 100644\n"
                f"(binary or unreadable — presence flagged for review)"
            )

    return "\n".join([tracked_diff] + synthetic)


def handle_claude_code_stop(
    reviewer: Optional[SecurityReviewer] = None,
    cwd: Optional[Path] = None,
) -> None:
    """Claude Code Stop hook entry point.

    Exit 0  → PERMIT  — session ends normally.
    Exit 2  → DENY    — blocks the stop; structured JSON on stdout is fed back
                         to Claude as context so it can auto-remediate.
    (Exit 1 would let the session proceed, which is fail-open — avoid it.)
    """
    try:
        json.load(sys.stdin)
    except Exception:
        pass

    # Pre-flight: surface a clear setup error rather than a confusing fail-closed verdict
    if not _api_key_available():
        config = load_config()
        if config.fail_open:
            sys.exit(0)
        print(json.dumps({"decision": "block", "reason": _MISSING_KEY_MESSAGE}))
        sys.exit(2)

    diff = get_diff(cwd=cwd)
    if reviewer is None:
        reviewer = SecurityReviewer()

    verdict = reviewer.run_review(diff)
    reviewer.write_audit_log(verdict, diff)

    if verdict["decision"] == "PERMIT":
        sys.exit(0)
    else:
        # Structured output: Claude Code Stop hook reads {"decision":"block","reason":"..."}
        # and feeds the reason back into the active session as context.
        print(json.dumps({
            "decision": "block",
            "reason": format_remediation_prompt(verdict),
        }))
        sys.exit(2)
