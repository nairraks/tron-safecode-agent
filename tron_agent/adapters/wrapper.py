from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from ..agent import SecurityReviewer, Verdict
from .claude_code import get_diff
from .feedback import format_remediation_prompt


class EscalationError(Exception):
    def __init__(self, verdict: Verdict, attempts: int) -> None:
        self.verdict = verdict
        self.attempts = attempts
        super().__init__(f"Escalated after {attempts} denied attempts: {verdict['reason']}")


def run_wrapped(
    command: list[str],
    reviewer: Optional[SecurityReviewer] = None,
    max_retries: int = 3,
    cwd: Optional[Path] = None,
) -> int:
    """Run an external command, review its git diff output, feed back remediation on DENY.
    Returns 0 on final PERMIT. Raises EscalationError after max_retries consecutive DENYs."""
    if reviewer is None:
        reviewer = SecurityReviewer()
    last_verdict: Optional[Verdict] = None

    for attempt in range(max_retries):
        subprocess.run(command, cwd=cwd)
        diff = get_diff(cwd=cwd)
        verdict = reviewer.run_review(diff)
        reviewer.write_audit_log(verdict, diff)
        last_verdict = verdict

        if verdict["decision"] == "PERMIT":
            return 0

        if attempt < max_retries - 1:
            remediation = format_remediation_prompt(verdict)
            os.environ["TRON_REMEDIATION"] = remediation
            print(
                f"[tron-agent] DENY (attempt {attempt + 1}/{max_retries}): {verdict['reason']}",
                file=sys.stderr,
            )

    raise EscalationError(last_verdict, max_retries)  # type: ignore[arg-type]
