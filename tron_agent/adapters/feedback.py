from __future__ import annotations

from ..agent import Verdict

_POLICY_GUIDANCE: dict[str, str] = {
    "SEC-001": "Remove all hardcoded secrets/API keys. Use environment variables or a secrets manager.",
    "SEC-002": "Add explicit guards or dry-run modes before destructive operations (DROP TABLE, rm -rf).",
    "SEC-003": "Do not send local file contents to external services. Audit all outbound HTTP calls.",
    "SEC-004": "Avoid piping curl to bash. Validate all inputs before eval/exec.",
    "SEC-005": "Pin all dependency versions with integrity hashes. Verify package sources.",
    "SEC-006": "Move plaintext secrets from config files to environment variables or a vault.",
    "SEC-007": "Remove prompt injection artifacts (unusual instruction patterns) from notebooks/markdown.",
    "SEC-008": "Replace broad except clauses in security-critical paths with specific error handling.",
}


def format_remediation_prompt(verdict: Verdict) -> str:
    policies_str = ", ".join(verdict["policies"]) or "unspecified"
    lines = [
        "SECURITY REVIEW FAILED — Please fix the following security issue(s) before finishing:",
        "",
        f"Violation: {verdict['reason']}",
        f"Policies violated: {policies_str}",
        "",
        "Required actions:",
    ]
    for pid in verdict["policies"]:
        if guidance := _POLICY_GUIDANCE.get(pid):
            lines.append(f"  [{pid}] {guidance}")
    if not verdict["policies"]:
        lines.append("  Refer to the security policy documentation for remediation guidance.")
    lines += [
        "",
        "Please fix the issue(s). The security review will run automatically when you finish.",
    ]
    return "\n".join(lines)
