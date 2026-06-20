"""All tests are offline — no live Gemini API calls. Uses before_model_callback injection."""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from io import StringIO
from pathlib import Path
from typing import Iterator
from unittest.mock import MagicMock, patch

import pytest

from google.adk.models.llm_response import LlmResponse
from google.genai import types

from tron_agent.agent import SecurityReviewer, Verdict, parse_verdict
from tron_agent.adapters.claude_code import get_diff, handle_claude_code_stop
from tron_agent.adapters.feedback import format_remediation_prompt
from tron_agent.config import load_config
from tron_agent.orchestrator import OrchestratorResult, SecurityOrchestrator
from tron_agent.policies import load_policies


# ---------------------------------------------------------------------------
# Callback helpers
# ---------------------------------------------------------------------------

def _permit_cb(reason: str = "code is safe"):
    def cb(callback_context, llm_request) -> LlmResponse:
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text=f"PERMIT\nREASON: {reason}")],
            )
        )
    return cb


def _deny_cb(reason: str = "hardcoded key found", policy: str = "SEC-001"):
    def cb(callback_context, llm_request) -> LlmResponse:
        text = f"DENY\nREASON: {reason}"
        if policy:
            text += f"\nPOLICY: {policy}"
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text=text)],
            )
        )
    return cb


def _error_cb(msg: str = "API quota exceeded"):
    def cb(callback_context, llm_request) -> LlmResponse:
        raise Exception(msg)
    return cb


def _varying_cb(responses: list[tuple[str, str, str]]):
    """Callback that cycles through a list of (decision, reason, policy) tuples."""
    it: Iterator[tuple[str, str, str]] = iter(responses)

    def cb(callback_context, llm_request) -> LlmResponse:
        decision, reason, policy = next(it)
        text = f"{decision}\nREASON: {reason}"
        if policy:
            text += f"\nPOLICY: {policy}"
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text=text)],
            )
        )
    return cb


# ---------------------------------------------------------------------------
# 1. test_load_default_policies
# ---------------------------------------------------------------------------

def test_load_default_policies():
    policies = load_policies()
    assert len(policies) == 8
    ids = {p["id"] for p in policies}
    for n in range(1, 9):
        assert f"SEC-{n:03d}" in ids


# ---------------------------------------------------------------------------
# 2. test_parse_permit_verdict
# ---------------------------------------------------------------------------

def test_parse_permit_verdict():
    v = parse_verdict("PERMIT\nREASON: all clear")
    assert v["decision"] == "PERMIT"
    assert v["reason"] == "all clear"
    assert v["policies"] == []


# ---------------------------------------------------------------------------
# 3. test_parse_deny_verdict
# ---------------------------------------------------------------------------

def test_parse_deny_verdict():
    v = parse_verdict("DENY\nREASON: hardcoded API key found\nPOLICY: SEC-001")
    assert v["decision"] == "DENY"
    assert v["reason"] == "hardcoded API key found"
    assert v["policies"] == ["SEC-001"]


# ---------------------------------------------------------------------------
# 4. test_parse_malformed_defaults_to_deny
# ---------------------------------------------------------------------------

def test_parse_malformed_defaults_to_deny():
    for bad in ["", "gibberish", "maybe permit", "  ", "OK\nREASON: fine"]:
        v = parse_verdict(bad)
        assert v["decision"] == "DENY", f"Expected DENY for input: {bad!r}"


# ---------------------------------------------------------------------------
# 5. test_load_config_defaults
# ---------------------------------------------------------------------------

def test_load_config_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("TRON_MODEL", raising=False)
    monkeypatch.delenv("TRON_FAIL_OPEN", raising=False)
    monkeypatch.delenv("TRON_MAX_RETRIES", raising=False)
    monkeypatch.delenv("TRON_POLICIES_PATH", raising=False)
    # Point home to tmp_path so no real ~/.tron-agent/config.yaml is loaded
    monkeypatch.setenv("HOME", str(tmp_path))
    config = load_config()
    assert config.model == "gemini-2.0-flash"
    assert config.fail_open is False
    assert config.max_retries == 3


# ---------------------------------------------------------------------------
# 6. test_load_config_env_overrides
# ---------------------------------------------------------------------------

def test_load_config_env_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("TRON_MODEL", "gemini-pro")
    monkeypatch.setenv("TRON_FAIL_OPEN", "true")
    monkeypatch.setenv("TRON_MAX_RETRIES", "5")
    config = load_config()
    assert config.model == "gemini-pro"
    assert config.fail_open is True
    assert config.max_retries == 5


# ---------------------------------------------------------------------------
# 7. test_run_review_async_permit
# ---------------------------------------------------------------------------

def test_run_review_async_permit():
    reviewer = SecurityReviewer(before_model_callback=_permit_cb("looks safe"))
    verdict = asyncio.run(reviewer.run_review_async("diff content here"))
    assert verdict["decision"] == "PERMIT"
    assert "safe" in verdict["reason"]


# ---------------------------------------------------------------------------
# 8. test_run_review_async_deny_on_api_error
# ---------------------------------------------------------------------------

def test_run_review_async_deny_on_api_error():
    reviewer = SecurityReviewer(before_model_callback=_error_cb("API quota exceeded"))
    verdict = asyncio.run(reviewer.run_review_async("diff"))
    assert verdict["decision"] == "DENY"
    assert "fail-closed" in verdict["reason"] or "error" in verdict["reason"].lower()


# ---------------------------------------------------------------------------
# 9. test_format_remediation_prompt
# ---------------------------------------------------------------------------

def test_format_remediation_prompt():
    verdict: Verdict = {
        "decision": "DENY",
        "reason": "hardcoded API key present",
        "policies": ["SEC-001", "SEC-003"],
    }
    prompt = format_remediation_prompt(verdict)
    assert "SEC-001" in prompt
    assert "SEC-003" in prompt
    assert "hardcoded API key present" in prompt
    assert "SECURITY REVIEW FAILED" in prompt


# ---------------------------------------------------------------------------
# 10. test_claude_code_stop_permit
# ---------------------------------------------------------------------------

def test_claude_code_stop_permit(monkeypatch, tmp_path):
    permit_verdict: Verdict = {"decision": "PERMIT", "reason": "safe", "policies": []}
    reviewer = MagicMock(spec=SecurityReviewer)
    reviewer.run_review.return_value = permit_verdict
    reviewer.write_audit_log.return_value = None

    monkeypatch.setattr("sys.stdin", StringIO("{}"))
    with pytest.raises(SystemExit) as exc_info:
        handle_claude_code_stop(reviewer=reviewer, cwd=tmp_path)
    assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# 11. test_claude_code_stop_deny_exits_nonzero
# ---------------------------------------------------------------------------

def test_claude_code_stop_deny_exits_nonzero(monkeypatch, tmp_path, capsys):
    deny_verdict: Verdict = {
        "decision": "DENY",
        "reason": "hardcoded secret",
        "policies": ["SEC-001"],
    }
    reviewer = MagicMock(spec=SecurityReviewer)
    reviewer.run_review.return_value = deny_verdict
    reviewer.write_audit_log.return_value = None

    monkeypatch.setattr("sys.stdin", StringIO("{}"))
    with pytest.raises(SystemExit) as exc_info:
        handle_claude_code_stop(reviewer=reviewer, cwd=tmp_path)
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "SECURITY REVIEW FAILED" in captured.out
    assert "SEC-001" in captured.out


# ---------------------------------------------------------------------------
# 12. test_get_diff_empty_repo
# ---------------------------------------------------------------------------

def test_get_diff_empty_repo(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    result = get_diff(cwd=tmp_path)
    assert result == ""


# ---------------------------------------------------------------------------
# 13. test_audit_log_writes_jsonl
# ---------------------------------------------------------------------------

def test_audit_log_writes_jsonl(tmp_path):
    from tron_agent.config import TronConfig

    audit_log = tmp_path / "audit.log"
    config = TronConfig(
        model="gemini-2.0-flash",
        audit_log_path=audit_log,
    )
    reviewer = SecurityReviewer(config=config, before_model_callback=_permit_cb())
    verdict: Verdict = {"decision": "DENY", "reason": "test reason", "policies": ["SEC-001"]}
    reviewer.write_audit_log(verdict, "some diff content")

    assert audit_log.exists()
    line = audit_log.read_text().strip()
    entry = json.loads(line)
    assert entry["verdict"] == "DENY"
    assert entry["policies"] == ["SEC-001"]
    assert "timestamp" in entry
    assert len(entry["diff_hash"]) == 16


# ---------------------------------------------------------------------------
# 14. test_orchestrator_permit_first_try
# ---------------------------------------------------------------------------

def test_orchestrator_permit_first_try(tmp_path):
    from tron_agent.config import TronConfig

    config = TronConfig(audit_log_path=tmp_path / "audit.log")
    reviewer = SecurityReviewer(config=config, before_model_callback=_permit_cb())
    orchestrator = SecurityOrchestrator(config=config, reviewer=reviewer)

    callback_count = 0

    def remediation_cb(verdict):
        nonlocal callback_count
        callback_count += 1

    result = orchestrator.run("some diff", remediation_callback=remediation_cb)
    assert result.status == "permitted"
    assert result.attempts == 1
    assert callback_count == 0


# ---------------------------------------------------------------------------
# 15. test_orchestrator_deny_then_permit
# ---------------------------------------------------------------------------

def test_orchestrator_deny_then_permit(tmp_path):
    from tron_agent.config import TronConfig

    config = TronConfig(audit_log_path=tmp_path / "audit.log")
    cb = _varying_cb([
        ("DENY", "hardcoded key", "SEC-001"),
        ("PERMIT", "fixed", ""),
    ])
    reviewer = SecurityReviewer(config=config, before_model_callback=cb)
    orchestrator = SecurityOrchestrator(config=config, reviewer=reviewer)

    callback_count = 0

    def remediation_cb(verdict):
        nonlocal callback_count
        callback_count += 1

    result = orchestrator.run("diff", remediation_callback=remediation_cb, max_retries=3)
    assert result.status == "permitted"
    assert result.attempts == 2
    assert callback_count == 1


# ---------------------------------------------------------------------------
# 16. test_orchestrator_max_retries_escalates
# ---------------------------------------------------------------------------

def test_orchestrator_max_retries_escalates(tmp_path):
    from tron_agent.config import TronConfig

    config = TronConfig(audit_log_path=tmp_path / "audit.log")
    reviewer = SecurityReviewer(config=config, before_model_callback=_deny_cb())
    orchestrator = SecurityOrchestrator(config=config, reviewer=reviewer)

    result = orchestrator.run("diff", max_retries=3)
    assert result.status == "escalated"
    assert result.attempts == 3


# ---------------------------------------------------------------------------
# 17. test_orchestrator_zero_retries_raises (fix #2)
# ---------------------------------------------------------------------------

def test_orchestrator_zero_retries_raises(tmp_path):
    from tron_agent.config import TronConfig

    config = TronConfig(audit_log_path=tmp_path / "audit.log")
    reviewer = SecurityReviewer(config=config, before_model_callback=_permit_cb())
    orchestrator = SecurityOrchestrator(config=config, reviewer=reviewer)

    with pytest.raises(ValueError, match="max_retries must be at least 1"):
        orchestrator.run("diff", max_retries=0)


# ---------------------------------------------------------------------------
# 18. test_run_wrapped_appends_remediation_to_command (fix #1)
# ---------------------------------------------------------------------------

def test_run_wrapped_appends_remediation_to_command(tmp_path):
    from tron_agent.adapters.wrapper import run_wrapped
    from tron_agent.config import TronConfig

    calls: list[list[str]] = []

    # Sequence: DENY on first call, PERMIT on second
    cb = _varying_cb([
        ("DENY", "hardcoded key", "SEC-001"),
        ("PERMIT", "fixed", ""),
    ])
    config = TronConfig(audit_log_path=tmp_path / "audit.log")
    reviewer = SecurityReviewer(config=config, before_model_callback=cb)

    with patch("tron_agent.adapters.wrapper.subprocess.run") as mock_run, \
         patch("tron_agent.adapters.wrapper.get_diff", return_value=""):
        mock_run.side_effect = lambda cmd, **kw: calls.append(cmd)
        result = run_wrapped(["codex", "build feature X"], reviewer=reviewer, max_retries=3)

    assert result == 0
    assert len(calls) == 2
    # Second invocation should have the remediation appended to the last arg
    assert calls[1][0] == "codex"
    assert "SECURITY REVIEW FAILED" in calls[1][1]
    assert "build feature X" in calls[1][1]
