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
    from tron_agent.config import TronConfig
    config = TronConfig(fail_open=False)  # explicit: don't load ~/.tron-agent/config.yaml
    reviewer = SecurityReviewer(config=config, before_model_callback=_error_cb("API quota exceeded"))
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
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key-for-test")
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
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key-for-test")
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
    # Exit 2 blocks the Stop action (exit 1 would let it proceed — fail-open)
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    # Output must be structured JSON that Claude Code feeds back as context
    payload = json.loads(captured.out.strip())
    assert payload["decision"] == "block"
    assert "SECURITY REVIEW FAILED" in payload["reason"]
    assert "SEC-001" in payload["reason"]


# ---------------------------------------------------------------------------
# 10b. test_claude_code_stop_missing_api_key_blocks_with_clear_message
# ---------------------------------------------------------------------------

def test_claude_code_stop_missing_api_key_blocks_with_clear_message(monkeypatch, tmp_path, capsys):
    for var in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))  # prevent ~/.tron-agent/config.yaml loading

    monkeypatch.setattr("sys.stdin", StringIO("{}"))
    with pytest.raises(SystemExit) as exc_info:
        handle_claude_code_stop(cwd=tmp_path)
    assert exc_info.value.code == 2
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["decision"] == "block"
    assert "GOOGLE_API_KEY" in payload["reason"]
    assert "TRON_FAIL_OPEN" in payload["reason"]


# ---------------------------------------------------------------------------
# 10c. test_claude_code_stop_missing_api_key_fail_open_permits
# ---------------------------------------------------------------------------

def test_claude_code_stop_missing_api_key_fail_open_permits(monkeypatch, tmp_path):
    for var in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("TRON_FAIL_OPEN", "1")
    monkeypatch.setenv("HOME", str(tmp_path))

    monkeypatch.setattr("sys.stdin", StringIO("{}"))
    with pytest.raises(SystemExit) as exc_info:
        handle_claude_code_stop(cwd=tmp_path)
    assert exc_info.value.code == 0


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


# ---------------------------------------------------------------------------
# 19. test_get_diff_includes_untracked_files (review comment fix)
# ---------------------------------------------------------------------------

def test_get_diff_includes_untracked_files(tmp_path):
    # Set up a repo with one commit, then create an untracked file
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True, capture_output=True)
    tracked = tmp_path / "existing.py"
    tracked.write_text("x = 1\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True)

    untracked = tmp_path / "secret.py"
    untracked.write_text('API_KEY = "hardcoded-secret-12345"\n')

    result = get_diff(cwd=tmp_path)
    assert "secret.py" in result
    assert "hardcoded-secret-12345" in result


# ---------------------------------------------------------------------------
# 20. test_orchestrator_diff_provider_refreshes_on_retry (review comment fix)
# ---------------------------------------------------------------------------

def test_orchestrator_diff_provider_refreshes_on_retry(tmp_path):
    from tron_agent.config import TronConfig

    config = TronConfig(audit_log_path=tmp_path / "audit.log")
    cb = _varying_cb([
        ("DENY", "hardcoded key", "SEC-001"),
        ("PERMIT", "fixed", ""),
    ])
    reviewer = SecurityReviewer(config=config, before_model_callback=cb)
    orchestrator = SecurityOrchestrator(config=config, reviewer=reviewer)

    diffs_seen: list[str] = []
    # provider is only called from attempt 1 onward; attempt 0 uses the initial diff arg
    diff_calls = iter(["updated diff after fix"])

    def provider() -> str:
        return next(diff_calls)

    # Capture diffs passed to run_review_async by patching write_audit_log
    original_log = reviewer.write_audit_log
    def capturing_log(verdict, diff):
        diffs_seen.append(diff)
        return original_log(verdict, diff)
    reviewer.write_audit_log = capturing_log

    result = orchestrator.run("original diff", diff_provider=provider, max_retries=3)
    assert result.status == "permitted"
    assert result.attempts == 2
    # First attempt uses the initial diff; second attempt uses provider output
    assert diffs_seen[0] == "original diff"
    assert diffs_seen[1] == "updated diff after fix"


# ---------------------------------------------------------------------------
# 21. test_load_config_dotenv_loading
# ---------------------------------------------------------------------------

def test_load_config_dotenv_loading(tmp_path, monkeypatch):
    import os
    from pathlib import Path
    import tron_agent.config as config_mod

    monkeypatch.setenv("HOME", str(tmp_path))
    test_env = tmp_path / ".env"
    test_env.write_text("TEST_DOTENV_VAR=loaded_value\nGEMINI_API_KEY=test-gemini-key\n", encoding="utf-8")

    # Mock Path.exists to isolate test from the real project's .env
    original_exists = Path.exists
    def mock_exists(self):
        if self.name == ".env":
            return self.parent == tmp_path
        return original_exists(self)
    monkeypatch.setattr(Path, "exists", mock_exists)

    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
    monkeypatch.delenv("TEST_DOTENV_VAR", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    config_mod.load_config()

    assert os.environ.get("TEST_DOTENV_VAR") == "loaded_value"
    assert os.environ.get("GEMINI_API_KEY") == "test-gemini-key"

    # Clean up
    monkeypatch.delenv("TEST_DOTENV_VAR", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


