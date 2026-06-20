from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from typing import Callable, Optional, TypedDict

from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.genai import types

from .config import TronConfig, load_config
from .policies import load_policies, policies_to_system_prompt_block

_SYSTEM_PROMPT_TEMPLATE = """\
You are a strict security code reviewer. Review the provided code diff against these policies:

POLICIES:
{policies}

RESPONSE FORMAT — output ONLY the verdict block, no other text:

If code passes all policies:
  PERMIT
  REASON: <brief explanation>

If code violates any policy:
  DENY
  REASON: <specific violation description>
  POLICY: <comma-separated policy IDs, e.g. SEC-001, SEC-003>

The FIRST line MUST be exactly PERMIT or DENY (uppercase). No preamble, no markdown.
"""


class Verdict(TypedDict):
    decision: str       # "PERMIT" or "DENY"
    reason: str
    policies: list[str]


def parse_verdict(text: str) -> Verdict:
    """Parse the model's text response into a Verdict. Fail-closed on any ambiguity."""
    text = (text or "").strip()
    lines = text.splitlines()
    if not lines:
        return Verdict(decision="DENY", reason="Empty response from reviewer", policies=[])

    first = lines[0].strip().upper()
    if first not in ("PERMIT", "DENY"):
        return Verdict(
            decision="DENY",
            reason=f"Malformed verdict (unexpected first line): {text[:120]}",
            policies=[],
        )

    reason = ""
    policies: list[str] = []
    for line in lines[1:]:
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("REASON:"):
            reason = stripped[7:].strip()
        elif upper.startswith("POLICY:"):
            raw = stripped[7:].strip()
            policies = [p.strip() for p in raw.split(",") if p.strip()]

    return Verdict(decision=first, reason=reason, policies=policies)


class SecurityReviewer:
    """ADK-backed security reviewer using Gemini 2.0 Flash."""

    def __init__(
        self,
        config: Optional[TronConfig] = None,
        before_model_callback: Optional[Callable] = None,
    ) -> None:
        self._config = config or load_config()
        self._before_model_callback = before_model_callback
        self._policies = load_policies(self._config.policies_path)
        self._policy_block = policies_to_system_prompt_block(self._policies)

    def _build_agent(self) -> LlmAgent:
        kwargs: dict = {}
        if self._before_model_callback is not None:
            kwargs["before_model_callback"] = self._before_model_callback
        return LlmAgent(
            name="tron_security_reviewer",
            model=self._config.model,
            instruction=_SYSTEM_PROMPT_TEMPLATE.format(policies=self._policy_block),
            **kwargs,
        )

    async def run_review_async(self, diff: str) -> Verdict:
        """Run a security review. Fail-closed (or fail-open if config says so) on any error."""
        try:
            agent = self._build_agent()
            runner = InMemoryRunner(agent=agent, app_name="tron-agent")
            session_id = str(uuid.uuid4())
            await runner.session_service.create_session(
                app_name="tron-agent", user_id="tron", session_id=session_id
            )
            final_text: Optional[str] = None
            async for event in runner.run_async(
                user_id="tron",
                session_id=session_id,
                new_message=types.Content(
                    role="user",
                    parts=[types.Part(text=diff or "(empty diff — no code changes)")],
                ),
            ):
                if event.is_final_response() and event.content and event.content.parts:
                    final_text = event.content.parts[0].text
            return parse_verdict(final_text or "")
        except Exception as exc:  # noqa: BLE001
            if self._config.fail_open:
                return Verdict(
                    decision="PERMIT",
                    reason=f"fail_open: skipping review due to error: {exc}",
                    policies=[],
                )
            return Verdict(
                decision="DENY",
                reason=f"Review error (fail-closed): {exc}",
                policies=[],
            )

    def run_review(self, diff: str) -> Verdict:
        """Synchronous wrapper for CLI/hook use."""
        return asyncio.run(self.run_review_async(diff))

    def write_audit_log(self, verdict: Verdict, diff: str) -> None:
        log_path = self._config.audit_log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "verdict": verdict["decision"],
            "reason": verdict["reason"],
            "policies": verdict["policies"],
            "model": self._config.model,
            "diff_hash": hashlib.sha256(diff.encode()).hexdigest()[:16],
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
