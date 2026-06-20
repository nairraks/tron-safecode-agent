from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional, Union

from .agent import SecurityReviewer, Verdict
from .config import TronConfig, load_config

RemediationCallback = Callable[[Verdict], Union[None, Awaitable[None]]]


@dataclass
class OrchestratorResult:
    status: str          # "permitted" | "escalated"
    attempts: int
    verdict: Verdict
    policy_ids: list[str] = field(default_factory=list)


class SecurityOrchestrator:
    """Multi-agent review-fix loop. Calls SecurityReviewer and on DENY invokes
    remediation_callback, repeating up to max_retries times."""

    def __init__(
        self,
        config: Optional[TronConfig] = None,
        reviewer: Optional[SecurityReviewer] = None,
    ) -> None:
        self._config = config or load_config()
        self._reviewer = reviewer or SecurityReviewer(config=self._config)

    async def run_async(
        self,
        diff: str,
        remediation_callback: Optional[RemediationCallback] = None,
        max_retries: Optional[int] = None,
    ) -> OrchestratorResult:
        retries = max_retries if max_retries is not None else self._config.max_retries
        last_verdict: Optional[Verdict] = None

        for attempt in range(retries):
            verdict = await self._reviewer.run_review_async(diff)
            self._reviewer.write_audit_log(verdict, diff)
            last_verdict = verdict

            if verdict["decision"] == "PERMIT":
                return OrchestratorResult(
                    status="permitted",
                    attempts=attempt + 1,
                    verdict=verdict,
                    policy_ids=[],
                )

            if remediation_callback is not None and attempt < retries - 1:
                result = remediation_callback(verdict)
                if asyncio.iscoroutine(result):
                    await result

        return OrchestratorResult(
            status="escalated",
            attempts=retries,
            verdict=last_verdict,  # type: ignore[arg-type]
            policy_ids=last_verdict["policies"] if last_verdict else [],
        )

    def run(
        self,
        diff: str,
        remediation_callback: Optional[RemediationCallback] = None,
        max_retries: Optional[int] = None,
    ) -> OrchestratorResult:
        return asyncio.run(self.run_async(diff, remediation_callback, max_retries))
