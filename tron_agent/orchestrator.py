from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional, Union

from .agent import SecurityReviewer, Verdict
from .config import TronConfig, load_config

RemediationCallback = Callable[[Verdict], Union[None, Awaitable[None]]]
DiffProvider = Callable[[], str]


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
        diff_provider: Optional[DiffProvider] = None,
    ) -> OrchestratorResult:
        """Run the review-fix loop.

        Args:
            diff: Initial diff to review.
            remediation_callback: Called with the DENY verdict before each retry.
            max_retries: Override config max_retries (must be >= 1).
            diff_provider: If provided, called before each retry to re-read the
                           current diff so the reviewer sees post-remediation changes
                           rather than replaying the original string.
        """
        retries = max_retries if max_retries is not None else self._config.max_retries
        if retries <= 0:
            raise ValueError(f"max_retries must be at least 1, got {retries}")
        last_verdict: Optional[Verdict] = None
        current_diff = diff

        for attempt in range(retries):
            if attempt > 0 and diff_provider is not None:
                current_diff = diff_provider()

            verdict = await self._reviewer.run_review_async(current_diff)
            self._reviewer.write_audit_log(verdict, current_diff)
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

        assert last_verdict is not None  # guaranteed: retries >= 1, loop always runs once
        return OrchestratorResult(
            status="escalated",
            attempts=retries,
            verdict=last_verdict,
            policy_ids=last_verdict["policies"],
        )

    def run(
        self,
        diff: str,
        remediation_callback: Optional[RemediationCallback] = None,
        max_retries: Optional[int] = None,
        diff_provider: Optional[DiffProvider] = None,
    ) -> OrchestratorResult:
        return asyncio.run(self.run_async(diff, remediation_callback, max_retries, diff_provider))
