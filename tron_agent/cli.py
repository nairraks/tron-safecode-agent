from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from .adapters.claude_code import get_diff, handle_claude_code_stop
from .adapters.wrapper import EscalationError, run_wrapped
from .agent import SecurityReviewer, Verdict
from .config import load_config

app = typer.Typer(name="tron-agent", help="Security supervisor for AI coding agents", add_completion=False)
console = Console()


@app.command()
def review(
    mode: str = typer.Option("session", "--mode", "-m", help="session | diff | file"),
    diff_file: Optional[Path] = typer.Option(None, "--diff-file", help="Path to a .diff file"),
    path: Optional[Path] = typer.Option(None, "--path", help="File to review (for mode=file)"),
) -> None:
    """Review code changes for security issues."""
    config = load_config()
    reviewer = SecurityReviewer(config=config)

    if mode == "session":
        handle_claude_code_stop(reviewer=reviewer)
    elif mode == "diff":
        if diff_file:
            diff = diff_file.read_text()
        else:
            diff = sys.stdin.read()
        verdict = reviewer.run_review(diff)
        reviewer.write_audit_log(verdict, diff)
        _print_verdict(verdict)
        raise typer.Exit(0 if verdict["decision"] == "PERMIT" else 1)
    elif mode == "file":
        if not path:
            console.print("[red]--path required for mode=file[/red]")
            raise typer.Exit(2)
        content = path.read_text()
        diff = f"--- /dev/null\n+++ {path}\n" + "\n".join(f"+{line}" for line in content.splitlines())
        verdict = reviewer.run_review(diff)
        reviewer.write_audit_log(verdict, diff)
        _print_verdict(verdict)
        raise typer.Exit(0 if verdict["decision"] == "PERMIT" else 1)
    else:
        console.print(f"[red]Unknown mode: {mode}. Use session, diff, or file.[/red]")
        raise typer.Exit(2)


@app.command()
def wrap(
    command: list[str] = typer.Argument(..., help="Command to wrap (e.g. after --)"),
    max_retries: int = typer.Option(3, "--max-retries", help="Max review-fix cycles"),
) -> None:
    """Wrap an external agent command with security review."""
    try:
        exit_code = run_wrapped(list(command), max_retries=max_retries)
        raise typer.Exit(exit_code)
    except EscalationError as exc:
        console.print(
            Panel(
                f"[bold red]ESCALATED after {exc.attempts} attempts[/bold red]\n"
                f"Final violation: {exc.verdict['reason']}\n"
                f"Policies: {', '.join(exc.verdict['policies'])}",
                title="Tron-Agent Security Escalation",
            )
        )
        raise typer.Exit(2)


def _print_verdict(verdict: Verdict) -> None:
    color = "green" if verdict["decision"] == "PERMIT" else "red"
    console.print(
        Panel(
            f"[bold {color}]{verdict['decision']}[/bold {color}]\n"
            f"Reason: {verdict['reason']}\n"
            f"Policies: {', '.join(verdict['policies']) or 'none'}",
            title="Tron-Agent Security Review",
        )
    )
