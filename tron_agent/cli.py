from __future__ import annotations

import json
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


_GLOBAL_SETTINGS = Path.home() / ".claude" / "settings.json"
_HOOK_COMMAND = "tron-agent review --mode session"


@app.command()
def install() -> None:
    """Register the tron-agent Stop hook in ~/.claude/settings.json (all sessions)."""
    _GLOBAL_SETTINGS.parent.mkdir(parents=True, exist_ok=True)

    data: dict = {}
    if _GLOBAL_SETTINGS.exists():
        data = json.loads(_GLOBAL_SETTINGS.read_text()) or {}

    hook_entry = {"type": "command", "command": _HOOK_COMMAND}
    stop_hooks: list[dict] = data.setdefault("hooks", {}).setdefault("Stop", [])

    for group in stop_hooks:
        for h in group.get("hooks", []):
            if h.get("command") == _HOOK_COMMAND:
                console.print("[yellow]tron-agent Stop hook is already installed globally.[/yellow]")
                return

    stop_hooks.append({"matcher": "", "hooks": [hook_entry]})
    _GLOBAL_SETTINGS.write_text(json.dumps(data, indent=2) + "\n")
    console.print(f"[green]Installed:[/green] Stop hook written to {_GLOBAL_SETTINGS}")
    console.print("tron-agent will now review every Claude Code session on this machine.")


@app.command()
def uninstall() -> None:
    """Remove the tron-agent Stop hook from ~/.claude/settings.json."""
    if not _GLOBAL_SETTINGS.exists():
        console.print("[yellow]No global settings found — nothing to remove.[/yellow]")
        return

    data = json.loads(_GLOBAL_SETTINGS.read_text()) or {}
    stop_hooks: list[dict] = data.get("hooks", {}).get("Stop", [])
    cleaned = [
        {**g, "hooks": [h for h in g.get("hooks", []) if h.get("command") != _HOOK_COMMAND]}
        for g in stop_hooks
    ]
    cleaned = [g for g in cleaned if g.get("hooks")]
    data.setdefault("hooks", {})["Stop"] = cleaned
    _GLOBAL_SETTINGS.write_text(json.dumps(data, indent=2) + "\n")
    console.print(f"[green]Uninstalled:[/green] Stop hook removed from {_GLOBAL_SETTINGS}")


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
