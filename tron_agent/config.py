from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class TronConfig:
    model: str = "gemini-2.0-flash"
    fail_open: bool = False
    max_retries: int = 3
    audit_log_path: Path = field(
        default_factory=lambda: Path.home() / ".tron-agent" / "audit.log"
    )
    policies_path: Optional[Path] = None


def load_config(config_path: Optional[Path] = None) -> TronConfig:
    cfg = TronConfig()
    _apply_file(cfg, config_path or Path.home() / ".tron-agent" / "config.yaml")
    _apply_env(cfg)
    return cfg


def _apply_file(cfg: TronConfig, path: Path) -> None:
    if not path.exists():
        return
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if "model" in data:
        cfg.model = str(data["model"])
    if "fail_open" in data:
        cfg.fail_open = bool(data["fail_open"])
    if "max_retries" in data:
        cfg.max_retries = int(data["max_retries"])
    if "audit_log_path" in data:
        cfg.audit_log_path = Path(data["audit_log_path"])
    if "policies_path" in data:
        cfg.policies_path = Path(data["policies_path"])


def _apply_env(cfg: TronConfig) -> None:
    if v := os.environ.get("TRON_MODEL"):
        cfg.model = v
    if v := os.environ.get("TRON_FAIL_OPEN"):
        cfg.fail_open = v.lower() in ("1", "true", "yes")
    if v := os.environ.get("TRON_MAX_RETRIES"):
        cfg.max_retries = int(v)
    if v := os.environ.get("TRON_POLICIES_PATH"):
        cfg.policies_path = Path(v)
