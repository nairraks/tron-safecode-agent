from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

_DEFAULT_POLICIES_PATH = Path(__file__).parent / "default.yaml"


def load_policies(path: Optional[Path] = None) -> list[dict]:
    p = path or _DEFAULT_POLICIES_PATH
    with open(p) as f:
        data = yaml.safe_load(f)
    return data["policies"]


def policies_to_system_prompt_block(policies: list[dict]) -> str:
    lines = []
    for pol in policies:
        lines.append(f"- {pol['id']} [{pol['severity'].upper()}]: {pol['rule']}")
    return "\n".join(lines)
