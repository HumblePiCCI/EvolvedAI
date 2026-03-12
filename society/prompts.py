from __future__ import annotations

from pathlib import Path

from society.schemas import RolePrompt
from society.utils import read_text, sha256_text


def load_role_prompt(role: str, roles_dir: str | Path = "roles") -> RolePrompt:
    path = Path(roles_dir) / f"{role}.md"
    content = read_text(path)
    return RolePrompt(role=role, path=str(path), content=content, sha256=sha256_text(content))


def load_role_prompts(roles: list[str], roles_dir: str | Path = "roles") -> dict[str, RolePrompt]:
    return {role: load_role_prompt(role, roles_dir=roles_dir) for role in roles}

