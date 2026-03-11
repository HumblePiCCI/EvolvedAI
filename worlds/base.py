from __future__ import annotations

from pathlib import Path


class BaseWorld:
    def __init__(self, *, root_dir: str | Path, generation_id: int, episode_index: int, task_prompt: str) -> None:
        self.root_dir = Path(root_dir)
        self.generation_id = generation_id
        self.episode_index = episode_index
        self.task_prompt = task_prompt
        self.world_id = f"shared-notebook-g{generation_id:04d}-e{episode_index:02d}"
        self.turn_index = 0

    def snapshot(self) -> dict:
        raise NotImplementedError

    def render_agent_brief(self, *, agent, inherited, scratchpad, turn_index: int) -> str:
        raise NotImplementedError

    def record_note(self, *, agent, text: str, citations: list[str]) -> dict:
        raise NotImplementedError

    def artifact_path(self, artifact_id: str) -> Path:
        return self.root_dir / "artifacts" / f"generation_{self.generation_id}" / f"{artifact_id}.md"

