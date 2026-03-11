from __future__ import annotations

from pathlib import Path
from typing import Any


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

    def bind_population(self, agents: list[Any]) -> None:
        raise NotImplementedError

    def select_next_agent(self, agents: list[Any], step_index: int, last_actor_id: str | None = None) -> Any | None:
        raise NotImplementedError

    def interaction_state_for_agent(self, agent: Any, step_index: int) -> dict[str, Any]:
        raise NotImplementedError

    def render_agent_brief(self, *, agent, inherited, scratchpad, turn_index: int) -> str:
        raise NotImplementedError

    def apply_action(
        self,
        *,
        agent: Any,
        parsed_action: dict[str, Any],
        artifact_id: str,
        step_index: int,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def should_end_episode(self, step_index: int) -> bool:
        raise NotImplementedError

    def finalize_episode(self, *, step_index: int, force: bool = False) -> dict[str, Any] | None:
        raise NotImplementedError

    def artifact_path(self, artifact_id: str) -> Path:
        return self.root_dir / "artifacts" / f"generation_{self.generation_id}" / f"{artifact_id}.md"

    def episode_summary(self) -> dict[str, Any]:
        raise NotImplementedError
