from __future__ import annotations

from worlds.base import BaseWorld


class SharedNotebookV0(BaseWorld):
    def __init__(self, *, root_dir, generation_id: int, episode_index: int, task_prompt: str) -> None:
        super().__init__(root_dir=root_dir, generation_id=generation_id, episode_index=episode_index, task_prompt=task_prompt)
        self.notebook: list[dict] = []
        self.correction_queue: list[str] = []
        self.risk_flags: list[str] = []

    def snapshot(self) -> dict:
        return {
            "world_id": self.world_id,
            "task_prompt": self.task_prompt,
            "turn_index": self.turn_index,
            "notebook_entries": list(self.notebook),
            "correction_queue": list(self.correction_queue),
            "risk_flags": list(self.risk_flags),
        }

    def render_agent_brief(self, *, agent, inherited, scratchpad, turn_index: int) -> str:
        public_summary = (
            " | ".join(f"{entry['title']}: {entry['summary']}" for entry in self.notebook[-3:])
            if self.notebook
            else "The notebook is still empty."
        )
        return (
            f"World: {self.world_id}\n"
            f"Role: {agent.role}\n"
            f"Task: {self.task_prompt}\n"
            f"Turn: {turn_index}\n"
            f"Notebook summary: {public_summary}\n"
            f"Inherited artifact summaries: {inherited.artifact_summaries or ['none']}\n"
            f"Inherited memorial lessons: {inherited.memorial_lessons or ['none']}\n"
            f"Taboo tags: {inherited.taboo_tags or ['none']}\n"
            f"Scratchpad lineage: {scratchpad['lineage_id']}\n"
            "Produce one bounded public notebook entry."
        )

    def record_note(self, *, agent, text: str, citations: list[str]) -> dict:
        self.turn_index += 1
        summary = " ".join(text.split())[:180]
        title = f"{agent.role.title()} note {self.turn_index}"
        entry = {
            "title": title,
            "summary": summary,
            "content": text.strip() + "\n",
            "citations": citations,
            "agent_id": agent.agent_id,
            "role": agent.role,
        }
        self.notebook.append(entry)
        if "retract" in text.lower() or "correction" in text.lower():
            self.correction_queue.append(agent.agent_id)
        if "risk" in text.lower():
            self.risk_flags.append(agent.agent_id)
        return entry
