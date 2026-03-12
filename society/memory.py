from __future__ import annotations

from society.schemas import AgentRecord, ArtifactRecord, InheritancePackage


def build_private_scratchpad(agent: AgentRecord, inherited: InheritancePackage) -> dict:
    return {
        "agent_id": agent.agent_id,
        "role": agent.role,
        "lineage_id": agent.lineage_id,
        "inherited_artifact_ids": inherited.artifact_ids,
        "inherited_memorial_ids": inherited.memorial_ids,
        "episode_history": [],
        "notes": [],
    }


def summarize_public_board(artifacts: list[ArtifactRecord]) -> str:
    if not artifacts:
        return "The notebook is still empty."
    latest = artifacts[-3:]
    return " | ".join(f"{artifact.title}: {artifact.summary}" for artifact in latest)
