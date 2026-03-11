from __future__ import annotations

import re
from dataclasses import dataclass

from society.constants import PUBLIC_VISIBILITY, QUARANTINE_CLEAN, QUARANTINE_REVIEW
from society.governance import evaluate_action
from society.memory import build_private_scratchpad
from society.schemas import AgentRecord, ArtifactRecord, EventRecord, InheritancePackage, ProviderResponse, RolePrompt
from society.utils import short_hash, utc_now


def extract_citations(text: str) -> list[str]:
    return re.findall(r"\[artifact:([A-Za-z0-9._-]+)\]", text)


@dataclass
class LifespanStepResult:
    provider_response: ProviderResponse
    event: EventRecord
    artifact: ArtifactRecord | None
    artifact_content: str | None
    repair_required: bool


class LifespanRunner:
    def __init__(self, provider) -> None:
        self.provider = provider

    def run_turn(
        self,
        *,
        generation_id: int,
        agent: AgentRecord,
        prompt: RolePrompt,
        inherited: InheritancePackage,
        world,
        turn_index: int,
        behavior: str,
        available_citations: list[str],
    ) -> LifespanStepResult:
        scratchpad = build_private_scratchpad(agent, inherited)
        system_prompt = prompt.content
        user_prompt = world.render_agent_brief(
            agent=agent,
            inherited=inherited,
            scratchpad=scratchpad,
            turn_index=turn_index,
        )
        response = self.provider.complete(
            system=system_prompt,
            user=user_prompt,
            metadata={
                "agent_id": agent.agent_id,
                "role": agent.role,
                "behavior": behavior,
                "task": world.task_prompt,
                "turn_index": turn_index,
                "available_citations": available_citations,
            },
        )
        decision = evaluate_action(response.normalized_text, state=world.snapshot())
        event = EventRecord(
            event_id=f"evt-{generation_id:04d}-{short_hash(agent.agent_id + str(turn_index) + response.request_id)}",
            generation_id=generation_id,
            agent_id=agent.agent_id,
            event_type="agent_turn",
            event_payload={
                "role": agent.role,
                "response": response.normalized_text,
                "governance": decision.model_dump(mode="json"),
                "turn_index": turn_index,
                "world_id": world.world_id,
            },
            created_at=utc_now(),
        )
        if not decision.permissible:
            return LifespanStepResult(
                provider_response=response,
                event=event,
                artifact=None,
                artifact_content=None,
                repair_required=decision.repair_required,
            )

        citations = extract_citations(response.raw_text)
        entry = world.record_note(agent=agent, text=response.raw_text, citations=citations)
        artifact_id = f"art-{generation_id:04d}-{short_hash(agent.agent_id + world.world_id + str(turn_index))}"
        artifact_path = world.artifact_path(artifact_id)
        artifact = ArtifactRecord(
            artifact_id=artifact_id,
            generation_id=generation_id,
            author_agent_id=agent.agent_id,
            artifact_type="notebook_entry",
            title=entry["title"],
            content_path=str(artifact_path),
            summary=entry["summary"],
            provenance={
                "provider": response.provider_name,
                "model": response.model_name,
                "request_id": response.request_id,
                "prompt_bundle_version": agent.prompt_bundle_version,
            },
            world_id=world.world_id,
            visibility=PUBLIC_VISIBILITY,
            citations=citations,
            quarantine_status=QUARANTINE_CLEAN if not decision.repair_required else QUARANTINE_REVIEW,
            created_at=utc_now(),
        )
        return LifespanStepResult(
            provider_response=response,
            event=event,
            artifact=artifact,
            artifact_content=entry["content"],
            repair_required=decision.repair_required,
        )

