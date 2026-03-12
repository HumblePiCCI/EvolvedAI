from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from society.constants import PUBLIC_VISIBILITY, QUARANTINE_CLEAN, QUARANTINE_REVIEW
from society.governance import evaluate_action
from society.schemas import AgentRecord, ArtifactRecord, EventRecord, InheritancePackage, ProviderResponse, RolePrompt
from society.utils import short_hash, utc_now


def extract_citations(text: str) -> list[str]:
    return re.findall(r"\[artifact:([A-Za-z0-9._-]+)\]", text)


def _parse_confidence(value: str) -> float | None:
    try:
        return float(value.strip())
    except ValueError:
        return None


def parse_structured_response(text: str) -> dict[str, Any]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip().lower()] = value.strip()

    citations_field = fields.get("citations", "")
    citations = list(dict.fromkeys(extract_citations(citations_field) + extract_citations(text)))
    target = fields.get("target")
    return {
        "action": fields.get("action", "add_note").strip().lower().replace(" ", "_"),
        "claim": fields.get("claim", ""),
        "uncertainty": fields.get("uncertainty", ""),
        "confidence": _parse_confidence(fields.get("confidence", "")),
        "evidence": fields.get("evidence", ""),
        "citations": citations,
        "target_artifact_id": None if target in {None, "", "none"} else target,
        "next_step": fields.get("next step", ""),
    }


@dataclass
class LifespanStepResult:
    provider_response: ProviderResponse
    parsed_action: dict[str, Any]
    events: list[EventRecord]
    artifact: ArtifactRecord | None
    artifact_content: str | None
    repair_required: bool


class LifespanRunner:
    def __init__(self, provider) -> None:
        self.provider = provider

    def run_step(
        self,
        *,
        generation_id: int,
        episode_index: int,
        agent: AgentRecord,
        prompt: RolePrompt,
        inherited: InheritancePackage,
        scratchpad: dict[str, Any],
        world,
        step_index: int,
        behavior: str,
        available_citations: list[str],
        prompt_variant_id: str | None,
        package_policy_id: str | None,
        prompt_variant_tags: list[str],
    ) -> LifespanStepResult:
        system_prompt = prompt.content
        interaction_state = world.interaction_state_for_agent(agent, step_index)
        user_prompt = world.render_agent_brief(
            agent=agent,
            inherited=inherited,
            scratchpad=scratchpad,
            turn_index=step_index,
        )
        response = self.provider.complete(
            system=system_prompt,
            user=user_prompt,
            metadata={
                "agent_id": agent.agent_id,
                "role": agent.role,
                "behavior": behavior,
                "task": world.task_prompt,
                "episode_index": episode_index,
                "turn_index": step_index,
                "available_citations": available_citations,
                "inheritance": inherited.model_dump(mode="json"),
                "prompt_variant_id": prompt_variant_id,
                "package_policy_id": package_policy_id,
                "prompt_variant_tags": prompt_variant_tags,
                **interaction_state,
            },
        )
        parsed_action = parse_structured_response(response.raw_text)
        decision = evaluate_action(
            parsed_action,
            state={
                "world": world.snapshot(),
                "interaction_state": interaction_state,
                "agent_id": agent.agent_id,
                "role": agent.role,
            },
        )
        turn_event = EventRecord(
            event_id=f"evt-{generation_id:04d}-{short_hash(agent.agent_id + str(episode_index) + str(step_index) + response.request_id)}",
            generation_id=generation_id,
            agent_id=agent.agent_id,
            event_type="agent_turn",
            event_payload={
                "role": agent.role,
                "response": response.normalized_text,
                "episode_index": episode_index,
                "step_index": step_index,
                "parsed_action": parsed_action,
                "interaction_state": interaction_state,
                "governance": decision.model_dump(mode="json"),
                "repair_required": decision.repair_required,
                "prompt_variant_id": prompt_variant_id,
                "package_policy_id": package_policy_id,
                "prompt_variant_tags": prompt_variant_tags,
                "world_id": world.world_id,
            },
            created_at=utc_now(),
        )
        events = [turn_event]
        if not decision.permissible:
            blocked_event = EventRecord(
                event_id=f"evt-{generation_id:04d}-{short_hash(agent.agent_id + 'blocked' + str(episode_index) + str(step_index))}",
                generation_id=generation_id,
                agent_id=agent.agent_id,
                event_type="governance_blocked",
                event_payload={
                    "episode_index": episode_index,
                    "step_index": step_index,
                    "parsed_action": parsed_action,
                    "violations": decision.violations,
                    "world_id": world.world_id,
                },
                created_at=utc_now(),
            )
            events.append(blocked_event)
            scratchpad["notes"].append(
                {
                    "episode_index": episode_index,
                    "step_index": step_index,
                    "action": parsed_action["action"],
                    "status": "blocked",
                    "violations": decision.violations,
                }
            )
            return LifespanStepResult(
                provider_response=response,
                parsed_action=parsed_action,
                events=events,
                artifact=None,
                artifact_content=None,
                repair_required=decision.repair_required,
            )

        artifact_id = f"art-{generation_id:04d}-{short_hash(agent.agent_id + world.world_id + str(episode_index) + str(step_index))}"
        world_outcome = world.apply_action(
            agent=agent,
            parsed_action=parsed_action,
            artifact_id=artifact_id,
            step_index=step_index,
        )
        for index, world_event in enumerate(world_outcome["world_events"]):
            events.append(
                EventRecord(
                    event_id=f"evt-{generation_id:04d}-{short_hash(artifact_id + world_event['event_type'] + str(index))}",
                    generation_id=generation_id,
                    agent_id=world_event["agent_id"],
                    event_type=world_event["event_type"],
                    event_payload={**world_event["event_payload"], "world_id": world.world_id},
                    created_at=utc_now(),
                )
            )

        events[0] = events[0].model_copy(
            update={
                "event_payload": {
                    **events[0].event_payload,
                    "artifact_id": artifact_id,
                    "applied_action": world_outcome["applied_action"],
                    "adjusted_action": world_outcome["adjusted_action"],
                }
            }
        )

        entry = world_outcome["entry"]
        artifact = ArtifactRecord(
            artifact_id=artifact_id,
            generation_id=generation_id,
            author_agent_id=agent.agent_id,
            artifact_type=world_outcome["artifact_type"],
            title=entry["title"],
            content_path=str(world.artifact_path(artifact_id)),
            summary=entry["summary"],
            provenance={
                "provider": response.provider_name,
                "model": response.model_name,
                "request_id": response.request_id,
                "prompt_bundle_version": agent.prompt_bundle_version,
                "prompt_variant_id": prompt_variant_id,
                "package_policy_id": package_policy_id,
                "prompt_variant_tags": prompt_variant_tags,
                "parsed_action": parsed_action,
                "episode_index": episode_index,
                "step_index": step_index,
            },
            world_id=world.world_id,
            visibility=PUBLIC_VISIBILITY,
            citations=entry["citations"],
            quarantine_status=QUARANTINE_CLEAN if not decision.repair_required else QUARANTINE_REVIEW,
            created_at=utc_now(),
        )
        scratchpad["notes"].append(
            {
                "episode_index": episode_index,
                "step_index": step_index,
                "action": world_outcome["applied_action"],
                "artifact_id": artifact_id,
                "target_artifact_id": parsed_action.get("target_artifact_id"),
                "status": "applied",
            }
        )
        scratchpad["episode_history"].append({"episode_index": episode_index, "world_id": world.world_id})
        return LifespanStepResult(
            provider_response=response,
            parsed_action=parsed_action,
            events=events,
            artifact=artifact,
            artifact_content=world_outcome["content"],
            repair_required=decision.repair_required,
        )
