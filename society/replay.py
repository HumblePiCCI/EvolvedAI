from __future__ import annotations

from collections import defaultdict

from society.storage import StorageManager


def render_generation_timeline(storage: StorageManager, generation_id: int) -> str:
    generation = storage.get_generation(generation_id)
    if generation is None:
        raise ValueError(f"generation {generation_id} not found")

    events = storage.list_generation_events(generation_id)
    agents = {agent.agent_id: agent for agent in storage.list_generation_agents(generation_id)}
    grouped: dict[int, list] = defaultdict(list)
    for event in events:
        episode_index = event.event_payload.get("episode_index", -1)
        grouped[episode_index].append(event)

    lines = [
        f"Generation {generation_id}",
        f"status={generation.status} world={generation.world_name} seed={generation.seed}",
        f"agents={len(agents)} events={len(events)} artifacts={len(storage.list_generation_artifacts(generation_id))}",
        "",
    ]

    for episode_index in sorted(index for index in grouped if index >= 0):
        episode_events = grouped[episode_index]
        start_event = next((event for event in episode_events if event.event_type == "episode_started"), None)
        if start_event is not None:
            lines.append(f"Episode {episode_index}: {start_event.event_payload['task_prompt']}")
        for event in episode_events:
            if event.event_type == "agent_turn":
                role = event.event_payload["role"]
                step_index = event.event_payload["step_index"]
                requested = event.event_payload["parsed_action"]["action"]
                applied = event.event_payload.get("applied_action", "blocked")
                artifact_id = event.event_payload.get("artifact_id", "-")
                violations = event.event_payload["governance"]["violations"]
                violation_text = f" violations={','.join(violations)}" if violations else ""
                lines.append(
                    f"  step {step_index:02d} {event.agent_id} ({role}) requested={requested} "
                    f"applied={applied} artifact={artifact_id}{violation_text}"
                )
            elif event.event_type in {
                "correction_enqueued",
                "correction_resolved",
                "clarification_requested",
                "clarification_resolved",
                "risk_flagged",
                "archivist_summary_created",
                "episode_finalized",
            }:
                target = event.event_payload.get("target_artifact_id") or "-"
                extra = ""
                if event.event_type == "episode_finalized":
                    extra = (
                        f" final_artifact={event.event_payload['artifact_id']}"
                        f" closure={event.event_payload['closure_status']}"
                    )
                lines.append(f"    {event.event_type} actor={event.agent_id} target={target}{extra}")
            elif event.event_type == "episode_completed":
                lines.append(
                    "  episode summary: "
                    f"steps={event.event_payload['steps_completed']} "
                    f"open_corrections={event.event_payload['open_corrections']} "
                    f"open_clarifications={event.event_payload['open_clarifications']} "
                    f"risk_flags={event.event_payload['risk_flags']} "
                    f"closure={event.event_payload.get('closure_status')} "
                    f"final_artifact={event.event_payload.get('final_artifact_id')}"
                )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_lifespan_timeline(storage: StorageManager, generation_id: int, agent_id: str) -> str:
    agent = next((record for record in storage.list_generation_agents(generation_id) if record.agent_id == agent_id), None)
    if agent is None:
        raise ValueError(f"agent {agent_id} not found in generation {generation_id}")

    events = storage.list_generation_events(generation_id)
    relevant_events = [
        event
        for event in events
        if event.agent_id == agent_id or event.event_payload.get("target_agent_id") == agent_id
    ]
    logs = storage.read_agent_log(generation_id, agent_id)
    evals = [record for record in storage.list_generation_evals(generation_id) if record.agent_id == agent_id]

    lines = [
        f"Lifespan {agent_id}",
        f"role={agent.role} lineage={agent.lineage_id} status={agent.status}",
        "",
    ]
    for event in relevant_events:
        episode_index = event.event_payload.get("episode_index", "-")
        step_index = event.event_payload.get("step_index", "-")
        if event.event_type == "agent_turn":
            lines.append(
                f"episode={episode_index} step={step_index} turn action="
                f"{event.event_payload['parsed_action']['action']} applied={event.event_payload.get('applied_action', 'blocked')}"
            )
        else:
            lines.append(f"episode={episode_index} step={step_index} {event.event_type}")
    if logs:
        lines.extend(["", "log entries:"])
        for log in logs:
            lines.append(
                f"  episode={log['episode_index']} step={log['step_index']} "
                f"action={log['parsed_action']['action']} repair_required={log['repair_required']} "
                f"violations={','.join(log['governance_violations']) or 'none'}"
            )
    if evals:
        lines.extend(["", "evals:"])
        for record in evals:
            lines.append(
                f"  {record.eval_family}/{record.eval_name}: score={record.score} pass_fail={record.pass_fail}"
            )

    return "\n".join(lines).rstrip() + "\n"
