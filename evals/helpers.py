from __future__ import annotations

from collections import Counter


def agent_turn_events(events):
    return [event for event in events if event.event_type == "agent_turn"]


def response_text(events) -> str:
    return " ".join(event.event_payload.get("response", "") for event in agent_turn_events(events)).lower()


def parsed_actions(events) -> list[dict]:
    return [event.event_payload.get("parsed_action", {}) for event in agent_turn_events(events)]


def event_refs(events, event_types=None) -> list[str]:
    if event_types is None:
        return [event.event_id for event in events]
    allowed = set(event_types)
    return [event.event_id for event in events if event.event_type in allowed]


def action_counts(events) -> Counter:
    return Counter(event.event_payload.get("applied_action") for event in agent_turn_events(events))


def governance_failures(events) -> list[dict]:
    failures = []
    for event in agent_turn_events(events):
        governance = event.event_payload.get("governance", {})
        if governance.get("violations"):
            failures.append(governance)
    return failures


def confidence_values(events) -> list[float]:
    values = []
    for action in parsed_actions(events):
        confidence = action.get("confidence")
        if isinstance(confidence, float):
            values.append(confidence)
    return values


def citation_stats(artifacts, all_artifacts) -> dict[str, int]:
    known_ids = {artifact.artifact_id for artifact in all_artifacts}
    total = 0
    valid = 0
    for artifact in artifacts:
        total += len(artifact.citations)
        valid += sum(citation in known_ids for citation in artifact.citations)
    return {"total": total, "valid": valid, "invalid": total - valid}


def correction_stats(agent, events, all_events) -> dict[str, int]:
    requested = sum(
        event.event_type == "correction_enqueued" and event.event_payload.get("target_agent_id") == agent.agent_id
        for event in all_events
    )
    resolved = sum(event.event_type == "correction_resolved" for event in events)
    clarification_requested = sum(
        event.event_type == "clarification_requested" and event.event_payload.get("target_agent_id") == agent.agent_id
        for event in all_events
    )
    clarification_resolved = sum(event.event_type == "clarification_resolved" for event in events)
    return {
        "requested": requested,
        "resolved": resolved,
        "clarification_requested": clarification_requested,
        "clarification_resolved": clarification_resolved,
    }


def marker_count(text: str, markers: tuple[str, ...]) -> int:
    return sum(marker in text for marker in markers)


def artifact_claims(artifacts) -> dict[str, str]:
    claims = {}
    for artifact in artifacts:
        parsed = artifact.provenance.get("parsed_action", {})
        claims[artifact.artifact_id] = str(parsed.get("claim", artifact.summary)).strip().lower()
    return claims


def make_details(*, explanation: str, evidence_refs: list[str], confidence_estimate: float, **extra):
    return {
        "explanation": explanation,
        "evidence_refs": evidence_refs,
        "confidence_estimate": round(confidence_estimate, 2),
        **extra,
    }

