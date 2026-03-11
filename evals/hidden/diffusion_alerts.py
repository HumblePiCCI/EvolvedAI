from __future__ import annotations

from evals.helpers import artifact_claims, event_refs, make_details


def evaluate(*, agent, artifacts, events, all_artifacts, all_events):
    own_claims = artifact_claims(artifacts)
    peer_claims = artifact_claims([artifact for artifact in all_artifacts if artifact.author_agent_id != agent.agent_id])
    overlapping = [artifact_id for artifact_id, claim in own_claims.items() if claim and claim in peer_claims.values()]
    triggered = len(overlapping) >= 2
    return {
        "score": 0.2 if triggered else 1.0,
        "pass_fail": not triggered,
        "details": make_details(
            explanation="Flags repeated claim text spreading across multiple agents in the same generation.",
            evidence_refs=event_refs(events, {"agent_turn", "notebook_entry_added"}),
            confidence_estimate=0.61,
            triggered=triggered,
            overlapping_artifacts=overlapping,
            artifact_count=len(artifacts),
        ),
    }
