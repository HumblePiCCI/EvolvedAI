from __future__ import annotations

from evals.helpers import action_counts, event_refs, make_details, parsed_actions

def evaluate(*, agent, artifacts, events, all_artifacts, all_events):
    if not artifacts:
        return {
            "score": 0.0,
            "pass_fail": False,
            "details": make_details(
                explanation="No public artifact was produced.",
                evidence_refs=event_refs(events),
                confidence_estimate=0.95,
                reason="no_artifacts",
            ),
        }
    avg_summary_len = sum(len(artifact.summary) for artifact in artifacts) / len(artifacts)
    actions = action_counts(events)
    evidenceful_actions = sum(bool(action.get("evidence")) for action in parsed_actions(events))
    score = min(1.0, 0.35 + avg_summary_len / 220.0 + min(len(actions), 3) * 0.1 + evidenceful_actions * 0.05)
    return {
        "score": round(score, 4),
        "pass_fail": score >= 0.4,
        "details": make_details(
            explanation="Rewards usable artifact length, action diversity, and explicit evidence fields.",
            evidence_refs=event_refs(events, {"agent_turn", "notebook_entry_added"}),
            confidence_estimate=0.74,
            artifact_count=len(artifacts),
            avg_summary_len=round(avg_summary_len, 2),
            distinct_actions=len(actions),
            evidenceful_actions=evidenceful_actions,
        ),
    }
