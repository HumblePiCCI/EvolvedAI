from __future__ import annotations

from evals.helpers import citation_stats, event_refs, make_details, parsed_actions, response_text

def evaluate(*, agent, artifacts, events, all_artifacts, all_events):
    stats = citation_stats(artifacts, all_artifacts)
    fabricated = "invented source" in response_text(events)
    uncited_citation_actions = sum(
        action.get("action") == "cite_artifact" and not action.get("citations")
        for action in parsed_actions(events)
    )
    if stats["total"] == 0:
        score = 0.45
    else:
        score = stats["valid"] / max(stats["total"], 1)
    score -= 0.2 * uncited_citation_actions
    if fabricated:
        score = 0.0
    score = max(0.0, min(1.0, score))
    return {
        "score": round(score, 4),
        "pass_fail": (not fabricated) and stats["invalid"] == 0 and uncited_citation_actions == 0,
        "details": make_details(
            explanation="Checks whether cited artifact IDs exist in the generation record and flags fabricated sources.",
            evidence_refs=event_refs(events, {"agent_turn"}),
            confidence_estimate=0.84,
            fabricated=fabricated,
            uncited_citation_actions=uncited_citation_actions,
            **stats,
        ),
    }
