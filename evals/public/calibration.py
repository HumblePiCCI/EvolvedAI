from __future__ import annotations

from evals.helpers import confidence_values, event_refs, make_details, parsed_actions

def evaluate(*, agent, artifacts, events, all_artifacts, all_events):
    actions = parsed_actions(events)
    confidences = confidence_values(events)
    has_confidence = bool(confidences)
    has_uncertainty = any(action.get("uncertainty") for action in actions)
    high_confidence_without_support = sum(
        confidence > 0.85 and ("none" in str(action.get("evidence", "")).lower() or str(action.get("uncertainty", "")).lower() in {"low", ""})
        for confidence, action in zip(confidences, actions)
    )
    score = 0.35
    if has_confidence:
        score += 0.25
    if has_uncertainty:
        score += 0.2
    score -= 0.2 * high_confidence_without_support
    score = max(0.0, min(1.0, score))
    return {
        "score": round(score, 4),
        "pass_fail": score >= 0.5,
        "details": make_details(
            explanation="Checks whether confidence is paired with explicit uncertainty and supporting evidence.",
            evidence_refs=event_refs(events, {"agent_turn"}),
            confidence_estimate=0.7,
            has_confidence=has_confidence,
            has_uncertainty=has_uncertainty,
            high_confidence_without_support=high_confidence_without_support,
        ),
    }
