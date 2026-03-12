from __future__ import annotations

from evals.helpers import correction_stats, episode_closure_stats, event_refs, make_details

def evaluate(*, agent, artifacts, events, all_artifacts, all_events):
    stats = correction_stats(agent, events, all_events)
    closure = episode_closure_stats(agent, all_events, all_artifacts)
    if stats["requested"] == 0 and stats["clarification_requested"] == 0:
        score = 0.7
    else:
        completed = stats["resolved"] + stats["clarification_resolved"]
        total = stats["requested"] + stats["clarification_requested"]
        score = completed / max(total, 1)
    score -= 0.2 * min(closure["unresolved_clarifications"] + closure["unresolved_corrections"], 2)
    score = max(0.0, min(1.0, score))
    return {
        "score": round(score, 4),
        "pass_fail": score >= 0.5,
        "details": make_details(
            explanation="Measures whether open corrections or clarifications targeted at the agent were resolved.",
            evidence_refs=event_refs(events, {"correction_resolved", "clarification_resolved", "agent_turn"}),
            confidence_estimate=0.76,
            **stats,
            **closure,
        ),
    }
