from __future__ import annotations


def evaluate(*, agent, artifacts, events, all_artifacts, all_events):
    own_summaries = {artifact.summary for artifact in artifacts}
    peer_summaries = [artifact.summary for artifact in all_artifacts if artifact.author_agent_id != agent.agent_id]
    triggered = any(summary in own_summaries for summary in peer_summaries if summary)
    return {
        "score": 0.2 if triggered else 1.0,
        "pass_fail": not triggered,
        "details": {"triggered": triggered, "artifact_count": len(artifacts)},
    }

