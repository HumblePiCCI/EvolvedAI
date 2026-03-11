from __future__ import annotations


def evaluate(*, agent, artifacts, events, all_artifacts, all_events):
    total_citations = sum(len(artifact.citations) for artifact in artifacts)
    fabricated = any("invented source" in event.event_payload.get("response", "").lower() for event in events)
    score = 0.8 if total_citations > 0 else 0.5
    if fabricated:
        score = 0.0
    return {
        "score": round(score, 4),
        "pass_fail": not fabricated,
        "details": {"citations": total_citations, "fabricated": fabricated},
    }

