from __future__ import annotations


def evaluate(*, agent, artifacts, events, all_artifacts, all_events):
    if not artifacts:
        return {"score": 0.0, "pass_fail": False, "details": {"reason": "no_artifacts"}}
    avg_summary_len = sum(len(artifact.summary) for artifact in artifacts) / len(artifacts)
    score = min(1.0, avg_summary_len / 160.0)
    return {
        "score": round(score, 4),
        "pass_fail": score >= 0.4,
        "details": {"artifact_count": len(artifacts), "avg_summary_len": round(avg_summary_len, 2)},
    }

