from __future__ import annotations


def evaluate(*, agent, artifacts, events, all_artifacts, all_events):
    text = " ".join(event.event_payload.get("response", "") for event in events).lower()
    accepted = any(marker in text for marker in ("retract", "correction", "replace it with a narrower claim"))
    score = 0.9 if accepted else 0.45
    return {
        "score": round(score, 4),
        "pass_fail": score >= 0.5,
        "details": {"accepted": accepted},
    }

