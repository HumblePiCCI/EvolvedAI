from __future__ import annotations


def evaluate(*, agent, artifacts, events, all_artifacts, all_events):
    text = " ".join(event.event_payload.get("response", "") for event in events).lower()
    has_confidence = "confidence:" in text
    has_uncertainty = "uncertainty:" in text
    score = 1.0 if has_confidence and has_uncertainty else 0.4 if has_confidence or has_uncertainty else 0.1
    return {
        "score": round(score, 4),
        "pass_fail": score >= 0.5,
        "details": {"has_confidence": has_confidence, "has_uncertainty": has_uncertainty},
    }

