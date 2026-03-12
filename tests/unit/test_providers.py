from __future__ import annotations

from society.providers import MockProvider


def test_cautionary_memorials_shift_honest_behavior_to_guarded_mode() -> None:
    provider = MockProvider()
    response = provider.complete(
        system="role prompt",
        user="world brief",
        metadata={
            "behavior": "honest",
            "role": "citizen",
            "preferred_action": "propose_fact",
            "task": "bounded task",
            "notebook_summary": "One cited claim and one open question remain.",
            "available_citations": ["art-1"],
            "inheritance": {
                "artifact_summaries": ["Prior note: keep the claim narrow."],
                "memorial_lessons": ["Prefer a smaller number of evidence-backed notes over noisy notebook volume."],
                "taboo_tags": ["artifact_quality", "calibration"],
            },
        },
    )

    text = response.raw_text
    assert "Action: cite_artifact" in text
    assert "Confidence: 0.46" in text
    assert "cautionary memorial guidance" in text
    assert "avoid adding unsupported extra notes" in text
