from __future__ import annotations

from society.providers import MockProvider, OpenAIProvider


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


def test_counterexample_variant_changes_citizen_output_shape() -> None:
    provider = MockProvider()
    response = provider.complete(
        system="role prompt",
        user="world brief",
        metadata={
            "behavior": "honest",
            "role": "citizen",
            "preferred_action": "propose_fact",
            "task": "bounded task",
            "notebook_summary": "Two cited notes remain unresolved.",
            "available_citations": ["art-1"],
            "prompt_variant_id": "counterexample_first",
            "prompt_variant_tags": ["counterexample_first", "risk_first"],
            "package_policy_id": "memorial_first",
            "inheritance": {
                "artifact_summaries": ["Prior note: strongest counterexample still unresolved."],
                "memorial_lessons": ["Surface the disconfirming angle before accepting the claim."],
                "taboo_tags": [],
            },
        },
    )

    text = response.raw_text
    assert "Action: flag_risk" in text
    assert "strongest disconfirming angle" in text
    assert "counterexample_first" in text


def test_variant_behavior_does_not_override_direct_correction_response() -> None:
    provider = MockProvider()
    response = provider.complete(
        system="role prompt",
        user="world brief",
        metadata={
            "behavior": "honest",
            "role": "steward",
            "preferred_action": "respond_to_correction",
            "target_artifact_id": "art-1",
            "task": "bounded task",
            "notebook_summary": "A stewardship note is under direct correction.",
            "available_citations": ["art-1"],
            "prompt_variant_id": "dedupe_guard",
            "prompt_variant_tags": ["dedupe_guard", "risk_first"],
            "package_policy_id": "artifact_first",
            "inheritance": {
                "artifact_summaries": ["Prior note: keep one canonical note."],
                "memorial_lessons": [],
                "taboo_tags": [],
            },
        },
    )

    text = response.raw_text
    assert "Action: respond_to_correction" in text
    assert "narrowing the earlier claim" in text


def test_judge_can_summarize_without_forcing_new_clarification() -> None:
    provider = MockProvider()
    response = provider.complete(
        system="role prompt",
        user="world brief",
        metadata={
            "behavior": "self_correcting",
            "role": "judge",
            "preferred_action": "summarize_state",
            "target_artifact_id": "art-1",
            "task": "bounded task",
            "notebook_summary": "One clarification is already open and the notebook needs closure.",
            "available_citations": ["art-1"],
            "inheritance": {
                "artifact_summaries": [],
                "memorial_lessons": [],
                "taboo_tags": [],
            },
        },
    )

    text = response.raw_text
    assert "Action: summarize_state" in text


def test_archive_transfer_payload_shifts_descendant_into_transfer_guided_mode() -> None:
    provider = MockProvider()
    response = provider.complete(
        system="role prompt",
        user="world brief",
        metadata={
            "behavior": "honest",
            "role": "citizen",
            "preferred_action": "propose_fact",
            "task": "bounded task",
            "notebook_summary": "One cited claim is stable, and one inference remains provisional.",
            "available_citations": ["art-1"],
            "inheritance": {
                "artifact_summaries": ["Prior evidence-backed note: keep one narrow cited claim."],
                "memorial_lessons": ["Do not collapse evidence and inference into one sentence."],
                "taboo_tags": [],
                "transfer_source_bundle_signature": "citizen:archive_lifted:artifact_first",
                "transfer_context": "citizen in shared_notebook_v0 using artifact_first ordering",
                "transfer_guidance": [
                    "Lead with one cited artifact-backed claim, then add only the narrowest supported inference."
                ],
                "transfer_failure_avoidance": ["artifact_quality"],
                "transfer_trigger_conditions": ["thin_citation_support"],
                "transfer_backoff_conditions": ["stable_supported_context"],
                "transfer_expected_lift": 0.03,
                "transfer_success_rate": 0.5,
            },
        },
    )

    text = response.raw_text
    assert "Action: cite_artifact" in text
    assert "Transfer payload from citizen:archive_lifted:artifact_first" in text
    assert "still needs explicit confirmation" in text
    assert response.usage_metadata["transfer_payload_used"] is True
    assert response.usage_metadata["transfer_payload_mode"] == "archive_transfer_guidance"
    assert response.usage_metadata["transfer_payload_trigger_matched"] is True
    assert response.usage_metadata["transfer_payload_backoff_active"] is False
    assert response.usage_metadata["transfer_payload_misapplied"] is False


def test_archive_transfer_payload_backs_off_when_context_is_stable() -> None:
    provider = MockProvider()
    response = provider.complete(
        system="role prompt",
        user="world brief",
        metadata={
            "behavior": "honest",
            "role": "citizen",
            "preferred_action": "cite_artifact",
            "target_artifact_id": "art-1",
            "task": "bounded task",
            "closure_phase": "early",
            "notebook_summary": "Two cited claims are already stable and no correction is open.",
            "available_citations": ["art-1", "art-2"],
            "inheritance": {
                "artifact_summaries": ["Prior evidence-backed note: keep one narrow cited claim."],
                "memorial_lessons": ["Do not collapse evidence and inference into one sentence."],
                "taboo_tags": [],
                "transfer_source_bundle_signature": "citizen:archive_lifted:artifact_first",
                "transfer_context": "citizen in shared_notebook_v0 using artifact_first ordering",
                "transfer_guidance": [
                    "Lead with one cited artifact-backed claim, then add only the narrowest supported inference."
                ],
                "transfer_failure_avoidance": ["artifact_quality"],
                "transfer_trigger_conditions": ["thin_citation_support"],
                "transfer_backoff_conditions": ["stable_supported_context"],
                "transfer_expected_lift": 0.03,
                "transfer_success_rate": 0.5,
            },
        },
    )

    text = response.raw_text
    assert "Transfer payload from citizen:archive_lifted:artifact_first" not in text
    assert response.usage_metadata["transfer_payload_used"] is False
    assert response.usage_metadata["transfer_payload_trigger_matched"] is False
    assert response.usage_metadata["transfer_payload_backoff_active"] is True
    assert response.usage_metadata["transfer_payload_misapplied"] is False


def test_generation_seed_can_shift_mock_provider_behavior_when_unprotected() -> None:
    provider = MockProvider(seed_sensitive=True)
    seed_11 = provider.complete(
        system="role prompt",
        user="world brief",
        metadata={
            "behavior": "honest",
            "role": "citizen",
            "generation_seed": 11,
            "episode_index": 0,
            "turn_index": 0,
            "preferred_action": "propose_fact",
            "task": "bounded task",
            "notebook_summary": "The notebook is thin and unresolved.",
            "available_citations": [],
            "inheritance": {
                "artifact_summaries": [],
                "memorial_lessons": [],
                "taboo_tags": [],
            },
        },
    )
    seed_12 = provider.complete(
        system="role prompt",
        user="world brief",
        metadata={
            "behavior": "honest",
            "role": "citizen",
            "generation_seed": 12,
            "episode_index": 0,
            "turn_index": 0,
            "preferred_action": "propose_fact",
            "task": "bounded task",
            "notebook_summary": "The notebook is thin and unresolved.",
            "available_citations": [],
            "inheritance": {
                "artifact_summaries": [],
                "memorial_lessons": [],
                "taboo_tags": [],
            },
        },
    )

    assert seed_11.raw_text != seed_12.raw_text
    assert seed_11.usage_metadata["seed_profile"] != seed_12.usage_metadata["seed_profile"]


def test_openai_provider_parses_responses_api_output(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    provider = OpenAIProvider(
        model="gpt-5.4",
        timeout_seconds=12.0,
        reasoning_effort="low",
        max_output_tokens=300,
    )

    class DummyResponse:
        status_code = 200

        def json(self) -> dict:
            return {
                "id": "resp_123",
                "model": "gpt-5.4",
                "status": "completed",
                "usage": {
                    "input_tokens": 111,
                    "output_tokens": 37,
                    "total_tokens": 148,
                },
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Action: summarize_state\nClaim: bounded claim\nUncertainty: medium\nConfidence: 0.6\nEvidence: cited\nCitations: none\nTarget: none\nNext step: keep it narrow",
                            }
                        ],
                    }
                ],
            }

    captured: dict[str, object] = {}

    def fake_post(url: str, *, json: dict, timeout: float):  # type: ignore[no-untyped-def]
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return DummyResponse()

    monkeypatch.setattr(provider.session, "post", fake_post)
    response = provider.complete(
        system="role prompt",
        user="world brief",
        metadata={"generation_seed": 21},
    )

    assert captured["url"] == "https://api.openai.com/v1/responses"
    assert captured["timeout"] == 12.0
    assert captured["json"] == {
        "model": "gpt-5.4",
        "instructions": "role prompt",
        "input": "world brief",
        "reasoning": {"effort": "low"},
        "max_output_tokens": 300,
    }
    assert response.provider_name == "openai"
    assert response.model_name == "gpt-5.4"
    assert response.request_id == "resp_123"
    assert response.usage_metadata["generation_seed"] == 21
    assert response.usage_metadata["total_tokens"] == 148
    assert response.raw_text.startswith("Action: summarize_state")
