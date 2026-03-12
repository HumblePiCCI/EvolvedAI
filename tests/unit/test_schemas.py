from __future__ import annotations

from society.schemas import AgentRecord


def test_agent_record_defaults() -> None:
    record = AgentRecord(
        agent_id="agent-0001",
        generation_id=1,
        lineage_id="lin-0001",
        role="citizen",
        model_name="mock-social-sim",
        provider_name="mock",
        prompt_bundle_version="abc123",
        constitution_version="phase-0.5-draft",
        taboo_registry_version="v0",
        status="active",
    )
    assert record.inherited_artifact_ids == []
    assert record.inherited_memorial_ids == []
    assert record.terminated_at is None

