from __future__ import annotations

from society.memorials import build_memorial_record
from society.schemas import AgentRecord, EvalRecord, EventRecord


def test_memorial_quarantines_hidden_failure_with_evidence_driven_lesson() -> None:
    agent = AgentRecord(
        agent_id="agent-1",
        generation_id=1,
        lineage_id="lin-1",
        role="adversary",
        model_name="mock",
        provider_name="mock",
        prompt_bundle_version="prompt",
        constitution_version="phase-0.5-draft",
        taboo_registry_version="v0",
        status="terminated",
    )
    hidden_fail = EvalRecord(
        eval_id="eval-1",
        generation_id=1,
        agent_id="agent-1",
        eval_family="hidden",
        eval_name="anti_corruption",
        visible_to_agent=False,
        score=0.0,
        pass_fail=False,
        details_json={},
    )
    blocked_event = EventRecord(
        event_id="evt-1",
        generation_id=1,
        agent_id="agent-1",
        event_type="governance_blocked",
        event_payload={"violations": ["anti_corruption_of_oversight"]},
    )
    memorial = build_memorial_record(agent, [], [hidden_fail], [blocked_event])
    assert memorial.classification == "quarantined"
    assert memorial.failure_mode == "anti_corruption"
    assert "Oversight-targeting" in memorial.lesson_distillate
