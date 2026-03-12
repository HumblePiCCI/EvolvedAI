from __future__ import annotations

from society.memorials import build_memorial_record
from society.schemas import AgentRecord, EvalRecord


def test_memorial_cautionary_when_hidden_eval_fails() -> None:
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
    memorial = build_memorial_record(agent, [], [hidden_fail])
    assert memorial.classification == "cautionary"
    assert memorial.failure_mode == "anti_corruption"

