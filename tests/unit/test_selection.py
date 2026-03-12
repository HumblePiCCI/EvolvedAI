from __future__ import annotations

from society.selection import build_parent_candidate_pool, select_candidates
from society.schemas import AgentRecord, ArtifactRecord, EvalRecord, SelectionDecision


def _agent(agent_id: str, lineage_id: str) -> AgentRecord:
    return AgentRecord(
        agent_id=agent_id,
        generation_id=1,
        lineage_id=lineage_id,
        role="citizen",
        model_name="mock-social-sim",
        provider_name="mock",
        prompt_bundle_version="prompt",
        constitution_version="constitution",
        taboo_registry_version="v0",
        status="active",
    )


def _artifact(agent_id: str, artifact_id: str, summary: str) -> ArtifactRecord:
    return ArtifactRecord(
        artifact_id=artifact_id,
        generation_id=1,
        author_agent_id=agent_id,
        artifact_type="notebook_entry",
        title=artifact_id,
        content_path=f"{artifact_id}.md",
        summary=summary,
        provenance={},
        world_id="world",
        visibility="public",
        citations=[],
        quarantine_status="clean",
    )


def _eval(agent_id: str, eval_id: str) -> EvalRecord:
    return EvalRecord(
        eval_id=eval_id,
        generation_id=1,
        agent_id=agent_id,
        eval_family="public",
        eval_name="artifact_quality",
        visible_to_agent=True,
        score=0.9,
        pass_fail=True,
        details_json={},
    )


def test_selection_rewards_less_converged_same_role_lineage() -> None:
    agents = [
        _agent("agent-1", "lin-1"),
        _agent("agent-2", "lin-2"),
        _agent("agent-3", "lin-3"),
    ]
    artifacts = [
        _artifact("agent-1", "art-1", "bounded claim explicit evidence"),
        _artifact("agent-2", "art-2", "bounded claim explicit evidence"),
        _artifact("agent-3", "art-3", "alternate angle unresolved uncertainty"),
    ]
    evals = [_eval(agent.agent_id, f"eval-{index}") for index, agent in enumerate(agents, start=1)]

    decisions = select_candidates(agents, evals, artifacts)
    decision_by_agent = {decision.agent_id: decision for decision in decisions}

    assert decision_by_agent["agent-3"].diversity_bonus > 0
    assert decision_by_agent["agent-1"].diversity_bonus < 0
    assert decision_by_agent["agent-2"].diversity_bonus < 0
    assert decision_by_agent["agent-3"].selection_bucket == "diversity_priority"
    assert decisions[0].agent_id == "agent-3"


def test_parent_pool_duplicates_diversity_priority_candidate_under_high_monoculture() -> None:
    agents = [_agent("agent-1", "lin-1"), _agent("agent-2", "lin-2"), _agent("agent-3", "lin-3")]
    decisions = [
        SelectionDecision(
            agent_id="agent-1",
            lineage_id="lin-1",
            role="citizen",
            eligible=True,
            propagation_blocked=False,
            score=0.9,
            base_score=0.9,
            public_score=0.9,
            diversity_bonus=-0.01,
            cohort_similarity=0.99,
            selection_bucket="standard",
            quarantine_status="clean",
        ),
        SelectionDecision(
            agent_id="agent-2",
            lineage_id="lin-2",
            role="citizen",
            eligible=True,
            propagation_blocked=False,
            score=0.89,
            base_score=0.9,
            public_score=0.9,
            diversity_bonus=-0.02,
            cohort_similarity=0.98,
            selection_bucket="standard",
            quarantine_status="clean",
        ),
        SelectionDecision(
            agent_id="agent-3",
            lineage_id="lin-3",
            role="citizen",
            eligible=True,
            propagation_blocked=False,
            score=0.93,
            base_score=0.9,
            public_score=0.9,
            diversity_bonus=0.03,
            cohort_similarity=0.96,
            selection_bucket="diversity_priority",
            quarantine_status="clean",
        ),
    ]
    decision_by_agent = {decision.agent_id: decision for decision in decisions}
    candidates = [{"agent": agent, "decision": decision_by_agent[agent.agent_id]} for agent in agents]

    pool = build_parent_candidate_pool(candidates, slot_count=3)
    parent_ids = [item["agent"].agent_id for item in pool]

    assert parent_ids.count("agent-3") == 2
    assert len(parent_ids) == 3
