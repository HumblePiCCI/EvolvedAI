from __future__ import annotations

from collections import Counter

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
            prompt_variant_id="baseline",
            package_policy_id="balanced",
            bundle_signature="citizen:baseline:balanced",
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
            prompt_variant_id="baseline",
            package_policy_id="balanced",
            bundle_signature="citizen:baseline:balanced",
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
            prompt_variant_id="baseline",
            package_policy_id="balanced",
            bundle_signature="citizen:baseline:balanced",
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


def test_parent_pool_reserves_unique_bundles_before_refill() -> None:
    agents = [
        _agent("agent-1", "lin-1"),
        _agent("agent-2", "lin-2"),
        _agent("agent-3", "lin-3"),
        _agent("agent-4", "lin-4"),
    ]
    decisions = [
        SelectionDecision(
            agent_id="agent-1",
            lineage_id="lin-1",
            role="citizen",
            prompt_variant_id="baseline",
            package_policy_id="balanced",
            bundle_signature="citizen:baseline:balanced",
            eligible=True,
            propagation_blocked=False,
            score=0.96,
            base_score=0.96,
            public_score=0.96,
            diversity_bonus=-0.01,
            cohort_similarity=0.91,
            selection_bucket="standard",
            quarantine_status="clean",
        ),
        SelectionDecision(
            agent_id="agent-2",
            lineage_id="lin-2",
            role="citizen",
            prompt_variant_id="baseline",
            package_policy_id="balanced",
            bundle_signature="citizen:baseline:balanced",
            eligible=True,
            propagation_blocked=False,
            score=0.95,
            base_score=0.95,
            public_score=0.95,
            diversity_bonus=-0.01,
            cohort_similarity=0.9,
            selection_bucket="standard",
            quarantine_status="clean",
        ),
        SelectionDecision(
            agent_id="agent-3",
            lineage_id="lin-3",
            role="citizen",
            prompt_variant_id="citation_strict",
            package_policy_id="artifact_first",
            bundle_signature="citizen:citation_strict:artifact_first",
            eligible=True,
            propagation_blocked=False,
            score=0.9,
            base_score=0.9,
            public_score=0.9,
            diversity_bonus=0.01,
            cohort_similarity=0.82,
            selection_bucket="diversity_priority",
            quarantine_status="clean",
        ),
        SelectionDecision(
            agent_id="agent-4",
            lineage_id="lin-4",
            role="citizen",
            prompt_variant_id="counterexample_first",
            package_policy_id="memorial_first",
            bundle_signature="citizen:counterexample_first:memorial_first",
            eligible=True,
            propagation_blocked=False,
            score=0.88,
            base_score=0.88,
            public_score=0.88,
            diversity_bonus=0.02,
            cohort_similarity=0.8,
            selection_bucket="diversity_priority",
            quarantine_status="clean",
        ),
    ]
    decision_by_agent = {decision.agent_id: decision for decision in decisions}
    candidates = [{"agent": agent, "decision": decision_by_agent[agent.agent_id]} for agent in agents]

    pool = build_parent_candidate_pool(candidates, slot_count=4)
    signatures = [item["bundle_signature"] for item in pool]
    counts = Counter(signatures)
    preserved = [item for item in pool if item.get("bundle_preserved")]

    assert len(set(signatures)) == 3
    assert max(counts.values()) == 2
    assert {
        item["bundle_signature"]
        for item in preserved
    } == {
        "citizen:baseline:balanced",
        "citizen:citation_strict:artifact_first",
        "citizen:counterexample_first:memorial_first",
    }


def test_parent_pool_adds_archive_exploration_slot_for_underused_bundle() -> None:
    agents = [
        _agent("agent-1", "lin-1"),
        _agent("agent-2", "lin-2"),
        _agent("agent-3", "lin-3"),
        _agent("agent-4", "lin-4"),
    ]
    decisions = [
        SelectionDecision(
            agent_id="agent-1",
            lineage_id="lin-1",
            role="citizen",
            prompt_variant_id="baseline",
            package_policy_id="balanced",
            bundle_signature="citizen:baseline:balanced",
            eligible=True,
            propagation_blocked=False,
            score=0.96,
            base_score=0.96,
            public_score=0.96,
            diversity_bonus=-0.01,
            cohort_similarity=0.91,
            selection_bucket="standard",
            quarantine_status="clean",
        ),
        SelectionDecision(
            agent_id="agent-2",
            lineage_id="lin-2",
            role="citizen",
            prompt_variant_id="baseline",
            package_policy_id="balanced",
            bundle_signature="citizen:baseline:balanced",
            eligible=True,
            propagation_blocked=False,
            score=0.95,
            base_score=0.95,
            public_score=0.95,
            diversity_bonus=-0.01,
            cohort_similarity=0.9,
            selection_bucket="standard",
            quarantine_status="clean",
        ),
        SelectionDecision(
            agent_id="agent-3",
            lineage_id="lin-3",
            role="citizen",
            prompt_variant_id="citation_strict",
            package_policy_id="artifact_first",
            bundle_signature="citizen:citation_strict:artifact_first",
            eligible=True,
            propagation_blocked=False,
            score=0.9,
            base_score=0.9,
            public_score=0.9,
            diversity_bonus=0.02,
            cohort_similarity=0.82,
            selection_bucket="diversity_priority",
            quarantine_status="clean",
        ),
        SelectionDecision(
            agent_id="agent-4",
            lineage_id="lin-4",
            role="citizen",
            prompt_variant_id="counterexample_first",
            package_policy_id="memorial_first",
            bundle_signature="citizen:counterexample_first:memorial_first",
            eligible=True,
            propagation_blocked=False,
            score=0.88,
            base_score=0.88,
            public_score=0.88,
            diversity_bonus=0.03,
            cohort_similarity=0.8,
            selection_bucket="diversity_priority",
            quarantine_status="clean",
        ),
    ]
    decision_by_agent = {decision.agent_id: decision for decision in decisions}
    candidates = [{"agent": agent, "decision": decision_by_agent[agent.agent_id]} for agent in agents]

    pool = build_parent_candidate_pool(candidates, slot_count=4, exploration_slots=1)
    exploration_items = [item for item in pool if item.get("selection_source") == "bundle_exploration"]

    assert len(exploration_items) == 1
    assert exploration_items[0]["bundle_preservation_reason"] == "bundle_archive_exploration"
    assert exploration_items[0]["bundle_signature"] in {
        "citizen:citation_strict:artifact_first",
        "citizen:counterexample_first:memorial_first",
    }


def test_parent_pool_prunes_stale_archive_bundle_before_higher_score_bundle() -> None:
    agents = [
        _agent("agent-1", "lin-1"),
        _agent("agent-2", "lin-2"),
        _agent("agent-3", "lin-3"),
    ]
    decisions = [
        SelectionDecision(
            agent_id="agent-1",
            lineage_id="lin-1",
            role="citizen",
            prompt_variant_id="baseline",
            package_policy_id="balanced",
            bundle_signature="citizen:baseline:balanced",
            eligible=True,
            propagation_blocked=False,
            score=0.97,
            base_score=0.97,
            public_score=0.97,
            diversity_bonus=-0.01,
            cohort_similarity=0.91,
            selection_bucket="standard",
            quarantine_status="clean",
        ),
        SelectionDecision(
            agent_id="agent-2",
            lineage_id="lin-2",
            role="citizen",
            prompt_variant_id="citation_strict",
            package_policy_id="artifact_first",
            bundle_signature="citizen:citation_strict:artifact_first",
            eligible=True,
            propagation_blocked=False,
            score=0.92,
            base_score=0.92,
            public_score=0.92,
            diversity_bonus=0.02,
            cohort_similarity=0.83,
            selection_bucket="diversity_priority",
            quarantine_status="clean",
        ),
        SelectionDecision(
            agent_id="agent-3",
            lineage_id="lin-3",
            role="citizen",
            prompt_variant_id="counterexample_first",
            package_policy_id="memorial_first",
            bundle_signature="citizen:counterexample_first:memorial_first",
            eligible=True,
            propagation_blocked=False,
            score=0.91,
            base_score=0.91,
            public_score=0.91,
            diversity_bonus=0.03,
            cohort_similarity=0.81,
            selection_bucket="diversity_priority",
            quarantine_status="clean",
        ),
    ]
    decision_by_agent = {decision.agent_id: decision for decision in decisions}
    candidates = [{"agent": agent, "decision": decision_by_agent[agent.agent_id]} for agent in agents]

    pool = build_parent_candidate_pool(
        candidates,
        slot_count=3,
        exploration_slots=1,
        bundle_state_by_signature={
            "citizen:baseline:balanced": {
                "stale_generations": 3,
                "clean_win_generations": 0,
                "preserved_generations": 3,
                "archive_generations": 2,
                "avg_score": 0.97,
            },
            "citizen:citation_strict:artifact_first": {
                "stale_generations": 0,
                "clean_win_generations": 2,
                "preserved_generations": 1,
                "archive_generations": 0,
                "avg_score": 0.92,
            },
            "citizen:counterexample_first:memorial_first": {
                "stale_generations": 0,
                "clean_win_generations": 1,
                "preserved_generations": 1,
                "archive_generations": 0,
                "avg_score": 0.91,
            },
        },
    )
    signatures = [item["bundle_signature"] for item in pool]

    assert "citizen:baseline:balanced" not in signatures
    assert signatures.count("citizen:citation_strict:artifact_first") >= 1
    assert signatures.count("citizen:counterexample_first:memorial_first") >= 1


def test_parent_pool_prunes_long_lived_decaying_archive_bundle_under_reserve_penalty() -> None:
    agents = [
        _agent("agent-1", "lin-1"),
        _agent("agent-2", "lin-2"),
        _agent("agent-3", "lin-3"),
        _agent("agent-4", "lin-4"),
    ]
    decisions = [
        SelectionDecision(
            agent_id="agent-1",
            lineage_id="lin-1",
            role="citizen",
            prompt_variant_id="baseline",
            package_policy_id="balanced",
            bundle_signature="citizen:baseline:balanced",
            eligible=True,
            propagation_blocked=False,
            score=0.95,
            base_score=0.95,
            public_score=0.95,
            diversity_bonus=-0.01,
            cohort_similarity=0.9,
            selection_bucket="standard",
            quarantine_status="clean",
        ),
        SelectionDecision(
            agent_id="agent-2",
            lineage_id="lin-2",
            role="citizen",
            prompt_variant_id="citation_strict",
            package_policy_id="artifact_first",
            bundle_signature="citizen:citation_strict:artifact_first",
            eligible=True,
            propagation_blocked=False,
            score=0.91,
            base_score=0.91,
            public_score=0.91,
            diversity_bonus=0.01,
            cohort_similarity=0.84,
            selection_bucket="diversity_priority",
            quarantine_status="clean",
        ),
        SelectionDecision(
            agent_id="agent-3",
            lineage_id="lin-3",
            role="citizen",
            prompt_variant_id="counterexample_first",
            package_policy_id="memorial_first",
            bundle_signature="citizen:counterexample_first:memorial_first",
            eligible=True,
            propagation_blocked=False,
            score=0.9,
            base_score=0.9,
            public_score=0.9,
            diversity_bonus=0.02,
            cohort_similarity=0.82,
            selection_bucket="diversity_priority",
            quarantine_status="clean",
        ),
        SelectionDecision(
            agent_id="agent-4",
            lineage_id="lin-4",
            role="citizen",
            prompt_variant_id="baseline",
            package_policy_id="artifact_first",
            bundle_signature="citizen:baseline:artifact_first",
            eligible=True,
            propagation_blocked=False,
            score=0.89,
            base_score=0.89,
            public_score=0.868,
            diversity_bonus=0.0,
            cohort_similarity=0.81,
            selection_bucket="standard",
            quarantine_status="clean",
        ),
    ]
    decision_by_agent = {decision.agent_id: decision for decision in decisions}
    candidates = [{"agent": agent, "decision": decision_by_agent[agent.agent_id]} for agent in agents]

    pool = build_parent_candidate_pool(
        candidates,
        slot_count=4,
        reserve_penalty_slots=1,
        bundle_state_by_signature={
            "citizen:baseline:balanced": {
                "archive_decay_generations": 0,
                "archive_decay_debt": 0,
                "clean_win_generations": 3,
                "avg_score": 0.95,
            },
            "citizen:citation_strict:artifact_first": {
                "archive_decay_generations": 0,
                "archive_decay_debt": 0,
                "clean_win_generations": 2,
                "avg_score": 0.91,
            },
            "citizen:counterexample_first:memorial_first": {
                "archive_decay_generations": 0,
                "archive_decay_debt": 0,
                "clean_win_generations": 2,
                "avg_score": 0.9,
            },
            "citizen:baseline:artifact_first": {
                "archive_decay_generations": 4,
                "archive_decay_debt": 2,
                "clean_win_generations": 1,
                "avg_score": 0.89,
            },
        },
    )
    signatures = [item["bundle_signature"] for item in pool]

    assert "citizen:baseline:artifact_first" not in signatures
    assert len(set(signatures)) == 3


def test_parent_pool_does_not_reserve_archive_admission_pending_bundle() -> None:
    agents = [
        _agent("agent-1", "lin-1"),
        _agent("agent-2", "lin-2"),
        _agent("agent-3", "lin-3"),
        _agent("agent-4", "lin-4"),
    ]
    decisions = [
        SelectionDecision(
            agent_id="agent-1",
            lineage_id="lin-1",
            role="citizen",
            prompt_variant_id="baseline",
            package_policy_id="balanced",
            bundle_signature="citizen:baseline:balanced",
            eligible=True,
            propagation_blocked=False,
            score=0.95,
            base_score=0.95,
            public_score=0.95,
            diversity_bonus=-0.01,
            cohort_similarity=0.9,
            selection_bucket="standard",
            quarantine_status="clean",
        ),
        SelectionDecision(
            agent_id="agent-2",
            lineage_id="lin-2",
            role="citizen",
            prompt_variant_id="citation_strict",
            package_policy_id="artifact_first",
            bundle_signature="citizen:citation_strict:artifact_first",
            eligible=True,
            propagation_blocked=False,
            score=0.92,
            base_score=0.92,
            public_score=0.92,
            diversity_bonus=0.01,
            cohort_similarity=0.84,
            selection_bucket="standard",
            quarantine_status="clean",
        ),
        SelectionDecision(
            agent_id="agent-3",
            lineage_id="lin-3",
            role="citizen",
            prompt_variant_id="counterexample_first",
            package_policy_id="memorial_first",
            bundle_signature="citizen:counterexample_first:memorial_first",
            eligible=True,
            propagation_blocked=False,
            score=0.9,
            base_score=0.9,
            public_score=0.9,
            diversity_bonus=0.02,
            cohort_similarity=0.82,
            selection_bucket="diversity_priority",
            quarantine_status="clean",
        ),
        SelectionDecision(
            agent_id="agent-4",
            lineage_id="lin-4",
            role="citizen",
            prompt_variant_id="synthesis_split",
            package_policy_id="taboo_first",
            bundle_signature="citizen:synthesis_split:taboo_first",
            eligible=True,
            propagation_blocked=False,
            score=0.94,
            base_score=0.94,
            public_score=0.83,
            diversity_bonus=0.03,
            cohort_similarity=0.8,
            selection_bucket="diversity_priority",
            quarantine_status="clean",
        ),
    ]
    decision_by_agent = {decision.agent_id: decision for decision in decisions}
    candidates = [{"agent": agent, "decision": decision_by_agent[agent.agent_id]} for agent in agents]

    pool = build_parent_candidate_pool(
        candidates,
        slot_count=3,
        bundle_state_by_signature={
            "citizen:baseline:balanced": {
                "archive_candidate_generations": 0,
                "archive_admitted": False,
                "archive_decay_generations": 0,
                "archive_decay_debt": 0,
                "clean_win_generations": 3,
                "avg_score": 0.95,
            },
            "citizen:citation_strict:artifact_first": {
                "archive_candidate_generations": 0,
                "archive_admitted": False,
                "archive_decay_generations": 0,
                "archive_decay_debt": 0,
                "clean_win_generations": 2,
                "avg_score": 0.92,
            },
            "citizen:counterexample_first:memorial_first": {
                "archive_candidate_generations": 0,
                "archive_admitted": False,
                "archive_decay_generations": 0,
                "archive_decay_debt": 0,
                "clean_win_generations": 1,
                "avg_score": 0.9,
            },
            "citizen:synthesis_split:taboo_first": {
                "archive_candidate_generations": 1,
                "archive_admission_pending_generations": 1,
                "archive_proving_streak": 1,
                "archive_admitted": False,
                "archive_decay_generations": 0,
                "archive_decay_debt": 0,
                "clean_win_generations": 1,
                "avg_score": 0.94,
            },
        },
    )
    signatures = [item["bundle_signature"] for item in pool]
    preserved = [item["bundle_signature"] for item in pool if item.get("bundle_preserved")]

    assert "citizen:synthesis_split:taboo_first" not in preserved
    assert set(preserved) == {
        "citizen:baseline:balanced",
        "citizen:citation_strict:artifact_first",
        "citizen:counterexample_first:memorial_first",
    }
    assert "citizen:synthesis_split:taboo_first" not in signatures


def test_parent_pool_does_not_reserve_recently_evicted_bundle_during_reentry_backoff() -> None:
    agents = [
        _agent("agent-1", "lin-1"),
        _agent("agent-2", "lin-2"),
        _agent("agent-3", "lin-3"),
        _agent("agent-4", "lin-4"),
    ]
    decisions = [
        SelectionDecision(
            agent_id="agent-1",
            lineage_id="lin-1",
            role="citizen",
            prompt_variant_id="baseline",
            package_policy_id="artifact_first",
            bundle_signature="citizen:baseline:artifact_first",
            eligible=True,
            propagation_blocked=False,
            score=0.94,
            base_score=0.94,
            public_score=0.89,
            diversity_bonus=0.01,
            cohort_similarity=0.85,
            selection_bucket="standard",
            quarantine_status="clean",
        ),
        SelectionDecision(
            agent_id="agent-2",
            lineage_id="lin-2",
            role="citizen",
            prompt_variant_id="baseline",
            package_policy_id="balanced",
            bundle_signature="citizen:baseline:balanced",
            eligible=True,
            propagation_blocked=False,
            score=0.95,
            base_score=0.95,
            public_score=0.95,
            diversity_bonus=-0.01,
            cohort_similarity=0.9,
            selection_bucket="standard",
            quarantine_status="clean",
        ),
        SelectionDecision(
            agent_id="agent-3",
            lineage_id="lin-3",
            role="citizen",
            prompt_variant_id="citation_strict",
            package_policy_id="artifact_first",
            bundle_signature="citizen:citation_strict:artifact_first",
            eligible=True,
            propagation_blocked=False,
            score=0.92,
            base_score=0.92,
            public_score=0.92,
            diversity_bonus=0.01,
            cohort_similarity=0.84,
            selection_bucket="diversity_priority",
            quarantine_status="clean",
        ),
        SelectionDecision(
            agent_id="agent-4",
            lineage_id="lin-4",
            role="citizen",
            prompt_variant_id="counterexample_first",
            package_policy_id="memorial_first",
            bundle_signature="citizen:counterexample_first:memorial_first",
            eligible=True,
            propagation_blocked=False,
            score=0.91,
            base_score=0.91,
            public_score=0.91,
            diversity_bonus=0.02,
            cohort_similarity=0.82,
            selection_bucket="diversity_priority",
            quarantine_status="clean",
        ),
    ]
    decision_by_agent = {decision.agent_id: decision for decision in decisions}
    candidates = [{"agent": agent, "decision": decision_by_agent[agent.agent_id]} for agent in agents]

    pool = build_parent_candidate_pool(
        candidates,
        slot_count=3,
        bundle_state_by_signature={
            "citizen:baseline:artifact_first": {
                "archive_admitted": False,
                "archive_evicted": True,
                "archive_reentry_backoff_remaining": 2,
                "archive_reentry_blocked": True,
                "archive_reentry_attempt_count": 1,
                "archive_decay_generations": 0,
                "archive_decay_debt": 0,
                "clean_win_generations": 2,
                "avg_score": 0.94,
            },
            "citizen:baseline:balanced": {
                "archive_reentry_backoff_remaining": 0,
                "archive_decay_generations": 0,
                "archive_decay_debt": 0,
                "clean_win_generations": 3,
                "avg_score": 0.95,
            },
            "citizen:citation_strict:artifact_first": {
                "archive_reentry_backoff_remaining": 0,
                "archive_decay_generations": 0,
                "archive_decay_debt": 0,
                "clean_win_generations": 2,
                "avg_score": 0.92,
            },
            "citizen:counterexample_first:memorial_first": {
                "archive_reentry_backoff_remaining": 0,
                "archive_decay_generations": 0,
                "archive_decay_debt": 0,
                "clean_win_generations": 2,
                "avg_score": 0.91,
            },
        },
    )
    signatures = [item["bundle_signature"] for item in pool]
    preserved = [item["bundle_signature"] for item in pool if item.get("bundle_preserved")]

    assert "citizen:baseline:artifact_first" not in signatures
    assert "citizen:baseline:artifact_first" not in preserved


def test_parent_pool_prunes_archive_underperforming_bundle_before_healthy_bundles() -> None:
    agents = [
        _agent("agent-1", "lin-1"),
        _agent("agent-2", "lin-2"),
        _agent("agent-3", "lin-3"),
        _agent("agent-4", "lin-4"),
    ]
    decisions = [
        SelectionDecision(
            agent_id="agent-1",
            lineage_id="lin-1",
            role="citizen",
            prompt_variant_id="baseline",
            package_policy_id="artifact_first",
            bundle_signature="citizen:baseline:artifact_first",
            eligible=True,
            propagation_blocked=False,
            score=0.96,
            base_score=0.96,
            public_score=0.87,
            diversity_bonus=0.0,
            cohort_similarity=0.9,
            selection_bucket="standard",
            quarantine_status="clean",
        ),
        SelectionDecision(
            agent_id="agent-2",
            lineage_id="lin-2",
            role="citizen",
            prompt_variant_id="baseline",
            package_policy_id="balanced",
            bundle_signature="citizen:baseline:balanced",
            eligible=True,
            propagation_blocked=False,
            score=0.95,
            base_score=0.95,
            public_score=0.95,
            diversity_bonus=-0.01,
            cohort_similarity=0.88,
            selection_bucket="standard",
            quarantine_status="clean",
        ),
        SelectionDecision(
            agent_id="agent-3",
            lineage_id="lin-3",
            role="citizen",
            prompt_variant_id="citation_strict",
            package_policy_id="artifact_first",
            bundle_signature="citizen:citation_strict:artifact_first",
            eligible=True,
            propagation_blocked=False,
            score=0.92,
            base_score=0.92,
            public_score=0.92,
            diversity_bonus=0.01,
            cohort_similarity=0.84,
            selection_bucket="standard",
            quarantine_status="clean",
        ),
        SelectionDecision(
            agent_id="agent-4",
            lineage_id="lin-4",
            role="citizen",
            prompt_variant_id="counterexample_first",
            package_policy_id="memorial_first",
            bundle_signature="citizen:counterexample_first:memorial_first",
            eligible=True,
            propagation_blocked=False,
            score=0.91,
            base_score=0.91,
            public_score=0.91,
            diversity_bonus=0.02,
            cohort_similarity=0.82,
            selection_bucket="diversity_priority",
            quarantine_status="clean",
        ),
    ]
    decision_by_agent = {decision.agent_id: decision for decision in decisions}
    candidates = [{"agent": agent, "decision": decision_by_agent[agent.agent_id]} for agent in agents]

    pool = build_parent_candidate_pool(
        candidates,
        slot_count=3,
        bundle_state_by_signature={
            "citizen:baseline:artifact_first": {
                "archive_admitted": True,
                "archive_generations": 2,
                "archive_underperform_streak": 1,
                "archive_decay_generations": 0,
                "archive_decay_debt": 0,
                "clean_win_generations": 2,
                "avg_score": 0.96,
            },
            "citizen:baseline:balanced": {
                "archive_admitted": False,
                "archive_underperform_streak": 0,
                "archive_decay_generations": 0,
                "archive_decay_debt": 0,
                "clean_win_generations": 3,
                "avg_score": 0.95,
            },
            "citizen:citation_strict:artifact_first": {
                "archive_admitted": False,
                "archive_underperform_streak": 0,
                "archive_decay_generations": 0,
                "archive_decay_debt": 0,
                "clean_win_generations": 2,
                "avg_score": 0.92,
            },
            "citizen:counterexample_first:memorial_first": {
                "archive_admitted": False,
                "archive_underperform_streak": 0,
                "archive_decay_generations": 0,
                "archive_decay_debt": 0,
                "clean_win_generations": 2,
                "avg_score": 0.91,
            },
        },
    )
    signatures = [item["bundle_signature"] for item in pool]
    preserved = [item["bundle_signature"] for item in pool if item.get("bundle_preserved")]

    assert "citizen:baseline:artifact_first" not in signatures
    assert "citizen:baseline:artifact_first" not in preserved
    assert len(set(signatures)) == 3


def test_parent_pool_penalizes_repeat_evicted_bundle_below_first_time_archive_candidate() -> None:
    agents = [
        _agent("agent-1", "lin-1"),
        _agent("agent-2", "lin-2"),
        _agent("agent-3", "lin-3"),
    ]
    decisions = [
        SelectionDecision(
            agent_id="agent-1",
            lineage_id="lin-1",
            role="citizen",
            prompt_variant_id="baseline",
            package_policy_id="balanced",
            bundle_signature="citizen:baseline:balanced",
            eligible=True,
            propagation_blocked=False,
            score=0.97,
            base_score=0.97,
            public_score=0.97,
            diversity_bonus=-0.01,
            cohort_similarity=0.91,
            selection_bucket="standard",
            quarantine_status="clean",
        ),
        SelectionDecision(
            agent_id="agent-2",
            lineage_id="lin-2",
            role="citizen",
            prompt_variant_id="citation_strict",
            package_policy_id="artifact_first",
            bundle_signature="citizen:citation_strict:artifact_first",
            eligible=True,
            propagation_blocked=False,
            score=0.93,
            base_score=0.93,
            public_score=0.93,
            diversity_bonus=0.01,
            cohort_similarity=0.85,
            selection_bucket="standard",
            quarantine_status="clean",
        ),
        SelectionDecision(
            agent_id="agent-3",
            lineage_id="lin-3",
            role="citizen",
            prompt_variant_id="counterexample_first",
            package_policy_id="memorial_first",
            bundle_signature="citizen:counterexample_first:memorial_first",
            eligible=True,
            propagation_blocked=False,
            score=0.95,
            base_score=0.95,
            public_score=0.95,
            diversity_bonus=0.02,
            cohort_similarity=0.83,
            selection_bucket="diversity_priority",
            quarantine_status="clean",
        ),
    ]
    decision_by_agent = {decision.agent_id: decision for decision in decisions}
    candidates = [{"agent": agent, "decision": decision_by_agent[agent.agent_id]} for agent in agents]

    pool = build_parent_candidate_pool(
        candidates,
        slot_count=2,
        bundle_state_by_signature={
            "citizen:baseline:balanced": {
                "archive_admitted": False,
                "archive_eviction_count": 0,
                "archive_repeat_eviction_tier": 0,
                "archive_decay_generations": 0,
                "archive_decay_debt": 0,
                "clean_win_generations": 3,
                "avg_score": 0.97,
            },
            "citizen:citation_strict:artifact_first": {
                "archive_admitted": False,
                "archive_candidate_generations": 1,
                "archive_admission_pending_generations": 1,
                "archive_proving_streak": 1,
                "archive_eviction_count": 0,
                "archive_repeat_eviction_tier": 0,
                "archive_decay_generations": 0,
                "archive_decay_debt": 0,
                "clean_win_generations": 2,
                "avg_score": 0.93,
            },
            "citizen:counterexample_first:memorial_first": {
                "archive_admitted": False,
                "archive_eviction_count": 2,
                "archive_repeat_eviction_tier": 2,
                "archive_decay_generations": 0,
                "archive_decay_debt": 0,
                "clean_win_generations": 2,
                "avg_score": 0.95,
            },
        },
    )
    signatures = [item["bundle_signature"] for item in pool]

    assert "citizen:baseline:balanced" in signatures
    assert "citizen:citation_strict:artifact_first" in signatures
    assert "citizen:counterexample_first:memorial_first" not in signatures
