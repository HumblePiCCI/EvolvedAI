from __future__ import annotations

from pathlib import Path

from society.analysis import build_experiment_report
from society.config import AutoCivConfig
from society.generation import GenerationRunner
from society.providers import build_provider
from society.schemas import GenerationRecord, SelectionDecision
from society.storage import StorageManager
from society.utils import utc_now

from tests.conftest import REPO_ROOT, minimal_config_data


def _bundle_diversity_config(tmp_path: Path) -> AutoCivConfig:
    data = minimal_config_data(tmp_path)
    data["generation"].update(
        {
            "population_size": 12,
            "islands": 2,
            "episodes_per_generation": 4,
            "max_turns_per_episode": 10,
            "seed": 42,
        }
    )
    data["worlds"]["shared_notebook_v0"]["task_pool"] = [
        "Draft a bounded research note on whether a prior claim should be accepted, rejected, or deferred.",
        "Summarize the strongest supported claim and the most important unresolved uncertainty.",
        "Prepare a correction-ready notebook entry with explicit citations and open risks.",
        "Produce a final artifact that distinguishes evidence, inference, and speculation.",
    ]
    data["roles"]["distribution"] = {
        "citizen": 6,
        "judge": 2,
        "steward": 2,
        "archivist": 1,
        "adversary": 1,
    }
    data["roles"]["behaviors"] = {
        "citizen": "honest",
        "judge": "self_correcting",
        "steward": "honest",
        "archivist": "honest",
        "adversary": "manipulative",
    }
    return AutoCivConfig.model_validate(data)


def test_bundle_archive_selection_adds_turnover_without_reintroducing_bundle_collapse(tmp_path: Path) -> None:
    config = _bundle_diversity_config(tmp_path)
    storage = StorageManager(root_dir=config.storage.root_dir, db_path=config.storage.db_path)
    provider = build_provider(config.provider.name, config.provider.model)
    try:
        runner = GenerationRunner(config=config, storage=storage, provider=provider, repo_root=REPO_ROOT)
        generation_ids = list(range(1, 14))
        for generation_id in generation_ids:
            runner.run(generation_id=generation_id)

        report = build_experiment_report(storage, generation_ids)
        post_root = report["generation_metrics"][1:]

        assert post_root
        assert max(metric["largest_bundle_share"] for metric in post_root) < 0.5
        assert max(metric["parent_bundle_concentration_index"] for metric in post_root) < 0.5
        assert all(metric["preserved_bundle_count"] >= 1 for metric in post_root)
        assert any(metric["bundle_survival_rate"] > 0.0 for metric in post_root)
        assert any(metric["bundle_archive_count"] >= 1 for metric in post_root)
        assert any(metric["archive_admission_pending_count"] > 0 for metric in post_root)
        assert any(metric["archive_proving_count"] > 0 for metric in post_root)
        assert any(metric["archive_admitted_count"] > 0 for metric in post_root)
        assert any(metric["newly_admitted_count"] > 0 for metric in post_root)
        assert any(metric["post_admission_grace_count"] > 0 for metric in post_root)
        assert any(metric["archive_reentry_block_count"] > 0 for metric in post_root)
        assert any(metric["archive_escalated_backoff_count"] > 0 for metric in post_root)
        assert any(metric["archive_reentry_attempt_count"] > 0 for metric in post_root)
        assert any(metric["archive_underperform_count"] > 0 for metric in post_root)
        assert any(metric["archive_positive_lift_count"] > 0 for metric in post_root)
        assert any(metric["archive_value_deficit_count"] > 0 for metric in post_root)
        assert any(metric["archive_incumbent_loss_count"] > 0 for metric in post_root)
        assert any(metric["archive_mean_comparative_lift"] != 0.0 for metric in post_root)
        assert any(metric["archive_eviction_count"] > 0 for metric in post_root)
        assert any(metric["repeat_eviction_count"] > 0 for metric in post_root)
        assert any(metric["archive_repeat_eviction_max_tier"] >= 2 for metric in post_root)
        assert any(metric["archive_admission_conversion_rate"] > 0.0 for metric in post_root)
        assert any(metric["archive_reentry_converted_count"] > 0 for metric in post_root)
        assert any(metric["archive_reentry_mean_gap_generations"] > 0.0 for metric in post_root)
        assert any(metric["archive_retired_count"] > 0 for metric in post_root)
        assert any(metric["bundle_archive_coexistence_budget_count"] > 0 for metric in post_root)
        assert any(metric["bundle_turnover_rate"] > 0.0 for metric in post_root)
        assert any(metric["new_bundle_win_rate"] > 0.0 for metric in post_root)
        assert any(metric["exploration_bundle_survival_rate"] > 0.0 for metric in post_root)
        assert any(metric["bundle_archive_reentry_backoff_roles"] for metric in post_root)
        assert any(metric["bundle_archive_reentry_block_roles"] for metric in post_root)
        assert any(metric["bundle_archive_escalated_backoff_roles"] for metric in post_root)
        assert any(metric["bundle_archive_coexistence_budget_roles"] for metric in post_root)
        assert any(metric["bundle_archive_positive_lift_roles"] for metric in post_root)
        assert any(metric["bundle_archive_value_deficit_roles"] for metric in post_root)
        assert any(metric["bundle_archive_eviction_roles"] for metric in post_root)
        assert any(metric["bundle_archive_repeat_eviction_roles"] for metric in post_root)
        assert any(metric["bundle_archive_retired_roles"] for metric in post_root)
        assert any(
            post_root[index - 1]["archive_admission_pending_count"] > 0
            and post_root[index]["bundle_archive_count"] == 0
            for index in range(1, len(post_root))
        )
        first_archive_generation = next(metric for metric in post_root if metric["bundle_archive_count"] > 0)
        assert first_archive_generation["archive_admitted_count"] == 0
        assert first_archive_generation["newly_admitted_count"] == 0
        assert first_archive_generation["post_admission_grace_count"] == 0
        assert first_archive_generation["bundle_archive_cooldown_count"] == 0
        admitted_generation_index = next(
            index for index, metric in enumerate(post_root) if metric["newly_admitted_count"] > 0
        )
        admitted_generation = post_root[admitted_generation_index]
        assert admitted_generation["archive_admitted_count"] > 0
        assert admitted_generation["post_admission_grace_count"] > 0
        assert admitted_generation["bundle_archive_cooldown_count"] == 0
        assert admitted_generation["bundle_archive_post_admission_grace_roles"]
        assert not admitted_generation["bundle_archive_cooldown_fresh_admission_roles"]
        assert not admitted_generation["bundle_archive_cooldown_long_lived_debt_roles"]
        eviction_generation = next(metric for metric in post_root if metric["archive_eviction_count"] > 0)
        assert eviction_generation["archive_underperform_count"] > 0
        assert eviction_generation["archive_admitted_count"] == 0
        assert eviction_generation["bundle_archive_cooldown_count"] == 0
        backoff_generation = next(metric for metric in post_root if metric["bundle_archive_reentry_backoff_roles"])
        assert backoff_generation["archive_admitted_count"] == 0
        blocked_generation = next(metric for metric in post_root if metric["archive_reentry_block_count"] > 0)
        assert blocked_generation["archive_admitted_count"] == 0
        assert blocked_generation["bundle_archive_reentry_block_roles"]
        assert blocked_generation["archive_reentry_attempt_count"] > 0
        reentry_generation = next(metric for metric in post_root if metric["archive_reentry_converted_count"] > 0)
        assert reentry_generation["archive_admitted_count"] > 0
        assert reentry_generation["newly_admitted_count"] > 0
        assert reentry_generation["archive_reentry_mean_gap_generations"] >= 4.0
        coexistence_readmission_generation = next(
            metric
            for metric in post_root
            if metric["bundle_archive_coexistence_budget_count"] > 0 and metric["archive_admitted_count"] > 0
        )
        assert coexistence_readmission_generation["bundle_archive_cooldown_count"] == 0
        assert coexistence_readmission_generation["bundle_archive_cooldown_fresh_admission_roles"] == []
        assert coexistence_readmission_generation["bundle_archive_cooldown_true_overload_count"] == 0
        repeat_eviction_generation = next(metric for metric in post_root if metric["repeat_eviction_count"] > 0)
        assert repeat_eviction_generation["archive_eviction_count"] > 0
        assert repeat_eviction_generation["archive_repeat_eviction_max_tier"] >= 2
        assert repeat_eviction_generation["archive_retired_count"] > 0
        assert repeat_eviction_generation["archive_retirement_reason_counts"].get("no_comparative_lift", 0) >= 1
        assert repeat_eviction_generation["bundle_archive_retired_roles"]
        assert repeat_eviction_generation["largest_bundle_share"] < 0.5
        retired_generation_index = next(
            index for index, metric in enumerate(post_root) if metric["archive_retired_count"] > 0
        )
        for metric in post_root[retired_generation_index + 1 :]:
            assert metric["archive_reentry_converted_count"] == 0
            assert metric["archive_admitted_count"] == 0
        assert max(metric["prompt_bundle_count"] for metric in post_root) <= 9
        assert any(
            post_root[index]["prompt_bundle_count"] <= post_root[index - 1]["prompt_bundle_count"]
            for index in range(1, len(post_root))
        )

        latest_generation = storage.get_generation(generation_ids[-1])
        assert latest_generation is not None
        latest_summary = latest_generation.summary_json
        citizen_bundles = latest_summary["selection_summary"]["preserved_bundles_by_role"]["citizen"]
        assert citizen_bundles
        assert latest_summary["selection_summary"]["role_parent_bundle_concentration_index"]["citizen"] < 0.5
        assert "archive_admission_pending_count" in latest_summary["selection_summary"]
        assert "archive_proving_count" in latest_summary["selection_summary"]
        assert "archive_reentry_block_count" in latest_summary["selection_summary"]
        assert "archive_escalated_backoff_count" in latest_summary["selection_summary"]
        assert "archive_reentry_attempt_count" in latest_summary["selection_summary"]
        assert "archive_underperform_count" in latest_summary["selection_summary"]
        assert "archive_positive_lift_count" in latest_summary["selection_summary"]
        assert "archive_value_deficit_count" in latest_summary["selection_summary"]
        assert "archive_incumbent_win_count" in latest_summary["selection_summary"]
        assert "archive_incumbent_loss_count" in latest_summary["selection_summary"]
        assert "archive_mean_comparative_lift" in latest_summary["selection_summary"]
        assert "archive_admitted_count" in latest_summary["selection_summary"]
        assert "newly_admitted_count" in latest_summary["selection_summary"]
        assert "post_admission_grace_count" in latest_summary["selection_summary"]
        assert "archive_eviction_count" in latest_summary["selection_summary"]
        assert "repeat_eviction_count" in latest_summary["selection_summary"]
        assert "archive_repeat_eviction_max_tier" in latest_summary["selection_summary"]
        assert "archive_admission_conversion_rate" in latest_summary["selection_summary"]
        assert "archive_reentry_converted_count" in latest_summary["selection_summary"]
        assert "archive_reentry_mean_gap_generations" in latest_summary["selection_summary"]
        assert "archive_reentry_max_gap_generations" in latest_summary["selection_summary"]
        assert "archive_retired_count" in latest_summary["selection_summary"]
        assert "archive_failed_admission_count" in latest_summary["selection_summary"]
        assert "bundle_archive_coexistence_budget_count" in latest_summary["selection_summary"]
        assert "bundle_archive_cooldown_true_overload_count" in latest_summary["selection_summary"]
        assert "bundle_archive_cooldown_avoidable_duplicate_count" in latest_summary["selection_summary"]
        assert "bundle_archive_post_admission_grace_roles" in latest_summary["selection_summary"]
        assert "bundle_archive_reentry_backoff_roles" in latest_summary["selection_summary"]
        assert "bundle_archive_reentry_block_roles" in latest_summary["selection_summary"]
        assert "bundle_archive_escalated_backoff_roles" in latest_summary["selection_summary"]
        assert "bundle_archive_coexistence_budget_roles" in latest_summary["selection_summary"]
        assert "bundle_archive_underperform_roles" in latest_summary["selection_summary"]
        assert "bundle_archive_positive_lift_roles" in latest_summary["selection_summary"]
        assert "bundle_archive_value_deficit_roles" in latest_summary["selection_summary"]
        assert "bundle_archive_eviction_roles" in latest_summary["selection_summary"]
        assert "bundle_archive_repeat_eviction_roles" in latest_summary["selection_summary"]
        assert "bundle_archive_retired_roles" in latest_summary["selection_summary"]
        assert "bundle_archive_cooldown_fresh_admission_roles" in latest_summary["selection_summary"]
        assert "bundle_archive_cooldown_long_lived_debt_roles" in latest_summary["selection_summary"]
        assert "bundle_archive_cooldown_true_overload_roles" in latest_summary["selection_summary"]
        assert "bundle_archive_cooldown_avoidable_duplicate_roles" in latest_summary["selection_summary"]
        assert "bundle_archive_cooldown_recovery_roles" in latest_summary["selection_summary"]
        assert "bundle_archive_cooldown_recovery_count" in latest_summary["selection_summary"]
        assert "bundle_archive_cooldown_recovery_max_generations" in latest_summary["selection_summary"]
        assert "stale_bundle_count" in latest_summary["selection_summary"]
        assert "decaying_bundle_count" in latest_summary["selection_summary"]
        assert "bundle_decay_prune_count" in latest_summary["selection_summary"]
        assert "archive_retirement_ready_count" in latest_summary["selection_summary"]
        assert "archive_retirement_reason_counts" in latest_summary["selection_summary"]
        assert "pruned_bundle_count" in latest_summary["selection_summary"]
        assert "bundle_archive_cooldown_count" in latest_summary["selection_summary"]
    finally:
        storage.close()


def test_archive_value_benchmark_retires_zero_lift_bundle_but_keeps_positive_lift_bundle(tmp_path: Path) -> None:
    config = _bundle_diversity_config(tmp_path)
    storage = StorageManager(root_dir=config.storage.root_dir, db_path=config.storage.db_path)
    provider = build_provider(config.provider.name, config.provider.model)
    try:
        storage.initialize()
        runner = GenerationRunner(config=config, storage=storage, provider=provider, repo_root=REPO_ROOT)
        storage.put_generation(
            GenerationRecord(
                generation_id=1,
                config_hash="config",
                world_name=config.world.name,
                population_size=config.generation.population_size,
                seed=config.generation.seed,
                status="completed",
                started_at=utc_now(),
                ended_at=utc_now(),
                summary_json={
                    "selection_summary": {
                        "bundle_state_by_role": {
                            "citizen": {
                                "citizen:archive_flat:artifact_first": {
                                    "archive_candidate_generations": 3,
                                    "archive_admission_pending_generations": 0,
                                    "archive_proving_streak": 0,
                                    "archive_admitted": True,
                                    "archive_generations": 1,
                                    "archive_post_admission_grace_remaining": 0,
                                    "archive_reentry_backoff_target": 2,
                                    "archive_reentry_backoff_remaining": 0,
                                    "archive_reentry_attempt_count": 1,
                                    "archive_eviction_count": 1,
                                    "archive_repeat_eviction_tier": 1,
                                    "archive_positive_lift_streak": 0,
                                    "archive_value_deficit_streak": 0,
                                    "archive_retired": False,
                                    "archive_retirement_reason": None,
                                    "archive_last_eviction_generation_id": 0,
                                },
                                "citizen:archive_lifted:taboo_first": {
                                    "archive_candidate_generations": 3,
                                    "archive_admission_pending_generations": 0,
                                    "archive_proving_streak": 0,
                                    "archive_admitted": True,
                                    "archive_generations": 1,
                                    "archive_post_admission_grace_remaining": 0,
                                    "archive_reentry_backoff_target": 2,
                                    "archive_reentry_backoff_remaining": 0,
                                    "archive_reentry_attempt_count": 1,
                                    "archive_eviction_count": 1,
                                    "archive_repeat_eviction_tier": 1,
                                    "archive_positive_lift_streak": 1,
                                    "archive_value_deficit_streak": 0,
                                    "archive_retired": False,
                                    "archive_retirement_reason": None,
                                    "archive_last_eviction_generation_id": 0,
                                },
                                "citizen:baseline:balanced": {
                                    "archive_candidate_generations": 0,
                                    "archive_admitted": False,
                                    "archive_eviction_count": 0,
                                    "archive_repeat_eviction_tier": 0,
                                },
                                "citizen:citation_strict:artifact_first": {
                                    "archive_candidate_generations": 0,
                                    "archive_admitted": False,
                                    "archive_eviction_count": 0,
                                    "archive_repeat_eviction_tier": 0,
                                },
                            }
                        }
                    }
                },
            )
        )

        lineage_updates = [
            {
                "agent_id": "agent-0002-000",
                "lineage_id": "lin-0002-000",
                "role": "citizen",
                "bundle_signature": "citizen:archive_flat:artifact_first",
                "variant_origin": "inherited",
            },
            {
                "agent_id": "agent-0002-001",
                "lineage_id": "lin-0002-001",
                "role": "citizen",
                "bundle_signature": "citizen:archive_lifted:taboo_first",
                "variant_origin": "inherited",
            },
            {
                "agent_id": "agent-0002-002",
                "lineage_id": "lin-0002-002",
                "role": "citizen",
                "bundle_signature": "citizen:baseline:balanced",
                "variant_origin": "inherited",
            },
            {
                "agent_id": "agent-0002-003",
                "lineage_id": "lin-0002-003",
                "role": "citizen",
                "bundle_signature": "citizen:citation_strict:artifact_first",
                "variant_origin": "inherited",
            },
        ]
        selection = [
            SelectionDecision(
                agent_id="agent-0002-000",
                lineage_id="lin-0002-000",
                role="citizen",
                prompt_variant_id="archive_flat",
                package_policy_id="artifact_first",
                bundle_signature="citizen:archive_flat:artifact_first",
                eligible=True,
                propagation_blocked=False,
                score=0.89,
                base_score=0.89,
                public_score=0.89,
                diversity_bonus=0.0,
                cohort_similarity=0.4,
                selection_bucket="standard",
                quarantine_status="clean",
            ),
            SelectionDecision(
                agent_id="agent-0002-001",
                lineage_id="lin-0002-001",
                role="citizen",
                prompt_variant_id="archive_lifted",
                package_policy_id="taboo_first",
                bundle_signature="citizen:archive_lifted:taboo_first",
                eligible=True,
                propagation_blocked=False,
                score=0.92,
                base_score=0.92,
                public_score=0.92,
                diversity_bonus=0.02,
                cohort_similarity=0.35,
                selection_bucket="diversity_priority",
                quarantine_status="clean",
            ),
            SelectionDecision(
                agent_id="agent-0002-002",
                lineage_id="lin-0002-002",
                role="citizen",
                prompt_variant_id="baseline",
                package_policy_id="balanced",
                bundle_signature="citizen:baseline:balanced",
                eligible=True,
                propagation_blocked=False,
                score=0.89,
                base_score=0.89,
                public_score=0.89,
                diversity_bonus=0.0,
                cohort_similarity=0.45,
                selection_bucket="standard",
                quarantine_status="clean",
            ),
            SelectionDecision(
                agent_id="agent-0002-003",
                lineage_id="lin-0002-003",
                role="citizen",
                prompt_variant_id="citation_strict",
                package_policy_id="artifact_first",
                bundle_signature="citizen:citation_strict:artifact_first",
                eligible=True,
                propagation_blocked=False,
                score=0.89,
                base_score=0.89,
                public_score=0.89,
                diversity_bonus=0.0,
                cohort_similarity=0.45,
                selection_bucket="standard",
                quarantine_status="clean",
            ),
        ]

        bundle_state_by_role = runner._bundle_state_by_role(
            generation_id=2,
            previous_generation_id=1,
            lineage_updates=lineage_updates,
            selection=selection,
        )
        no_lift_state = bundle_state_by_role["citizen"]["citizen:archive_flat:artifact_first"]
        positive_lift_state = bundle_state_by_role["citizen"]["citizen:archive_lifted:taboo_first"]

        assert no_lift_state["archive_retired"] is True
        assert no_lift_state["archive_retirement_reason"] == "no_comparative_lift"
        assert no_lift_state["archive_comparative_lift"] == 0.0
        assert no_lift_state["archive_positive_lift_streak"] == 0
        assert positive_lift_state["archive_admitted"] is True
        assert positive_lift_state["archive_retired"] is False
        assert positive_lift_state["archive_comparative_lift"] > 0.0
        assert positive_lift_state["archive_positive_lift_streak"] >= 2
        assert positive_lift_state["archive_value_qualified"] is True
    finally:
        storage.close()
