from __future__ import annotations

from pathlib import Path

from society.analysis import build_experiment_report
from society.config import AutoCivConfig
from society.generation import GenerationRunner
from society.providers import build_provider
from society.storage import StorageManager

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
        generation_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
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
        assert any(metric["archive_underperform_count"] > 0 for metric in post_root)
        assert any(metric["archive_eviction_count"] > 0 for metric in post_root)
        assert any(metric["archive_admission_conversion_rate"] > 0.0 for metric in post_root)
        assert any(metric["bundle_turnover_rate"] > 0.0 for metric in post_root)
        assert any(metric["new_bundle_win_rate"] > 0.0 for metric in post_root)
        assert any(metric["exploration_bundle_survival_rate"] > 0.0 for metric in post_root)
        assert any(metric["bundle_archive_underperform_roles"] for metric in post_root)
        assert any(metric["bundle_archive_eviction_roles"] for metric in post_root)
        assert all(metric["bundle_archive_cooldown_count"] == 0 for metric in post_root)
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
        assert admitted_generation["bundle_archive_cooldown_fresh_admission_roles"]
        assert not admitted_generation["bundle_archive_cooldown_long_lived_debt_roles"]
        eviction_generation = next(metric for metric in post_root if metric["archive_eviction_count"] > 0)
        assert eviction_generation["archive_underperform_count"] > 0
        assert eviction_generation["archive_admitted_count"] == 0
        assert eviction_generation["bundle_archive_underperform_roles"]
        assert eviction_generation["bundle_archive_eviction_roles"]
        assert eviction_generation["bundle_archive_cooldown_count"] == 0
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
        assert "citizen" in latest_summary["selection_summary"]["bundle_archive_candidate_roles"]
        assert (
            "citizen" in latest_summary["selection_summary"]["bundle_archive_pending_roles"]
            or "citizen" in latest_summary["selection_summary"]["bundle_archive_proving_roles"]
            or "citizen" in latest_summary["selection_summary"]["bundle_archive_cooldown_roles"]
            or "citizen" in latest_summary["selection_summary"]["bundle_archive_roles"]
        )
        assert latest_summary["selection_summary"]["role_parent_bundle_concentration_index"]["citizen"] < 0.5
        assert "archive_admission_pending_count" in latest_summary["selection_summary"]
        assert "archive_proving_count" in latest_summary["selection_summary"]
        assert "archive_underperform_count" in latest_summary["selection_summary"]
        assert "archive_admitted_count" in latest_summary["selection_summary"]
        assert "newly_admitted_count" in latest_summary["selection_summary"]
        assert "post_admission_grace_count" in latest_summary["selection_summary"]
        assert "archive_eviction_count" in latest_summary["selection_summary"]
        assert "archive_admission_conversion_rate" in latest_summary["selection_summary"]
        assert "archive_failed_admission_count" in latest_summary["selection_summary"]
        assert "bundle_archive_post_admission_grace_roles" in latest_summary["selection_summary"]
        assert "bundle_archive_underperform_roles" in latest_summary["selection_summary"]
        assert "bundle_archive_eviction_roles" in latest_summary["selection_summary"]
        assert "bundle_archive_cooldown_fresh_admission_roles" in latest_summary["selection_summary"]
        assert "bundle_archive_cooldown_long_lived_debt_roles" in latest_summary["selection_summary"]
        assert "bundle_archive_cooldown_recovery_roles" in latest_summary["selection_summary"]
        assert "bundle_archive_cooldown_recovery_count" in latest_summary["selection_summary"]
        assert "bundle_archive_cooldown_recovery_max_generations" in latest_summary["selection_summary"]
        assert "stale_bundle_count" in latest_summary["selection_summary"]
        assert "decaying_bundle_count" in latest_summary["selection_summary"]
        assert "bundle_decay_prune_count" in latest_summary["selection_summary"]
        assert "archive_retirement_ready_count" in latest_summary["selection_summary"]
        assert "pruned_bundle_count" in latest_summary["selection_summary"]
        assert "bundle_archive_cooldown_count" in latest_summary["selection_summary"]
    finally:
        storage.close()
