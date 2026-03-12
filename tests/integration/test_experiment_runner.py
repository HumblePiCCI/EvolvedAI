from __future__ import annotations

from pathlib import Path

from society.analysis import build_lineage_report
from society.experiment import run_experiment_from_config
from society.storage import StorageManager

from tests.conftest import REPO_ROOT


def test_run_experiment_exports_batch_and_lineage_history(config_path: Path) -> None:
    output_prefix = config_path.parent / "exports" / "batch"
    report = run_experiment_from_config(
        config_path=config_path,
        repo_root=REPO_ROOT,
        generations=2,
        start_generation_id=1,
        output_prefix=str(output_prefix),
    )

    assert report["generation_ids"] == [1, 2]
    assert len(report["generation_metrics"]) == 2
    assert Path(report["exports"]["json_path"]).exists()
    assert Path(report["exports"]["markdown_path"]).exists()
    assert report["lineages"]
    assert "inheritance_effect" in report
    assert "transfer_score" in report["inheritance_effect"]
    assert any(metric["memorial_transfer_score"] > 0.0 for metric in report["generation_metrics"][1:])
    assert "monoculture_index" in report["generation_metrics"][0]
    assert "most_converged_role" in report["generation_metrics"][0]
    assert "parent_concentration_index" in report["generation_metrics"][0]
    assert "most_reused_parent_role" in report["generation_metrics"][0]
    assert "prompt_variant_count" in report["generation_metrics"][0]
    assert "prompt_bundle_count" in report["generation_metrics"][0]
    assert "largest_variant_share" in report["generation_metrics"][0]
    assert "largest_bundle_share" in report["generation_metrics"][0]
    assert "bundle_concentration_index" in report["generation_metrics"][0]
    assert "bundle_survival_rate" in report["generation_metrics"][0]
    assert "bundle_turnover_rate" in report["generation_metrics"][0]
    assert "new_bundle_win_rate" in report["generation_metrics"][0]
    assert "exploration_bundle_survival_rate" in report["generation_metrics"][0]
    assert "preserved_bundle_count" in report["generation_metrics"][0]
    assert "bundle_archive_count" in report["generation_metrics"][0]
    assert "archive_admission_pending_count" in report["generation_metrics"][0]
    assert "archive_proving_count" in report["generation_metrics"][0]
    assert "archive_reentry_block_count" in report["generation_metrics"][0]
    assert "archive_escalated_backoff_count" in report["generation_metrics"][0]
    assert "archive_reentry_attempt_count" in report["generation_metrics"][0]
    assert "archive_underperform_count" in report["generation_metrics"][0]
    assert "archive_positive_lift_count" in report["generation_metrics"][0]
    assert "archive_value_deficit_count" in report["generation_metrics"][0]
    assert "archive_incumbent_win_count" in report["generation_metrics"][0]
    assert "archive_incumbent_loss_count" in report["generation_metrics"][0]
    assert "archive_mean_comparative_lift" in report["generation_metrics"][0]
    assert "archive_transfer_success_count" in report["generation_metrics"][0]
    assert "archive_transfer_failure_count" in report["generation_metrics"][0]
    assert "archive_transfer_success_rate" in report["generation_metrics"][0]
    assert "archive_parent_vs_child_lift_retention" in report["generation_metrics"][0]
    assert "archive_admitted_count" in report["generation_metrics"][0]
    assert "newly_admitted_count" in report["generation_metrics"][0]
    assert "post_admission_grace_count" in report["generation_metrics"][0]
    assert "archive_eviction_count" in report["generation_metrics"][0]
    assert "repeat_eviction_count" in report["generation_metrics"][0]
    assert "archive_repeat_eviction_max_tier" in report["generation_metrics"][0]
    assert "archive_admission_conversion_rate" in report["generation_metrics"][0]
    assert "archive_reentry_converted_count" in report["generation_metrics"][0]
    assert "archive_reentry_mean_gap_generations" in report["generation_metrics"][0]
    assert "archive_reentry_max_gap_generations" in report["generation_metrics"][0]
    assert "archive_retired_count" in report["generation_metrics"][0]
    assert "archive_failed_admission_count" in report["generation_metrics"][0]
    assert "bundle_archive_cooldown_count" in report["generation_metrics"][0]
    assert "bundle_archive_coexistence_budget_count" in report["generation_metrics"][0]
    assert "bundle_archive_cooldown_true_overload_count" in report["generation_metrics"][0]
    assert "bundle_archive_cooldown_avoidable_duplicate_count" in report["generation_metrics"][0]
    assert "bundle_archive_reentry_backoff_roles" in report["generation_metrics"][0]
    assert "bundle_archive_reentry_block_roles" in report["generation_metrics"][0]
    assert "bundle_archive_escalated_backoff_roles" in report["generation_metrics"][0]
    assert "bundle_archive_coexistence_budget_roles" in report["generation_metrics"][0]
    assert "bundle_archive_underperform_roles" in report["generation_metrics"][0]
    assert "bundle_archive_positive_lift_roles" in report["generation_metrics"][0]
    assert "bundle_archive_value_deficit_roles" in report["generation_metrics"][0]
    assert "bundle_archive_transfer_success_roles" in report["generation_metrics"][0]
    assert "bundle_archive_transfer_failure_roles" in report["generation_metrics"][0]
    assert "bundle_archive_eviction_roles" in report["generation_metrics"][0]
    assert "bundle_archive_repeat_eviction_roles" in report["generation_metrics"][0]
    assert "bundle_archive_retired_roles" in report["generation_metrics"][0]
    assert "bundle_archive_post_admission_grace_roles" in report["generation_metrics"][0]
    assert "bundle_archive_cooldown_fresh_admission_roles" in report["generation_metrics"][0]
    assert "bundle_archive_cooldown_long_lived_debt_roles" in report["generation_metrics"][0]
    assert "bundle_archive_cooldown_true_overload_roles" in report["generation_metrics"][0]
    assert "bundle_archive_cooldown_avoidable_duplicate_roles" in report["generation_metrics"][0]
    assert "bundle_archive_cooldown_recovery_roles" in report["generation_metrics"][0]
    assert "bundle_archive_cooldown_recovery_count" in report["generation_metrics"][0]
    assert "bundle_archive_cooldown_recovery_max_generations" in report["generation_metrics"][0]
    assert "bundle_decay_prune_count" in report["generation_metrics"][0]
    assert "stale_bundle_count" in report["generation_metrics"][0]
    assert "decaying_bundle_count" in report["generation_metrics"][0]
    assert "archive_retirement_ready_count" in report["generation_metrics"][0]
    assert "archive_retirement_reason_counts" in report["generation_metrics"][0]
    assert "pruned_bundle_count" in report["generation_metrics"][0]
    assert "drift_pressure_lineages" in report["generation_metrics"][0]

    storage = StorageManager(
        root_dir=config_path.parent / "data",
        db_path=config_path.parent / "data" / "db.sqlite",
    )
    try:
        lineage_report = build_lineage_report(storage, report["lineages"][0]["lineage_id"])
        assert lineage_report["survival_history"]
        assert lineage_report["selected_lineage"]["outcome"] in {
            "propagated",
            "eligible_pending",
            "eligible_unselected",
            "reviewed",
            "blocked",
            "not_selected",
            "unknown",
        }
        assert "inherited_artifacts" in lineage_report["selected_lineage"]
        assert "inherited_memorials" in lineage_report["selected_lineage"]
        assert "warning_outcome" in lineage_report["selected_lineage"]
        assert "selection_bucket" in lineage_report["selected_lineage"]
        assert "prompt_variant_id" in lineage_report["selected_lineage"]
        assert "bundle_signature" in lineage_report["selected_lineage"]
        assert "inheritance_source_bundle_signature" in lineage_report["selected_lineage"]
        assert "inheritance_source_selection_source" in lineage_report["selected_lineage"]
    finally:
        storage.close()
