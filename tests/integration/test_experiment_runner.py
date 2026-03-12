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
    assert "archive_underperform_count" in report["generation_metrics"][0]
    assert "archive_admitted_count" in report["generation_metrics"][0]
    assert "newly_admitted_count" in report["generation_metrics"][0]
    assert "post_admission_grace_count" in report["generation_metrics"][0]
    assert "archive_eviction_count" in report["generation_metrics"][0]
    assert "archive_admission_conversion_rate" in report["generation_metrics"][0]
    assert "archive_failed_admission_count" in report["generation_metrics"][0]
    assert "bundle_archive_cooldown_count" in report["generation_metrics"][0]
    assert "bundle_archive_underperform_roles" in report["generation_metrics"][0]
    assert "bundle_archive_eviction_roles" in report["generation_metrics"][0]
    assert "bundle_archive_post_admission_grace_roles" in report["generation_metrics"][0]
    assert "bundle_archive_cooldown_fresh_admission_roles" in report["generation_metrics"][0]
    assert "bundle_archive_cooldown_long_lived_debt_roles" in report["generation_metrics"][0]
    assert "bundle_archive_cooldown_recovery_roles" in report["generation_metrics"][0]
    assert "bundle_archive_cooldown_recovery_count" in report["generation_metrics"][0]
    assert "bundle_archive_cooldown_recovery_max_generations" in report["generation_metrics"][0]
    assert "bundle_decay_prune_count" in report["generation_metrics"][0]
    assert "stale_bundle_count" in report["generation_metrics"][0]
    assert "decaying_bundle_count" in report["generation_metrics"][0]
    assert "archive_retirement_ready_count" in report["generation_metrics"][0]
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
