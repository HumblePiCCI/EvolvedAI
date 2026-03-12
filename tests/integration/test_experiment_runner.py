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
    finally:
        storage.close()
