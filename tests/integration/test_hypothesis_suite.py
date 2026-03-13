from __future__ import annotations

from pathlib import Path

from society.experiment import run_hypothesis_suite_from_config

from tests.conftest import REPO_ROOT


def _generation_entries(report: dict, *, generation_id: int) -> list[dict]:
    return [entry for entry in report["lineages"] if entry["generation_id"] == generation_id]


def test_hypothesis_suite_runs_all_experiment_modes_and_preserves_mode_specific_channels(config_path: Path) -> None:
    output_prefix = config_path.parent / "exports" / "hypothesis_suite"
    report = run_hypothesis_suite_from_config(
        config_path=config_path,
        repo_root=REPO_ROOT,
        generations=2,
        start_generation_id=1,
        seed=11,
        output_prefix=str(output_prefix),
    )

    assert report["modes"] == [
        "inheritance_on",
        "inheritance_off",
        "memorials_only",
        "taboo_registry_only",
        "isolated_baseline",
    ]
    assert Path(report["exports"]["json_path"]).exists()
    assert Path(report["exports"]["markdown_path"]).exists()
    assert len(report["mode_summaries"]) == 5
    assert all("final_public_eval_average" in item for item in report["mode_summaries"])

    reports_by_mode = report["reports_by_mode"]
    assert set(reports_by_mode) == set(report["modes"])
    assert all(
        all(metric["experiment_mode"] == mode for metric in mode_report["generation_metrics"])
        for mode, mode_report in reports_by_mode.items()
    )

    inheritance_on_entries = _generation_entries(reports_by_mode["inheritance_on"], generation_id=2)
    assert any(entry["parent_lineage_ids"] for entry in inheritance_on_entries)
    assert any(
        entry["inherited_artifact_ids"]
        or entry["inherited_memorial_ids"]
        or entry["taboo_tags"]
        or entry["transfer_payload_active"]
        for entry in inheritance_on_entries
    )

    inheritance_off_entries = _generation_entries(reports_by_mode["inheritance_off"], generation_id=2)
    assert any(entry["parent_lineage_ids"] for entry in inheritance_off_entries)
    assert all(not entry["inherited_artifact_ids"] for entry in inheritance_off_entries)
    assert all(not entry["inherited_memorial_ids"] for entry in inheritance_off_entries)
    assert all(not entry["taboo_tags"] for entry in inheritance_off_entries)
    assert all(entry["transfer_payload_active"] is False for entry in inheritance_off_entries)

    memorial_only_entries = _generation_entries(reports_by_mode["memorials_only"], generation_id=2)
    assert any(entry["inherited_memorial_ids"] for entry in memorial_only_entries)
    assert all(not entry["inherited_artifact_ids"] for entry in memorial_only_entries)
    assert all(not entry["taboo_tags"] for entry in memorial_only_entries)
    assert all(entry["transfer_payload_active"] is False for entry in memorial_only_entries)

    taboo_only_entries = _generation_entries(reports_by_mode["taboo_registry_only"], generation_id=2)
    assert all(not entry["inherited_artifact_ids"] for entry in taboo_only_entries)
    assert all(not entry["inherited_memorial_ids"] for entry in taboo_only_entries)
    assert any(entry["taboo_tags"] for entry in taboo_only_entries if entry["role"] == "adversary")
    assert all(entry["transfer_payload_active"] is False for entry in taboo_only_entries)

    isolated_entries = _generation_entries(reports_by_mode["isolated_baseline"], generation_id=2)
    assert isolated_entries
    assert all(not entry["parent_lineage_ids"] for entry in isolated_entries)
    assert all(entry["inheritance_source_agent_id"] is None for entry in isolated_entries)
    assert all(entry["inheritance_source_generation_id"] is None for entry in isolated_entries)
    assert all(not entry["inherited_artifact_ids"] for entry in isolated_entries)
    assert all(not entry["inherited_memorial_ids"] for entry in isolated_entries)
    assert all(not entry["taboo_tags"] for entry in isolated_entries)
    assert all(entry["transfer_payload_active"] is False for entry in isolated_entries)

    isolated_summary = next(item for item in report["mode_summaries"] if item["mode"] == "isolated_baseline")
    inheritance_on_summary = next(item for item in report["mode_summaries"] if item["mode"] == "inheritance_on")
    assert isolated_summary["delta_vs_isolated_public_eval_average"] == 0.0
    assert inheritance_on_summary["mode"] == "inheritance_on"
    assert "delta_vs_isolated_public_eval_average" in inheritance_on_summary
