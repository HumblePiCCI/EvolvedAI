from __future__ import annotations

from pathlib import Path

from society.experiment import run_comparative_batches_from_config

from tests.conftest import REPO_ROOT


def test_comparative_batches_export_requested_mode_deltas(config_path: Path) -> None:
    output_prefix = config_path.parent / "exports" / "comparative_batches"
    report = run_comparative_batches_from_config(
        config_path=config_path,
        repo_root=REPO_ROOT,
        generations=2,
        seeds=[11, 12],
        start_generation_id=1,
        output_prefix=str(output_prefix),
    )

    assert report["seeds"] == [11, 12]
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

    by_mode = {item["mode"]: item for item in report["mode_summaries"]}
    isolated = by_mode["isolated_baseline"]
    inheritance_off = by_mode["inheritance_off"]
    inheritance_on = by_mode["inheritance_on"]

    for mode, item in by_mode.items():
        assert item["seed_count"] == 2
        assert len(item["seed_metrics"]) == 2
        assert "cooperative_truthfulness_score" in item["batch_means"]
        assert "correction_acceptance_average" in item["batch_means"]
        assert "hidden_eval_failure_rate" in item["batch_means"]
        assert "failure_recurrence_rate" in item["batch_means"]
        assert "lineage_diffusion_index" in item["batch_means"]
        assert "deltas_vs_isolated_baseline" in item
        assert "deltas_vs_inheritance_off" in item

    assert isolated["deltas_vs_isolated_baseline"]["cooperative_truthfulness_score_delta"] == 0.0
    assert isolated["deltas_vs_isolated_baseline"]["hidden_eval_failure_rate_reduction"] == 0.0
    assert inheritance_off["deltas_vs_inheritance_off"]["recurrence_reduction"] == 0.0
    assert "cooperative_truthfulness_score_delta" in inheritance_on["deltas_vs_isolated_baseline"]
    assert "recurrence_reduction" in inheritance_on["deltas_vs_inheritance_off"]
    assert len({item["cooperative_truthfulness_score"] for item in inheritance_on["seed_metrics"]}) > 1
