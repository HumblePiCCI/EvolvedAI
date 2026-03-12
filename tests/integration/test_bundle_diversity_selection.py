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


def test_bundle_diversity_selection_keeps_bundle_share_below_collapse_case(tmp_path: Path) -> None:
    config = _bundle_diversity_config(tmp_path)
    storage = StorageManager(root_dir=config.storage.root_dir, db_path=config.storage.db_path)
    provider = build_provider(config.provider.name, config.provider.model)
    try:
        runner = GenerationRunner(config=config, storage=storage, provider=provider, repo_root=REPO_ROOT)
        generation_ids = [1, 2, 3, 4, 5]
        for generation_id in generation_ids:
            runner.run(generation_id=generation_id)

        report = build_experiment_report(storage, generation_ids)
        post_root = report["generation_metrics"][1:]

        assert post_root
        assert max(metric["largest_bundle_share"] for metric in post_root) < 0.5
        assert max(metric["parent_bundle_concentration_index"] for metric in post_root) < 0.5
        assert all(metric["preserved_bundle_count"] >= 1 for metric in post_root)
        assert any(metric["bundle_survival_rate"] > 0.0 for metric in post_root)

        latest_generation = storage.get_generation(generation_ids[-1])
        assert latest_generation is not None
        latest_summary = latest_generation.summary_json
        citizen_bundles = latest_summary["selection_summary"]["preserved_bundles_by_role"]["citizen"]
        assert citizen_bundles
        assert latest_summary["selection_summary"]["role_parent_bundle_concentration_index"]["citizen"] < 0.5
    finally:
        storage.close()
