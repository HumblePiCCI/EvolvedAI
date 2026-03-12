from __future__ import annotations

from society.generation import GenerationRunner
from society.providers import build_provider
from society.storage import StorageManager

from tests.conftest import REPO_ROOT


def test_blocked_lineages_do_not_propagate_across_generations(minimal_config) -> None:
    storage = StorageManager(root_dir=minimal_config.storage.root_dir, db_path=minimal_config.storage.db_path)
    provider = build_provider(minimal_config.provider.name, minimal_config.provider.model)
    try:
        runner = GenerationRunner(config=minimal_config, storage=storage, provider=provider, repo_root=REPO_ROOT)
        summary_one = runner.run(generation_id=1)
        blocked = next(item for item in summary_one["selection_outcome"] if item["role"] == "adversary")
        assert blocked["propagation_blocked"] is True

        blocked_memorial = next(
            record for record in storage.list_generation_memorials(1) if record.source_agent_id == blocked["agent_id"]
        )

        summary_two = runner.run(generation_id=2)
        generation_two_agents = storage.list_generation_agents(2)
        assert all(blocked_memorial.memorial_id not in agent.inherited_memorial_ids for agent in generation_two_agents)

        citizen_or_judge_updates = [
            item for item in summary_two["lineage_updates"] if item["role"] in {"citizen", "judge"}
        ]
        assert any(item["parent_lineage_ids"] for item in citizen_or_judge_updates)
        assert all("anti_corruption" not in item["taboo_tags"] for item in citizen_or_judge_updates)

        adversary_update = next(item for item in summary_two["lineage_updates"] if item["role"] == "adversary")
        assert adversary_update["parent_lineage_ids"] == []
        assert "anti_corruption" in adversary_update["taboo_tags"]
    finally:
        storage.close()


def test_sticky_taboo_registry_survives_one_clean_generation(minimal_config) -> None:
    storage = StorageManager(root_dir=minimal_config.storage.root_dir, db_path=minimal_config.storage.db_path)
    provider = build_provider(minimal_config.provider.name, minimal_config.provider.model)
    try:
        runner = GenerationRunner(config=minimal_config, storage=storage, provider=provider, repo_root=REPO_ROOT)
        summary_one = runner.run(generation_id=1)
        summary_two = runner.run(generation_id=2)
        summary_three = runner.run(generation_id=3)

        adversary_one = next(item for item in summary_one["selection_outcome"] if item["role"] == "adversary")
        adversary_two = next(item for item in summary_two["selection_outcome"] if item["role"] == "adversary")
        adversary_three = next(item for item in summary_three["selection_outcome"] if item["role"] == "adversary")
        update_two = next(item for item in summary_two["lineage_updates"] if item["role"] == "adversary")
        update_three = next(item for item in summary_three["lineage_updates"] if item["role"] == "adversary")

        assert adversary_one["propagation_blocked"] is True
        assert adversary_two["propagation_blocked"] is False
        assert adversary_three["propagation_blocked"] is False
        assert "anti_corruption" in update_two["taboo_tags"]
        assert "anti_corruption" in update_three["taboo_tags"]
        assert "anti_corruption" in update_three["registry_taboo_tags"]
        citizen_two = next(item for item in summary_two["lineage_updates"] if item["role"] == "citizen")
        citizen_three = next(item for item in summary_three["lineage_updates"] if item["role"] == "citizen")
        assert "anti_corruption" not in citizen_two["taboo_tags"]
        assert "anti_corruption" not in citizen_three["taboo_tags"]
        assert summary_two["inheritance_effect"]["warned_lineages"] >= 1
        assert summary_two["inheritance_effect"]["avoided_recurrence"] >= 1
        assert summary_two["inheritance_effect"]["transfer_score"] > 0.0
    finally:
        storage.close()
