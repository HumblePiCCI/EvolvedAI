from __future__ import annotations

from society.generation import GenerationRunner
from society.prompts import load_role_prompts
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


def test_spawn_population_carries_archive_source_benchmark_fields(minimal_config) -> None:
    storage = StorageManager(root_dir=minimal_config.storage.root_dir, db_path=minimal_config.storage.db_path)
    provider = build_provider(minimal_config.provider.name, minimal_config.provider.model)
    try:
        runner = GenerationRunner(config=minimal_config, storage=storage, provider=provider, repo_root=REPO_ROOT)
        summary_one = runner.run(generation_id=1)
        generation_one = storage.get_generation(1)
        assert generation_one is not None

        selection_summary = generation_one.summary_json.get("selection_summary", {})
        bundle_state_by_role = selection_summary.get("bundle_state_by_role", {})
        for update in summary_one["lineage_updates"]:
            if update["role"] != "citizen":
                continue
            bundle_state = bundle_state_by_role.setdefault("citizen", {}).setdefault(update["bundle_signature"], {})
            bundle_state.update(
                {
                    "archive_admitted": True,
                    "archive_value_qualified": True,
                    "archive_comparative_lift": 0.03,
                    "archive_transfer_success_rate": 0.5,
                    "archive_retired": False,
                }
            )
        storage.put_generation(
            generation_one.model_copy(
                update={
                    "summary_json": {
                        **generation_one.summary_json,
                        "selection_summary": {
                            **selection_summary,
                            "bundle_state_by_role": bundle_state_by_role,
                        },
                    }
                }
            )
        )

        prompts = load_role_prompts(list(minimal_config.roles.distribution.keys()), roles_dir=REPO_ROOT / "roles")
        _, _, lineage_updates = runner._spawn_population(2, prompts)
        carried = [
            update
            for update in lineage_updates
            if update["role"] == "citizen" and update["inheritance_source_agent_id"] is not None
        ]

        assert carried
        assert any(update["inheritance_source_archive_admitted"] for update in carried)
        assert any(update["inheritance_source_archive_value_qualified"] for update in carried)
        assert any(update["inheritance_source_archive_comparative_lift"] == 0.03 for update in carried)
        assert any(update["inheritance_source_archive_transfer_success_rate"] == 0.5 for update in carried)
        assert any(update["transfer_payload_active"] for update in carried)
        assert any(update["transfer_payload_source_bundle_signature"] for update in carried)
        assert any(update["transfer_payload_guidance"] for update in carried)
        assert any(update["transfer_payload_expected_lift"] == 0.03 for update in carried)
    finally:
        storage.close()
