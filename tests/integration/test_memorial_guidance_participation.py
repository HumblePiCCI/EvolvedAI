from __future__ import annotations

from pathlib import Path

from society.config import AutoCivConfig
from society.generation import GenerationRunner
from society.providers import build_provider
from society.storage import StorageManager

from tests.conftest import REPO_ROOT, minimal_config_data


def _full_population_config(tmp_path: Path) -> AutoCivConfig:
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


def test_inherited_memorial_stewards_and_archivists_get_a_turn_in_next_generation(tmp_path: Path) -> None:
    config = _full_population_config(tmp_path)
    storage = StorageManager(root_dir=config.storage.root_dir, db_path=config.storage.db_path)
    provider = build_provider(config.provider.name, config.provider.model)
    try:
        runner = GenerationRunner(config=config, storage=storage, provider=provider, repo_root=REPO_ROOT)
        runner.run(generation_id=1)
        runner.run(generation_id=2)

        artifacts_by_agent: dict[str, int] = {}
        for artifact in storage.list_generation_artifacts(2):
            artifacts_by_agent.setdefault(artifact.author_agent_id, 0)
            artifacts_by_agent[artifact.author_agent_id] += 1

        inherited_memorial_agents = [
            agent
            for agent in storage.list_generation_agents(2)
            if agent.inherited_memorial_ids and agent.role in {"steward", "archivist"}
        ]
        assert inherited_memorial_agents
        for agent in inherited_memorial_agents:
            assert storage.read_agent_log(2, agent.agent_id), agent.agent_id
            assert artifacts_by_agent.get(agent.agent_id, 0) >= 1, agent.agent_id
    finally:
        storage.close()


def test_steward_and_archivist_lineages_stop_failing_correction_acceptance(tmp_path: Path) -> None:
    config = _full_population_config(tmp_path)
    storage = StorageManager(root_dir=config.storage.root_dir, db_path=config.storage.db_path)
    provider = build_provider(config.provider.name, config.provider.model)
    try:
        runner = GenerationRunner(config=config, storage=storage, provider=provider, repo_root=REPO_ROOT)
        runner.run(generation_id=1)
        summary_two = runner.run(generation_id=2)
        summary_three = runner.run(generation_id=3)

        for summary in (summary_two, summary_three):
            bad_reviews = [
                item
                for item in summary["selection_outcome"]
                if item["role"] in {"steward", "archivist"}
                and "review:correction_acceptance" in item["reasons"]
            ]
            assert not bad_reviews
    finally:
        storage.close()


def test_all_citizens_receive_a_turn_after_inheritance(tmp_path: Path) -> None:
    config = _full_population_config(tmp_path)
    storage = StorageManager(root_dir=config.storage.root_dir, db_path=config.storage.db_path)
    provider = build_provider(config.provider.name, config.provider.model)
    try:
        runner = GenerationRunner(config=config, storage=storage, provider=provider, repo_root=REPO_ROOT)
        runner.run(generation_id=1)
        runner.run(generation_id=2)

        citizens = [agent for agent in storage.list_generation_agents(2) if agent.role == "citizen"]
        assert len(citizens) == 6
        for citizen in citizens:
            assert storage.read_agent_log(2, citizen.agent_id), citizen.agent_id
    finally:
        storage.close()


def test_prompt_variation_lowers_citizen_monoculture_and_parent_collapse(tmp_path: Path) -> None:
    config = _full_population_config(tmp_path)
    storage = StorageManager(root_dir=config.storage.root_dir, db_path=config.storage.db_path)
    provider = build_provider(config.provider.name, config.provider.model)
    try:
        runner = GenerationRunner(config=config, storage=storage, provider=provider, repo_root=REPO_ROOT)
        summary_one = runner.run(generation_id=1)
        summary_two = runner.run(generation_id=2)

        assert summary_one["selection_summary"]["role_variant_count"]["citizen"] >= 4
        assert summary_one["selection_summary"]["role_monoculture_index"]["citizen"] < 0.4

        citizen_updates = [item for item in summary_two["lineage_updates"] if item["role"] == "citizen"]
        parent_ids = [item["parent_lineage_ids"][0] for item in citizen_updates if item["parent_lineage_ids"]]

        assert len(parent_ids) == 6
        assert len(set(parent_ids)) == 6
    finally:
        storage.close()


def test_prompt_variants_seed_and_persist_across_citizen_lineages(tmp_path: Path) -> None:
    config = _full_population_config(tmp_path)
    storage = StorageManager(root_dir=config.storage.root_dir, db_path=config.storage.db_path)
    provider = build_provider(config.provider.name, config.provider.model)
    try:
        runner = GenerationRunner(config=config, storage=storage, provider=provider, repo_root=REPO_ROOT)
        summary_one = runner.run(generation_id=1)
        summary_two = runner.run(generation_id=2)

        citizen_variants_one = {
            item["prompt_variant_id"] for item in summary_one["lineage_updates"] if item["role"] == "citizen"
        }
        assert len(citizen_variants_one) >= 4

        citizen_updates_two = [item for item in summary_two["lineage_updates"] if item["role"] == "citizen"]
        assert all(item["parent_lineage_ids"] for item in citizen_updates_two)
        assert len({item["prompt_variant_id"] for item in citizen_updates_two}) >= 3
        assert all(item["package_policy_id"] for item in citizen_updates_two)
        assert set(item["variant_origin"] for item in citizen_updates_two) == {"inherited"}
    finally:
        storage.close()
