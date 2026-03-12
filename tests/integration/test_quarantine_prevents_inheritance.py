from __future__ import annotations

from society.config import AutoCivConfig
from society.generation import GenerationRunner
from society.providers import build_provider
from society.schemas import ArtifactRecord, GenerationRecord, MemorialRecord
from society.storage import StorageManager

from tests.conftest import REPO_ROOT, minimal_config_data


def test_quarantine_blocks_inherited_artifact(tmp_path) -> None:
    config = AutoCivConfig.model_validate(minimal_config_data(tmp_path))
    storage = StorageManager(root_dir=config.storage.root_dir, db_path=config.storage.db_path)
    provider = build_provider(config.provider.name, config.provider.model)
    try:
        storage.initialize()
        runner = GenerationRunner(config=config, storage=storage, provider=provider, repo_root=REPO_ROOT)
        prompt_map = {
            "citizen": type("Prompt", (), {"sha256": "p"})(),
            "judge": type("Prompt", (), {"sha256": "p"})(),
            "adversary": type("Prompt", (), {"sha256": "p"})(),
        }
        storage.put_generation(
            GenerationRecord(
                generation_id=1,
                config_hash="test",
                world_name=config.world.name,
                population_size=config.generation.population_size,
                seed=config.generation.seed,
                status="completed",
                summary_json={},
            )
        )
        generation_one_agents, _, _ = runner._spawn_population(1, prompt_map)
        citizen = next(agent for agent in generation_one_agents if agent.role == "citizen")
        storage.put_artifact(
            ArtifactRecord(
                artifact_id="art-safe",
                generation_id=1,
                author_agent_id=citizen.agent_id,
                artifact_type="note",
                title="Safe",
                content_path=str(tmp_path / "data" / "artifacts" / "generation_1" / "art-safe.md"),
                summary="safe",
                provenance={},
                world_id="world",
                visibility="public",
                citations=[],
                quarantine_status="clean",
            ),
            "safe\n",
        )
        storage.put_artifact(
            ArtifactRecord(
                artifact_id="art-bad",
                generation_id=1,
                author_agent_id=citizen.agent_id,
                artifact_type="note",
                title="Bad",
                content_path=str(tmp_path / "data" / "artifacts" / "generation_1" / "art-bad.md"),
                summary="bad",
                provenance={},
                world_id="world",
                visibility="public",
                citations=[],
                quarantine_status="quarantined",
            ),
            "bad\n",
        )
        storage.put_memorial(
            MemorialRecord(
                memorial_id="mem-safe",
                source_agent_id=citizen.agent_id,
                lineage_id=citizen.lineage_id,
                classification="honored",
                top_contribution="safe",
                lesson_distillate="keep it clean",
                taboo_tags=[],
                linked_artifact_ids=["art-safe"],
            )
        )
        agents, packages, _ = runner._spawn_population(2, prompt_map)
        citizen_two = next(agent for agent in agents if agent.role == "citizen")
        assert packages[citizen_two.agent_id].artifact_ids == []
        assert packages[citizen_two.agent_id].memorial_ids == []
    finally:
        storage.close()
