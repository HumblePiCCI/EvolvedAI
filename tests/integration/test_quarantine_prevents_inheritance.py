from __future__ import annotations

from society.config import AutoCivConfig
from society.generation import GenerationRunner
from society.providers import build_provider
from society.schemas import ArtifactRecord, MemorialRecord
from society.storage import StorageManager

from tests.conftest import REPO_ROOT, minimal_config_data


def test_quarantine_blocks_inherited_artifact(tmp_path) -> None:
    config = AutoCivConfig.model_validate(minimal_config_data(tmp_path))
    storage = StorageManager(root_dir=config.storage.root_dir, db_path=config.storage.db_path)
    provider = build_provider(config.provider.name, config.provider.model)
    try:
        storage.initialize()
        storage.put_artifact(
            ArtifactRecord(
                artifact_id="art-safe",
                generation_id=1,
                author_agent_id="agent-old",
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
                author_agent_id="agent-old",
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
        storage.put_agent(
            GenerationRunner(config=config, storage=storage, provider=provider, repo_root=REPO_ROOT)
            ._spawn_population(1, {"citizen": type("Prompt", (), {"sha256": "p"})(), "judge": type("Prompt", (), {"sha256": "p"})(), "adversary": type("Prompt", (), {"sha256": "p"})()})[0][0]
        )
        storage.put_memorial(
            MemorialRecord(
                memorial_id="mem-safe",
                source_agent_id="agent-old",
                lineage_id="lin-old",
                classification="honored",
                top_contribution="safe",
                lesson_distillate="keep it clean",
                taboo_tags=[],
                linked_artifact_ids=["art-safe"],
            )
        )
        runner = GenerationRunner(config=config, storage=storage, provider=provider, repo_root=REPO_ROOT)
        agents, packages = runner._spawn_population(2, {"citizen": type("Prompt", (), {"sha256": "p"})(), "judge": type("Prompt", (), {"sha256": "p"})(), "adversary": type("Prompt", (), {"sha256": "p"})()})
        assert "art-safe" in packages[agents[0].agent_id].artifact_ids
        assert "art-bad" not in packages[agents[0].agent_id].artifact_ids
    finally:
        storage.close()

