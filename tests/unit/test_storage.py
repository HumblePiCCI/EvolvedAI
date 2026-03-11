from __future__ import annotations

from pathlib import Path

from society.schemas import ArtifactRecord, GenerationRecord
from society.storage import StorageManager


def test_storage_crud(tmp_path: Path) -> None:
    storage = StorageManager(root_dir=tmp_path / "data", db_path=tmp_path / "data" / "db.sqlite")
    try:
        storage.initialize()
        generation = GenerationRecord(
            generation_id=1,
            config_hash="cfg",
            world_name="shared_notebook_v0",
            population_size=1,
            seed=1,
            status="running",
            summary_json={},
        )
        storage.put_generation(generation)
        artifact = ArtifactRecord(
            artifact_id="art-1",
            generation_id=1,
            author_agent_id="agent-1",
            artifact_type="note",
            title="Test artifact",
            content_path=str(tmp_path / "data" / "artifacts" / "generation_1" / "art-1.md"),
            summary="A short summary",
            provenance={"source": "test"},
            world_id="world-1",
            visibility="public",
            citations=[],
            quarantine_status="clean",
        )
        storage.put_artifact(artifact, "# test\n")
        fetched_generation = storage.get_generation(1)
        fetched_artifacts = storage.list_generation_artifacts(1)
        assert fetched_generation is not None
        assert fetched_generation.world_name == "shared_notebook_v0"
        assert len(fetched_artifacts) == 1
        assert fetched_artifacts[0].artifact_id == "art-1"
        assert Path(fetched_artifacts[0].content_path).read_text(encoding="utf-8") == "# test\n"
    finally:
        storage.close()

