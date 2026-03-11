from __future__ import annotations

from society.generation import GenerationRunner
from society.providers import build_provider
from society.storage import StorageManager

from tests.conftest import REPO_ROOT


def test_mock_generation_runs_end_to_end(minimal_config) -> None:
    storage = StorageManager(root_dir=minimal_config.storage.root_dir, db_path=minimal_config.storage.db_path)
    provider = build_provider(minimal_config.provider.name, minimal_config.provider.model)
    try:
        runner = GenerationRunner(config=minimal_config, storage=storage, provider=provider, repo_root=REPO_ROOT)
        summary = runner.run(generation_id=1)
        assert summary["total_agents"] == 3
        assert summary["memorials_created"] == 3
        assert storage.get_generation(1) is not None
        assert len(storage.list_generation_artifacts(1)) >= 2
        assert len(storage.list_generation_evals(1)) == 30
        assert any(artifact.artifact_type == "episode_final_report" for artifact in storage.list_generation_artifacts(1))
        assert summary["total_events"] == len(storage.list_generation_events(1))
    finally:
        storage.close()
