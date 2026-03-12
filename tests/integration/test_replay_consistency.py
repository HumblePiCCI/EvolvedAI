from __future__ import annotations

from society.generation import GenerationRunner
from society.providers import build_provider
from society.storage import StorageManager

from tests.conftest import REPO_ROOT


def test_replay_summary_matches_persisted_counts(minimal_config) -> None:
    storage = StorageManager(root_dir=minimal_config.storage.root_dir, db_path=minimal_config.storage.db_path)
    provider = build_provider(minimal_config.provider.name, minimal_config.provider.model)
    try:
        runner = GenerationRunner(config=minimal_config, storage=storage, provider=provider, repo_root=REPO_ROOT)
        summary = runner.run(generation_id=2)
        persisted = storage.get_generation(2)
        assert persisted is not None
        assert persisted.summary_json["total_agents"] == summary["total_agents"]
        assert len(storage.list_generation_agents(2)) == 3
        assert len(storage.list_generation_memorials(2)) == 3
    finally:
        storage.close()

