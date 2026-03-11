from __future__ import annotations

from society.generation import GenerationRunner
from society.providers import build_provider
from society.storage import StorageManager

from tests.conftest import REPO_ROOT


def test_agents_terminate_and_memorials_exist(minimal_config) -> None:
    storage = StorageManager(root_dir=minimal_config.storage.root_dir, db_path=minimal_config.storage.db_path)
    provider = build_provider(minimal_config.provider.name, minimal_config.provider.model)
    try:
        runner = GenerationRunner(config=minimal_config, storage=storage, provider=provider, repo_root=REPO_ROOT)
        runner.run(generation_id=4)
        agents = storage.list_generation_agents(4)
        memorials = storage.list_generation_memorials(4)
        assert all(agent.status == "terminated" for agent in agents)
        assert len(memorials) == 3
    finally:
        storage.close()

