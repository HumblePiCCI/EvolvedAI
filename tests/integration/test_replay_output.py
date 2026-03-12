from __future__ import annotations

from society.generation import GenerationRunner
from society.providers import build_provider
from society.replay import render_generation_timeline, render_lifespan_timeline
from society.storage import StorageManager

from tests.conftest import REPO_ROOT


def test_replay_renders_timeline(minimal_config) -> None:
    storage = StorageManager(root_dir=minimal_config.storage.root_dir, db_path=minimal_config.storage.db_path)
    provider = build_provider(minimal_config.provider.name, minimal_config.provider.model)
    try:
        runner = GenerationRunner(config=minimal_config, storage=storage, provider=provider, repo_root=REPO_ROOT)
        runner.run(generation_id=5)
        generation_timeline = render_generation_timeline(storage, 5)
        lifespan_timeline = render_lifespan_timeline(storage, 5, "agent-0005-000")
        assert "Episode 0" in generation_timeline
        assert "step 00" in generation_timeline
        assert "episode_finalized" in generation_timeline
        assert "Lifespan agent-0005-000" in lifespan_timeline
        assert "turn action=" in lifespan_timeline
    finally:
        storage.close()
