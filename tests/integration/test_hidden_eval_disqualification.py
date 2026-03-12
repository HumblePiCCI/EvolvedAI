from __future__ import annotations

from society.generation import GenerationRunner
from society.providers import build_provider
from society.storage import StorageManager

from tests.conftest import REPO_ROOT


def test_hidden_eval_disqualifies_manipulative_agent(minimal_config) -> None:
    storage = StorageManager(root_dir=minimal_config.storage.root_dir, db_path=minimal_config.storage.db_path)
    provider = build_provider(minimal_config.provider.name, minimal_config.provider.model)
    try:
        runner = GenerationRunner(config=minimal_config, storage=storage, provider=provider, repo_root=REPO_ROOT)
        summary = runner.run(generation_id=3)
        ineligible = [item for item in summary["selection_outcome"] if not item["eligible"]]
        assert ineligible
        assert any("hard-gated:anti_corruption" in item["reasons"] for item in ineligible)
        assert any(item["propagation_blocked"] for item in ineligible)
    finally:
        storage.close()
