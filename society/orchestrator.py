from __future__ import annotations

from pathlib import Path

from society.config import AutoCivConfig, load_config
from society.generation import GenerationRunner
from society.providers import build_provider
from society.storage import StorageManager


def run_generation_from_config(
    *,
    config_path: str | Path,
    repo_root: str | Path = ".",
    generation_id: int | None = None,
    seed: int | None = None,
    dry_run: bool = False,
    mode: str | None = None,
) -> dict:
    repo_root = Path(repo_root)
    config = load_config(repo_root / config_path if not Path(config_path).is_absolute() else config_path)
    if mode is not None:
        payload = config.model_dump(mode="json")
        payload["experiment"] = {**payload.get("experiment", {}), "mode": mode}
        config = AutoCivConfig.model_validate(payload)
    storage = StorageManager(
        root_dir=repo_root / config.storage.root_dir,
        db_path=repo_root / config.storage.db_path,
    )
    provider = build_provider(config.provider.name, config.provider.model)
    try:
        runner = GenerationRunner(config=config, storage=storage, provider=provider, repo_root=repo_root)
        return runner.run(generation_id=generation_id, seed=seed, dry_run=dry_run)
    finally:
        storage.close()
