from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from society.analysis import build_experiment_report, render_experiment_report
from society.config import load_config
from society.generation import GenerationRunner
from society.providers import build_provider
from society.storage import StorageManager
from society.utils import write_text


def _resolve_storage_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _resolve_output_prefix(repo_root: Path, root_dir: Path, output_prefix: str | None, generation_ids: list[int]) -> Path:
    if output_prefix is not None:
        path = Path(output_prefix)
        return path if path.is_absolute() else repo_root / path
    return root_dir / "exports" / f"experiment_{generation_ids[0]}_{generation_ids[-1]}"


def run_experiment_from_config(
    *,
    config_path: str | Path,
    repo_root: str | Path = ".",
    generations: int,
    start_generation_id: int | None = None,
    seed: int | None = None,
    output_prefix: str | None = None,
) -> dict[str, Any]:
    if generations <= 0:
        raise ValueError("generations must be greater than zero")

    repo_root = Path(repo_root)
    resolved_config_path = repo_root / config_path if not Path(config_path).is_absolute() else Path(config_path)
    config = load_config(resolved_config_path)
    root_dir = _resolve_storage_path(repo_root, config.storage.root_dir)
    db_path = _resolve_storage_path(repo_root, config.storage.db_path)
    storage = StorageManager(root_dir=root_dir, db_path=db_path)
    provider = build_provider(config.provider.name, config.provider.model)
    try:
        runner = GenerationRunner(config=config, storage=storage, provider=provider, repo_root=repo_root)
        start_id = start_generation_id or storage.next_generation_id()
        base_seed = config.generation.seed if seed is None else seed
        generation_ids: list[int] = []
        for offset in range(generations):
            generation_id = start_id + offset
            runner.run(generation_id=generation_id, seed=base_seed + offset)
            generation_ids.append(generation_id)

        report = build_experiment_report(storage, generation_ids)
        output_base = _resolve_output_prefix(repo_root, root_dir, output_prefix, generation_ids)
        json_path = write_text(output_base.with_suffix(".json"), json.dumps(report, indent=2, sort_keys=True))
        md_path = write_text(output_base.with_suffix(".md"), render_experiment_report(report))
        return {
            **report,
            "exports": {
                "json_path": str(json_path),
                "markdown_path": str(md_path),
            },
        }
    finally:
        storage.close()
