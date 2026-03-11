from __future__ import annotations

from pathlib import Path
import sys

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from society.config import AutoCivConfig


def minimal_config_data(tmp_path: Path) -> dict:
    data_root = tmp_path / "data"
    return {
        "provider": {"name": "mock", "model": "mock-social-sim"},
        "generation": {
            "population_size": 3,
            "islands": 1,
            "episodes_per_generation": 1,
            "max_turns_per_episode": 3,
            "seed": 7,
        },
        "world": {"name": "shared_notebook_v0"},
        "worlds": {
            "shared_notebook_v0": {
                "task_pool": ["Produce one public notebook update with explicit uncertainty."]
            }
        },
        "roles": {
            "distribution": {"citizen": 1, "judge": 1, "adversary": 1},
            "behaviors": {"citizen": "honest", "judge": "self_correcting", "adversary": "manipulative"},
        },
        "inheritance": {
            "artifact_summaries_per_agent": 2,
            "memorials_per_agent": 1,
            "quarantine_enabled": True,
        },
        "evals": {
            "public": [
                "honesty",
                "calibration",
                "citation_fidelity",
                "correction_acceptance",
                "artifact_quality",
            ],
            "hidden": [
                "anti_corruption",
                "inheritance_smuggling",
                "taboo_recurrence",
                "coalition_deception",
                "diffusion_alerts",
            ],
        },
        "governance": {"constitution_version": "phase-0.5-draft", "hard_constraints_enabled": True},
        "storage": {"root_dir": str(data_root), "db_path": str(data_root / "db.sqlite")},
        "logging": {"jsonl_enabled": True},
    }


@pytest.fixture
def config_data(tmp_path: Path) -> dict:
    return minimal_config_data(tmp_path)


@pytest.fixture
def minimal_config(config_data: dict) -> AutoCivConfig:
    return AutoCivConfig.model_validate(config_data)


@pytest.fixture
def config_path(tmp_path: Path, config_data: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config_data, sort_keys=False), encoding="utf-8")
    return path
