from __future__ import annotations

import pytest

from society.config import AutoCivConfig, load_config


def test_load_config(config_path) -> None:
    config = load_config(config_path)
    assert config.generation.population_size == 3
    assert config.world.name == "shared_notebook_v0"


def test_config_rejects_role_population_mismatch(config_data: dict) -> None:
    broken = dict(config_data)
    broken["generation"] = dict(config_data["generation"])
    broken["generation"]["population_size"] = 4
    with pytest.raises(ValueError):
        AutoCivConfig.model_validate(broken)

