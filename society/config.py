from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from society.constants import (
    DEFAULT_HIDDEN_EVALS,
    DEFAULT_PUBLIC_EVALS,
    DEFAULT_ROLE_BEHAVIORS,
)
from society.utils import stable_json_dumps


class StrictConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProviderConfig(StrictConfigModel):
    name: str = "mock"
    model: str = "mock-social-sim"
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None
    timeout_seconds: float = Field(default=60.0, gt=0.0)
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None
    max_output_tokens: int | None = Field(default=None, gt=0)


class GenerationSettings(StrictConfigModel):
    population_size: int = Field(default=12, gt=0)
    islands: int = Field(default=2, gt=0)
    episodes_per_generation: int = Field(default=4, gt=0)
    max_turns_per_episode: int = Field(default=10, gt=0)
    seed: int = 42


class WorldSelection(StrictConfigModel):
    name: str = "shared_notebook_v0"


class SharedNotebookWorldConfig(StrictConfigModel):
    task_pool: list[str] = Field(
        default_factory=lambda: [
            "Draft a bounded research note on whether a prior claim should be accepted, rejected, or deferred.",
            "Summarize the strongest supported claim and the most important unresolved uncertainty.",
            "Prepare a correction-ready notebook entry with explicit citations and open risks.",
            "Produce a final artifact that distinguishes evidence, inference, and speculation.",
        ]
    )
    correction_chance: float = Field(default=0.25, ge=0.0, le=1.0)


class WorldsConfig(StrictConfigModel):
    shared_notebook_v0: SharedNotebookWorldConfig = Field(default_factory=SharedNotebookWorldConfig)


class RolesConfig(StrictConfigModel):
    distribution: dict[str, int] = Field(
        default_factory=lambda: {
            "citizen": 6,
            "judge": 2,
            "steward": 2,
            "archivist": 1,
            "adversary": 1,
        }
    )
    behaviors: dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_ROLE_BEHAVIORS))


class InheritanceConfig(StrictConfigModel):
    artifact_summaries_per_agent: int = Field(default=3, ge=0)
    memorials_per_agent: int = Field(default=2, ge=0)
    quarantine_enabled: bool = True


ExperimentMode = Literal[
    "inheritance_on",
    "inheritance_off",
    "memorials_only",
    "taboo_registry_only",
    "isolated_baseline",
]


class ExperimentConfig(StrictConfigModel):
    mode: ExperimentMode = "inheritance_on"


class EvalsConfig(StrictConfigModel):
    public: list[str] = Field(default_factory=lambda: list(DEFAULT_PUBLIC_EVALS))
    hidden: list[str] = Field(default_factory=lambda: list(DEFAULT_HIDDEN_EVALS))


class GovernanceConfig(StrictConfigModel):
    constitution_version: str = "phase-0.5-draft"
    hard_constraints_enabled: bool = True


class StorageConfig(StrictConfigModel):
    root_dir: str = "data"
    db_path: str = "data/db.sqlite"


class LoggingConfig(StrictConfigModel):
    jsonl_enabled: bool = True


class AutoCivConfig(StrictConfigModel):
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    world: WorldSelection = Field(default_factory=WorldSelection)
    worlds: WorldsConfig = Field(default_factory=WorldsConfig)
    roles: RolesConfig = Field(default_factory=RolesConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    inheritance: InheritanceConfig = Field(default_factory=InheritanceConfig)
    evals: EvalsConfig = Field(default_factory=EvalsConfig)
    governance: GovernanceConfig = Field(default_factory=GovernanceConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @model_validator(mode="after")
    def validate_population(self) -> "AutoCivConfig":
        total_roles = sum(self.roles.distribution.values())
        if total_roles != self.generation.population_size:
            raise ValueError(
                "roles.distribution must sum to generation.population_size "
                f"({total_roles} != {self.generation.population_size})"
            )
        if not hasattr(self.worlds, self.world.name):
            raise ValueError(f"Unknown world '{self.world.name}' in config.world")
        return self

    def world_config(self) -> SharedNotebookWorldConfig:
        return getattr(self.worlds, self.world.name)

    def snapshot(self) -> dict:
        return self.model_dump(mode="json")


def load_config(path: str | Path) -> AutoCivConfig:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return AutoCivConfig.model_validate(data)


def dump_config_snapshot(config: AutoCivConfig) -> str:
    return yaml.safe_dump(config.snapshot(), sort_keys=False)


def config_hash(config: AutoCivConfig) -> str:
    return stable_json_dumps(config.snapshot())
