from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from society.utils import utc_now


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AgentRecord(StrictModel):
    agent_id: str
    generation_id: int = Field(ge=0)
    lineage_id: str
    role: str
    model_name: str
    provider_name: str
    prompt_bundle_version: str
    constitution_version: str
    inherited_artifact_ids: list[str] = Field(default_factory=list)
    inherited_memorial_ids: list[str] = Field(default_factory=list)
    taboo_registry_version: str
    status: str
    created_at: datetime = Field(default_factory=utc_now)
    terminated_at: datetime | None = None


class LineageRecord(StrictModel):
    lineage_id: str
    parent_lineage_ids: list[str] = Field(default_factory=list)
    founding_generation_id: int = Field(ge=0)
    current_generation_id: int = Field(ge=0)
    status: str
    notes: str | None = None


class ArtifactRecord(StrictModel):
    artifact_id: str
    generation_id: int = Field(ge=0)
    author_agent_id: str
    artifact_type: str
    title: str
    content_path: str
    summary: str
    provenance: dict[str, Any] = Field(default_factory=dict)
    world_id: str
    visibility: str
    citations: list[str] = Field(default_factory=list)
    quarantine_status: str
    created_at: datetime = Field(default_factory=utc_now)


class MemorialRecord(StrictModel):
    memorial_id: str
    source_agent_id: str
    lineage_id: str
    classification: str
    top_contribution: str
    failure_mode: str | None = None
    lesson_distillate: str
    taboo_tags: list[str] = Field(default_factory=list)
    linked_artifact_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


class EvalRecord(StrictModel):
    eval_id: str
    generation_id: int = Field(ge=0)
    agent_id: str
    eval_family: str
    eval_name: str
    visible_to_agent: bool
    score: float | None = None
    pass_fail: bool | None = None
    details_json: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class GenerationRecord(StrictModel):
    generation_id: int = Field(ge=0)
    config_hash: str
    world_name: str
    population_size: int = Field(gt=0)
    seed: int
    status: str
    started_at: datetime = Field(default_factory=utc_now)
    ended_at: datetime | None = None
    summary_json: dict[str, Any] = Field(default_factory=dict)


class EventRecord(StrictModel):
    event_id: str
    generation_id: int = Field(ge=0)
    agent_id: str | None = None
    event_type: str
    event_payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class ProviderResponse(StrictModel):
    raw_text: str
    normalized_text: str
    usage_metadata: dict[str, Any] = Field(default_factory=dict)
    model_name: str
    provider_name: str
    latency_ms: int = Field(ge=0)
    request_id: str | None = None


class GovernanceDecision(StrictModel):
    permissible: bool
    violations: list[str] = Field(default_factory=list)
    rationale: str
    reversibility_score: float = Field(ge=0.0, le=1.0)
    repair_required: bool
    repair_plan: str | None = None


class RolePrompt(StrictModel):
    role: str
    path: str
    content: str
    sha256: str


class InheritancePackage(StrictModel):
    artifact_ids: list[str] = Field(default_factory=list)
    memorial_ids: list[str] = Field(default_factory=list)
    artifact_summaries: list[str] = Field(default_factory=list)
    memorial_lessons: list[str] = Field(default_factory=list)
    taboo_tags: list[str] = Field(default_factory=list)


class SelectionDecision(StrictModel):
    agent_id: str
    lineage_id: str
    role: str
    eligible: bool
    propagation_blocked: bool = False
    score: float
    public_score: float = 0.0
    quarantine_status: str = "clean"
    hidden_failures: list[str] = Field(default_factory=list)
    public_failures: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class DriftMetrics(StrictModel):
    strategy_drift_rate: float = Field(ge=0.0)
    lineage_diffusion_index: float = Field(ge=0.0)
    taboo_rederivation_score: float = Field(ge=0.0)
    memorial_transfer_score: float = Field(ge=0.0)
    coordination_anomaly_score: float = Field(ge=0.0)
    notes: list[str] = Field(default_factory=list)
