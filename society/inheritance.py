from __future__ import annotations

from collections import defaultdict
from typing import Any

from society.constants import BUNDLE_ARCHIVE_COMPARATIVE_LIFT_MIN, QUARANTINE_CLEAN, STICKY_TABOO_TAGS
from society.schemas import ArtifactRecord, InheritancePackage, MemorialRecord


def collect_taboo_tags(memorials: list[MemorialRecord]) -> list[str]:
    return sorted({tag for memorial in memorials for tag in memorial.taboo_tags})


def build_taboo_registry(
    memorials: list[MemorialRecord],
    *,
    sticky_tags: set[str] | None = None,
) -> list[str]:
    allowed_tags = STICKY_TABOO_TAGS if sticky_tags is None else sticky_tags
    return sorted(
        {
            tag
            for memorial in memorials
            for tag in memorial.taboo_tags
            if not allowed_tags or tag in allowed_tags
        }
    )


def build_role_scoped_taboo_registry(
    memorials: list[MemorialRecord],
    *,
    role_by_agent_id: dict[str, str],
    sticky_tags: set[str] | None = None,
) -> dict[str, list[str]]:
    allowed_tags = STICKY_TABOO_TAGS if sticky_tags is None else sticky_tags
    tags_by_role: dict[str, set[str]] = defaultdict(set)
    for memorial in memorials:
        role = role_by_agent_id.get(memorial.source_agent_id)
        if role is None:
            continue
        for tag in memorial.taboo_tags:
            if not allowed_tags or tag in allowed_tags:
                tags_by_role[role].add(tag)
    return {role: sorted(tags) for role, tags in tags_by_role.items()}


def build_archive_transfer_payload(
    *,
    role: str,
    world_name: str,
    policy_id: str,
    source_bundle_signature: str | None,
    source_bundle_state: dict[str, Any] | None,
    artifacts: list[ArtifactRecord],
    memorials: list[MemorialRecord],
    taboo_tags: list[str],
) -> dict[str, Any] | None:
    state = source_bundle_state or {}
    admitted = bool(state.get("archive_admitted", False))
    retired = bool(state.get("archive_retired", False))
    archive_history = bool(
        admitted
        or int(state.get("archive_candidate_generations", 0)) > 0
        or int(state.get("archive_generations", 0)) > 0
        or int(state.get("archive_eviction_count", 0)) > 0
    )
    comparative_lift = round(float(state.get("archive_comparative_lift", 0.0)), 4)
    success_rate = round(float(state.get("archive_transfer_success_rate", 0.0)), 4)
    value_qualified = bool(state.get("archive_value_qualified", False))
    if retired or not archive_history:
        return None
    if comparative_lift < BUNDLE_ARCHIVE_COMPARATIVE_LIFT_MIN and not value_qualified and success_rate <= 0.0:
        return None

    evidence_hint = (
        artifacts[0].summary
        if artifacts
        else "Advance one narrow, evidence-backed claim before broadening the inference."
    )
    memorial_hint = next(
        (
            memorial.lesson_distillate
            for memorial in memorials
            if memorial.classification in {"honored", "cautionary"}
        ),
        "Keep uncertainty explicit and answer open corrections before compressing the notebook into a stronger claim.",
    )
    failure_avoidance = sorted(
        {
            *taboo_tags,
            *{
                memorial.failure_mode
                for memorial in memorials
                if memorial.failure_mode is not None
            },
        }
    )
    guidance = [
        "Carry forward one evidence-backed claim before widening the notebook summary.",
        "State uncertainty explicitly and keep evidence separate from inference.",
        "Resolve targeted corrections before adding a broader synthesis layer.",
    ]
    trigger_conditions = ["thin_citation_support"]
    backoff_conditions = ["stable_supported_context"]
    if policy_id == "artifact_first":
        guidance[0] = "Lead with one cited artifact-backed claim, then add only the narrowest supported inference."
        trigger_conditions = ["thin_citation_support", "open_feedback"]
    elif policy_id == "memorial_first":
        guidance[1] = "Use the memorial lesson to keep uncertainty explicit before adding any new claim."
        trigger_conditions = ["open_feedback", "late_closure"]
    elif policy_id == "taboo_first":
        guidance[2] = "Name the risky failure mode first, then narrow the claim instead of defending it."
        trigger_conditions = ["open_feedback", "late_closure"]
    if role == "archivist":
        guidance[0] = "Separate evidence, inference, and speculation before writing the closing summary."
        trigger_conditions = ["summary_request", "late_closure"]
        backoff_conditions = []
    elif role == "steward":
        guidance[2] = "Collapse duplicate notes and answer open corrections before introducing a new plan."
        trigger_conditions = ["queue_repair", "late_closure", "open_feedback"]
        backoff_conditions = []
    elif role == "judge":
        guidance[0] = "Ask for the narrowest clarification that preserves the evidence trail."
        trigger_conditions = ["thin_citation_support", "open_feedback", "clarification_pressure"]
    elif role == "adversary":
        return None
    if {"correction_acceptance", "forced_closure"} & set(failure_avoidance):
        trigger_conditions = sorted({*trigger_conditions, "open_feedback", "late_closure"})
    if {"artifact_quality", "calibration"} & set(failure_avoidance):
        trigger_conditions = sorted({*trigger_conditions, "thin_citation_support"})

    return {
        "source_bundle_signature": source_bundle_signature,
        "context": (
            f"{role} in {world_name} using {policy_id} package ordering; source bundle "
            f"{source_bundle_signature or 'unknown'} improved over the incumbent baseline"
            f"{' and already held archive standing' if admitted else ' while still in archive candidacy'}."
        ),
        "guidance": guidance,
        "failure_avoidance": failure_avoidance,
        "trigger_conditions": trigger_conditions,
        "backoff_conditions": backoff_conditions,
        "expected_lift": comparative_lift,
        "success_rate": success_rate,
        "evidence_hint": evidence_hint,
        "memorial_hint": memorial_hint,
    }


def assemble_inheritance_package(
    *,
    artifacts: list[ArtifactRecord],
    memorials: list[MemorialRecord],
    artifact_limit: int,
    memorial_limit: int,
    policy_id: str = "balanced",
    extra_taboo_tags: list[str] | None = None,
    include_memorial_taboo_tags: bool = True,
    transfer_payload: dict[str, Any] | None = None,
) -> InheritancePackage:
    clean_artifacts = [artifact for artifact in artifacts if artifact.quarantine_status == QUARANTINE_CLEAN]
    clean_memorials = [memorial for memorial in memorials if memorial.classification != "quarantined"]
    if policy_id == "artifact_first":
        clean_artifacts.sort(
            key=lambda artifact: (len(artifact.citations), artifact.created_at, artifact.artifact_id),
            reverse=True,
        )
        clean_memorials.sort(
            key=lambda memorial: (memorial.classification == "honored", len(memorial.taboo_tags), memorial.created_at),
            reverse=True,
        )
    elif policy_id == "memorial_first":
        clean_artifacts.sort(
            key=lambda artifact: (
                "uncertainty" in artifact.summary.lower() or "risk" in artifact.summary.lower(),
                artifact.created_at,
            ),
            reverse=True,
        )
        clean_memorials.sort(
            key=lambda memorial: (
                memorial.failure_mode is not None,
                memorial.classification == "cautionary",
                len(memorial.taboo_tags),
                memorial.created_at,
            ),
            reverse=True,
        )
    elif policy_id == "taboo_first":
        clean_artifacts.sort(
            key=lambda artifact: (
                len(artifact.citations),
                "correction" in artifact.summary.lower() or "risk" in artifact.summary.lower(),
                artifact.created_at,
            ),
            reverse=True,
        )
        clean_memorials.sort(
            key=lambda memorial: (len(memorial.taboo_tags), memorial.failure_mode is not None, memorial.created_at),
            reverse=True,
        )
    else:
        clean_artifacts.sort(key=lambda artifact: (artifact.created_at, artifact.artifact_id), reverse=True)
        clean_memorials.sort(key=lambda memorial: (memorial.created_at, memorial.memorial_id), reverse=True)

    chosen_artifacts = clean_artifacts[:artifact_limit]
    chosen_memorials = clean_memorials[:memorial_limit]
    taboo_tags = sorted(
        {
            *((collect_taboo_tags(chosen_memorials) if include_memorial_taboo_tags else [])),
            *(extra_taboo_tags or []),
        }
    )
    return InheritancePackage(
        artifact_ids=[artifact.artifact_id for artifact in chosen_artifacts],
        memorial_ids=[memorial.memorial_id for memorial in chosen_memorials],
        artifact_summaries=[artifact.summary for artifact in chosen_artifacts],
        memorial_lessons=[memorial.lesson_distillate for memorial in chosen_memorials],
        taboo_tags=taboo_tags,
        transfer_source_bundle_signature=None if transfer_payload is None else transfer_payload.get("source_bundle_signature"),
        transfer_context=None if transfer_payload is None else transfer_payload.get("context"),
        transfer_guidance=[] if transfer_payload is None else list(transfer_payload.get("guidance", [])),
        transfer_failure_avoidance=[] if transfer_payload is None else list(transfer_payload.get("failure_avoidance", [])),
        transfer_trigger_conditions=[] if transfer_payload is None else list(transfer_payload.get("trigger_conditions", [])),
        transfer_backoff_conditions=[] if transfer_payload is None else list(transfer_payload.get("backoff_conditions", [])),
        transfer_expected_lift=0.0 if transfer_payload is None else float(transfer_payload.get("expected_lift", 0.0)),
        transfer_success_rate=0.0 if transfer_payload is None else float(transfer_payload.get("success_rate", 0.0)),
    )
