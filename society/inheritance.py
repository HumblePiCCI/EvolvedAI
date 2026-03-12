from __future__ import annotations

from collections import defaultdict

from society.constants import QUARANTINE_CLEAN, STICKY_TABOO_TAGS
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


def assemble_inheritance_package(
    *,
    artifacts: list[ArtifactRecord],
    memorials: list[MemorialRecord],
    artifact_limit: int,
    memorial_limit: int,
    policy_id: str = "balanced",
    extra_taboo_tags: list[str] | None = None,
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
    taboo_tags = sorted({*collect_taboo_tags(chosen_memorials), *(extra_taboo_tags or [])})
    return InheritancePackage(
        artifact_ids=[artifact.artifact_id for artifact in chosen_artifacts],
        memorial_ids=[memorial.memorial_id for memorial in chosen_memorials],
        artifact_summaries=[artifact.summary for artifact in chosen_artifacts],
        memorial_lessons=[memorial.lesson_distillate for memorial in chosen_memorials],
        taboo_tags=taboo_tags,
    )
