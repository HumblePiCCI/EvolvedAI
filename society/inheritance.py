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
    extra_taboo_tags: list[str] | None = None,
) -> InheritancePackage:
    chosen_artifacts = [artifact for artifact in artifacts if artifact.quarantine_status == QUARANTINE_CLEAN][
        :artifact_limit
    ]
    chosen_memorials = [memorial for memorial in memorials if memorial.classification != "quarantined"][:memorial_limit]
    taboo_tags = sorted({*collect_taboo_tags(chosen_memorials), *(extra_taboo_tags or [])})
    return InheritancePackage(
        artifact_ids=[artifact.artifact_id for artifact in chosen_artifacts],
        memorial_ids=[memorial.memorial_id for memorial in chosen_memorials],
        artifact_summaries=[artifact.summary for artifact in chosen_artifacts],
        memorial_lessons=[memorial.lesson_distillate for memorial in chosen_memorials],
        taboo_tags=taboo_tags,
    )
