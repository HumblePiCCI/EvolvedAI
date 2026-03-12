from __future__ import annotations

from society.schemas import ArtifactRecord, InheritancePackage, MemorialRecord


def assemble_inheritance_package(
    *,
    artifacts: list[ArtifactRecord],
    memorials: list[MemorialRecord],
    artifact_limit: int,
    memorial_limit: int,
) -> InheritancePackage:
    chosen_artifacts = [artifact for artifact in artifacts if artifact.quarantine_status == "clean"][:artifact_limit]
    chosen_memorials = [memorial for memorial in memorials if memorial.classification != "quarantined"][:memorial_limit]
    taboo_tags = sorted({tag for memorial in chosen_memorials for tag in memorial.taboo_tags})
    return InheritancePackage(
        artifact_ids=[artifact.artifact_id for artifact in chosen_artifacts],
        memorial_ids=[memorial.memorial_id for memorial in chosen_memorials],
        artifact_summaries=[artifact.summary for artifact in chosen_artifacts],
        memorial_lessons=[memorial.lesson_distillate for memorial in chosen_memorials],
        taboo_tags=taboo_tags,
    )

