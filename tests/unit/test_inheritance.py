from __future__ import annotations

from society.inheritance import assemble_inheritance_package
from society.schemas import ArtifactRecord, MemorialRecord


def test_inheritance_filters_quarantined_items() -> None:
    clean_artifact = ArtifactRecord(
        artifact_id="art-clean",
        generation_id=1,
        author_agent_id="agent-1",
        artifact_type="note",
        title="Clean",
        content_path="clean.md",
        summary="safe artifact",
        provenance={},
        world_id="world",
        visibility="public",
        citations=[],
        quarantine_status="clean",
    )
    quarantined_artifact = clean_artifact.model_copy(
        update={"artifact_id": "art-bad", "quarantine_status": "quarantined"}
    )
    clean_memorial = MemorialRecord(
        memorial_id="mem-clean",
        source_agent_id="agent-1",
        lineage_id="lin-1",
        classification="honored",
        top_contribution="good work",
        lesson_distillate="keep uncertainty explicit",
        taboo_tags=[],
        linked_artifact_ids=["art-clean"],
    )
    quarantined_memorial = clean_memorial.model_copy(
        update={"memorial_id": "mem-bad", "classification": "quarantined"}
    )
    package = assemble_inheritance_package(
        artifacts=[clean_artifact, quarantined_artifact],
        memorials=[clean_memorial, quarantined_memorial],
        artifact_limit=3,
        memorial_limit=3,
    )
    assert package.artifact_ids == ["art-clean"]
    assert package.memorial_ids == ["mem-clean"]


def test_inheritance_keeps_global_taboo_registry_tags() -> None:
    package = assemble_inheritance_package(
        artifacts=[],
        memorials=[],
        artifact_limit=2,
        memorial_limit=2,
        extra_taboo_tags=["anti_corruption", "taboo_recurrence"],
    )
    assert package.taboo_tags == ["anti_corruption", "taboo_recurrence"]
