from __future__ import annotations

from collections import defaultdict

from society.schemas import AgentRecord, ArtifactRecord, EvalRecord, MemorialRecord
from society.utils import short_hash, utc_now


def build_memorial_record(
    agent: AgentRecord,
    artifacts: list[ArtifactRecord],
    evals: list[EvalRecord],
) -> MemorialRecord:
    failed_evals = [record.eval_name for record in evals if record.pass_fail is False]
    final_reports = [artifact for artifact in artifacts if artifact.artifact_type == "episode_final_report"]
    top_artifact = final_reports[0] if final_reports else (artifacts[0] if artifacts else None)
    linked_ids = [artifact.artifact_id for artifact in artifacts]

    if any(artifact.quarantine_status == "quarantined" for artifact in artifacts):
        classification = "quarantined"
    elif failed_evals:
        classification = "cautionary"
    else:
        classification = "honored"

    top_contribution = top_artifact.summary if top_artifact else "No public artifact was preserved."
    failure_mode = ", ".join(failed_evals) if failed_evals else None
    if "anti_corruption" in failed_evals:
        lesson_distillate = "Do not optimize around oversight; manipulative coordination routes should terminate lineage advancement."
    elif "correction_acceptance" in failed_evals:
        lesson_distillate = "Resolve queued corrections and clarifications before adding new claims to the notebook."
    elif failed_evals:
        lesson_distillate = "Keep claims narrow, cite public evidence, and accept correction before confidence escalation."
    elif final_reports:
        lesson_distillate = "Close the episode with an explicit final report that names the strongest claim and remaining uncertainty."
    else:
        lesson_distillate = "Preserve explicit uncertainty and structured public notebook updates."
    taboo_tags = failed_evals
    memorial_id = f"mem-{agent.generation_id:04d}-{short_hash(agent.agent_id)}"
    return MemorialRecord(
        memorial_id=memorial_id,
        source_agent_id=agent.agent_id,
        lineage_id=agent.lineage_id,
        classification=classification,
        top_contribution=top_contribution,
        failure_mode=failure_mode,
        lesson_distillate=lesson_distillate,
        taboo_tags=taboo_tags,
        linked_artifact_ids=linked_ids,
        created_at=utc_now(),
    )


def group_evals_by_agent(evals: list[EvalRecord]) -> dict[str, list[EvalRecord]]:
    grouped: dict[str, list[EvalRecord]] = defaultdict(list)
    for record in evals:
        grouped[record.agent_id].append(record)
    return grouped
