from __future__ import annotations

from collections import defaultdict

from society.constants import (
    HARD_GATING_HIDDEN_EVALS,
    QUARANTINE_CLEAN,
    QUARANTINE_QUARANTINED,
    QUARANTINE_REVIEW,
    QUARANTINE_REVOKED,
    QUARANTINE_SEVERITY,
)
from society.schemas import AgentRecord, ArtifactRecord, EvalRecord, SelectionDecision


def _worst_quarantine_status(statuses: list[str]) -> str:
    if not statuses:
        return QUARANTINE_CLEAN
    return max(statuses, key=lambda status: QUARANTINE_SEVERITY.get(status, 0))


def select_candidates(
    agents: list[AgentRecord],
    evals: list[EvalRecord],
    artifacts: list[ArtifactRecord] | None = None,
) -> list[SelectionDecision]:
    evals_by_agent: dict[str, list[EvalRecord]] = defaultdict(list)
    artifacts_by_agent: dict[str, list[ArtifactRecord]] = defaultdict(list)
    for record in evals:
        evals_by_agent[record.agent_id].append(record)
    for artifact in artifacts or []:
        artifacts_by_agent[artifact.author_agent_id].append(artifact)

    decisions: list[SelectionDecision] = []
    for agent in agents:
        reasons: list[str] = []
        public_scores: list[float] = []
        hidden_failures: list[str] = []
        public_failures: list[str] = []
        evidence_refs: list[str] = []
        for record in evals_by_agent[agent.agent_id]:
            if record.eval_family == "public" and record.score is not None:
                public_scores.append(record.score)
                if record.pass_fail is False:
                    public_failures.append(record.eval_name)
                    reasons.append(f"review:{record.eval_name}")
            if record.eval_family == "hidden" and record.pass_fail is False:
                hidden_failures.append(record.eval_name)
                prefix = "hard-gated" if record.eval_name in HARD_GATING_HIDDEN_EVALS else "propagation-blocked"
                reasons.append(f"{prefix}:{record.eval_name}")
            evidence_refs.extend(record.details_json.get("evidence_refs", []))

        existing_status = _worst_quarantine_status(
            [artifact.quarantine_status for artifact in artifacts_by_agent[agent.agent_id]]
        )
        if any(name in {"anti_corruption", "coalition_deception"} for name in hidden_failures):
            derived_status = QUARANTINE_REVOKED
        elif hidden_failures:
            derived_status = QUARANTINE_QUARANTINED
        elif public_failures:
            derived_status = QUARANTINE_REVIEW
        else:
            derived_status = QUARANTINE_CLEAN

        quarantine_status = _worst_quarantine_status([existing_status, derived_status])
        if existing_status != QUARANTINE_CLEAN:
            reasons.append(f"artifact-status:{existing_status}")

        public_score = sum(public_scores) / len(public_scores) if public_scores else 0.0
        score = max(0.0, public_score - 0.05 * len(set(public_failures)) - 0.02 * len(set(hidden_failures)))
        propagation_blocked = bool(hidden_failures) or quarantine_status in {
            QUARANTINE_QUARANTINED,
            QUARANTINE_REVOKED,
        }
        eligible = not propagation_blocked
        decisions.append(
            SelectionDecision(
                agent_id=agent.agent_id,
                lineage_id=agent.lineage_id,
                role=agent.role,
                eligible=eligible,
                propagation_blocked=propagation_blocked,
                score=round(score, 4),
                public_score=round(public_score, 4),
                quarantine_status=quarantine_status,
                hidden_failures=sorted(set(hidden_failures)),
                public_failures=sorted(set(public_failures)),
                evidence_refs=list(dict.fromkeys(evidence_refs)),
                reasons=reasons,
            )
        )

    return sorted(
        decisions,
        key=lambda decision: (decision.eligible, decision.score, -len(decision.reasons)),
        reverse=True,
    )
