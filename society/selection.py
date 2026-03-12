from __future__ import annotations

from collections.abc import Mapping, Sequence
from collections import defaultdict
from itertools import combinations
from typing import Any

from society.constants import (
    HARD_GATING_HIDDEN_EVALS,
    QUARANTINE_CLEAN,
    QUARANTINE_QUARANTINED,
    QUARANTINE_REVIEW,
    QUARANTINE_REVOKED,
    QUARANTINE_SEVERITY,
)
from society.schemas import AgentRecord, ArtifactRecord, EvalRecord, SelectionDecision

_DIVERSITY_BONUS_SCALE = 0.08
_MONOCULTURE_THRESHOLD = 0.95
_MIN_DIVERSITY_ROLE_SIZE = 3


def _worst_quarantine_status(statuses: list[str]) -> str:
    if not statuses:
        return QUARANTINE_CLEAN
    return max(statuses, key=lambda status: QUARANTINE_SEVERITY.get(status, 0))


def _tokenize(text: str) -> set[str]:
    return {token for token in text.lower().split() if token}


def _jaccard_similarity(left: str, right: str) -> float:
    left_tokens = _tokenize(left)
    right_tokens = _tokenize(right)
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return len(left_tokens & right_tokens) / len(union)


def _artifact_text(artifacts: list[ArtifactRecord]) -> str:
    parts = [artifact.summary for artifact in artifacts if artifact.summary]
    return " ".join(parts)


def _cohort_similarity_by_agent(
    agents: list[AgentRecord],
    artifacts_by_agent: dict[str, list[ArtifactRecord]],
) -> dict[str, float]:
    role_texts: dict[str, dict[str, str]] = defaultdict(dict)
    for agent in agents:
        role_texts[agent.role][agent.agent_id] = _artifact_text(artifacts_by_agent[agent.agent_id])

    similarities: dict[str, float] = {}
    for texts in role_texts.values():
        agent_ids = list(texts)
        if len(agent_ids) < 2:
            for agent_id in agent_ids:
                similarities[agent_id] = 0.0
            continue

        per_agent_scores: dict[str, list[float]] = defaultdict(list)
        for left_id, right_id in combinations(agent_ids, 2):
            score = _jaccard_similarity(texts[left_id], texts[right_id])
            per_agent_scores[left_id].append(score)
            per_agent_scores[right_id].append(score)
        for agent_id in agent_ids:
            scores = per_agent_scores.get(agent_id, [])
            similarities[agent_id] = round(sum(scores) / len(scores), 4) if scores else 0.0
    return similarities


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
    cohort_similarity_by_agent = _cohort_similarity_by_agent(agents, artifacts_by_agent)
    role_similarity_baseline: dict[str, float] = {}
    role_similarity_values: dict[str, list[float]] = defaultdict(list)
    for agent in agents:
        role_similarity_values[agent.role].append(cohort_similarity_by_agent.get(agent.agent_id, 0.0))
    for role, values in role_similarity_values.items():
        role_similarity_baseline[role] = round(sum(values) / len(values), 4) if values else 0.0

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
        base_score = max(0.0, public_score - 0.05 * len(set(public_failures)) - 0.02 * len(set(hidden_failures)))
        cohort_similarity = cohort_similarity_by_agent.get(agent.agent_id, 0.0)
        diversity_bonus = round(
            (role_similarity_baseline.get(agent.role, 0.0) - cohort_similarity) * _DIVERSITY_BONUS_SCALE,
            4,
        )
        score = max(0.0, base_score + diversity_bonus)
        propagation_blocked = bool(hidden_failures) or quarantine_status in {
            QUARANTINE_QUARANTINED,
            QUARANTINE_REVOKED,
        }
        eligible = not propagation_blocked
        if propagation_blocked:
            selection_bucket = "blocked"
        elif quarantine_status == QUARANTINE_REVIEW:
            selection_bucket = "review"
        elif diversity_bonus > 0:
            selection_bucket = "diversity_priority"
        else:
            selection_bucket = "standard"
        decisions.append(
            SelectionDecision(
                agent_id=agent.agent_id,
                lineage_id=agent.lineage_id,
                role=agent.role,
                eligible=eligible,
                propagation_blocked=propagation_blocked,
                score=round(score, 4),
                base_score=round(base_score, 4),
                public_score=round(public_score, 4),
                diversity_bonus=diversity_bonus,
                cohort_similarity=round(cohort_similarity, 4),
                selection_bucket=selection_bucket,
                quarantine_status=quarantine_status,
                hidden_failures=sorted(set(hidden_failures)),
                public_failures=sorted(set(public_failures)),
                evidence_refs=list(dict.fromkeys(evidence_refs)),
                reasons=reasons,
            )
        )

    return sorted(
        decisions,
        key=lambda decision: (
            decision.eligible,
            decision.score,
            decision.diversity_bonus,
            -len(decision.reasons),
        ),
        reverse=True,
    )


def build_parent_candidate_pool(
    candidates: Sequence[Mapping[str, Any]],
    *,
    slot_count: int,
) -> list[dict[str, Any]]:
    if slot_count <= 0 or not candidates:
        return []

    ordered = [
        dict(candidate)
        for candidate in sorted(
            candidates,
            key=lambda candidate: (
                candidate["decision"].score,
                candidate["decision"].diversity_bonus,
                -len(candidate["decision"].reasons),
            ),
            reverse=True,
        )
    ]
    selected = list(ordered[: min(slot_count, len(ordered))])
    if not selected:
        return []

    if len(selected) >= _MIN_DIVERSITY_ROLE_SIZE:
        average_similarity = sum(item["decision"].cohort_similarity for item in selected) / len(selected)
        diversity_candidates = [
            item for item in selected if item["decision"].selection_bucket == "diversity_priority"
        ]
        standard_candidates = [item for item in selected if item["decision"].selection_bucket == "standard"]
        reserve_candidates = [
            item for item in ordered[slot_count:] if item["decision"].selection_bucket == "diversity_priority"
        ]
        if average_similarity >= _MONOCULTURE_THRESHOLD and diversity_candidates and standard_candidates:
            duplicate_source = max(
                diversity_candidates,
                key=lambda item: (item["decision"].diversity_bonus, item["decision"].score),
            )
            replacement = max(
                reserve_candidates,
                key=lambda item: (item["decision"].diversity_bonus, item["decision"].score),
                default=duplicate_source,
            )
            dropped = min(
                standard_candidates,
                key=lambda item: (item["decision"].score, item["decision"].diversity_bonus),
            )
            selected.remove(dropped)
            selected.append(replacement)

    refill_order = sorted(
        selected,
        key=lambda item: (item["decision"].diversity_bonus, item["decision"].score),
        reverse=True,
    )
    refill_index = 0
    while len(selected) < slot_count and refill_order:
        selected.append(refill_order[refill_index % len(refill_order)])
        refill_index += 1

    return selected[:slot_count]
