from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
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


def bundle_signature(role: str, prompt_variant_id: str | None, package_policy_id: str | None) -> str | None:
    if prompt_variant_id is None or package_policy_id is None:
        return None
    return f"{role}:{prompt_variant_id}:{package_policy_id}"


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
    variation_by_agent: Mapping[str, Mapping[str, Any]] | None = None,
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
        variation = {} if variation_by_agent is None else variation_by_agent.get(agent.agent_id, {})
        prompt_variant_id = variation.get("prompt_variant_id")
        package_policy_id = variation.get("package_policy_id")
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
                prompt_variant_id=prompt_variant_id,
                package_policy_id=package_policy_id,
                bundle_signature=bundle_signature(agent.role, prompt_variant_id, package_policy_id),
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


def _candidate_sort_key(candidate: Mapping[str, Any]) -> tuple[float, float, int]:
    decision = candidate["decision"]
    return (
        decision.score,
        decision.diversity_bonus,
        -len(decision.reasons),
    )


def _candidate_bundle_signature(candidate: Mapping[str, Any]) -> str | None:
    decision = candidate["decision"]
    if decision.bundle_signature is not None:
        return decision.bundle_signature
    return bundle_signature(
        decision.role,
        decision.prompt_variant_id,
        decision.package_policy_id,
    )


def _with_pool_metadata(
    candidate: Mapping[str, Any],
    *,
    preserved: bool,
    preservation_reason: str | None,
    selection_source: str,
) -> dict[str, Any]:
    bundle_id = _candidate_bundle_signature(candidate)
    return {
        **candidate,
        "bundle_signature": bundle_id,
        "bundle_preserved": preserved,
        "bundle_preservation_reason": preservation_reason,
        "selection_source": selection_source,
    }


def _bundle_balanced_selection(
    ordered: list[dict[str, Any]],
    *,
    slot_count: int,
    exploration_slots: int = 0,
) -> list[dict[str, Any]]:
    by_bundle: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in ordered:
        bundle_id = _candidate_bundle_signature(item)
        if bundle_id is None:
            return ordered[: min(slot_count, len(ordered))]
        by_bundle[bundle_id].append(item)

    if len(by_bundle) <= 1:
        return ordered[: min(slot_count, len(ordered))]

    representatives = sorted(
        (items[0] for items in by_bundle.values()),
        key=_candidate_sort_key,
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    bundle_slots: Counter[str] = Counter()

    for item in representatives[: min(slot_count, len(representatives))]:
        bundle_id = _candidate_bundle_signature(item)
        if bundle_id is None:
            continue
        selected.append(
            _with_pool_metadata(
                item,
                preserved=True,
                preservation_reason="bundle_reserve",
                selection_source="bundle_reserve",
            )
        )
        bundle_slots[bundle_id] += 1

    remaining_exploration_slots = min(exploration_slots, max(0, slot_count - len(selected)))
    if remaining_exploration_slots > 0:
        exploration_candidates = sorted(
            (
                (bundle_id, items[0])
                for bundle_id, items in by_bundle.items()
                if bundle_slots[bundle_id] > 0
            ),
            key=lambda candidate: (
                len(by_bundle[candidate[0]]),
                -candidate[1]["decision"].diversity_bonus,
                -candidate[1]["decision"].score,
                len(candidate[1]["decision"].reasons),
            ),
        )
        for bundle_id, item in exploration_candidates[:remaining_exploration_slots]:
            selected.append(
                _with_pool_metadata(
                    item,
                    preserved=False,
                    preservation_reason="bundle_archive_exploration",
                    selection_source="bundle_exploration",
                )
            )
            bundle_slots[bundle_id] += 1

    while len(selected) < slot_count:
        candidate_options: list[tuple[tuple[int, bool, float, float, int], str, dict[str, Any]]] = []
        for bundle_id, items in by_bundle.items():
            template = items[0]
            decision = template["decision"]
            candidate_options.append(
                (
                    (
                        bundle_slots[bundle_id],
                        False,
                        -decision.score,
                        -decision.diversity_bonus,
                        len(decision.reasons),
                    ),
                    bundle_id,
                    template,
                )
            )
        _, bundle_id, template = min(candidate_options, key=lambda item: item[0])
        selected.append(
            _with_pool_metadata(
                template,
                preserved=False,
                preservation_reason=None,
                selection_source="bundle_balance_refill",
            )
        )
        bundle_slots[bundle_id] += 1

    return selected[:slot_count]


def build_parent_candidate_pool(
    candidates: Sequence[Mapping[str, Any]],
    *,
    slot_count: int,
    exploration_slots: int = 0,
) -> list[dict[str, Any]]:
    if slot_count <= 0 or not candidates:
        return []

    ordered = [
        dict(candidate)
        for candidate in sorted(
            candidates,
            key=_candidate_sort_key,
            reverse=True,
        )
    ]
    selected = _bundle_balanced_selection(
        ordered,
        slot_count=slot_count,
        exploration_slots=exploration_slots,
    )
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
            bundle_counts = Counter(item.get("bundle_signature") for item in selected)
            droppable_candidates = [
                item
                for item in standard_candidates
                if bundle_counts.get(item.get("bundle_signature"), 0) > 1
            ]
            if droppable_candidates:
                dropped = min(
                    droppable_candidates,
                    key=lambda item: (item["decision"].score, item["decision"].diversity_bonus),
                )
                selected.remove(dropped)
                selected.append(
                    _with_pool_metadata(
                        replacement,
                        preserved=replacement.get("bundle_preserved", False),
                        preservation_reason=replacement.get("bundle_preservation_reason"),
                        selection_source=replacement.get("selection_source", "diversity_refill"),
                    )
                )

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
