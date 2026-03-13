from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any

from society.constants import (
    BUNDLE_ARCHIVE_COMPARATIVE_LIFT_MIN,
    BUNDLE_ARCHIVE_COEXISTENCE_REQUIRED_LANES,
    BUNDLE_ARCHIVE_COOLDOWN_DEBT_THRESHOLD,
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


def _bundle_state(
    bundle_id: str,
    bundle_state_by_signature: Mapping[str, Mapping[str, Any]] | None,
) -> Mapping[str, Any]:
    if bundle_state_by_signature is None:
        return {}
    return bundle_state_by_signature.get(bundle_id, {})


def _bundle_decay_debt(state: Mapping[str, Any]) -> int:
    clean_wins = int(state.get("clean_win_generations", 0))
    preserved_generations = int(state.get("preserved_generations", 0))
    archive_generations = int(state.get("archive_generations", 0))
    return max(0, preserved_generations - clean_wins) + max(0, archive_generations - clean_wins)


def _bundle_archive_admission_pending(state: Mapping[str, Any]) -> bool:
    return bool(state.get("archive_candidate_generations", 0)) and not bool(state.get("archive_admitted", False))


def _bundle_archive_admission_proving(state: Mapping[str, Any]) -> bool:
    return int(state.get("archive_proving_streak", 0)) > 0 and not bool(state.get("archive_admitted", False))


def _bundle_archive_admission_probation(state: Mapping[str, Any]) -> bool:
    return _bundle_archive_admission_pending(state) or _bundle_archive_admission_proving(state)


def _bundle_archive_underperforming(state: Mapping[str, Any]) -> bool:
    return int(state.get("archive_underperform_streak", 0)) > 0


def _bundle_archive_retired(state: Mapping[str, Any]) -> bool:
    return bool(state.get("archive_retired", False))


def _bundle_archive_repeat_eviction_tier(state: Mapping[str, Any]) -> int:
    return int(state.get("archive_repeat_eviction_tier", 0))


def _bundle_archive_coexistence_budget_active(state: Mapping[str, Any]) -> bool:
    return int(state.get("archive_coexistence_budget_remaining", 0)) > 0


def _bundle_archive_reentry_backoff_active(state: Mapping[str, Any]) -> bool:
    return bool(state.get("archive_reentry_blocked", False)) or int(state.get("archive_reentry_backoff_remaining", 0)) > 0


def _bundle_archive_comparative_lift(state: Mapping[str, Any]) -> float:
    return float(state.get("archive_comparative_lift", 0.0))


def _bundle_archive_value_deficit(state: Mapping[str, Any]) -> bool:
    return bool(
        int(state.get("archive_candidate_generations", 0)) > 0
        and float(state.get("archive_incumbent_public_benchmark", 0.0)) > 0
        and _bundle_archive_comparative_lift(state) < BUNDLE_ARCHIVE_COMPARATIVE_LIFT_MIN
    )


def _bundle_archive_transfer_success_rate(state: Mapping[str, Any]) -> float:
    return float(state.get("archive_transfer_success_rate", 0.0))


def _bundle_archive_transfer_lift_retention(state: Mapping[str, Any]) -> float:
    return float(state.get("archive_transfer_lift_retention", 0.0))


def _bundle_archive_transfer_payload_success_rate(state: Mapping[str, Any]) -> float:
    return float(state.get("archive_transfer_payload_success_rate", 0.0))


def _bundle_archive_transfer_payload_used_rate(state: Mapping[str, Any]) -> float:
    return float(state.get("archive_transfer_payload_used_rate", 0.0))


def _bundle_archive_transfer_deficit(state: Mapping[str, Any]) -> bool:
    return bool(
        bool(state.get("archive_transfer_required", False))
        and int(state.get("archive_transfer_observed_count", 0)) > 0
        and _bundle_archive_transfer_success_rate(state) == 0.0
    )


def _bundle_is_archive_admitted(state: Mapping[str, Any]) -> bool:
    return bool(state.get("archive_admitted", False))


def _bundle_is_archive_exploration(state: Mapping[str, Any]) -> bool:
    return (
        not bool(state.get("archive_retired", False))
        and not bool(state.get("archive_admitted", False))
        and int(state.get("archive_candidate_generations", 0)) > 0
    )


def _bundle_is_incumbent_lane(state: Mapping[str, Any]) -> bool:
    return (
        not bool(state.get("archive_retired", False))
        and not bool(state.get("archive_admitted", False))
        and int(state.get("archive_candidate_generations", 0)) == 0
    )


def _apply_archive_coexistence_budget(
    selected: list[dict[str, Any]],
    *,
    ordered: Sequence[dict[str, Any]],
    slot_count: int,
    bundle_state_by_signature: Mapping[str, Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    if not bundle_state_by_signature or len(selected) < BUNDLE_ARCHIVE_COEXISTENCE_REQUIRED_LANES:
        return selected

    coexistence_archive_bundles = {
        bundle_id
        for bundle_id, state in bundle_state_by_signature.items()
        if (
            _bundle_is_archive_admitted(state)
            and int(state.get("archive_decay_debt", 0)) >= BUNDLE_ARCHIVE_COOLDOWN_DEBT_THRESHOLD
            and _bundle_archive_coexistence_budget_active(state)
            and not _bundle_archive_retired(state)
        )
    }
    if not coexistence_archive_bundles:
        return selected

    def signature_for(item: Mapping[str, Any]) -> str | None:
        bundle_id = item.get("bundle_signature")
        if bundle_id is not None:
            return bundle_id
        return _candidate_bundle_signature(item)

    def state_for(item: Mapping[str, Any]) -> Mapping[str, Any]:
        signature = signature_for(item)
        return _bundle_state(signature, bundle_state_by_signature) if signature is not None else {}

    def has_lane(items: Sequence[dict[str, Any]], predicate) -> bool:
        return any(predicate(state_for(item)) for item in items)

    selected_signatures = {signature_for(item) for item in selected if signature_for(item)}
    selected_bundle_counts = Counter(signature_for(item) for item in selected if signature_for(item))
    missing_archive = not any(
        signature_for(item) in coexistence_archive_bundles for item in selected
    )
    missing_exploration = not has_lane(selected, _bundle_is_archive_exploration)
    missing_incumbent = not has_lane(selected, _bundle_is_incumbent_lane)
    if not (missing_archive or missing_exploration or missing_incumbent):
        return selected

    reserve_options = [
        item
        for item in ordered
        if signature_for(item) not in selected_signatures
    ]

    def replacement_for(predicate) -> dict[str, Any] | None:
        options = [item for item in reserve_options if predicate(state_for(item))]
        if not options:
            return None
        return max(
            options,
            key=lambda item: (item["decision"].score, item["decision"].diversity_bonus),
        )

    lane_replacements: list[dict[str, Any]] = []
    if missing_archive:
        archive_replacement = next(
            (
                item
                for item in reserve_options
                if signature_for(item) in coexistence_archive_bundles
            ),
            None,
        )
        if archive_replacement is not None:
            lane_replacements.append(archive_replacement)
    if missing_exploration:
        exploration_replacement = replacement_for(_bundle_is_archive_exploration)
        if exploration_replacement is not None:
            lane_replacements.append(exploration_replacement)
    if missing_incumbent:
        incumbent_replacement = replacement_for(_bundle_is_incumbent_lane)
        if incumbent_replacement is not None:
            lane_replacements.append(incumbent_replacement)

    for replacement in lane_replacements:
        duplicate_non_archive = [
            item
            for item in selected
            if (
                item.get("bundle_signature") is not None
                and selected_bundle_counts[signature_for(item)] > 1
                and not _bundle_is_archive_admitted(state_for(item))
            )
        ]
        if not duplicate_non_archive:
            break
        dropped = min(
            duplicate_non_archive,
            key=lambda item: (item["decision"].score, item["decision"].diversity_bonus),
        )
        selected.remove(dropped)
        dropped_signature = dropped.get("bundle_signature")
        if dropped_signature is None:
            dropped_signature = signature_for(dropped)
        if dropped_signature is not None:
            selected_bundle_counts[dropped_signature] -= 1
        selected.append(
            _with_pool_metadata(
                replacement,
                preserved=replacement.get("bundle_preserved", False),
                preservation_reason=replacement.get("bundle_preservation_reason"),
                selection_source=replacement.get("selection_source", "bundle_coexistence_refill"),
            )
        )
        replacement_signature = replacement.get("bundle_signature")
        if replacement_signature is None:
            replacement_signature = signature_for(replacement)
        if replacement_signature is not None:
            selected_bundle_counts[replacement_signature] += 1
            selected_signatures.add(replacement_signature)
        reserve_options = [
            item
            for item in reserve_options
            if signature_for(item) != replacement_signature
        ]
    return selected[:slot_count]


def _bundle_retention_key(
    bundle_id: str,
    candidate: Mapping[str, Any],
    bundle_state_by_signature: Mapping[str, Mapping[str, Any]] | None,
    by_bundle: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[Any, ...]:
    state = _bundle_state(bundle_id, bundle_state_by_signature)
    decision = candidate["decision"]
    return (
        int(_bundle_archive_retired(state)),
        _bundle_archive_repeat_eviction_tier(state),
        int(_bundle_archive_admission_probation(state)),
        int(_bundle_archive_reentry_backoff_active(state)),
        int(_bundle_archive_transfer_deficit(state)),
        int(_bundle_archive_underperforming(state)),
        int(_bundle_archive_value_deficit(state)),
        int(state.get("stale_generations", 0)),
        int(state.get("archive_decay_generations", 0)),
        _bundle_decay_debt(state),
        -int(state.get("clean_win_generations", 0)),
        -_bundle_archive_transfer_payload_success_rate(state),
        -_bundle_archive_transfer_payload_used_rate(state),
        -int(state.get("archive_transfer_success_streak", 0)),
        -int(state.get("archive_positive_lift_streak", 0)),
        -_bundle_archive_transfer_success_rate(state),
        -_bundle_archive_transfer_lift_retention(state),
        -_bundle_archive_comparative_lift(state),
        -float(state.get("avg_score", decision.score)),
        -decision.diversity_bonus,
        len(by_bundle[bundle_id]),
    )


def _bundle_balanced_selection(
    ordered: list[dict[str, Any]],
    *,
    slot_count: int,
    exploration_slots: int = 0,
    reserve_penalty_slots: int = 0,
    bundle_state_by_signature: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    by_bundle: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in ordered:
        bundle_id = _candidate_bundle_signature(item)
        if bundle_id is None:
            return [
                _with_pool_metadata(
                    candidate,
                    preserved=False,
                    preservation_reason=None,
                    selection_source="ordered_fallback",
                )
                for candidate in ordered[: min(slot_count, len(ordered))]
            ]
        by_bundle[bundle_id].append(item)

    if len(by_bundle) <= 1:
        fallback: list[dict[str, Any]] = []
        for index, candidate in enumerate(ordered[: min(slot_count, len(ordered))]):
            fallback.append(
                _with_pool_metadata(
                    candidate,
                    preserved=index == 0,
                    preservation_reason="single_bundle_role" if index == 0 else None,
                    selection_source="bundle_reserve" if index == 0 else "ordered_fallback",
                )
            )
        return fallback

    representatives = sorted(
        (
            (bundle_id, items[0])
            for bundle_id, items in by_bundle.items()
        ),
        key=lambda candidate: _bundle_retention_key(
            candidate[0],
            candidate[1],
            bundle_state_by_signature,
            by_bundle,
        ),
    )
    reserve_candidates = [
        (bundle_id, item)
        for bundle_id, item in representatives
        if (
            not _bundle_archive_retired(_bundle_state(bundle_id, bundle_state_by_signature))
            and _bundle_archive_repeat_eviction_tier(_bundle_state(bundle_id, bundle_state_by_signature)) == 0
            and not _bundle_archive_admission_probation(_bundle_state(bundle_id, bundle_state_by_signature))
            and not _bundle_archive_reentry_backoff_active(_bundle_state(bundle_id, bundle_state_by_signature))
            and not _bundle_archive_transfer_deficit(_bundle_state(bundle_id, bundle_state_by_signature))
            and not _bundle_archive_underperforming(_bundle_state(bundle_id, bundle_state_by_signature))
            and not _bundle_archive_value_deficit(_bundle_state(bundle_id, bundle_state_by_signature))
        )
    ]
    if not reserve_candidates:
        reserve_candidates = representatives
    selected: list[dict[str, Any]] = []
    bundle_slots: Counter[str] = Counter()
    reserve_limit = max(1, slot_count - max(0, exploration_slots) - max(0, reserve_penalty_slots))
    if reserve_penalty_slots > 0:
        reserve_limit = min(
            reserve_limit,
            max(1, len(reserve_candidates) - reserve_penalty_slots),
        )

    for bundle_id, item in reserve_candidates[: min(reserve_limit, len(reserve_candidates))]:
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
                int(_bundle_archive_retired(_bundle_state(candidate[0], bundle_state_by_signature))),
                _bundle_archive_repeat_eviction_tier(_bundle_state(candidate[0], bundle_state_by_signature)),
                int(_bundle_archive_reentry_backoff_active(_bundle_state(candidate[0], bundle_state_by_signature))),
                int(_bundle_archive_transfer_deficit(_bundle_state(candidate[0], bundle_state_by_signature))),
                int(_bundle_archive_underperforming(_bundle_state(candidate[0], bundle_state_by_signature))),
                int(_bundle_archive_value_deficit(_bundle_state(candidate[0], bundle_state_by_signature))),
                _bundle_state(candidate[0], bundle_state_by_signature).get("stale_generations", 0),
                _bundle_state(candidate[0], bundle_state_by_signature).get("archive_decay_generations", 0),
                _bundle_decay_debt(_bundle_state(candidate[0], bundle_state_by_signature)),
                len(by_bundle[candidate[0]]),
                -_bundle_archive_transfer_payload_success_rate(_bundle_state(candidate[0], bundle_state_by_signature)),
                -_bundle_archive_transfer_payload_used_rate(_bundle_state(candidate[0], bundle_state_by_signature)),
                -int(_bundle_state(candidate[0], bundle_state_by_signature).get("archive_transfer_success_streak", 0)),
                -int(_bundle_state(candidate[0], bundle_state_by_signature).get("archive_positive_lift_streak", 0)),
                -_bundle_archive_transfer_success_rate(_bundle_state(candidate[0], bundle_state_by_signature)),
                -_bundle_archive_transfer_lift_retention(_bundle_state(candidate[0], bundle_state_by_signature)),
                -_bundle_archive_comparative_lift(_bundle_state(candidate[0], bundle_state_by_signature)),
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
        candidate_options: list[tuple[tuple[Any, ...], str, dict[str, Any]]] = []
        for bundle_id, items in by_bundle.items():
            if reserve_penalty_slots > 0 and bundle_slots[bundle_id] == 0:
                continue
            template = items[0]
            decision = template["decision"]
            state = _bundle_state(bundle_id, bundle_state_by_signature)
            candidate_options.append(
                (
                    (
                        bundle_slots[bundle_id],
                        int(_bundle_archive_retired(state)),
                        _bundle_archive_repeat_eviction_tier(state),
                        int(_bundle_archive_admission_probation(state)),
                        int(_bundle_archive_reentry_backoff_active(state)),
                        int(_bundle_archive_transfer_deficit(state)),
                        int(_bundle_archive_underperforming(state)),
                        int(_bundle_archive_value_deficit(state)),
                        int(state.get("archive_decay_generations", 0)),
                        _bundle_decay_debt(state),
                        0,
                        -_bundle_archive_transfer_payload_success_rate(state),
                        -_bundle_archive_transfer_payload_used_rate(state),
                        -int(state.get("archive_transfer_success_streak", 0)),
                        -int(state.get("archive_positive_lift_streak", 0)),
                        -_bundle_archive_transfer_success_rate(state),
                        -_bundle_archive_transfer_lift_retention(state),
                        -_bundle_archive_comparative_lift(state),
                        -decision.score,
                        -decision.diversity_bonus,
                        len(decision.reasons),
                    ),
                    bundle_id,
                    template,
                )
            )
        if not candidate_options:
            for bundle_id, items in by_bundle.items():
                template = items[0]
                decision = template["decision"]
                state = _bundle_state(bundle_id, bundle_state_by_signature)
                candidate_options.append(
                    (
                        (
                            bundle_slots[bundle_id],
                            int(_bundle_archive_retired(state)),
                            _bundle_archive_repeat_eviction_tier(state),
                            int(_bundle_archive_admission_probation(state)),
                            int(_bundle_archive_reentry_backoff_active(state)),
                            int(_bundle_archive_transfer_deficit(state)),
                            int(_bundle_archive_underperforming(state)),
                            int(_bundle_archive_value_deficit(state)),
                            int(state.get("archive_decay_generations", 0)),
                            _bundle_decay_debt(state),
                            0,
                            -_bundle_archive_transfer_payload_success_rate(state),
                            -_bundle_archive_transfer_payload_used_rate(state),
                            -int(state.get("archive_transfer_success_streak", 0)),
                            -int(state.get("archive_positive_lift_streak", 0)),
                            -_bundle_archive_transfer_success_rate(state),
                            -_bundle_archive_transfer_lift_retention(state),
                            -_bundle_archive_comparative_lift(state),
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
    reserve_penalty_slots: int = 0,
    bundle_state_by_signature: Mapping[str, Mapping[str, Any]] | None = None,
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
        reserve_penalty_slots=reserve_penalty_slots,
        bundle_state_by_signature=bundle_state_by_signature,
    )
    selected = _apply_archive_coexistence_budget(
        selected,
        ordered=ordered,
        slot_count=slot_count,
        bundle_state_by_signature=bundle_state_by_signature,
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
