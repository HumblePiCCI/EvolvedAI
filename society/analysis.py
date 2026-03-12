from __future__ import annotations

from collections import Counter, defaultdict, deque
from typing import Any

from society.constants import QUARANTINE_REVIEW
from society.storage import StorageManager
from society.trust import classify_warning_outcome, summarize_warning_effect, warning_labels


def _artifact_brief(storage: StorageManager, artifact_id: str) -> dict[str, Any] | None:
    artifact = storage.get_artifact(artifact_id)
    if artifact is None:
        return None
    return {
        "artifact_id": artifact.artifact_id,
        "generation_id": artifact.generation_id,
        "title": artifact.title,
        "summary": artifact.summary,
        "artifact_type": artifact.artifact_type,
        "quarantine_status": artifact.quarantine_status,
        "citations": artifact.citations,
    }


def _memorial_brief(storage: StorageManager, memorial_id: str) -> dict[str, Any] | None:
    memorial = storage.get_memorial(memorial_id)
    if memorial is None:
        return None
    return {
        "memorial_id": memorial.memorial_id,
        "lineage_id": memorial.lineage_id,
        "classification": memorial.classification,
        "failure_mode": memorial.failure_mode,
        "lesson_distillate": memorial.lesson_distillate,
        "taboo_tags": memorial.taboo_tags,
        "top_contribution": memorial.top_contribution,
    }


def _outcome_for_entry(entry: dict[str, Any], max_generation_id: int | None) -> str:
    if entry["child_lineage_ids"]:
        return "propagated"
    if entry["propagation_blocked"]:
        return "blocked"
    if entry["quarantine_status"] == QUARANTINE_REVIEW:
        return "reviewed"
    if entry["eligible"]:
        if max_generation_id is not None and entry["generation_id"] == max_generation_id:
            return "eligible_pending"
        return "eligible_unselected"
    if entry["evidence_refs"] or entry["reasons"]:
        return "not_selected"
    return "unknown"


def _parent_concentration(
    lineage_updates: list[dict[str, Any]],
) -> tuple[float, str | None]:
    by_role: dict[str, list[str]] = defaultdict(list)
    for update in lineage_updates:
        parents = update.get("parent_lineage_ids", [])
        if parents:
            by_role[update["role"]].append(parents[0])

    highest_role = None
    highest_value = 0.0
    for role, parent_ids in by_role.items():
        if len(parent_ids) < 3:
            continue
        counts = Counter(parent_ids)
        concentration = max(counts.values()) / len(parent_ids)
        if concentration > highest_value:
            highest_value = concentration
            highest_role = role
    return round(highest_value, 4), highest_role


def _bundle_set(lineage_updates: list[dict[str, Any]]) -> set[tuple[str, str, str]]:
    return {
        (item["role"], item["prompt_variant_id"], item["package_policy_id"])
        for item in lineage_updates
        if item.get("role") and item.get("prompt_variant_id") and item.get("package_policy_id")
    }


def lineage_entries(
    storage: StorageManager,
    generation_ids: list[int] | None = None,
) -> list[dict[str, Any]]:
    generation_filter = None if generation_ids is None else set(generation_ids)
    generations = {
        generation.generation_id: generation
        for generation in storage.list_generations()
        if generation_filter is None or generation.generation_id in generation_filter
    }
    max_generation_id = None if not generations else max(generations)
    lineages = storage.list_lineages()
    children_by_parent: dict[str, list[str]] = defaultdict(list)
    for lineage in lineages:
        for parent in lineage.parent_lineage_ids:
            children_by_parent[parent].append(lineage.lineage_id)

    entries: list[dict[str, Any]] = []
    for lineage in sorted(lineages, key=lambda item: (item.current_generation_id, item.lineage_id)):
        if generation_filter is not None and lineage.current_generation_id not in generation_filter:
            continue
        generation = generations.get(lineage.current_generation_id)
        summary = {} if generation is None else generation.summary_json
        selection: dict[str, Any] = next(
            (
                item
                for item in summary.get("selection_outcome", [])
                if item.get("lineage_id") == lineage.lineage_id
            ),
            {},
        )
        lineage_update: dict[str, Any] = next(
            (
                item
                for item in summary.get("lineage_updates", [])
                if item.get("lineage_id") == lineage.lineage_id
            ),
            {},
        )
        agents = storage.list_agents_by_lineage(lineage.lineage_id)
        agent = next(
            (candidate for candidate in agents if candidate.generation_id == lineage.current_generation_id),
            None,
        )
        inherited_artifact_ids = lineage_update.get(
            "inherited_artifact_ids",
            [] if agent is None else agent.inherited_artifact_ids,
        )
        inherited_memorial_ids = lineage_update.get(
            "inherited_memorial_ids",
            [] if agent is None else agent.inherited_memorial_ids,
        )
        inherited_artifacts = [
            detail
            for detail in (_artifact_brief(storage, artifact_id) for artifact_id in inherited_artifact_ids)
            if detail is not None
        ]
        inherited_memorials = [
            detail
            for detail in (_memorial_brief(storage, memorial_id) for memorial_id in inherited_memorial_ids)
            if detail is not None
        ]
        child_lineages = sorted(children_by_parent.get(lineage.lineage_id, []))
        warning_labels_for_entry = warning_labels(
            taboo_tags=lineage_update.get("taboo_tags", []),
            memorials=inherited_memorials,
        )
        current_failures = set(selection.get("hidden_failures", [])) | set(selection.get("public_failures", []))
        entry = {
            "lineage_id": lineage.lineage_id,
            "generation_id": lineage.current_generation_id,
            "agent_id": None if agent is None else agent.agent_id,
            "role": None if agent is None else agent.role,
            "status": lineage.status,
            "notes": lineage.notes,
            "parent_lineage_ids": lineage.parent_lineage_ids,
            "child_lineage_ids": child_lineages,
            "inheritance_source_agent_id": lineage_update.get("inheritance_source_agent_id"),
            "inheritance_source_generation_id": lineage_update.get("inheritance_source_generation_id"),
            "inheritance_source_bundle_signature": lineage_update.get("inheritance_source_bundle_signature"),
            "inheritance_source_bundle_preserved": lineage_update.get("inheritance_source_bundle_preserved", False),
            "inheritance_source_bundle_reason": lineage_update.get("inheritance_source_bundle_reason"),
            "inheritance_source_selection_source": lineage_update.get("inheritance_source_selection_source"),
            "inherited_artifact_ids": inherited_artifact_ids,
            "inherited_memorial_ids": inherited_memorial_ids,
            "inherited_artifacts": inherited_artifacts,
            "inherited_memorials": inherited_memorials,
            "taboo_tags": lineage_update.get("taboo_tags", []),
            "eligible": selection.get("eligible"),
            "propagation_blocked": selection.get("propagation_blocked", False),
            "quarantine_status": selection.get("quarantine_status"),
            "hidden_failures": selection.get("hidden_failures", []),
            "public_failures": selection.get("public_failures", []),
            "reasons": selection.get("reasons", []),
            "evidence_refs": selection.get("evidence_refs", []),
            "warning_labels": sorted(warning_labels_for_entry),
            "base_score": selection.get("base_score", selection.get("score", 0.0)),
            "diversity_bonus": selection.get("diversity_bonus", 0.0),
            "cohort_similarity": selection.get("cohort_similarity", 0.0),
            "selection_bucket": selection.get("selection_bucket", "standard"),
            "prompt_variant_id": lineage_update.get("prompt_variant_id"),
            "prompt_variant_tags": lineage_update.get("prompt_variant_tags", []),
            "package_policy_id": lineage_update.get("package_policy_id"),
            "bundle_signature": lineage_update.get("bundle_signature"),
            "bundle_preserved": selection.get("bundle_preserved", False),
            "bundle_preservation_reason": selection.get("bundle_preservation_reason"),
            "variant_origin": lineage_update.get("variant_origin"),
        }
        entry["warning_outcome"] = classify_warning_outcome(
            warning_labels=warning_labels_for_entry,
            current_failures=current_failures,
        )
        entry["outcome"] = _outcome_for_entry(entry, max_generation_id)
        entries.append(entry)
    return entries


def build_lineage_report(storage: StorageManager, lineage_id: str) -> dict[str, Any]:
    entries = lineage_entries(storage)
    entry_by_id = {entry["lineage_id"]: entry for entry in entries}
    if lineage_id not in entry_by_id:
        raise ValueError(f"lineage {lineage_id} not found")

    queue: deque[str] = deque([lineage_id])
    visited: set[str] = set()
    family_history: list[dict[str, Any]] = []
    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        entry = entry_by_id.get(current)
        if entry is None:
            continue
        family_history.append(entry)
        for parent in entry["parent_lineage_ids"]:
            queue.append(parent)
        for child in entry["child_lineage_ids"]:
            queue.append(child)

    family_history.sort(key=lambda entry: (entry["generation_id"], entry["lineage_id"]))
    root = min(family_history, key=lambda entry: (entry["generation_id"], entry["lineage_id"]))
    outcome_counts = Counter(entry["outcome"] for entry in family_history)
    return {
        "lineage_id": lineage_id,
        "selected_lineage": entry_by_id[lineage_id],
        "family_root": root,
        "survival_history": family_history,
        "outcome_counts": dict(outcome_counts),
    }


def render_lineage_report(report: dict[str, Any]) -> str:
    selected = report["selected_lineage"]
    root = report["family_root"]
    lines = [
        f"Lineage {report['lineage_id']}",
        (
            f"selected_generation={selected['generation_id']} role={selected['role']} "
            f"outcome={selected['outcome']} parent_source={selected['inheritance_source_agent_id'] or 'root'}"
        ),
        (
            f"family_root={root['lineage_id']} g{root['generation_id']} "
            f"root_parents={','.join(root['parent_lineage_ids']) or 'root'}"
        ),
        f"selected_parents={','.join(selected['parent_lineage_ids']) or 'root'} selected_children={','.join(selected['child_lineage_ids']) or 'none'}",
        "",
        "Survival history:",
    ]
    for entry in report["survival_history"]:
        lines.append(
            f"- g{entry['generation_id']} {entry['lineage_id']} role={entry['role']} outcome={entry['outcome']} "
            f"selection={'eligible' if entry['eligible'] else 'ineligible'} "
            f"status={entry['quarantine_status'] or entry['status']} "
            f"warning_outcome={entry['warning_outcome']} "
            f"bucket={entry['selection_bucket']} "
            f"variant={entry['prompt_variant_id'] or 'none'} "
            f"reasons={','.join(entry['reasons']) or 'none'}"
        )
        lines.append(
            f"  inherited_artifacts={len(entry['inherited_artifacts'])} inherited_memorials={len(entry['inherited_memorials'])} "
            f"taboo_tags={','.join(entry['taboo_tags']) or 'none'} "
            f"warning_labels={','.join(entry['warning_labels']) or 'none'} "
            f"package_policy={entry['package_policy_id'] or 'none'} origin={entry['variant_origin'] or 'none'} "
            f"bundle={entry['bundle_signature'] or 'none'} parent_bundle={entry['inheritance_source_bundle_signature'] or 'none'} "
            f"parent_preserved={entry['inheritance_source_bundle_preserved']} "
            f"parent_source={entry['inheritance_source_selection_source'] or 'none'}"
        )
        lines.append(
            f"  base_score={entry['base_score']} diversity_bonus={entry['diversity_bonus']} "
            f"cohort_similarity={entry['cohort_similarity']} "
            f"bundle_preserved={entry['bundle_preserved']} "
            f"bundle_reason={entry['bundle_preservation_reason'] or 'none'}"
        )
        for artifact in entry["inherited_artifacts"]:
            lines.append(
                f"  artifact {artifact['artifact_id']} [{artifact['artifact_type']}]: {artifact['summary']}"
            )
        for memorial in entry["inherited_memorials"]:
            lines.append(
                f"  memorial {memorial['memorial_id']} [{memorial['classification']}]: "
                f"failure={memorial['failure_mode'] or 'none'} lesson={memorial['lesson_distillate']}"
            )
        if entry["evidence_refs"]:
            lines.append(f"  evidence_refs={','.join(entry['evidence_refs'])}")
    lines.extend(["", "Outcome counts:"])
    for outcome, count in sorted(report["outcome_counts"].items()):
        lines.append(f"- {outcome}: {count}")
    return "\n".join(lines) + "\n"


def build_experiment_report(storage: StorageManager, generation_ids: list[int]) -> dict[str, Any]:
    generations = []
    for generation_id in generation_ids:
        generation = storage.get_generation(generation_id)
        if generation is None:
            raise ValueError(f"generation {generation_id} not found")
        generations.append(generation)

    generation_metrics = []
    previous_bundles: set[tuple[str, str, str]] | None = None
    previous_exploratory_bundles: set[tuple[str, str, str]] = set()
    for generation in generations:
        summary = generation.summary_json
        hidden_counts = summary.get("hidden_eval_failure_counts", {})
        selection_summary = summary.get("selection_summary", {})
        role_monoculture = selection_summary.get("role_monoculture_index", {})
        role_counts = Counter(item.get("role") for item in summary.get("selection_outcome", []))
        major_roles = [role for role, count in role_counts.items() if count >= 3 and role in role_monoculture]
        major_role_set = set(major_roles)
        scoped_monoculture = {role: role_monoculture[role] for role in major_roles}
        role_variant_count = selection_summary.get("role_variant_count", {})
        role_bundle_count = selection_summary.get("role_bundle_count", {})
        role_bundle_concentration = selection_summary.get("role_bundle_concentration_index", {})
        role_parent_bundle_concentration = selection_summary.get("role_parent_bundle_concentration_index", {})
        scoped_bundle_concentration = {
            role: role_bundle_concentration[role]
            for role in major_roles
            if role in role_bundle_concentration
        }
        scoped_parent_bundle_concentration = {
            role: role_parent_bundle_concentration[role]
            for role in major_roles
            if role in role_parent_bundle_concentration
        }
        preserved_bundles = selection_summary.get("preserved_bundles", [])
        all_variants = {
            item.get("prompt_variant_id")
            for item in summary.get("lineage_updates", [])
            if item.get("prompt_variant_id")
        }
        all_bundles = {
            (item.get("prompt_variant_id"), item.get("package_policy_id"))
            for item in summary.get("lineage_updates", [])
            if item.get("prompt_variant_id") and item.get("package_policy_id")
        }
        largest_variant_share = 0.0
        largest_bundle_share = 0.0
        most_common_variant_role = None
        most_common_bundle_role = None
        for role, count in role_counts.items():
            if count < 3:
                continue
            role_variants = [
                item.get("prompt_variant_id")
                for item in summary.get("lineage_updates", [])
                if item.get("role") == role and item.get("prompt_variant_id")
            ]
            if not role_variants:
                continue
            share = max(Counter(role_variants).values()) / len(role_variants)
            if share > largest_variant_share:
                largest_variant_share = share
                most_common_variant_role = role
            role_bundles = [
                (item.get("prompt_variant_id"), item.get("package_policy_id"))
                for item in summary.get("lineage_updates", [])
                if item.get("role") == role and item.get("prompt_variant_id") and item.get("package_policy_id")
            ]
            if role_bundles:
                bundle_share = max(Counter(role_bundles).values()) / len(role_bundles)
                if bundle_share > largest_bundle_share:
                    largest_bundle_share = bundle_share
                    most_common_bundle_role = role
        current_bundles = {
            bundle
            for bundle in _bundle_set(summary.get("lineage_updates", []))
            if bundle[0] in major_role_set
        }
        bundle_survival = set() if previous_bundles is None else current_bundles & previous_bundles
        new_bundles = current_bundles if previous_bundles is None else current_bundles - previous_bundles
        new_bundle_signatures = {
            f"{role}:{variant}:{policy}" for role, variant, policy in new_bundles
        }
        new_bundle_lineages = [
            item
            for item in summary.get("selection_outcome", [])
            if item.get("role") in major_role_set and item.get("bundle_signature") in new_bundle_signatures
        ]
        exploratory_bundles = {
            (
                item.get("role"),
                item.get("prompt_variant_id"),
                item.get("package_policy_id"),
            )
            for item in summary.get("lineage_updates", [])
            if item.get("variant_origin") == "bundle_archive_exploration"
            and item.get("role") in major_role_set
            and item.get("role")
            and item.get("prompt_variant_id")
            and item.get("package_policy_id")
        }
        exploration_bundle_survival = current_bundles & previous_exploratory_bundles
        bundle_concentration_index = max(scoped_bundle_concentration.values(), default=0.0)
        most_bundle_concentrated_role = (
            max(scoped_bundle_concentration, key=lambda role: scoped_bundle_concentration[role])
            if scoped_bundle_concentration
            else None
        )
        parent_bundle_concentration_index = max(scoped_parent_bundle_concentration.values(), default=0.0)
        most_parent_bundle_concentrated_role = (
            max(scoped_parent_bundle_concentration, key=lambda role: scoped_parent_bundle_concentration[role])
            if scoped_parent_bundle_concentration
            else None
        )
        parent_concentration, most_reused_parent_role = _parent_concentration(summary.get("lineage_updates", []))
        generation_metrics.append(
            {
                "generation_id": generation.generation_id,
                "public_eval_average": summary.get("public_eval_average", 0.0),
                "eligible": selection_summary.get("eligible", 0),
                "propagation_blocked": selection_summary.get("propagation_blocked", 0),
                "review_only": selection_summary.get("review_only", 0),
                "diversity_priority_count": selection_summary.get("diversity_priority_count", 0),
                "diffusion_alerts": hidden_counts.get("diffusion_alerts", 0),
                "anti_corruption": hidden_counts.get("anti_corruption", 0),
                "taboo_recurrence": hidden_counts.get("taboo_recurrence", 0),
                "warned_lineages": summary.get("inheritance_effect", {}).get("warned_lineages", 0),
                "memorial_transfer_score": summary.get("inheritance_effect", {}).get(
                    "transfer_score",
                    summary.get("drift", {}).get("memorial_transfer_score", 0.0),
                ),
                "monoculture_index": max(scoped_monoculture.values(), default=0.0),
                "most_converged_role": (
                    max(scoped_monoculture, key=lambda role: scoped_monoculture[role]) if scoped_monoculture else None
                ),
                "prompt_variant_count": len(all_variants),
                "prompt_bundle_count": len(all_bundles),
                "largest_variant_share": round(largest_variant_share, 4),
                "largest_bundle_share": round(largest_bundle_share, 4),
                "most_common_variant_role": most_common_variant_role,
                "most_common_bundle_role": most_common_bundle_role,
                "role_variant_count": role_variant_count,
                "role_bundle_count": role_bundle_count,
                "bundle_concentration_index": round(bundle_concentration_index, 4),
                "most_bundle_concentrated_role": most_bundle_concentrated_role,
                "parent_bundle_concentration_index": round(parent_bundle_concentration_index, 4),
                "most_parent_bundle_concentrated_role": most_parent_bundle_concentrated_role,
                "preserved_bundle_count": len(preserved_bundles),
                "preserved_bundles": [
                    item["bundle_signature"]
                    for item in preserved_bundles
                    if item.get("bundle_signature")
                ],
                "bundle_archive_count": selection_summary.get("bundle_archive_count", 0),
                "bundle_archive_candidate_roles": selection_summary.get("bundle_archive_candidate_roles", []),
                "bundle_archive_roles": selection_summary.get("bundle_archive_roles", []),
                "bundle_archive_pending_roles": selection_summary.get("bundle_archive_pending_roles", []),
                "bundle_archive_proving_roles": selection_summary.get("bundle_archive_proving_roles", []),
                "bundle_archive_post_admission_grace_roles": selection_summary.get(
                    "bundle_archive_post_admission_grace_roles",
                    [],
                ),
                "bundle_archive_reentry_backoff_roles": selection_summary.get(
                    "bundle_archive_reentry_backoff_roles",
                    [],
                ),
                "bundle_archive_reentry_block_roles": selection_summary.get(
                    "bundle_archive_reentry_block_roles",
                    [],
                ),
                "bundle_archive_escalated_backoff_roles": selection_summary.get(
                    "bundle_archive_escalated_backoff_roles",
                    [],
                ),
                "bundle_archive_coexistence_budget_roles": selection_summary.get(
                    "bundle_archive_coexistence_budget_roles",
                    [],
                ),
                "bundle_archive_underperform_roles": selection_summary.get(
                    "bundle_archive_underperform_roles",
                    [],
                ),
                "bundle_archive_positive_lift_roles": selection_summary.get(
                    "bundle_archive_positive_lift_roles",
                    [],
                ),
                "bundle_archive_value_deficit_roles": selection_summary.get(
                    "bundle_archive_value_deficit_roles",
                    [],
                ),
                "bundle_archive_transfer_success_roles": selection_summary.get(
                    "bundle_archive_transfer_success_roles",
                    [],
                ),
                "bundle_archive_transfer_failure_roles": selection_summary.get(
                    "bundle_archive_transfer_failure_roles",
                    [],
                ),
                "bundle_archive_eviction_roles": selection_summary.get(
                    "bundle_archive_eviction_roles",
                    [],
                ),
                "bundle_archive_repeat_eviction_roles": selection_summary.get(
                    "bundle_archive_repeat_eviction_roles",
                    [],
                ),
                "bundle_archive_retired_roles": selection_summary.get(
                    "bundle_archive_retired_roles",
                    [],
                ),
                "bundle_archive_cooldown_roles": selection_summary.get("bundle_archive_cooldown_roles", []),
                "bundle_archive_cooldown_fresh_admission_roles": selection_summary.get(
                    "bundle_archive_cooldown_fresh_admission_roles",
                    [],
                ),
                "bundle_archive_cooldown_long_lived_debt_roles": selection_summary.get(
                    "bundle_archive_cooldown_long_lived_debt_roles",
                    [],
                ),
                "bundle_archive_cooldown_true_overload_roles": selection_summary.get(
                    "bundle_archive_cooldown_true_overload_roles",
                    [],
                ),
                "bundle_archive_cooldown_avoidable_duplicate_roles": selection_summary.get(
                    "bundle_archive_cooldown_avoidable_duplicate_roles",
                    [],
                ),
                "bundle_archive_cooldown_recovery_roles": selection_summary.get(
                    "bundle_archive_cooldown_recovery_roles",
                    [],
                ),
                "bundle_archive_cooldown_count": selection_summary.get("bundle_archive_cooldown_count", 0),
                "bundle_archive_coexistence_budget_count": selection_summary.get(
                    "bundle_archive_coexistence_budget_count",
                    0,
                ),
                "bundle_archive_cooldown_true_overload_count": selection_summary.get(
                    "bundle_archive_cooldown_true_overload_count",
                    0,
                ),
                "bundle_archive_cooldown_avoidable_duplicate_count": selection_summary.get(
                    "bundle_archive_cooldown_avoidable_duplicate_count",
                    0,
                ),
                "bundle_archive_cooldown_recovery_count": selection_summary.get(
                    "bundle_archive_cooldown_recovery_count",
                    0,
                ),
                "bundle_archive_cooldown_recovery_max_generations": selection_summary.get(
                    "bundle_archive_cooldown_recovery_max_generations",
                    0,
                ),
                "archive_admission_pending_count": selection_summary.get("archive_admission_pending_count", 0),
                "archive_proving_count": selection_summary.get("archive_proving_count", 0),
                "archive_reentry_block_count": selection_summary.get("archive_reentry_block_count", 0),
                "archive_escalated_backoff_count": selection_summary.get("archive_escalated_backoff_count", 0),
                "archive_reentry_attempt_count": selection_summary.get("archive_reentry_attempt_count", 0),
                "archive_underperform_count": selection_summary.get("archive_underperform_count", 0),
                "archive_positive_lift_count": selection_summary.get("archive_positive_lift_count", 0),
                "archive_value_deficit_count": selection_summary.get("archive_value_deficit_count", 0),
                "archive_incumbent_win_count": selection_summary.get("archive_incumbent_win_count", 0),
                "archive_incumbent_loss_count": selection_summary.get("archive_incumbent_loss_count", 0),
                "archive_mean_comparative_lift": selection_summary.get("archive_mean_comparative_lift", 0.0),
                "archive_transfer_success_count": selection_summary.get("archive_transfer_success_count", 0),
                "archive_transfer_failure_count": selection_summary.get("archive_transfer_failure_count", 0),
                "archive_transfer_success_rate": selection_summary.get("archive_transfer_success_rate", 0.0),
                "archive_parent_vs_child_lift_retention": selection_summary.get(
                    "archive_parent_vs_child_lift_retention",
                    0.0,
                ),
                "archive_admitted_count": selection_summary.get("archive_admitted_count", 0),
                "newly_admitted_count": selection_summary.get("newly_admitted_count", 0),
                "post_admission_grace_count": selection_summary.get("post_admission_grace_count", 0),
                "archive_eviction_count": selection_summary.get("archive_eviction_count", 0),
                "repeat_eviction_count": selection_summary.get("repeat_eviction_count", 0),
                "archive_repeat_eviction_max_tier": selection_summary.get("archive_repeat_eviction_max_tier", 0),
                "archive_admission_conversion_rate": selection_summary.get("archive_admission_conversion_rate", 0.0),
                "archive_reentry_converted_count": selection_summary.get("archive_reentry_converted_count", 0),
                "archive_reentry_mean_gap_generations": selection_summary.get(
                    "archive_reentry_mean_gap_generations",
                    0.0,
                ),
                "archive_reentry_max_gap_generations": selection_summary.get(
                    "archive_reentry_max_gap_generations",
                    0,
                ),
                "archive_retired_count": selection_summary.get("archive_retired_count", 0),
                "archive_retirement_reason_counts": selection_summary.get("archive_retirement_reason_counts", {}),
                "archive_failed_admission_count": selection_summary.get("archive_failed_admission_count", 0),
                "bundle_decay_prune_roles": selection_summary.get("bundle_decay_prune_roles", []),
                "bundle_decay_prune_count": selection_summary.get("bundle_decay_prune_count", 0),
                "stale_bundle_count": selection_summary.get("stale_bundle_count", 0),
                "decaying_bundle_count": selection_summary.get("decaying_bundle_count", 0),
                "archive_retirement_ready_count": selection_summary.get("archive_retirement_ready_count", 0),
                "pruned_bundle_count": selection_summary.get("pruned_bundle_count", 0),
                "pruned_bundles": selection_summary.get("pruned_bundles", []),
                "bundle_survival_count": len(bundle_survival),
                "bundle_survival_rate": (
                    round(len(bundle_survival) / len(previous_bundles), 4)
                    if previous_bundles
                    else 0.0
                ),
                "bundle_turnover_count": len(new_bundles),
                "bundle_turnover_rate": round(len(new_bundles) / len(current_bundles), 4) if current_bundles else 0.0,
                "new_bundle_win_rate": (
                    round(
                        sum(
                            int(item.get("eligible") and item.get("quarantine_status") == "clean")
                            for item in new_bundle_lineages
                        )
                        / len(new_bundle_lineages),
                        4,
                    )
                    if new_bundle_lineages
                    else 0.0
                ),
                "exploration_bundle_survival_count": len(exploration_bundle_survival),
                "exploration_bundle_survival_rate": (
                    round(len(exploration_bundle_survival) / len(previous_exploratory_bundles), 4)
                    if previous_exploratory_bundles
                    else 0.0
                ),
                "parent_concentration_index": parent_concentration,
                "most_reused_parent_role": most_reused_parent_role,
                "strategy_drift_rate": summary.get("drift", {}).get("strategy_drift_rate", 0.0),
                "drift_pressure_lineages": selection_summary.get("variant_origin_counts", {}).get("drift_pressure", 0),
            }
        )
        previous_bundles = current_bundles
        previous_exploratory_bundles = exploratory_bundles

    lineages = lineage_entries(storage, generation_ids=generation_ids)
    outcome_counts = Counter(entry["outcome"] for entry in lineages)
    blocked_lineages = [entry for entry in lineages if entry["outcome"] == "blocked"]
    reviewed_lineages = [entry for entry in lineages if entry["outcome"] == "reviewed"]
    propagated_lineages = [entry for entry in lineages if entry["outcome"] == "propagated"]
    pending_lineages = [entry for entry in lineages if entry["outcome"] == "eligible_pending"]

    inheritance_effect = summarize_warning_effect(
        (entry["warning_labels"], set(entry["hidden_failures"]) | set(entry["public_failures"]))
        for entry in lineages
    )
    warned_lineages = int(inheritance_effect["warned_lineages"])
    avoided_recurrence = int(inheritance_effect["avoided_recurrence"])
    repeated_warning = int(inheritance_effect["repeated_warning"])
    shifted_failure = int(inheritance_effect["shifted_failure"])

    notes: list[str] = []
    if generation_metrics:
        first = generation_metrics[0]
        last = generation_metrics[-1]
        diffusion_delta = last["diffusion_alerts"] - first["diffusion_alerts"]
        anti_corruption_delta = last["anti_corruption"] - first["anti_corruption"]
        memorial_delta = round(last["memorial_transfer_score"] - first["memorial_transfer_score"], 4)
        monoculture_delta = round(last["monoculture_index"] - first["monoculture_index"], 4)
        parent_reuse_delta = round(last["parent_concentration_index"] - first["parent_concentration_index"], 4)
        variant_delta = last["prompt_variant_count"] - first["prompt_variant_count"]
        bundle_delta = last["prompt_bundle_count"] - first["prompt_bundle_count"]
        bundle_survival_delta = round(last["bundle_survival_rate"] - first["bundle_survival_rate"], 4)
        bundle_turnover_delta = round(last["bundle_turnover_rate"] - first["bundle_turnover_rate"], 4)
        archive_admission_pending_delta = (
            last["archive_admission_pending_count"] - first["archive_admission_pending_count"]
        )
        archive_proving_delta = last["archive_proving_count"] - first["archive_proving_count"]
        archive_reentry_block_delta = last["archive_reentry_block_count"] - first["archive_reentry_block_count"]
        archive_escalated_backoff_delta = (
            last["archive_escalated_backoff_count"] - first["archive_escalated_backoff_count"]
        )
        archive_reentry_attempt_delta = last["archive_reentry_attempt_count"] - first["archive_reentry_attempt_count"]
        archive_underperform_delta = last["archive_underperform_count"] - first["archive_underperform_count"]
        archive_admitted_delta = last["archive_admitted_count"] - first["archive_admitted_count"]
        newly_admitted_delta = last["newly_admitted_count"] - first["newly_admitted_count"]
        post_admission_grace_delta = last["post_admission_grace_count"] - first["post_admission_grace_count"]
        archive_eviction_delta = last["archive_eviction_count"] - first["archive_eviction_count"]
        repeat_eviction_delta = last["repeat_eviction_count"] - first["repeat_eviction_count"]
        repeat_eviction_tier_delta = (
            last["archive_repeat_eviction_max_tier"] - first["archive_repeat_eviction_max_tier"]
        )
        archive_conversion_delta = round(
            last["archive_admission_conversion_rate"] - first["archive_admission_conversion_rate"],
            4,
        )
        archive_reentry_converted_delta = (
            last["archive_reentry_converted_count"] - first["archive_reentry_converted_count"]
        )
        archive_reentry_gap_delta = round(
            last["archive_reentry_mean_gap_generations"] - first["archive_reentry_mean_gap_generations"],
            4,
        )
        archive_retired_delta = last["archive_retired_count"] - first["archive_retired_count"]
        archive_failed_delta = last["archive_failed_admission_count"] - first["archive_failed_admission_count"]
        bundle_archive_cooldown_delta = (
            last["bundle_archive_cooldown_count"] - first["bundle_archive_cooldown_count"]
        )
        bundle_archive_coexistence_budget_delta = (
            last["bundle_archive_coexistence_budget_count"] - first["bundle_archive_coexistence_budget_count"]
        )
        bundle_archive_cooldown_true_overload_delta = (
            last["bundle_archive_cooldown_true_overload_count"]
            - first["bundle_archive_cooldown_true_overload_count"]
        )
        bundle_archive_cooldown_avoidable_duplicate_delta = (
            last["bundle_archive_cooldown_avoidable_duplicate_count"]
            - first["bundle_archive_cooldown_avoidable_duplicate_count"]
        )
        bundle_archive_cooldown_recovery_delta = (
            last["bundle_archive_cooldown_recovery_count"] - first["bundle_archive_cooldown_recovery_count"]
        )
        bundle_decay_prune_delta = last["bundle_decay_prune_count"] - first["bundle_decay_prune_count"]
        stale_bundle_delta = last["stale_bundle_count"] - first["stale_bundle_count"]
        decaying_bundle_delta = last["decaying_bundle_count"] - first["decaying_bundle_count"]
        archive_retirement_ready_delta = (
            last["archive_retirement_ready_count"] - first["archive_retirement_ready_count"]
        )
        pruned_bundle_delta = last["pruned_bundle_count"] - first["pruned_bundle_count"]
        if diffusion_delta < 0:
            notes.append(f"Diffusion alerts fell by {-diffusion_delta} between the first and last generation.")
        elif diffusion_delta > 0:
            notes.append(f"Diffusion alerts increased by {diffusion_delta} across the batch.")
        else:
            notes.append("Diffusion alerts stayed flat across the batch.")
        if anti_corruption_delta < 0:
            notes.append(f"Anti-corruption failures fell by {-anti_corruption_delta} across the batch.")
        elif anti_corruption_delta > 0:
            notes.append(f"Anti-corruption failures increased by {anti_corruption_delta} across the batch.")
        else:
            notes.append("Anti-corruption failures stayed flat across the batch.")
        notes.append(f"Memorial transfer score changed by {memorial_delta}.")
        if monoculture_delta < 0:
            notes.append(f"Selection monoculture index fell by {-monoculture_delta} across the batch.")
        elif monoculture_delta > 0:
            notes.append(f"Selection monoculture index increased by {monoculture_delta} across the batch.")
        else:
            notes.append("Selection monoculture index stayed flat across the batch.")
        if parent_reuse_delta < 0:
            notes.append(f"Parent concentration index fell by {-parent_reuse_delta} across the batch.")
        elif parent_reuse_delta > 0:
            notes.append(f"Parent concentration index increased by {parent_reuse_delta} across the batch.")
        else:
            notes.append("Parent concentration index stayed flat across the batch.")
        if variant_delta > 0:
            notes.append(f"Prompt variant coverage increased by {variant_delta} across the batch.")
        elif variant_delta < 0:
            notes.append(f"Prompt variant coverage fell by {-variant_delta} across the batch.")
        else:
            notes.append("Prompt variant coverage stayed flat across the batch.")
        if bundle_delta > 0:
            notes.append(f"Prompt bundle coverage increased by {bundle_delta} across the batch.")
        elif bundle_delta < 0:
            notes.append(f"Prompt bundle coverage fell by {-bundle_delta} across the batch.")
        else:
            notes.append("Prompt bundle coverage stayed flat across the batch.")
        if bundle_survival_delta > 0:
            notes.append(f"Bundle survival rate increased by {bundle_survival_delta} across the batch.")
        elif bundle_survival_delta < 0:
            notes.append(f"Bundle survival rate fell by {-bundle_survival_delta} across the batch.")
        else:
            notes.append("Bundle survival rate stayed flat across the batch.")
        if bundle_turnover_delta > 0:
            notes.append(f"Bundle turnover rate increased by {bundle_turnover_delta} across the batch.")
        elif bundle_turnover_delta < 0:
            notes.append(f"Bundle turnover rate fell by {-bundle_turnover_delta} across the batch.")
        else:
            notes.append("Bundle turnover rate stayed flat across the batch.")
        if archive_admission_pending_delta > 0:
            notes.append(
                f"Archive admission probation expanded by {archive_admission_pending_delta} bundles across the batch."
            )
        elif archive_admission_pending_delta < 0:
            notes.append(
                f"Archive admission probation fell by {-archive_admission_pending_delta} bundles across the batch."
            )
        else:
            notes.append("Archive admission probation stayed flat across the batch.")
        if archive_proving_delta > 0:
            notes.append(f"Archive proving pressure increased by {archive_proving_delta} bundles across the batch.")
        elif archive_proving_delta < 0:
            notes.append(f"Archive proving pressure fell by {-archive_proving_delta} bundles across the batch.")
        else:
            notes.append("Archive proving pressure stayed flat across the batch.")
        if archive_reentry_block_delta > 0:
            notes.append(
                f"Archive re-entry backoff blocked {archive_reentry_block_delta} more bundles across the batch."
            )
        elif archive_reentry_block_delta < 0:
            notes.append(
                f"Archive re-entry backoff blocked {-archive_reentry_block_delta} fewer bundles across the batch."
            )
        else:
            notes.append("Archive re-entry backoff pressure stayed flat across the batch.")
        if archive_escalated_backoff_delta > 0:
            notes.append(
                f"Escalated archive backoff covered {archive_escalated_backoff_delta} more bundles across the batch."
            )
        elif archive_escalated_backoff_delta < 0:
            notes.append(
                f"Escalated archive backoff covered {-archive_escalated_backoff_delta} fewer bundles across the batch."
            )
        else:
            notes.append("Escalated archive backoff coverage stayed flat across the batch.")
        if archive_reentry_attempt_delta > 0:
            notes.append(
                f"Archive re-entry attempts increased by {archive_reentry_attempt_delta} across the batch."
            )
        elif archive_reentry_attempt_delta < 0:
            notes.append(
                f"Archive re-entry attempts fell by {-archive_reentry_attempt_delta} across the batch."
            )
        else:
            notes.append("Archive re-entry attempts stayed flat across the batch.")
        if archive_underperform_delta > 0:
            notes.append(
                f"Archive underperformance pressure increased by {archive_underperform_delta} bundles across the batch."
            )
        elif archive_underperform_delta < 0:
            notes.append(
                f"Archive underperformance pressure fell by {-archive_underperform_delta} bundles across the batch."
            )
        else:
            notes.append("Archive underperformance pressure stayed flat across the batch.")
        if archive_admitted_delta > 0:
            notes.append(f"Archive admissions increased by {archive_admitted_delta} bundles across the batch.")
        elif archive_admitted_delta < 0:
            notes.append(f"Archive admissions fell by {-archive_admitted_delta} bundles across the batch.")
        else:
            notes.append("Archive admissions stayed flat across the batch.")
        if newly_admitted_delta > 0:
            notes.append(f"New archive admissions increased by {newly_admitted_delta} bundles across the batch.")
        elif newly_admitted_delta < 0:
            notes.append(f"New archive admissions fell by {-newly_admitted_delta} bundles across the batch.")
        else:
            notes.append("New archive admissions stayed flat across the batch.")
        if post_admission_grace_delta > 0:
            notes.append(
                f"Post-admission grace coverage increased by {post_admission_grace_delta} bundles across the batch."
            )
        elif post_admission_grace_delta < 0:
            notes.append(
                f"Post-admission grace coverage fell by {-post_admission_grace_delta} bundles across the batch."
            )
        else:
            notes.append("Post-admission grace coverage stayed flat across the batch.")
        if archive_eviction_delta > 0:
            notes.append(f"Archive evictions increased by {archive_eviction_delta} bundles across the batch.")
        elif archive_eviction_delta < 0:
            notes.append(f"Archive evictions fell by {-archive_eviction_delta} bundles across the batch.")
        else:
            notes.append("Archive evictions stayed flat across the batch.")
        if repeat_eviction_delta > 0:
            notes.append(f"Repeat archive evictions increased by {repeat_eviction_delta} bundles across the batch.")
        elif repeat_eviction_delta < 0:
            notes.append(f"Repeat archive evictions fell by {-repeat_eviction_delta} bundles across the batch.")
        else:
            notes.append("Repeat archive evictions stayed flat across the batch.")
        if repeat_eviction_tier_delta > 0:
            notes.append(f"Repeat-eviction tier rose by {repeat_eviction_tier_delta} across the batch.")
        elif repeat_eviction_tier_delta < 0:
            notes.append(f"Repeat-eviction tier fell by {-repeat_eviction_tier_delta} across the batch.")
        else:
            notes.append("Repeat-eviction tier stayed flat across the batch.")
        if archive_conversion_delta > 0:
            notes.append(f"Archive admission conversion rate increased by {archive_conversion_delta} across the batch.")
        elif archive_conversion_delta < 0:
            notes.append(f"Archive admission conversion rate fell by {-archive_conversion_delta} across the batch.")
        else:
            notes.append("Archive admission conversion rate stayed flat across the batch.")
        if archive_reentry_converted_delta > 0:
            notes.append(
                f"Archive re-entry readmissions increased by {archive_reentry_converted_delta} across the batch."
            )
        elif archive_reentry_converted_delta < 0:
            notes.append(
                f"Archive re-entry readmissions fell by {-archive_reentry_converted_delta} across the batch."
            )
        else:
            notes.append("Archive re-entry readmissions stayed flat across the batch.")
        if archive_reentry_gap_delta > 0:
            notes.append(
                f"Mean time-to-reentry increased by {archive_reentry_gap_delta} generations across the batch."
            )
        elif archive_reentry_gap_delta < 0:
            notes.append(
                f"Mean time-to-reentry fell by {-archive_reentry_gap_delta} generations across the batch."
            )
        else:
            notes.append("Mean time-to-reentry stayed flat across the batch.")
        if archive_retired_delta > 0:
            notes.append(f"Retired archive bundle count increased by {archive_retired_delta} across the batch.")
        elif archive_retired_delta < 0:
            notes.append(f"Retired archive bundle count fell by {-archive_retired_delta} across the batch.")
        else:
            notes.append("Retired archive bundle count stayed flat across the batch.")
        if archive_failed_delta > 0:
            notes.append(f"Failed archive admissions increased by {archive_failed_delta} across the batch.")
        elif archive_failed_delta < 0:
            notes.append(f"Failed archive admissions fell by {-archive_failed_delta} across the batch.")
        else:
            notes.append("Failed archive admissions stayed flat across the batch.")
        if bundle_archive_cooldown_delta > 0:
            notes.append(f"Archive cooldown blocked {bundle_archive_cooldown_delta} more roles by the end of the batch.")
        elif bundle_archive_cooldown_delta < 0:
            notes.append(f"Archive cooldown pressure fell by {-bundle_archive_cooldown_delta} roles across the batch.")
        else:
            notes.append("Archive cooldown pressure stayed flat across the batch.")
        if bundle_archive_coexistence_budget_delta > 0:
            notes.append(
                f"Archive coexistence budget coverage increased by {bundle_archive_coexistence_budget_delta} roles across the batch."
            )
        elif bundle_archive_coexistence_budget_delta < 0:
            notes.append(
                f"Archive coexistence budget coverage fell by {-bundle_archive_coexistence_budget_delta} roles across the batch."
            )
        else:
            notes.append("Archive coexistence budget coverage stayed flat across the batch.")
        if bundle_archive_cooldown_true_overload_delta > 0:
            notes.append(
                f"True archive overload cooldowns increased by {bundle_archive_cooldown_true_overload_delta} roles across the batch."
            )
        elif bundle_archive_cooldown_true_overload_delta < 0:
            notes.append(
                f"True archive overload cooldowns fell by {-bundle_archive_cooldown_true_overload_delta} roles across the batch."
            )
        else:
            notes.append("True archive overload cooldowns stayed flat across the batch.")
        if bundle_archive_cooldown_avoidable_duplicate_delta > 0:
            notes.append(
                f"Avoidable duplicate cooldown pressure increased by {bundle_archive_cooldown_avoidable_duplicate_delta} roles across the batch."
            )
        elif bundle_archive_cooldown_avoidable_duplicate_delta < 0:
            notes.append(
                f"Avoidable duplicate cooldown pressure fell by {-bundle_archive_cooldown_avoidable_duplicate_delta} roles across the batch."
            )
        else:
            notes.append("Avoidable duplicate cooldown pressure stayed flat across the batch.")
        if bundle_archive_cooldown_recovery_delta > 0:
            notes.append(
                f"Archive cooldown recoveries increased by {bundle_archive_cooldown_recovery_delta} roles across the batch."
            )
        elif bundle_archive_cooldown_recovery_delta < 0:
            notes.append(
                f"Archive cooldown recoveries fell by {-bundle_archive_cooldown_recovery_delta} roles across the batch."
            )
        else:
            notes.append("Archive cooldown recoveries stayed flat across the batch.")
        if bundle_decay_prune_delta > 0:
            notes.append(f"Archive decay pruning expanded to {bundle_decay_prune_delta} more roles across the batch.")
        elif bundle_decay_prune_delta < 0:
            notes.append(f"Archive decay pruning fell by {-bundle_decay_prune_delta} roles across the batch.")
        else:
            notes.append("Archive decay pruning stayed flat across the batch.")
        if stale_bundle_delta > 0:
            notes.append(f"Stale bundle count increased by {stale_bundle_delta} across the batch.")
        elif stale_bundle_delta < 0:
            notes.append(f"Stale bundle count fell by {-stale_bundle_delta} across the batch.")
        else:
            notes.append("Stale bundle count stayed flat across the batch.")
        if decaying_bundle_delta > 0:
            notes.append(f"Archive decay pressure increased by {decaying_bundle_delta} bundles across the batch.")
        elif decaying_bundle_delta < 0:
            notes.append(f"Archive decay pressure fell by {-decaying_bundle_delta} bundles across the batch.")
        else:
            notes.append("Archive decay pressure stayed flat across the batch.")
        if archive_retirement_ready_delta > 0:
            notes.append(
                f"Archive retirement readiness increased by {archive_retirement_ready_delta} bundles across the batch."
            )
        elif archive_retirement_ready_delta < 0:
            notes.append(
                f"Archive retirement readiness fell by {-archive_retirement_ready_delta} bundles across the batch."
            )
        else:
            notes.append("Archive retirement readiness stayed flat across the batch.")
        if pruned_bundle_delta > 0:
            notes.append(f"Bundle pruning count increased by {pruned_bundle_delta} across the batch.")
        elif pruned_bundle_delta < 0:
            notes.append(f"Bundle pruning count fell by {-pruned_bundle_delta} across the batch.")
        else:
            notes.append("Bundle pruning count stayed flat across the batch.")
    if warned_lineages:
        notes.append(
            "Inheritance warning effect: "
            f"{avoided_recurrence}/{warned_lineages} warned lineages avoided the warned failure, "
            f"{repeated_warning} repeated it, {shifted_failure} also failed in a different way."
        )

    return {
        "generation_ids": generation_ids,
        "generation_metrics": generation_metrics,
        "lineage_outcomes": dict(outcome_counts),
        "blocked_lineages": blocked_lineages,
        "reviewed_lineages": reviewed_lineages,
        "propagated_lineages": propagated_lineages,
        "pending_lineages": pending_lineages,
        "lineages": lineages,
        "inheritance_effect": {
            "warned_lineages": warned_lineages,
            "avoided_recurrence": avoided_recurrence,
            "repeated_warning": repeated_warning,
            "shifted_failure": shifted_failure,
            "transfer_score": inheritance_effect["transfer_score"],
        },
        "notes": notes,
    }


def render_experiment_report(report: dict[str, Any]) -> str:
    lines = [
        f"# Experiment {report['generation_ids'][0]}-{report['generation_ids'][-1]}",
        "",
        "## Generations",
        "",
    ]
    for metric in report["generation_metrics"]:
        lines.append(
            f"- g{metric['generation_id']}: public_eval_average={metric['public_eval_average']} "
            f"eligible={metric['eligible']} blocked={metric['propagation_blocked']} "
            f"review_only={metric['review_only']} diffusion_alerts={metric['diffusion_alerts']} "
            f"anti_corruption={metric['anti_corruption']} warned_lineages={metric['warned_lineages']} "
            f"memorial_transfer_score={metric['memorial_transfer_score']} "
            f"monoculture_index={metric['monoculture_index']} "
            f"prompt_variant_count={metric['prompt_variant_count']} "
            f"prompt_bundle_count={metric['prompt_bundle_count']} "
            f"largest_variant_share={metric['largest_variant_share']} "
            f"largest_bundle_share={metric['largest_bundle_share']} "
            f"bundle_concentration_index={metric['bundle_concentration_index']} "
            f"parent_bundle_concentration_index={metric['parent_bundle_concentration_index']} "
            f"bundle_survival_rate={metric['bundle_survival_rate']} "
            f"bundle_turnover_rate={metric['bundle_turnover_rate']} "
            f"new_bundle_win_rate={metric['new_bundle_win_rate']} "
            f"exploration_bundle_survival_rate={metric['exploration_bundle_survival_rate']} "
            f"preserved_bundle_count={metric['preserved_bundle_count']} "
            f"bundle_archive_count={metric['bundle_archive_count']} "
            f"archive_admission_pending_count={metric['archive_admission_pending_count']} "
            f"archive_proving_count={metric['archive_proving_count']} "
            f"archive_reentry_block_count={metric['archive_reentry_block_count']} "
            f"archive_escalated_backoff_count={metric['archive_escalated_backoff_count']} "
            f"archive_reentry_attempt_count={metric['archive_reentry_attempt_count']} "
            f"archive_underperform_count={metric['archive_underperform_count']} "
            f"archive_positive_lift_count={metric['archive_positive_lift_count']} "
            f"archive_value_deficit_count={metric['archive_value_deficit_count']} "
            f"archive_incumbent_win_count={metric['archive_incumbent_win_count']} "
            f"archive_incumbent_loss_count={metric['archive_incumbent_loss_count']} "
            f"archive_mean_comparative_lift={metric['archive_mean_comparative_lift']} "
            f"archive_transfer_success_count={metric['archive_transfer_success_count']} "
            f"archive_transfer_failure_count={metric['archive_transfer_failure_count']} "
            f"archive_transfer_success_rate={metric['archive_transfer_success_rate']} "
            f"archive_parent_vs_child_lift_retention={metric['archive_parent_vs_child_lift_retention']} "
            f"archive_admitted_count={metric['archive_admitted_count']} "
            f"newly_admitted_count={metric['newly_admitted_count']} "
            f"post_admission_grace_count={metric['post_admission_grace_count']} "
            f"archive_eviction_count={metric['archive_eviction_count']} "
            f"repeat_eviction_count={metric['repeat_eviction_count']} "
            f"archive_repeat_eviction_max_tier={metric['archive_repeat_eviction_max_tier']} "
            f"archive_admission_conversion_rate={metric['archive_admission_conversion_rate']} "
            f"archive_reentry_converted_count={metric['archive_reentry_converted_count']} "
            f"archive_reentry_mean_gap_generations={metric['archive_reentry_mean_gap_generations']} "
            f"archive_reentry_max_gap_generations={metric['archive_reentry_max_gap_generations']} "
            f"archive_retired_count={metric['archive_retired_count']} "
            f"archive_failed_admission_count={metric['archive_failed_admission_count']} "
            f"bundle_archive_cooldown_count={metric['bundle_archive_cooldown_count']} "
            f"bundle_archive_coexistence_budget_count={metric['bundle_archive_coexistence_budget_count']} "
            f"bundle_archive_cooldown_true_overload_count={metric['bundle_archive_cooldown_true_overload_count']} "
            f"bundle_archive_cooldown_avoidable_duplicate_count={metric['bundle_archive_cooldown_avoidable_duplicate_count']} "
            f"bundle_archive_cooldown_recovery_count={metric['bundle_archive_cooldown_recovery_count']} "
            f"bundle_archive_cooldown_recovery_max_generations={metric['bundle_archive_cooldown_recovery_max_generations']} "
            f"bundle_decay_prune_count={metric['bundle_decay_prune_count']} "
            f"stale_bundle_count={metric['stale_bundle_count']} "
            f"decaying_bundle_count={metric['decaying_bundle_count']} "
            f"archive_retirement_ready_count={metric['archive_retirement_ready_count']} "
            f"pruned_bundle_count={metric['pruned_bundle_count']} "
            f"parent_concentration_index={metric['parent_concentration_index']} "
            f"drift_pressure_lineages={metric['drift_pressure_lineages']} "
            f"diversity_priority_count={metric['diversity_priority_count']} "
            f"most_converged_role={metric['most_converged_role'] or 'none'} "
            f"most_common_variant_role={metric['most_common_variant_role'] or 'none'} "
            f"most_common_bundle_role={metric['most_common_bundle_role'] or 'none'} "
            f"most_bundle_concentrated_role={metric['most_bundle_concentrated_role'] or 'none'} "
            f"most_parent_bundle_concentrated_role={metric['most_parent_bundle_concentrated_role'] or 'none'} "
            f"most_reused_parent_role={metric['most_reused_parent_role'] or 'none'}"
        )
        if metric["preserved_bundles"]:
            lines.append(f"  preserved_bundles={','.join(metric['preserved_bundles'])}")
        if metric["bundle_archive_roles"]:
            lines.append(f"  bundle_archive_roles={','.join(metric['bundle_archive_roles'])}")
        if metric["bundle_archive_pending_roles"]:
            lines.append(
                f"  bundle_archive_pending_roles={','.join(metric['bundle_archive_pending_roles'])}"
            )
        if metric["bundle_archive_proving_roles"]:
            lines.append(
                f"  bundle_archive_proving_roles={','.join(metric['bundle_archive_proving_roles'])}"
            )
        if metric["bundle_archive_post_admission_grace_roles"]:
            lines.append(
                "  bundle_archive_post_admission_grace_roles="
                + ",".join(metric["bundle_archive_post_admission_grace_roles"])
            )
        if metric["bundle_archive_reentry_backoff_roles"]:
            lines.append(
                "  bundle_archive_reentry_backoff_roles="
                + ",".join(metric["bundle_archive_reentry_backoff_roles"])
            )
        if metric["bundle_archive_reentry_block_roles"]:
            lines.append(
                "  bundle_archive_reentry_block_roles="
                + ",".join(metric["bundle_archive_reentry_block_roles"])
            )
        if metric["bundle_archive_escalated_backoff_roles"]:
            lines.append(
                "  bundle_archive_escalated_backoff_roles="
                + ",".join(metric["bundle_archive_escalated_backoff_roles"])
            )
        if metric["bundle_archive_coexistence_budget_roles"]:
            lines.append(
                "  bundle_archive_coexistence_budget_roles="
                + ",".join(metric["bundle_archive_coexistence_budget_roles"])
            )
        if metric["bundle_archive_underperform_roles"]:
            lines.append(
                "  bundle_archive_underperform_roles="
                + ",".join(metric["bundle_archive_underperform_roles"])
            )
        if metric["bundle_archive_positive_lift_roles"]:
            lines.append(
                "  bundle_archive_positive_lift_roles="
                + ",".join(metric["bundle_archive_positive_lift_roles"])
            )
        if metric["bundle_archive_value_deficit_roles"]:
            lines.append(
                "  bundle_archive_value_deficit_roles="
                + ",".join(metric["bundle_archive_value_deficit_roles"])
            )
        if metric["bundle_archive_transfer_success_roles"]:
            lines.append(
                "  bundle_archive_transfer_success_roles="
                + ",".join(metric["bundle_archive_transfer_success_roles"])
            )
        if metric["bundle_archive_transfer_failure_roles"]:
            lines.append(
                "  bundle_archive_transfer_failure_roles="
                + ",".join(metric["bundle_archive_transfer_failure_roles"])
            )
        if metric["bundle_archive_eviction_roles"]:
            lines.append(
                "  bundle_archive_eviction_roles="
                + ",".join(metric["bundle_archive_eviction_roles"])
            )
        if metric["bundle_archive_repeat_eviction_roles"]:
            lines.append(
                "  bundle_archive_repeat_eviction_roles="
                + ",".join(metric["bundle_archive_repeat_eviction_roles"])
            )
        if metric["bundle_archive_retired_roles"]:
            lines.append(
                "  bundle_archive_retired_roles="
                + ",".join(metric["bundle_archive_retired_roles"])
            )
        if metric["bundle_archive_cooldown_roles"]:
            lines.append(
                f"  bundle_archive_cooldown_roles={','.join(metric['bundle_archive_cooldown_roles'])}"
            )
        if metric["bundle_archive_cooldown_fresh_admission_roles"]:
            lines.append(
                "  bundle_archive_cooldown_fresh_admission_roles="
                + ",".join(metric["bundle_archive_cooldown_fresh_admission_roles"])
            )
        if metric["bundle_archive_cooldown_long_lived_debt_roles"]:
            lines.append(
                "  bundle_archive_cooldown_long_lived_debt_roles="
                + ",".join(metric["bundle_archive_cooldown_long_lived_debt_roles"])
            )
        if metric["bundle_archive_cooldown_true_overload_roles"]:
            lines.append(
                "  bundle_archive_cooldown_true_overload_roles="
                + ",".join(metric["bundle_archive_cooldown_true_overload_roles"])
            )
        if metric["bundle_archive_cooldown_avoidable_duplicate_roles"]:
            lines.append(
                "  bundle_archive_cooldown_avoidable_duplicate_roles="
                + ",".join(metric["bundle_archive_cooldown_avoidable_duplicate_roles"])
            )
        if metric["bundle_archive_cooldown_recovery_roles"]:
            lines.append(
                "  bundle_archive_cooldown_recovery_roles="
                + ",".join(metric["bundle_archive_cooldown_recovery_roles"])
            )
        if metric["archive_retirement_reason_counts"]:
            reason_parts = [
                f"{reason}:{count}"
                for reason, count in sorted(metric["archive_retirement_reason_counts"].items())
            ]
            lines.append("  archive_retirement_reason_counts=" + ",".join(reason_parts))
        if metric["bundle_decay_prune_roles"]:
            lines.append(
                f"  bundle_decay_prune_roles={','.join(metric['bundle_decay_prune_roles'])}"
            )
        if metric["pruned_bundles"]:
            lines.append(
                "  pruned_bundles="
                + ",".join(
                    f"{item['role']}:{item['bundle_signature']}:{item.get('pruned_reason', 'bundle_pressure_pruned')}"
                    for item in metric["pruned_bundles"]
                )
            )
    lines.extend(["", "## Lineage outcomes", ""])
    for outcome, count in sorted(report["lineage_outcomes"].items()):
        lines.append(f"- {outcome}: {count}")
    effect = report["inheritance_effect"]
    lines.extend(
        [
            "",
            "## Inheritance effect",
            "",
            f"- warned_lineages: {effect['warned_lineages']}",
            f"- avoided_recurrence: {effect['avoided_recurrence']}",
            f"- repeated_warning: {effect['repeated_warning']}",
            f"- shifted_failure: {effect['shifted_failure']}",
            f"- transfer_score: {effect['transfer_score']}",
            "",
            "## Notes",
            "",
        ]
    )
    for note in report["notes"]:
        lines.append(f"- {note}")
    if report["blocked_lineages"]:
        lines.extend(["", "## Blocked lineages", ""])
        for entry in report["blocked_lineages"][:5]:
            lines.append(
                f"- {entry['lineage_id']} g{entry['generation_id']} role={entry['role']} "
                f"reasons={','.join(entry['reasons']) or 'none'}"
            )
    if report["reviewed_lineages"]:
        lines.extend(["", "## Reviewed lineages", ""])
        for entry in report["reviewed_lineages"][:5]:
            lines.append(
                f"- {entry['lineage_id']} g{entry['generation_id']} role={entry['role']} "
                f"reasons={','.join(entry['reasons']) or 'none'}"
            )
    if report["propagated_lineages"]:
        lines.extend(["", "## Propagated lineages", ""])
        for entry in report["propagated_lineages"][:5]:
            lines.append(
                f"- {entry['lineage_id']} g{entry['generation_id']} role={entry['role']} "
                f"children={','.join(entry['child_lineage_ids']) or 'none'}"
            )
    if report["pending_lineages"]:
        lines.extend(["", "## Pending eligible lineages", ""])
        for entry in report["pending_lineages"][:5]:
            lines.append(
                f"- {entry['lineage_id']} g{entry['generation_id']} role={entry['role']} "
                f"taboo_tags={','.join(entry['taboo_tags']) or 'none'}"
            )
    return "\n".join(lines) + "\n"
