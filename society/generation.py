from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from evals import run_eval_suite
from society.config import AutoCivConfig, dump_config_snapshot
from society.constants import (
    ACTIVE_STATUS,
    BUNDLE_ARCHIVE_ADMISSION_PUBLIC_SCORE_FLOOR,
    BUNDLE_ARCHIVE_ADMISSION_USEFUL_STREAK,
    BUNDLE_ARCHIVE_COMPARATIVE_LIFT_MIN,
    BUNDLE_ARCHIVE_COMPARATIVE_LIFT_STREAK,
    BUNDLE_ARCHIVE_COEXISTENCE_BUDGET_GENERATIONS,
    BUNDLE_ARCHIVE_COEXISTENCE_REQUIRED_LANES,
    BUNDLE_ARCHIVE_COOLDOWN_DEBT_THRESHOLD,
    BUNDLE_ARCHIVE_DECAY_PRUNE_GENERATION_THRESHOLD,
    BUNDLE_ARCHIVE_DECAY_PRUNE_SLOTS,
    BUNDLE_ARCHIVE_EXPLORATION_SLOTS,
    BUNDLE_ARCHIVE_MIN_ROLE_SIZE,
    BUNDLE_ARCHIVE_MONOCULTURE_THRESHOLD,
    BUNDLE_ARCHIVE_POST_ADMISSION_GRACE_GENERATIONS,
    BUNDLE_ARCHIVE_REENTRY_BACKOFF_GENERATIONS,
    BUNDLE_ARCHIVE_REPEAT_EVICTION_BACKOFF_ESCALATION,
    BUNDLE_ARCHIVE_REPEAT_EVICTION_PROVING_STREAK_BONUS,
    BUNDLE_ARCHIVE_REPEAT_EVICTION_RETIREMENT_THRESHOLD,
    BUNDLE_ARCHIVE_RETIREMENT_USEFUL_SCORE_FLOOR,
    BUNDLE_ARCHIVE_RETIREMENT_USEFUL_STREAK,
    BUNDLE_ARCHIVE_UNDERPERFORM_EVICTION_STREAK,
    BUNDLE_ARCHIVE_UNDERPERFORM_PUBLIC_SCORE_MARGIN,
    BUNDLE_STALE_GENERATION_THRESHOLD,
    COMPLETED_STATUS,
    CONSTITUTION_VERSION,
    DRIFT_PRESSURE_EXPLORATION_SLOTS,
    DRIFT_PRESSURE_MIN_ROLE_SIZE,
    DRIFT_PRESSURE_MONOCULTURE_THRESHOLD,
    QUARANTINE_CLEAN,
    QUARANTINE_REVIEW,
    QUARANTINE_SEVERITY,
    ROLE_ORDER,
    RUNNING_STATUS,
    TABOO_REGISTRY_VERSION,
    TERMINATED_STATUS,
)
from society.inheritance import assemble_inheritance_package, build_role_scoped_taboo_registry
from society.lifespan import LifespanRunner
from society.memorials import build_memorial_record, group_evals_by_agent
from society.memory import build_private_scratchpad
from society.prompts import load_role_prompts
from society.schemas import (
    AgentRecord,
    ArtifactRecord,
    EventRecord,
    GenerationRecord,
    InheritancePackage,
    LineageRecord,
    RolePrompt,
)
from society.selection import build_parent_candidate_pool, select_candidates
from society.storage import StorageManager
from society.trust import compute_drift_metrics, summarize_warning_effect, warning_labels
from society.utils import sha256_data, utc_now
from society.variation import (
    PACKAGE_POLICIES,
    materialize_prompt_bundle,
    mutate_package_policy,
    mutate_variant,
    root_variant,
    variant_by_id,
    variants_for_role,
)
from worlds.shared_notebook_v0 import SharedNotebookV0


class GenerationRunner:
    def __init__(
        self,
        *,
        config: AutoCivConfig,
        storage: StorageManager,
        provider,
        repo_root: str | Path = ".",
        roles_dir: str | Path = "roles",
    ) -> None:
        self.config = config
        self.storage = storage
        self.provider = provider
        self.repo_root = Path(repo_root)
        self.roles_dir = self.repo_root / roles_dir
        self.lifespan = LifespanRunner(provider)
        self.prompt_bundles_by_agent: dict[str, RolePrompt] = {}

    def _store_event(self, event: EventRecord, all_events: list[EventRecord]) -> None:
        self.storage.put_event(event)
        all_events.append(event)

    def _worst_quarantine_status(self, *statuses: str) -> str:
        valid = [status for status in statuses if status]
        if not valid:
            return QUARANTINE_CLEAN
        return max(valid, key=lambda status: QUARANTINE_SEVERITY.get(status, 0))

    def _relevant_events_for_agent(self, agent_id: str, events: list[EventRecord]) -> list[EventRecord]:
        return [
            event
            for event in events
            if event.agent_id == agent_id
            or event.event_payload.get("target_agent_id") == agent_id
            or event.event_payload.get("resolved_by") == agent_id
        ]

    def _persist_artifact(
        self,
        *,
        generation_id: int,
        artifact: ArtifactRecord,
        artifact_content: str,
        all_artifacts: list[ArtifactRecord],
    ) -> None:
        prior_artifacts = list(all_artifacts)
        self.storage.put_artifact(artifact, artifact_content)
        all_artifacts.append(artifact)
        for cited_artifact_id in artifact.citations:
            cited = next((candidate for candidate in prior_artifacts if candidate.artifact_id == cited_artifact_id), None)
            if cited is None or cited.author_agent_id == artifact.author_agent_id:
                continue
            self.storage.put_communication(
                generation_id=generation_id,
                source_agent_id=artifact.author_agent_id,
                target_agent_id=cited.author_agent_id,
                message_type="citation",
                artifact_id=artifact.artifact_id,
            )
            self.storage.put_trust_edge(
                generation_id=generation_id,
                source_agent_id=artifact.author_agent_id,
                target_agent_id=cited.author_agent_id,
                weight=1.0,
                evidence_json={"artifact_id": artifact.artifact_id, "cited_artifact_id": cited_artifact_id},
            )

    def _record_episode_finalization(
        self,
        *,
        generation_id: int,
        world: SharedNotebookV0,
        finalization: dict[str, Any] | None,
        all_events: list[EventRecord],
        all_artifacts: list[ArtifactRecord],
    ) -> None:
        if finalization is None:
            return
        artifact_payload = finalization["artifact"]
        artifact = ArtifactRecord(
            artifact_id=artifact_payload["artifact_id"],
            generation_id=generation_id,
            author_agent_id=artifact_payload["author_agent_id"],
            artifact_type=artifact_payload["artifact_type"],
            title=artifact_payload["title"],
            content_path=str(world.artifact_path(artifact_payload["artifact_id"])),
            summary=artifact_payload["summary"],
            provenance=artifact_payload["provenance"],
            world_id=world.world_id,
            visibility="public",
            citations=artifact_payload["citations"],
            quarantine_status="clean",
            created_at=utc_now(),
        )
        self._persist_artifact(
            generation_id=generation_id,
            artifact=artifact,
            artifact_content=artifact_payload["content"],
            all_artifacts=all_artifacts,
        )
        for index, world_event in enumerate(finalization["events"]):
            self._store_event(
                EventRecord(
                    event_id=f"evt-{generation_id:04d}-{world.episode_index:02d}-{index}-final-{artifact.artifact_id}",
                    generation_id=generation_id,
                    agent_id=world_event["agent_id"],
                    event_type=world_event["event_type"],
                    event_payload={**world_event["event_payload"], "world_id": world.world_id},
                    created_at=utc_now(),
                ),
                all_events,
            )

    def _parent_context(self, generation_id: int) -> dict[str, Any]:
        previous_generation_id = self.storage.latest_generation_id_before(generation_id)
        if previous_generation_id is None:
            return {
                "previous_generation_id": None,
                "eligible_by_role": {},
                "artifacts_by_agent": {},
                "memorials_by_agent": {},
                "registry_taboo_tags_by_role": {},
                "prompt_variant_by_agent": {},
                "package_policy_by_agent": {},
                "prior_role_monoculture": {},
            }

        previous_agents = self.storage.list_generation_agents(previous_generation_id)
        previous_artifacts = self.storage.list_generation_artifacts(previous_generation_id)
        previous_memorials = self.storage.list_generation_memorials(previous_generation_id)
        previous_evals = self.storage.list_generation_evals(previous_generation_id)
        previous_generation = self.storage.get_generation(previous_generation_id)
        previous_lineage_updates = [] if previous_generation is None else previous_generation.summary_json.get("lineage_updates", [])
        prior_registry_by_role = (
            {} if previous_generation is None else previous_generation.summary_json.get("registry_taboo_tags_by_role", {})
        )
        prior_bundle_state_by_role = (
            {}
            if previous_generation is None
            else previous_generation.summary_json.get("selection_summary", {}).get("bundle_state_by_role", {})
        )

        role_by_agent_id = {agent.agent_id: agent.role for agent in previous_agents}
        artifacts_by_agent: dict[str, list[ArtifactRecord]] = defaultdict(list)
        memorials_by_agent: dict[str, list[Any]] = defaultdict(list)
        taboo_tags_by_agent: dict[str, list[str]] = {
            update["agent_id"]: update.get("taboo_tags", [])
            for update in previous_lineage_updates
        }
        prompt_variant_by_agent = {
            update["agent_id"]: update.get("prompt_variant_id")
            for update in previous_lineage_updates
            if update.get("prompt_variant_id")
        }
        package_policy_by_agent = {
            update["agent_id"]: update.get("package_policy_id")
            for update in previous_lineage_updates
            if update.get("package_policy_id")
        }
        bundle_signatures_by_role = {
            role: {
                update.get("bundle_signature")
                for update in previous_lineage_updates
                if update.get("role") == role and update.get("bundle_signature")
            }
            for role in sorted({update.get("role") for update in previous_lineage_updates if update.get("role")})
        }
        previous_selection = select_candidates(
            previous_agents,
            previous_evals,
            previous_artifacts,
            variation_by_agent=self._variation_by_agent(previous_lineage_updates),
        )
        prior_role_monoculture = (
            {} if previous_generation is None else previous_generation.summary_json.get("selection_summary", {}).get("role_monoculture_index", {})
        )
        for artifact in previous_artifacts:
            artifacts_by_agent[artifact.author_agent_id].append(artifact)
        for memorial in previous_memorials:
            memorials_by_agent[memorial.source_agent_id].append(memorial)

        parent_pool_summary = self._parent_pool_summary(
            previous_agents,
            previous_selection,
            role_monoculture_index=prior_role_monoculture,
            bundle_state_by_role=prior_bundle_state_by_role,
        )
        eligible_by_role = parent_pool_summary["pools_by_role"]

        current_registry_by_role = build_role_scoped_taboo_registry(
            previous_memorials,
            role_by_agent_id=role_by_agent_id,
        )
        registry_taboo_tags_by_role = {
            role: sorted({*prior_registry_by_role.get(role, []), *current_registry_by_role.get(role, [])})
            for role in set(prior_registry_by_role) | set(current_registry_by_role)
        }

        return {
            "previous_generation_id": previous_generation_id,
            "eligible_by_role": dict(eligible_by_role),
            "artifacts_by_agent": dict(artifacts_by_agent),
            "memorials_by_agent": dict(memorials_by_agent),
            "taboo_tags_by_agent": taboo_tags_by_agent,
            "registry_taboo_tags_by_role": registry_taboo_tags_by_role,
            "prompt_variant_by_agent": prompt_variant_by_agent,
            "package_policy_by_agent": package_policy_by_agent,
            "bundle_signatures_by_role": bundle_signatures_by_role,
            "prior_role_monoculture": prior_role_monoculture,
            "bundle_archive_roles": parent_pool_summary["bundle_archive_roles"],
        }

    def _variation_by_agent(self, lineage_updates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {
            update["agent_id"]: {
                "prompt_variant_id": update.get("prompt_variant_id"),
                "package_policy_id": update.get("package_policy_id"),
            }
            for update in lineage_updates
        }

    def _parent_pool_summary(
        self,
        agents: list[AgentRecord],
        selection: list[Any],
        *,
        role_monoculture_index: dict[str, float] | None = None,
        bundle_state_by_role: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        agent_by_id = {agent.agent_id: agent for agent in agents}
        pools_by_role: dict[str, list[dict[str, Any]]] = {}
        role_parent_bundle_concentration_index: dict[str, float] = {}
        preserved_bundles: list[dict[str, Any]] = []
        pruned_bundles: list[dict[str, Any]] = []
        preserved_signatures: set[str] = set()
        role_counts = Counter(agent.role for agent in agents)
        if role_monoculture_index is None:
            role_monoculture_index = {
                role: round(
                    statistics.fmean([decision.cohort_similarity for decision in selection if decision.role == role]),
                    4,
                )
                if any(decision.role == role for decision in selection)
                else 0.0
                for role in sorted({decision.role for decision in selection})
            }
        bundle_archive_candidate_roles = {
            role
            for role, score in role_monoculture_index.items()
            if score >= BUNDLE_ARCHIVE_MONOCULTURE_THRESHOLD
            and role_counts.get(role, 0) >= BUNDLE_ARCHIVE_MIN_ROLE_SIZE
        }
        bundle_archive_pending_roles = {
            role
            for role in bundle_archive_candidate_roles
            if any(
                bool(state.get("archive_candidate_generations", 0)) and not bool(state.get("archive_admitted", False))
                for state in (bundle_state_by_role or {}).get(role, {}).values()
            )
        }
        bundle_archive_proving_roles = {
            role
            for role in bundle_archive_candidate_roles
            if any(
                int(state.get("archive_proving_streak", 0)) > 0 and not bool(state.get("archive_admitted", False))
                for state in (bundle_state_by_role or {}).get(role, {}).values()
            )
        }
        bundle_archive_post_admission_grace_roles = {
            role
            for role in bundle_archive_candidate_roles
            if any(
                int(state.get("archive_post_admission_grace_remaining", 0)) > 0
                for state in (bundle_state_by_role or {}).get(role, {}).values()
            )
        }
        bundle_archive_reentry_backoff_roles = {
            role
            for role in bundle_archive_candidate_roles
            if any(
                int(state.get("archive_reentry_backoff_remaining", 0)) > 0
                for state in (bundle_state_by_role or {}).get(role, {}).values()
            )
        }
        bundle_archive_reentry_block_roles = {
            role
            for role in bundle_archive_candidate_roles
            if any(
                bool(state.get("archive_reentry_blocked", False))
                for state in (bundle_state_by_role or {}).get(role, {}).values()
            )
        }
        bundle_archive_escalated_backoff_roles = {
            role
            for role in bundle_archive_candidate_roles
            if any(
                int(state.get("archive_reentry_backoff_target", 0)) > BUNDLE_ARCHIVE_REENTRY_BACKOFF_GENERATIONS
                for state in (bundle_state_by_role or {}).get(role, {}).values()
            )
        }
        bundle_archive_coexistence_budget_roles = {
            role
            for role in bundle_archive_candidate_roles
            if any(
                int(state.get("archive_coexistence_budget_remaining", 0)) > 0
                for state in (bundle_state_by_role or {}).get(role, {}).values()
            )
        }
        bundle_archive_underperform_roles = {
            role
            for role in bundle_archive_candidate_roles
            if any(
                int(state.get("archive_underperform_streak", 0)) > 0
                for state in (bundle_state_by_role or {}).get(role, {}).values()
            )
        }
        bundle_archive_positive_lift_roles = {
            role
            for role in bundle_archive_candidate_roles
            if any(
                float(state.get("archive_comparative_lift", 0.0)) >= BUNDLE_ARCHIVE_COMPARATIVE_LIFT_MIN
                and float(state.get("archive_incumbent_public_benchmark", 0.0)) > 0
                for state in (bundle_state_by_role or {}).get(role, {}).values()
            )
        }
        bundle_archive_value_deficit_roles = {
            role
            for role in bundle_archive_candidate_roles
            if any(
                float(state.get("archive_comparative_lift", 0.0)) < BUNDLE_ARCHIVE_COMPARATIVE_LIFT_MIN
                and float(state.get("archive_incumbent_public_benchmark", 0.0)) > 0
                and int(state.get("archive_candidate_generations", 0)) > 0
                for state in (bundle_state_by_role or {}).get(role, {}).values()
            )
        }
        bundle_archive_eviction_roles = {
            role
            for role in bundle_archive_candidate_roles
            if any(
                bool(state.get("archive_evicted", False))
                for state in (bundle_state_by_role or {}).get(role, {}).values()
            )
        }
        bundle_archive_repeat_eviction_roles = {
            role
            for role, role_states in (bundle_state_by_role or {}).items()
            if any(int(state.get("archive_eviction_count", 0)) > 1 for state in role_states.values())
        }
        bundle_archive_retired_roles = {
            role
            for role, role_states in (bundle_state_by_role or {}).items()
            if any(bool(state.get("archive_retired", False)) for state in role_states.values())
        }
        bundle_archive_cooldown_fresh_admission_roles: set[str] = set()
        bundle_archive_cooldown_roles: set[str] = set()
        bundle_archive_cooldown_long_lived_debt_roles: set[str] = set()
        bundle_archive_cooldown_true_overload_roles: set[str] = set()
        bundle_archive_cooldown_avoidable_duplicate_roles: set[str] = set()
        bundle_archive_roles: set[str] = set()
        preliminary_bundle_archive_roles = (
            bundle_archive_candidate_roles
            - bundle_archive_pending_roles
            - bundle_archive_underperform_roles
        )
        bundle_decay_prune_roles = {
            role
            for role, role_states in (bundle_state_by_role or {}).items()
            if role_counts.get(role, 0) >= BUNDLE_ARCHIVE_MIN_ROLE_SIZE
            and any(
                int(state.get("archive_decay_generations", 0)) >= BUNDLE_ARCHIVE_DECAY_PRUNE_GENERATION_THRESHOLD
                for state in role_states.values()
            )
        }

        for role in sorted({agent.role for agent in agents}):
            candidates = [
                {"agent": agent_by_id[decision.agent_id], "decision": decision}
                for decision in selection
                if decision.role == role and decision.eligible
            ]
            if not candidates:
                continue
            pool = build_parent_candidate_pool(
                candidates,
                slot_count=self.config.roles.distribution.get(role, len(candidates)),
                exploration_slots=(
                    BUNDLE_ARCHIVE_EXPLORATION_SLOTS if role in preliminary_bundle_archive_roles else 0
                ),
                reserve_penalty_slots=BUNDLE_ARCHIVE_DECAY_PRUNE_SLOTS if role in bundle_decay_prune_roles else 0,
                bundle_state_by_signature=(
                    {} if bundle_state_by_role is None else bundle_state_by_role.get(role, {})
                ),
            )
            pools_by_role[role] = pool
            candidate_bundles = {
                item["decision"].bundle_signature
                for item in candidates
                if item["decision"].bundle_signature
            }
            selected_bundles = {
                item["bundle_signature"]
                for item in pool
                if item.get("bundle_signature")
            }
            bundle_counts = Counter(
                item["bundle_signature"] for item in pool if item.get("bundle_signature")
            )
            role_parent_bundle_concentration_index[role] = (
                round(max(bundle_counts.values()) / len(pool), 4) if bundle_counts and pool else 0.0
            )
            stale_state = {} if bundle_state_by_role is None else bundle_state_by_role.get(role, {})
            admitted_selected = [
                signature
                for signature in selected_bundles
                if bool(stale_state.get(signature, {}).get("archive_admitted", False))
            ]
            exploratory_selected = [
                signature
                for signature in selected_bundles
                if (
                    int(stale_state.get(signature, {}).get("archive_candidate_generations", 0)) > 0
                    and not bool(stale_state.get(signature, {}).get("archive_admitted", False))
                    and not bool(stale_state.get(signature, {}).get("archive_retired", False))
                )
            ]
            incumbent_selected = [
                signature
                for signature in selected_bundles
                if (
                    int(stale_state.get(signature, {}).get("archive_candidate_generations", 0)) == 0
                    and not bool(stale_state.get(signature, {}).get("archive_admitted", False))
                    and not bool(stale_state.get(signature, {}).get("archive_retired", False))
                )
            ]
            non_archive_duplicate_count = sum(
                count - 1
                for bundle_signature, count in bundle_counts.items()
                if (
                    count > 1
                    and not bool(stale_state.get(bundle_signature, {}).get("archive_admitted", False))
                )
            )
            cooldown_fresh_admission = any(
                int(state.get("archive_post_admission_grace_remaining", 0)) > 0
                and max(
                    0,
                    int(state.get("archive_generations", 0))
                    - int(state.get("archive_retirement_credit", 0)),
                ) >= BUNDLE_ARCHIVE_COOLDOWN_DEBT_THRESHOLD
                for state in stale_state.values()
            )
            cooldown_candidate = any(
                int(state.get("archive_decay_debt", 0)) >= BUNDLE_ARCHIVE_COOLDOWN_DEBT_THRESHOLD
                for state in stale_state.values()
            )
            coexistence_budget_active = any(
                int(state.get("archive_coexistence_budget_remaining", 0)) > 0
                for state in stale_state.values()
            )
            coexistence_supported = bool(
                cooldown_candidate
                and coexistence_budget_active
                and len(pool) >= BUNDLE_ARCHIVE_COEXISTENCE_REQUIRED_LANES
                and admitted_selected
                and exploratory_selected
                and incumbent_selected
            )
            if cooldown_candidate:
                if coexistence_supported:
                    if non_archive_duplicate_count > 0:
                        bundle_archive_cooldown_avoidable_duplicate_roles.add(role)
                else:
                    bundle_archive_cooldown_roles.add(role)
                    bundle_archive_cooldown_true_overload_roles.add(role)
                    if cooldown_fresh_admission:
                        bundle_archive_cooldown_fresh_admission_roles.add(role)
                    else:
                        bundle_archive_cooldown_long_lived_debt_roles.add(role)
            for bundle_signature in sorted(candidate_bundles - selected_bundles):
                state = stale_state.get(bundle_signature, {})
                if int(state.get("stale_generations", 0)) >= BUNDLE_STALE_GENERATION_THRESHOLD:
                    pruned_reason = "stale_bundle_pruned"
                elif bool(state.get("archive_evicted", False)):
                    pruned_reason = "archive_eviction_pruned"
                elif int(state.get("archive_underperform_streak", 0)) > 0 and bool(state.get("archive_admitted", False)):
                    pruned_reason = "archive_underperform_pruned"
                elif bool(state.get("archive_reentry_blocked", False)) or int(
                    state.get("archive_reentry_backoff_remaining", 0)
                ) > 0:
                    pruned_reason = "archive_reentry_backoff_pruned"
                elif bool(state.get("archive_retired", False)):
                    pruned_reason = "archive_retired_pruned"
                elif bool(state.get("archive_candidate_generations", 0)) and not bool(state.get("archive_admitted", False)):
                    pruned_reason = "archive_admission_pruned"
                elif int(state.get("archive_decay_generations", 0)) >= BUNDLE_ARCHIVE_DECAY_PRUNE_GENERATION_THRESHOLD:
                    pruned_reason = "long_lived_decay_pruned"
                elif int(state.get("archive_decay_debt", 0)) > 0:
                    pruned_reason = "archive_decay_pruned"
                elif int(state.get("retention_debt", 0)) > 0:
                    pruned_reason = "retention_decay_pruned"
                else:
                    pruned_reason = "bundle_pressure_pruned"
                pruned_bundles.append(
                    {
                        "role": role,
                        "bundle_signature": bundle_signature,
                        "stale_generations": int(state.get("stale_generations", 0)),
                        "clean_win_generations": int(state.get("clean_win_generations", 0)),
                        "retention_debt": int(state.get("retention_debt", 0)),
                        "archive_candidate_generations": int(state.get("archive_candidate_generations", 0)),
                        "archive_admission_pending_generations": int(
                            state.get("archive_admission_pending_generations", 0)
                        ),
                        "archive_proving_streak": int(state.get("archive_proving_streak", 0)),
                        "archive_admission_converted": bool(state.get("archive_admission_converted", False)),
                        "archive_admitted": bool(state.get("archive_admitted", False)),
                        "archive_post_admission_grace_remaining": int(
                            state.get("archive_post_admission_grace_remaining", 0)
                        ),
                        "archive_coexistence_budget_remaining": int(
                            state.get("archive_coexistence_budget_remaining", 0)
                        ),
                        "archive_reentry_backoff_remaining": int(
                            state.get("archive_reentry_backoff_remaining", 0)
                        ),
                        "archive_reentry_backoff_target": int(state.get("archive_reentry_backoff_target", 0)),
                        "archive_reentry_required_proving_streak": int(
                            state.get("archive_reentry_required_proving_streak", 0)
                        ),
                        "archive_reentry_attempt_count": int(state.get("archive_reentry_attempt_count", 0)),
                        "archive_reentry_blocked": bool(state.get("archive_reentry_blocked", False)),
                        "archive_repeat_eviction_tier": int(state.get("archive_repeat_eviction_tier", 0)),
                        "archive_retired": bool(state.get("archive_retired", False)),
                        "archive_decay_debt": int(state.get("archive_decay_debt", 0)),
                        "archive_decay_generations": int(state.get("archive_decay_generations", 0)),
                        "archive_useful_clean_streak": int(state.get("archive_useful_clean_streak", 0)),
                        "archive_retirement_credit": int(state.get("archive_retirement_credit", 0)),
                        "avg_public_score": float(state.get("avg_public_score", 0.0)),
                        "pruned_reason": pruned_reason,
                    }
                )
            for item in pool:
                if not item.get("bundle_preserved"):
                    continue
                bundle_signature = item.get("bundle_signature")
                if bundle_signature is None or bundle_signature in preserved_signatures:
                    continue
                preserved_signatures.add(bundle_signature)
                decision = item["decision"]
                preserved_bundles.append(
                    {
                        "role": decision.role,
                        "agent_id": decision.agent_id,
                        "lineage_id": decision.lineage_id,
                        "bundle_signature": bundle_signature,
                        "prompt_variant_id": decision.prompt_variant_id,
                        "package_policy_id": decision.package_policy_id,
                        "preservation_reason": item.get("bundle_preservation_reason"),
                    }
                )

        bundle_archive_roles = (
            bundle_archive_candidate_roles
            - bundle_archive_cooldown_roles
            - bundle_archive_pending_roles
            - bundle_archive_underperform_roles
        )

        return {
            "pools_by_role": pools_by_role,
            "role_parent_bundle_concentration_index": role_parent_bundle_concentration_index,
            "preserved_bundles": preserved_bundles,
            "pruned_bundles": pruned_bundles,
            "bundle_archive_roles": sorted(bundle_archive_roles),
            "bundle_archive_candidate_roles": sorted(bundle_archive_candidate_roles),
            "bundle_archive_pending_roles": sorted(bundle_archive_pending_roles),
            "bundle_archive_proving_roles": sorted(bundle_archive_proving_roles),
            "bundle_archive_post_admission_grace_roles": sorted(bundle_archive_post_admission_grace_roles),
            "bundle_archive_reentry_backoff_roles": sorted(bundle_archive_reentry_backoff_roles),
            "bundle_archive_reentry_block_roles": sorted(bundle_archive_reentry_block_roles),
            "bundle_archive_escalated_backoff_roles": sorted(bundle_archive_escalated_backoff_roles),
            "bundle_archive_coexistence_budget_roles": sorted(bundle_archive_coexistence_budget_roles),
            "bundle_archive_underperform_roles": sorted(bundle_archive_underperform_roles),
            "bundle_archive_positive_lift_roles": sorted(bundle_archive_positive_lift_roles),
            "bundle_archive_value_deficit_roles": sorted(bundle_archive_value_deficit_roles),
            "bundle_archive_eviction_roles": sorted(bundle_archive_eviction_roles),
            "bundle_archive_repeat_eviction_roles": sorted(bundle_archive_repeat_eviction_roles),
            "bundle_archive_retired_roles": sorted(bundle_archive_retired_roles),
            "bundle_archive_cooldown_fresh_admission_roles": sorted(bundle_archive_cooldown_fresh_admission_roles),
            "bundle_archive_cooldown_roles": sorted(bundle_archive_cooldown_roles),
            "bundle_archive_cooldown_long_lived_debt_roles": sorted(bundle_archive_cooldown_long_lived_debt_roles),
            "bundle_archive_cooldown_true_overload_roles": sorted(bundle_archive_cooldown_true_overload_roles),
            "bundle_archive_cooldown_avoidable_duplicate_roles": sorted(
                bundle_archive_cooldown_avoidable_duplicate_roles
            ),
            "bundle_decay_prune_roles": sorted(bundle_decay_prune_roles),
        }

    def _bundle_state_by_role(
        self,
        *,
        generation_id: int,
        previous_generation_id: int | None,
        lineage_updates: list[dict[str, Any]],
        selection: list[Any],
    ) -> dict[str, dict[str, Any]]:
        previous_state_by_role = {}
        if previous_generation_id is not None:
            previous_generation = self.storage.get_generation(previous_generation_id)
            if previous_generation is not None:
                previous_state_by_role = previous_generation.summary_json.get("selection_summary", {}).get(
                    "bundle_state_by_role",
                    {},
                )

        role_survivor_public_benchmark: dict[str, float] = {}
        clean_public_scores_by_role: dict[str, list[float]] = defaultdict(list)
        for decision in selection:
            if decision.eligible and decision.quarantine_status == QUARANTINE_CLEAN:
                clean_public_scores_by_role[decision.role].append(decision.public_score)
        for role, scores in clean_public_scores_by_role.items():
            ordered_scores = sorted(scores, reverse=True)
            survivor_window = max(
                1,
                min(
                    len(ordered_scores),
                    max(1, (self.config.roles.distribution.get(role, len(ordered_scores)) + 1) // 2),
                ),
            )
            role_survivor_public_benchmark[role] = round(
                statistics.fmean(ordered_scores[:survivor_window]),
                4,
            )

        decisions_by_bundle: dict[tuple[str, str], list[Any]] = defaultdict(list)
        for decision in selection:
            if decision.bundle_signature is None:
                continue
            decisions_by_bundle[(decision.role, decision.bundle_signature)].append(decision)

        updates_by_bundle: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for update in lineage_updates:
            bundle_signature = update.get("bundle_signature")
            if bundle_signature is None:
                continue
            updates_by_bundle[(update["role"], bundle_signature)].append(update)

        role_incumbent_public_benchmark: dict[str, float] = {}
        incumbent_clean_scores_by_role: dict[str, list[float]] = defaultdict(list)
        for decision in selection:
            if (
                decision.bundle_signature is None
                or not decision.eligible
                or decision.quarantine_status != QUARANTINE_CLEAN
            ):
                continue
            prior_state = previous_state_by_role.get(decision.role, {}).get(decision.bundle_signature, {})
            bundle_updates = updates_by_bundle.get((decision.role, decision.bundle_signature), [])
            archive_generated = any(
                update.get("variant_origin") == "bundle_archive_exploration"
                for update in bundle_updates
            )
            has_archive_history = bool(
                archive_generated
                or int(prior_state.get("archive_candidate_generations", 0)) > 0
                or bool(prior_state.get("archive_admitted", False))
                or int(prior_state.get("archive_generations", 0)) > 0
                or int(prior_state.get("archive_eviction_count", 0)) > 0
                or bool(prior_state.get("archive_retired", False))
            )
            if not has_archive_history:
                incumbent_clean_scores_by_role[decision.role].append(decision.public_score)
        for role, scores in incumbent_clean_scores_by_role.items():
            ordered_scores = sorted(scores, reverse=True)
            survivor_window = max(
                1,
                min(
                    len(ordered_scores),
                    max(1, (self.config.roles.distribution.get(role, len(ordered_scores)) + 1) // 2),
                ),
            )
            role_incumbent_public_benchmark[role] = round(
                statistics.fmean(ordered_scores[:survivor_window]),
                4,
            )

        bundle_state_by_role: dict[str, dict[str, Any]] = {}
        for role in sorted({update["role"] for update in lineage_updates}):
            role_states: dict[str, Any] = {}
            role_signatures = {
                bundle_signature
                for update_role, bundle_signature in updates_by_bundle
                if update_role == role
            }
            for bundle_signature in sorted(role_signatures):
                prior_state = previous_state_by_role.get(role, {}).get(bundle_signature, {})
                bundle_decisions = decisions_by_bundle.get((role, bundle_signature), [])
                bundle_updates = updates_by_bundle.get((role, bundle_signature), [])
                clean_win = any(
                    decision.eligible and decision.quarantine_status == QUARANTINE_CLEAN
                    for decision in bundle_decisions
                )
                preserved = any(decision.bundle_preserved for decision in bundle_decisions)
                archive_generated = any(
                    update.get("variant_origin") == "bundle_archive_exploration"
                    for update in bundle_updates
                )
                prior_archive_candidate_generations = int(prior_state.get("archive_candidate_generations", 0))
                prior_archive_admission_pending_generations = int(
                    prior_state.get("archive_admission_pending_generations", 0)
                )
                prior_archive_proving_streak = int(prior_state.get("archive_proving_streak", 0))
                prior_archive_post_admission_grace_remaining = int(
                    prior_state.get("archive_post_admission_grace_remaining", 0)
                )
                prior_archive_coexistence_budget_remaining = int(
                    prior_state.get("archive_coexistence_budget_remaining", 0)
                )
                prior_archive_reentry_backoff_remaining = int(
                    prior_state.get("archive_reentry_backoff_remaining", 0)
                )
                prior_archive_reentry_backoff_target = int(
                    prior_state.get("archive_reentry_backoff_target", BUNDLE_ARCHIVE_REENTRY_BACKOFF_GENERATIONS)
                )
                prior_archive_reentry_attempt_count = int(prior_state.get("archive_reentry_attempt_count", 0))
                prior_archive_underperform_streak = int(prior_state.get("archive_underperform_streak", 0))
                prior_archive_eviction_count = int(prior_state.get("archive_eviction_count", 0))
                prior_archive_last_eviction_generation_id = prior_state.get("archive_last_eviction_generation_id")
                prior_archive_retired = bool(prior_state.get("archive_retired", False))
                prior_archive_retired_generation_id = prior_state.get("archive_retired_generation_id")
                prior_archive_retirement_reason = prior_state.get("archive_retirement_reason")
                prior_archive_admitted = bool(prior_state.get("archive_admitted", False))
                prior_archive_positive_lift_streak = int(prior_state.get("archive_positive_lift_streak", 0))
                prior_archive_value_deficit_streak = int(prior_state.get("archive_value_deficit_streak", 0))
                role_public_benchmark = role_survivor_public_benchmark.get(role, 0.0)
                role_incumbent_benchmark = role_incumbent_public_benchmark.get(role, 0.0)
                avg_public_score = round(
                    statistics.fmean([decision.public_score for decision in bundle_decisions]),
                    4,
                ) if bundle_decisions else 0.0
                archive_candidate = (
                    not prior_archive_retired
                    and (
                        archive_generated
                        or prior_archive_candidate_generations > 0
                        or prior_archive_admitted
                        or int(prior_state.get("archive_generations", 0)) > 0
                        or prior_archive_eviction_count > 0
                    )
                )
                base_useful_admission_signal = (
                    archive_candidate
                    and clean_win
                    and avg_public_score >= BUNDLE_ARCHIVE_ADMISSION_PUBLIC_SCORE_FLOOR
                )
                archive_reentry_required_proving_streak = (
                    BUNDLE_ARCHIVE_ADMISSION_USEFUL_STREAK
                    + (prior_archive_eviction_count * BUNDLE_ARCHIVE_REPEAT_EVICTION_PROVING_STREAK_BONUS)
                )
                archive_reentry_attempted = bool(
                    prior_archive_eviction_count > 0 and base_useful_admission_signal
                )
                archive_reentry_blocked = bool(
                    prior_archive_reentry_backoff_remaining > 0 and archive_reentry_attempted
                )
                useful_admission_signal = base_useful_admission_signal and not archive_reentry_blocked
                archive_proving_streak = (
                    prior_archive_proving_streak + 1
                    if useful_admission_signal and not prior_archive_admitted
                    else prior_archive_proving_streak
                    if prior_archive_admitted
                    else 0
                )
                archive_admitted = prior_archive_admitted or (
                    useful_admission_signal
                    and archive_proving_streak >= archive_reentry_required_proving_streak
                )
                archive_admission_converted = bool(not prior_archive_admitted and archive_admitted)
                clean_win_generations = int(prior_state.get("clean_win_generations", 0)) + int(clean_win)
                preserved_generations = int(prior_state.get("preserved_generations", 0)) + int(preserved)
                archive_candidate_generations = prior_archive_candidate_generations + int(archive_candidate)
                archive_admission_pending_generations = (
                    0
                    if not archive_candidate or archive_admitted
                    else prior_archive_admission_pending_generations + 1
                )
                archive_generations = int(prior_state.get("archive_generations", 0)) + int(archive_admitted)
                archive_post_admission_grace_remaining = (
                    BUNDLE_ARCHIVE_POST_ADMISSION_GRACE_GENERATIONS
                    if archive_admission_converted
                    else max(0, prior_archive_post_admission_grace_remaining - 1)
                )
                archive_comparative_lift = (
                    round(avg_public_score - role_incumbent_benchmark, 4)
                    if archive_candidate and role_incumbent_benchmark > 0
                    else 0.0
                )
                archive_positive_lift = bool(
                    archive_candidate
                    and clean_win
                    and role_incumbent_benchmark > 0
                    and archive_comparative_lift >= BUNDLE_ARCHIVE_COMPARATIVE_LIFT_MIN
                )
                archive_positive_lift_streak = (
                    prior_archive_positive_lift_streak + 1
                    if archive_positive_lift
                    else 0
                )
                archive_value_qualified = bool(
                    archive_admitted
                    and archive_positive_lift_streak >= BUNDLE_ARCHIVE_COMPARATIVE_LIFT_STREAK
                )
                archive_value_deficit = bool(
                    archive_admitted
                    and archive_post_admission_grace_remaining == 0
                    and role_incumbent_benchmark > 0
                    and not archive_value_qualified
                )
                archive_value_deficit_streak = (
                    prior_archive_value_deficit_streak + 1
                    if archive_value_deficit
                    else 0
                )
                archive_underperform_margin = (
                    round(max(0.0, role_public_benchmark - avg_public_score), 4)
                    if archive_admitted and role_public_benchmark > 0
                    else 0.0
                )
                archive_underperform = (
                    archive_admitted
                    and archive_post_admission_grace_remaining == 0
                    and role_public_benchmark > 0
                    and avg_public_score + BUNDLE_ARCHIVE_UNDERPERFORM_PUBLIC_SCORE_MARGIN < role_public_benchmark
                )
                archive_underperform_streak = (
                    prior_archive_underperform_streak + 1
                    if archive_underperform
                    else 0
                )
                archive_eviction_reason = None
                if (
                    archive_admitted
                    and archive_underperform_streak >= BUNDLE_ARCHIVE_UNDERPERFORM_EVICTION_STREAK
                ):
                    archive_eviction_reason = "underperform"
                elif archive_admitted and archive_value_deficit_streak > 0:
                    archive_eviction_reason = "no_comparative_lift"
                archive_evicted = archive_eviction_reason is not None
                archive_eviction_count = prior_archive_eviction_count + int(archive_evicted)
                archive_repeat_eviction_tier = archive_eviction_count
                archive_reentry_backoff_target = (
                    BUNDLE_ARCHIVE_REENTRY_BACKOFF_GENERATIONS
                    + max(0, archive_eviction_count - 1) * BUNDLE_ARCHIVE_REPEAT_EVICTION_BACKOFF_ESCALATION
                    if archive_eviction_count > 0
                    else prior_archive_reentry_backoff_target
                )
                archive_retired = bool(
                    prior_archive_retired
                    or archive_eviction_count >= BUNDLE_ARCHIVE_REPEAT_EVICTION_RETIREMENT_THRESHOLD
                )
                archive_retirement_reason = (
                    prior_archive_retirement_reason
                    if prior_archive_retired
                    else archive_eviction_reason
                    if archive_retired and archive_evicted
                    else None
                )
                if archive_evicted:
                    archive_admitted = False
                    archive_admission_pending_generations = (
                        0
                        if archive_retired
                        else max(
                            1,
                            prior_archive_admission_pending_generations + 1,
                        )
                    )
                    archive_proving_streak = 0
                    archive_generations = 0
                    archive_post_admission_grace_remaining = 0
                    archive_positive_lift_streak = 0
                    archive_value_deficit_streak = 0
                archive_reentry_backoff_remaining = (
                    0
                    if archive_retired
                    else archive_reentry_backoff_target
                    if archive_evicted
                    else max(0, prior_archive_reentry_backoff_remaining - 1)
                )
                archive_reentry_attempt_count = (
                    prior_archive_reentry_attempt_count + int(archive_reentry_attempted)
                )
                archive_last_eviction_generation_id = (
                    generation_id if archive_evicted else prior_archive_last_eviction_generation_id
                )
                archive_retired_generation_id = (
                    generation_id
                    if archive_retired and not prior_archive_retired
                    else prior_archive_retired_generation_id
                )
                archive_reentry_converted = bool(
                    archive_admission_converted
                    and prior_archive_eviction_count > 0
                    and prior_archive_last_eviction_generation_id is not None
                )
                archive_reentry_gap_generations = (
                    generation_id - int(prior_archive_last_eviction_generation_id)
                    if archive_reentry_converted
                    else None
                )
                archive_coexistence_budget_remaining = (
                    BUNDLE_ARCHIVE_COEXISTENCE_BUDGET_GENERATIONS
                    if archive_reentry_converted
                    else max(0, prior_archive_coexistence_budget_remaining - 1)
                )
                if archive_retired:
                    archive_candidate_generations = prior_archive_candidate_generations
                    archive_admission_pending_generations = 0
                    archive_proving_streak = 0
                    archive_coexistence_budget_remaining = 0
                    if archive_retirement_reason is None:
                        archive_retirement_reason = "repeat_eviction_threshold"
                useful_clean = (
                    archive_admitted
                    and clean_win
                    and avg_public_score >= BUNDLE_ARCHIVE_RETIREMENT_USEFUL_SCORE_FLOOR
                )
                ever_archived = archive_generations > 0 or archive_admitted
                archive_useful_clean_streak = (
                    int(prior_state.get("archive_useful_clean_streak", 0)) + 1
                    if useful_clean and ever_archived
                    else 0
                )
                archive_retirement_credit = (
                    max(
                        0,
                        archive_useful_clean_streak - BUNDLE_ARCHIVE_RETIREMENT_USEFUL_STREAK + 1,
                    )
                    if ever_archived
                    else 0
                )
                retention_debt = max(0, preserved_generations - clean_win_generations)
                archive_grace_credit = min(
                    archive_generations,
                    BUNDLE_ARCHIVE_POST_ADMISSION_GRACE_GENERATIONS,
                )
                archive_decay_debt = max(
                    0,
                    archive_generations - archive_retirement_credit - archive_grace_credit,
                )
                archive_decay_generations = (
                    int(prior_state.get("archive_decay_generations", 0)) + 1
                    if archive_decay_debt > 0
                    else 0
                )
                avg_score = round(
                    statistics.fmean([decision.score for decision in bundle_decisions]),
                    4,
                ) if bundle_decisions else 0.0
                role_states[bundle_signature] = {
                    "age_generations": int(prior_state.get("age_generations", 0)) + 1,
                    "clean_win_generations": clean_win_generations,
                    "stale_generations": 0 if clean_win else int(prior_state.get("stale_generations", 0)) + 1,
                    "preserved_generations": preserved_generations,
                    "archive_generations": archive_generations,
                    "archive_candidate_generations": archive_candidate_generations,
                    "archive_admission_pending_generations": archive_admission_pending_generations,
                    "archive_proving_streak": archive_proving_streak,
                    "archive_admission_converted": archive_admission_converted,
                    "archive_admitted": archive_admitted,
                    "archive_post_admission_grace_remaining": archive_post_admission_grace_remaining,
                    "archive_coexistence_budget_remaining": archive_coexistence_budget_remaining,
                    "archive_reentry_backoff_remaining": archive_reentry_backoff_remaining,
                    "archive_reentry_backoff_target": archive_reentry_backoff_target,
                    "archive_reentry_required_proving_streak": archive_reentry_required_proving_streak,
                    "archive_reentry_attempt_count": archive_reentry_attempt_count,
                    "archive_reentry_blocked": archive_reentry_blocked,
                    "archive_reentry_converted": archive_reentry_converted,
                    "archive_reentry_gap_generations": archive_reentry_gap_generations,
                    "archive_repeat_eviction_tier": archive_repeat_eviction_tier,
                    "archive_underperform_streak": archive_underperform_streak,
                    "archive_underperform_margin": archive_underperform_margin,
                    "archive_survivor_public_benchmark": role_public_benchmark,
                    "archive_incumbent_public_benchmark": role_incumbent_benchmark,
                    "archive_comparative_lift": archive_comparative_lift,
                    "archive_positive_lift_streak": archive_positive_lift_streak,
                    "archive_value_qualified": archive_value_qualified,
                    "archive_value_deficit_streak": archive_value_deficit_streak,
                    "archive_evicted": archive_evicted,
                    "archive_eviction_reason": archive_eviction_reason,
                    "archive_eviction_count": archive_eviction_count,
                    "archive_last_eviction_generation_id": archive_last_eviction_generation_id,
                    "archive_retired": archive_retired,
                    "archive_retired_generation_id": archive_retired_generation_id,
                    "archive_retirement_reason": archive_retirement_reason,
                    "retention_debt": retention_debt,
                    "archive_decay_debt": archive_decay_debt,
                    "archive_decay_generations": archive_decay_generations,
                    "archive_useful_clean_streak": archive_useful_clean_streak,
                    "archive_retirement_credit": archive_retirement_credit,
                    "avg_score": avg_score,
                    "avg_public_score": avg_public_score,
                    "last_seen_generation_id": generation_id,
                    "last_clean_generation_id": generation_id if clean_win else prior_state.get("last_clean_generation_id"),
                }
            bundle_state_by_role[role] = role_states
        return bundle_state_by_role

    def _drift_pressure_roles(self, parent_context: dict[str, Any]) -> set[str]:
        prior_role_monoculture = parent_context.get("prior_role_monoculture", {})
        return {
            role
            for role, score in prior_role_monoculture.items()
            if score >= DRIFT_PRESSURE_MONOCULTURE_THRESHOLD
            and self.config.roles.distribution.get(role, 0) >= DRIFT_PRESSURE_MIN_ROLE_SIZE
        }

    def _rebalance_bundle_choice(
        self,
        *,
        role: str,
        variant,
        package_policy_id: str,
        parent_variant_id: str | None,
        parent_package_policy_id: str | None,
        base_steps: int,
        used_bundle_signatures: set[str],
        forbidden_bundle_signatures: set[str] | None = None,
    ) -> tuple[Any, str]:
        forbidden_bundle_signatures = forbidden_bundle_signatures or set()
        signature = f"{role}:{variant.variant_id}:{package_policy_id}"
        if (
            signature not in used_bundle_signatures
            and signature not in forbidden_bundle_signatures
        ) or parent_variant_id is None:
            return variant, package_policy_id

        max_attempts = len(variants_for_role(role)) * len(PACKAGE_POLICIES)
        policy_seed = parent_package_policy_id or package_policy_id
        for extra_steps in range(1, max_attempts + 1):
            candidate_variant = mutate_variant(role, parent_variant_id, steps=base_steps + extra_steps)
            candidate_policy = mutate_package_policy(policy_seed, steps=base_steps + extra_steps)
            candidate_signature = f"{role}:{candidate_variant.variant_id}:{candidate_policy}"
            if (
                candidate_signature not in used_bundle_signatures
                and candidate_signature not in forbidden_bundle_signatures
            ):
                return candidate_variant, candidate_policy
        return variant, package_policy_id

    def run(self, *, generation_id: int | None = None, seed: int | None = None, dry_run: bool = False) -> dict[str, Any]:
        self.storage.initialize()
        generation_seed = self.config.generation.seed if seed is None else seed
        generation_id = generation_id or self.storage.next_generation_id()
        config_payload = self.config.snapshot()
        config_hash = sha256_data(config_payload)
        generation = GenerationRecord(
            generation_id=generation_id,
            config_hash=config_hash,
            world_name=self.config.world.name,
            population_size=self.config.generation.population_size,
            seed=generation_seed,
            status=RUNNING_STATUS,
            started_at=utc_now(),
            summary_json={},
        )
        self.storage.put_generation(generation)
        self.storage.save_config_snapshot(generation_id, dump_config_snapshot(self.config))

        prompts = load_role_prompts(list(self.config.roles.distribution.keys()), roles_dir=self.roles_dir)
        agents, inheritance_packages, lineage_updates = self._spawn_population(generation_id, prompts)
        scratchpads = {
            agent.agent_id: build_private_scratchpad(
                agent,
                inheritance_packages[agent.agent_id],
                variation={
                    "prompt_variant_id": next(
                        item["prompt_variant_id"] for item in lineage_updates if item["agent_id"] == agent.agent_id
                    ),
                    "package_policy_id": next(
                        item["package_policy_id"] for item in lineage_updates if item["agent_id"] == agent.agent_id
                    ),
                },
            )
            for agent in agents
        }

        all_artifacts: list[ArtifactRecord] = []
        all_events: list[EventRecord] = []
        episode_summaries: list[dict[str, Any]] = []
        participation_counts = {agent.agent_id: 0 for agent in agents}

        if not dry_run:
            task_pool = self.config.world_config().task_pool
            for episode_index in range(self.config.generation.episodes_per_generation):
                episode_agents = self._active_agents_for_episode(
                    agents,
                    episode_index,
                    participation_counts=participation_counts,
                    inheritance_packages=inheritance_packages,
                )
                world = SharedNotebookV0(
                    root_dir=self.storage.root_dir,
                    generation_id=generation_id,
                    episode_index=episode_index,
                    task_prompt=task_pool[episode_index % len(task_pool)],
                    max_steps=self.config.generation.max_turns_per_episode,
                )
                world.bind_population(episode_agents)
                start_event = EventRecord(
                    event_id=f"evt-{generation_id:04d}-episode-{episode_index:02d}",
                    generation_id=generation_id,
                    agent_id=None,
                    event_type="episode_started",
                    event_payload={
                        "episode_index": episode_index,
                        "task_prompt": world.task_prompt,
                        "world_id": world.world_id,
                        "participants": [agent.agent_id for agent in episode_agents],
                    },
                    created_at=utc_now(),
                )
                self._store_event(start_event, all_events)

                last_actor_id: str | None = None
                episode_finalized = False
                for step_index in range(world.step_budget()):
                    agent = world.select_next_agent(episode_agents, step_index, last_actor_id=last_actor_id)
                    if agent is None:
                        break
                    inherited = inheritance_packages[agent.agent_id]
                    available_citations = [artifact.artifact_id for artifact in all_artifacts]
                    result = self.lifespan.run_step(
                        generation_id=generation_id,
                        episode_index=episode_index,
                        agent=agent,
                        prompt=self.prompt_bundles_by_agent[agent.agent_id],
                        inherited=inherited,
                        scratchpad=scratchpads[agent.agent_id],
                        world=world,
                        step_index=step_index,
                        behavior=self.config.roles.behaviors.get(agent.role, "honest"),
                        available_citations=available_citations,
                        prompt_variant_id=scratchpads[agent.agent_id].get("prompt_variant_id"),
                        package_policy_id=scratchpads[agent.agent_id].get("package_policy_id"),
                        prompt_variant_tags=next(
                            item["prompt_variant_tags"]
                            for item in lineage_updates
                            if item["agent_id"] == agent.agent_id
                        ),
                    )
                    for event in result.events:
                        self._store_event(event, all_events)
                    participation_counts[agent.agent_id] += 1
                    self.storage.append_agent_log(
                        generation_id,
                        agent.agent_id,
                        {
                            "episode_index": episode_index,
                            "step_index": step_index,
                            "world_id": world.world_id,
                            "event_ids": [event.event_id for event in result.events],
                            "response": result.provider_response.model_dump(mode="json"),
                            "parsed_action": result.parsed_action,
                            "repair_required": result.repair_required,
                            "governance_violations": result.events[0].event_payload["governance"]["violations"],
                            "scratchpad_size": len(scratchpads[agent.agent_id]["notes"]),
                            "prompt_variant_id": scratchpads[agent.agent_id].get("prompt_variant_id"),
                            "package_policy_id": scratchpads[agent.agent_id].get("package_policy_id"),
                        },
                    )
                    last_actor_id = agent.agent_id
                    if result.artifact is not None and result.artifact_content is not None:
                        self._persist_artifact(
                            generation_id=generation_id,
                            artifact=result.artifact,
                            artifact_content=result.artifact_content,
                            all_artifacts=all_artifacts,
                        )
                    if world.should_end_episode(step_index):
                        finalization = world.finalize_episode(
                            step_index=step_index,
                            force=not world._ready_for_finalization(step_index),
                        )
                        self._record_episode_finalization(
                            generation_id=generation_id,
                            world=world,
                            finalization=finalization,
                            all_events=all_events,
                            all_artifacts=all_artifacts,
                        )
                        episode_finalized = True
                        break
                if not episode_finalized:
                    finalization = world.finalize_episode(
                        step_index=max(world.step_budget() - 1, 0),
                        force=True,
                    )
                    self._record_episode_finalization(
                        generation_id=generation_id,
                        world=world,
                        finalization=finalization,
                        all_events=all_events,
                        all_artifacts=all_artifacts,
                    )
                episode_summary = world.episode_summary()
                episode_summaries.append(episode_summary)
                end_event = EventRecord(
                    event_id=f"evt-{generation_id:04d}-episode-{episode_index:02d}-end",
                    generation_id=generation_id,
                    agent_id=None,
                    event_type="episode_completed",
                    event_payload=episode_summary,
                    created_at=utc_now(),
                )
                self._store_event(end_event, all_events)

        evals = run_eval_suite(
            config=self.config,
            generation_id=generation_id,
            agents=agents,
            artifacts=all_artifacts,
            events=all_events,
        )
        for record in evals:
            self.storage.put_eval(record)

        evals_by_agent = group_evals_by_agent(evals)
        selection = select_candidates(
            agents,
            evals,
            all_artifacts,
            variation_by_agent=self._variation_by_agent(lineage_updates),
        )
        selection_by_agent = {decision.agent_id: decision for decision in selection}
        memorials = []
        quarantine_report = []
        agent_artifacts: dict[str, list[ArtifactRecord]] = defaultdict(list)
        for artifact in all_artifacts:
            agent_artifacts[artifact.author_agent_id].append(artifact)
        for agent in agents:
            decision = selection_by_agent[agent.agent_id]
            updated_artifacts: list[ArtifactRecord] = []
            for artifact in agent_artifacts.get(agent.agent_id, []):
                status = self._worst_quarantine_status(artifact.quarantine_status, decision.quarantine_status)
                updated = artifact.model_copy(update={"quarantine_status": status})
                self.storage.put_artifact(updated)
                updated_artifacts.append(updated)
            agent_artifacts[agent.agent_id] = updated_artifacts

            if decision.quarantine_status != QUARANTINE_CLEAN or updated_artifacts:
                quarantined_ids = [
                    artifact.artifact_id
                    for artifact in updated_artifacts
                    if artifact.quarantine_status != QUARANTINE_CLEAN
                ]
                if decision.quarantine_status != QUARANTINE_CLEAN or quarantined_ids:
                    quarantine_report.append(
                        {
                            "agent_id": agent.agent_id,
                            "lineage_id": agent.lineage_id,
                            "role": agent.role,
                            "quarantine_status": decision.quarantine_status,
                            "propagation_blocked": decision.propagation_blocked,
                            "reasons": decision.reasons,
                            "artifact_ids": quarantined_ids,
                        }
                    )

            memorial = build_memorial_record(
                agent,
                agent_artifacts.get(agent.agent_id, []),
                evals_by_agent[agent.agent_id],
                self._relevant_events_for_agent(agent.agent_id, all_events),
            )
            self.storage.put_memorial(memorial)
            memorials.append(memorial)

            lineage = self.storage.get_lineage(agent.lineage_id)
            if lineage is not None:
                lineage_status = (
                    decision.quarantine_status
                    if decision.quarantine_status != QUARANTINE_CLEAN
                    else COMPLETED_STATUS
                )
                lineage_notes = f"Role {agent.role}; reasons={', '.join(decision.reasons) or 'none'}"
                self.storage.put_lineage(
                    lineage.model_copy(update={"status": lineage_status, "notes": lineage_notes})
                )

            terminated_agent = agent.model_copy(update={"status": TERMINATED_STATUS, "terminated_at": utc_now()})
            self.storage.put_agent(terminated_agent)

        all_artifacts = sorted(
            [artifact for artifacts in agent_artifacts.values() for artifact in artifacts],
            key=lambda artifact: artifact.created_at,
        )
        inheritance_effect = self._build_inheritance_effect(lineage_updates=lineage_updates, selection=selection)
        drift = compute_drift_metrics(
            all_artifacts,
            memorials,
            communications=self.storage.list_generation_communications(generation_id),
            inheritance_effect=inheritance_effect,
        )
        summary = self._build_summary(
            generation_id=generation_id,
            agents=agents,
            artifacts=all_artifacts,
            evals=evals,
            memorials=memorials,
            selection=selection,
            previous_generation_id=self.storage.latest_generation_id_before(generation_id),
            lineage_updates=lineage_updates,
            quarantine_report=quarantine_report,
            inheritance_effect=inheritance_effect,
            drift=drift.model_dump(mode="json"),
            episode_summaries=episode_summaries,
            total_events=len(self.storage.list_generation_events(generation_id)),
        )

        generation = generation.model_copy(
            update={
                "status": COMPLETED_STATUS,
                "ended_at": utc_now(),
                "summary_json": summary,
            }
        )
        self.storage.put_generation(generation)
        self.storage.save_generation_summary(generation_id, summary, self._render_markdown_summary(summary))
        return summary

    def _spawn_population(
        self,
        generation_id: int,
        prompts: dict[str, Any],
    ) -> tuple[list[AgentRecord], dict[str, InheritancePackage], list[dict[str, Any]]]:
        agents: list[AgentRecord] = []
        inheritance_packages: dict[str, InheritancePackage] = {}
        lineage_updates: list[dict[str, Any]] = []
        prompt_bundles_by_agent: dict[str, RolePrompt] = {}
        roles: list[str] = []
        parent_context = self._parent_context(generation_id)
        parent_indexes: dict[str, int] = defaultdict(int)
        role_ordinals: dict[str, int] = defaultdict(int)
        parent_reuse_counts: dict[tuple[str, str], int] = defaultdict(int)
        used_bundle_signatures_by_role: dict[str, set[str]] = defaultdict(set)
        drift_pressure_roles = self._drift_pressure_roles(parent_context)
        drift_pressure_used: dict[str, int] = defaultdict(int)
        for role, count in self.config.roles.distribution.items():
            roles.extend([role] * count)
        for index, role in enumerate(roles):
            lineage_id = f"lin-{generation_id:04d}-{index:03d}"
            agent_id = f"agent-{generation_id:04d}-{index:03d}"
            role_ordinal = role_ordinals[role]
            role_ordinals[role] += 1
            parent_candidates = parent_context["eligible_by_role"].get(role, [])
            parent_assignment = None
            if parent_candidates:
                parent_assignment = parent_candidates[parent_indexes[role] % len(parent_candidates)]
                parent_indexes[role] += 1

            parent_agent = None if parent_assignment is None else parent_assignment["agent"]
            parent_lineage_ids = [] if parent_agent is None else [parent_agent.lineage_id]
            parent_taboo_tags = [] if parent_agent is None else parent_context["taboo_tags_by_agent"].get(parent_agent.agent_id, [])
            registry_taboo_tags = parent_context["registry_taboo_tags_by_role"].get(role, [])
            inherited_taboo_tags = sorted({*registry_taboo_tags, *parent_taboo_tags})
            parent_variant_id = None if parent_agent is None else parent_context["prompt_variant_by_agent"].get(parent_agent.agent_id)
            parent_package_policy_id = None if parent_agent is None else parent_context["package_policy_by_agent"].get(parent_agent.agent_id)
            if parent_agent is None:
                variant = root_variant(role, role_ordinal)
                package_policy_id = variant.default_package_policy
                variant_origin = "seeded"
                base_steps = 0
            else:
                reuse_key = (role, parent_agent.agent_id)
                reuse_count = parent_reuse_counts[reuse_key]
                parent_reuse_counts[reuse_key] += 1
                archive_exploration_active = parent_assignment is not None and parent_assignment.get("selection_source") == "bundle_exploration"
                drift_pressure_active = (
                    reuse_count == 0
                    and not archive_exploration_active
                    and role in drift_pressure_roles
                    and drift_pressure_used[role] < DRIFT_PRESSURE_EXPLORATION_SLOTS
                )
                if archive_exploration_active:
                    base_variant = variant_by_id(role, parent_variant_id)
                    exploration_steps = max(1, reuse_count + 1)
                    variant = mutate_variant(role, base_variant.variant_id, steps=exploration_steps)
                    package_policy_id = mutate_package_policy(
                        parent_package_policy_id or base_variant.default_package_policy,
                        steps=exploration_steps + 1,
                    )
                    variant_origin = "bundle_archive_exploration"
                    base_steps = exploration_steps
                elif reuse_count > 0:
                    variant = mutate_variant(role, parent_variant_id, steps=reuse_count)
                    package_policy_id = mutate_package_policy(
                        parent_package_policy_id or variant.default_package_policy,
                        steps=reuse_count,
                    )
                    variant_origin = "mutated_on_parent_reuse"
                    base_steps = reuse_count
                elif drift_pressure_active:
                    base_variant = variant_by_id(role, parent_variant_id)
                    variant = mutate_variant(role, base_variant.variant_id, steps=1)
                    package_policy_id = mutate_package_policy(
                        parent_package_policy_id or base_variant.default_package_policy,
                        steps=1,
                    )
                    variant_origin = "drift_pressure"
                    drift_pressure_used[role] += 1
                    base_steps = 1
                else:
                    variant = variant_by_id(role, parent_variant_id)
                    package_policy_id = parent_package_policy_id or variant.default_package_policy
                    variant_origin = "inherited"
                    base_steps = 0
                variant, package_policy_id = self._rebalance_bundle_choice(
                    role=role,
                    variant=variant,
                    package_policy_id=package_policy_id,
                    parent_variant_id=parent_variant_id,
                    parent_package_policy_id=parent_package_policy_id,
                    base_steps=base_steps,
                    used_bundle_signatures=used_bundle_signatures_by_role[role],
                    forbidden_bundle_signatures=(
                        parent_context["bundle_signatures_by_role"].get(role, set())
                        if archive_exploration_active
                        else set()
                    ),
                )
            current_bundle_signature = f"{role}:{variant.variant_id}:{package_policy_id}"
            used_bundle_signatures_by_role[role].add(current_bundle_signature)
            inherited = assemble_inheritance_package(
                artifacts=[]
                if parent_agent is None
                else parent_context["artifacts_by_agent"].get(parent_agent.agent_id, []),
                memorials=[]
                if parent_agent is None
                else parent_context["memorials_by_agent"].get(parent_agent.agent_id, []),
                artifact_limit=self.config.inheritance.artifact_summaries_per_agent,
                memorial_limit=self.config.inheritance.memorials_per_agent,
                policy_id=package_policy_id,
                extra_taboo_tags=inherited_taboo_tags,
            )
            prompt_bundle = materialize_prompt_bundle(
                base_prompt=prompts[role],
                variant=variant,
                package_policy_id=package_policy_id,
            )
            agent = AgentRecord(
                agent_id=agent_id,
                generation_id=generation_id,
                lineage_id=lineage_id,
                role=role,
                model_name=self.config.provider.model,
                provider_name=self.provider.name(),
                prompt_bundle_version=prompt_bundle.sha256,
                constitution_version=CONSTITUTION_VERSION,
                inherited_artifact_ids=inherited.artifact_ids,
                inherited_memorial_ids=inherited.memorial_ids,
                taboo_registry_version=TABOO_REGISTRY_VERSION,
                status=ACTIVE_STATUS,
                created_at=utc_now(),
            )
            lineage = LineageRecord(
                lineage_id=lineage_id,
                parent_lineage_ids=parent_lineage_ids,
                founding_generation_id=generation_id,
                current_generation_id=generation_id,
                status=ACTIVE_STATUS,
                notes=(
                    f"Role {role}; root lineage"
                    if parent_agent is None
                    else f"Role {role}; parent={parent_agent.lineage_id}; prior_generation={parent_context['previous_generation_id']}"
                ),
            )
            self.storage.put_lineage(lineage)
            self.storage.put_agent(agent)
            agents.append(agent)
            prompt_bundles_by_agent[agent_id] = prompt_bundle
            inheritance_packages[agent_id] = inherited
            lineage_updates.append(
                {
                    "agent_id": agent_id,
                    "lineage_id": lineage_id,
                    "role": role,
                    "parent_lineage_ids": parent_lineage_ids,
                    "inheritance_source_agent_id": None if parent_agent is None else parent_agent.agent_id,
                    "inheritance_source_generation_id": parent_context["previous_generation_id"],
                    "inheritance_source_bundle_signature": None if parent_assignment is None else parent_assignment.get("bundle_signature"),
                    "inheritance_source_bundle_preserved": False if parent_assignment is None else parent_assignment.get("bundle_preserved", False),
                    "inheritance_source_bundle_reason": None if parent_assignment is None else parent_assignment.get("bundle_preservation_reason"),
                    "inheritance_source_selection_source": None if parent_assignment is None else parent_assignment.get("selection_source"),
                    "lineage_taboo_tags": parent_taboo_tags,
                    "registry_taboo_tags": registry_taboo_tags,
                    "inherited_artifact_ids": inherited.artifact_ids,
                    "inherited_memorial_ids": inherited.memorial_ids,
                    "taboo_tags": inherited.taboo_tags,
                    "prompt_variant_id": variant.variant_id,
                    "prompt_variant_tags": list(variant.tags),
                    "package_policy_id": package_policy_id,
                    "bundle_signature": current_bundle_signature,
                    "variant_origin": variant_origin,
                }
            )
        self.prompt_bundles_by_agent = prompt_bundles_by_agent
        return agents, inheritance_packages, lineage_updates

    def _active_agents_for_episode(
        self,
        agents: list[AgentRecord],
        episode_index: int,
        *,
        participation_counts: dict[str, int],
        inheritance_packages: dict[str, InheritancePackage],
    ) -> list[AgentRecord]:
        if not agents:
            return []

        role_buckets: dict[str, list[AgentRecord]] = defaultdict(list)
        for agent in agents:
            role_buckets[agent.role].append(agent)

        ordered: list[AgentRecord] = []
        for role in ROLE_ORDER:
            bucket = role_buckets.get(role, [])
            if not bucket:
                continue
            bucket.sort(
                key=lambda agent: (
                    participation_counts.get(agent.agent_id, 0),
                    0 if inheritance_packages[agent.agent_id].memorial_ids else 1,
                    0 if inheritance_packages[agent.agent_id].taboo_tags else 1,
                    (int(agent.agent_id.rsplit("-", 1)[-1]) - episode_index) % max(len(bucket), 1),
                )
            )
            ordered.extend(bucket)
        return ordered

    def _build_inheritance_effect(
        self,
        *,
        lineage_updates: list[dict[str, Any]],
        selection: list,
    ) -> dict[str, Any]:
        selection_by_lineage = {decision.lineage_id: decision for decision in selection}
        records: list[tuple[set[str], set[str]]] = []
        for update in lineage_updates:
            inherited_memorials = [
                memorial
                for memorial_id in update.get("inherited_memorial_ids", [])
                if (memorial := self.storage.get_memorial(memorial_id)) is not None
            ]
            labels = warning_labels(
                taboo_tags=update.get("taboo_tags", []),
                memorials=inherited_memorials,
            )
            decision = selection_by_lineage.get(update["lineage_id"])
            failures = set()
            if decision is not None:
                failures = set(decision.hidden_failures) | set(decision.public_failures)
            records.append((labels, failures))
        return summarize_warning_effect(records)

    def _build_summary(
        self,
        *,
        generation_id: int,
        agents: list[AgentRecord],
        artifacts: list[ArtifactRecord],
        evals: list,
        memorials: list,
        selection: list,
        previous_generation_id: int | None,
        lineage_updates: list[dict[str, Any]],
        quarantine_report: list[dict[str, Any]],
        inheritance_effect: dict[str, Any],
        drift: dict[str, Any],
        episode_summaries: list[dict[str, Any]],
        total_events: int,
    ) -> dict[str, Any]:
        public_scores = [record.score for record in evals if record.eval_family == "public" and record.score is not None]
        hidden_failures = [record.eval_name for record in evals if record.eval_family == "hidden" and record.pass_fail is False]
        hidden_failure_counts = dict(Counter(hidden_failures))
        quarantines = [artifact.artifact_id for artifact in artifacts if artifact.quarantine_status != QUARANTINE_CLEAN]
        previous_selection_summary: dict[str, Any] = {}
        if previous_generation_id is not None:
            previous_generation = self.storage.get_generation(previous_generation_id)
            if previous_generation is not None:
                previous_selection_summary = previous_generation.summary_json.get("selection_summary", {})
        suspicious_lineages = sorted(
            {
                decision.lineage_id
                for decision in selection
                if decision.propagation_blocked or "diffusion_alerts" in decision.hidden_failures
            }
        )
        honored_contributions = [memorial.top_contribution for memorial in memorials if memorial.classification == "honored"]
        notable_failures = [memorial.failure_mode for memorial in memorials if memorial.failure_mode]
        bundle_state_by_role = self._bundle_state_by_role(
            generation_id=generation_id,
            previous_generation_id=previous_generation_id,
            lineage_updates=lineage_updates,
            selection=selection,
        )
        parent_pool_summary = self._parent_pool_summary(
            agents,
            selection,
            bundle_state_by_role=bundle_state_by_role,
        )
        preserved_by_agent = {
            item["agent_id"]: {
                "bundle_preserved": True,
                "bundle_preservation_reason": item["preservation_reason"],
            }
            for item in parent_pool_summary["preserved_bundles"]
        }
        augmented_selection = [
            decision.model_copy(update=preserved_by_agent.get(decision.agent_id, {}))
            for decision in selection
        ]
        role_monoculture_index = {
            role: round(
                statistics.fmean([decision.cohort_similarity for decision in augmented_selection if decision.role == role]),
                4,
            )
            if any(decision.role == role for decision in augmented_selection)
            else 0.0
            for role in sorted({decision.role for decision in augmented_selection})
        }
        diversity_priority_lineages = [
            decision.lineage_id for decision in augmented_selection if decision.selection_bucket == "diversity_priority"
        ]
        role_variant_count = {
            role: len({item["prompt_variant_id"] for item in lineage_updates if item["role"] == role})
            for role in sorted({item["role"] for item in lineage_updates})
        }
        role_bundle_count = {
            role: len(
                {
                    (item["prompt_variant_id"], item["package_policy_id"])
                    for item in lineage_updates
                    if item["role"] == role
                }
            )
            for role in sorted({item["role"] for item in lineage_updates})
        }
        role_bundle_concentration_index = {
            role: round(
                max(
                    Counter(
                        (
                            item["prompt_variant_id"],
                            item["package_policy_id"],
                        )
                        for item in lineage_updates
                        if item["role"] == role
                    ).values(),
                    default=0,
                )
                / max(1, sum(1 for item in lineage_updates if item["role"] == role)),
                4,
            )
            for role in sorted({item["role"] for item in lineage_updates})
        }
        variant_origin_counts = dict(Counter(item["variant_origin"] for item in lineage_updates))
        bundle_archive_lineages = [
            item["lineage_id"]
            for item in lineage_updates
            if item["variant_origin"] == "bundle_archive_exploration"
        ]
        archive_admission_pending_bundles = [
            {
                "role": role,
                "bundle_signature": bundle_signature,
                "archive_candidate_generations": state["archive_candidate_generations"],
                "archive_admission_pending_generations": state["archive_admission_pending_generations"],
                "archive_proving_streak": state["archive_proving_streak"],
                "avg_public_score": state["avg_public_score"],
            }
            for role, role_states in bundle_state_by_role.items()
            for bundle_signature, state in role_states.items()
            if state["archive_candidate_generations"] > 0 and not state["archive_admitted"]
        ]
        archive_proving_bundles = [
            {
                "role": role,
                "bundle_signature": bundle_signature,
                "archive_candidate_generations": state["archive_candidate_generations"],
                "archive_admission_pending_generations": state["archive_admission_pending_generations"],
                "archive_proving_streak": state["archive_proving_streak"],
                "archive_incumbent_public_benchmark": state["archive_incumbent_public_benchmark"],
                "archive_comparative_lift": state["archive_comparative_lift"],
                "archive_positive_lift_streak": state["archive_positive_lift_streak"],
                "avg_public_score": state["avg_public_score"],
            }
            for role, role_states in bundle_state_by_role.items()
            for bundle_signature, state in role_states.items()
            if state["archive_proving_streak"] > 0 and not state["archive_admitted"]
        ]
        post_admission_grace_bundles = [
            {
                "role": role,
                "bundle_signature": bundle_signature,
                "archive_generations": state["archive_generations"],
                "archive_post_admission_grace_remaining": state["archive_post_admission_grace_remaining"],
                "avg_public_score": state["avg_public_score"],
            }
            for role, role_states in bundle_state_by_role.items()
            for bundle_signature, state in role_states.items()
            if state["archive_post_admission_grace_remaining"] > 0
        ]
        archive_reentry_block_bundles = [
            {
                "role": role,
                "bundle_signature": bundle_signature,
                "archive_reentry_backoff_remaining": state["archive_reentry_backoff_remaining"],
                "archive_reentry_backoff_target": state["archive_reentry_backoff_target"],
                "archive_reentry_required_proving_streak": state["archive_reentry_required_proving_streak"],
                "archive_reentry_attempt_count": state["archive_reentry_attempt_count"],
                "archive_repeat_eviction_tier": state["archive_repeat_eviction_tier"],
                "avg_public_score": state["avg_public_score"],
            }
            for role, role_states in bundle_state_by_role.items()
            for bundle_signature, state in role_states.items()
            if state["archive_reentry_blocked"]
        ]
        escalated_backoff_bundles = [
            {
                "role": role,
                "bundle_signature": bundle_signature,
                "archive_reentry_backoff_target": state["archive_reentry_backoff_target"],
                "archive_repeat_eviction_tier": state["archive_repeat_eviction_tier"],
                "avg_public_score": state["avg_public_score"],
            }
            for role, role_states in bundle_state_by_role.items()
            for bundle_signature, state in role_states.items()
            if state["archive_reentry_backoff_target"] > BUNDLE_ARCHIVE_REENTRY_BACKOFF_GENERATIONS
        ]
        archive_underperform_bundles = [
            {
                "role": role,
                "bundle_signature": bundle_signature,
                "archive_underperform_streak": state["archive_underperform_streak"],
                "archive_underperform_margin": state["archive_underperform_margin"],
                "archive_survivor_public_benchmark": state["archive_survivor_public_benchmark"],
                "archive_incumbent_public_benchmark": state["archive_incumbent_public_benchmark"],
                "archive_comparative_lift": state["archive_comparative_lift"],
                "archive_value_deficit_streak": state["archive_value_deficit_streak"],
                "archive_post_admission_grace_remaining": state["archive_post_admission_grace_remaining"],
                "avg_public_score": state["avg_public_score"],
            }
            for role, role_states in bundle_state_by_role.items()
            for bundle_signature, state in role_states.items()
            if state["archive_underperform_streak"] > 0
        ]
        archive_evicted_bundles = [
            {
                "role": role,
                "bundle_signature": bundle_signature,
                "archive_underperform_streak": state["archive_underperform_streak"],
                "archive_underperform_margin": state["archive_underperform_margin"],
                "archive_survivor_public_benchmark": state["archive_survivor_public_benchmark"],
                "archive_candidate_generations": state["archive_candidate_generations"],
                "archive_admission_pending_generations": state["archive_admission_pending_generations"],
                "archive_reentry_backoff_target": state["archive_reentry_backoff_target"],
                "archive_reentry_required_proving_streak": state["archive_reentry_required_proving_streak"],
                "archive_reentry_attempt_count": state["archive_reentry_attempt_count"],
                "archive_repeat_eviction_tier": state["archive_repeat_eviction_tier"],
                "archive_eviction_reason": state["archive_eviction_reason"],
                "archive_retired": state["archive_retired"],
                "archive_incumbent_public_benchmark": state["archive_incumbent_public_benchmark"],
                "archive_comparative_lift": state["archive_comparative_lift"],
                "archive_positive_lift_streak": state["archive_positive_lift_streak"],
                "archive_value_deficit_streak": state["archive_value_deficit_streak"],
                "avg_public_score": state["avg_public_score"],
            }
            for role, role_states in bundle_state_by_role.items()
            for bundle_signature, state in role_states.items()
            if state["archive_evicted"]
        ]
        repeat_eviction_bundles = [
            {
                "role": role,
                "bundle_signature": bundle_signature,
                "archive_eviction_count": state["archive_eviction_count"],
                "archive_repeat_eviction_tier": state["archive_repeat_eviction_tier"],
                "archive_reentry_backoff_target": state["archive_reentry_backoff_target"],
                "archive_reentry_attempt_count": state["archive_reentry_attempt_count"],
                "archive_retired": state["archive_retired"],
                "avg_public_score": state["avg_public_score"],
            }
            for role, role_states in bundle_state_by_role.items()
            for bundle_signature, state in role_states.items()
            if state["archive_eviction_count"] > 1
        ]
        archive_admitted_bundles = [
            {
                "role": role,
                "bundle_signature": bundle_signature,
                "archive_generations": state["archive_generations"],
                "archive_candidate_generations": state["archive_candidate_generations"],
                "archive_proving_streak": state["archive_proving_streak"],
                "archive_admission_converted": state["archive_admission_converted"],
                "archive_post_admission_grace_remaining": state["archive_post_admission_grace_remaining"],
                "archive_reentry_required_proving_streak": state["archive_reentry_required_proving_streak"],
                "archive_reentry_backoff_target": state["archive_reentry_backoff_target"],
                "archive_reentry_attempt_count": state["archive_reentry_attempt_count"],
                "archive_reentry_gap_generations": state["archive_reentry_gap_generations"],
                "archive_repeat_eviction_tier": state["archive_repeat_eviction_tier"],
                "archive_underperform_streak": state["archive_underperform_streak"],
                "archive_underperform_margin": state["archive_underperform_margin"],
                "archive_survivor_public_benchmark": state["archive_survivor_public_benchmark"],
                "archive_incumbent_public_benchmark": state["archive_incumbent_public_benchmark"],
                "archive_comparative_lift": state["archive_comparative_lift"],
                "archive_positive_lift_streak": state["archive_positive_lift_streak"],
                "archive_value_qualified": state["archive_value_qualified"],
                "archive_value_deficit_streak": state["archive_value_deficit_streak"],
                "archive_useful_clean_streak": state["archive_useful_clean_streak"],
                "archive_retirement_credit": state["archive_retirement_credit"],
                "avg_public_score": state["avg_public_score"],
            }
            for role, role_states in bundle_state_by_role.items()
            for bundle_signature, state in role_states.items()
            if state["archive_admitted"]
        ]
        archive_reentry_converted_bundles = [
            {
                "role": role,
                "bundle_signature": bundle_signature,
                "archive_reentry_gap_generations": state["archive_reentry_gap_generations"],
                "archive_reentry_backoff_target": state["archive_reentry_backoff_target"],
                "archive_reentry_attempt_count": state["archive_reentry_attempt_count"],
                "archive_repeat_eviction_tier": state["archive_repeat_eviction_tier"],
                "archive_incumbent_public_benchmark": state["archive_incumbent_public_benchmark"],
                "archive_comparative_lift": state["archive_comparative_lift"],
                "archive_positive_lift_streak": state["archive_positive_lift_streak"],
                "avg_public_score": state["avg_public_score"],
            }
            for role, role_states in bundle_state_by_role.items()
            for bundle_signature, state in role_states.items()
            if state["archive_reentry_converted"]
        ]
        archive_retired_bundles = [
            {
                "role": role,
                "bundle_signature": bundle_signature,
                "archive_eviction_count": state["archive_eviction_count"],
                "archive_repeat_eviction_tier": state["archive_repeat_eviction_tier"],
                "archive_reentry_backoff_target": state["archive_reentry_backoff_target"],
                "archive_reentry_attempt_count": state["archive_reentry_attempt_count"],
                "archive_retired_generation_id": state["archive_retired_generation_id"],
                "archive_retirement_reason": state["archive_retirement_reason"],
                "archive_incumbent_public_benchmark": state["archive_incumbent_public_benchmark"],
                "archive_comparative_lift": state["archive_comparative_lift"],
                "avg_public_score": state["avg_public_score"],
            }
            for role, role_states in bundle_state_by_role.items()
            for bundle_signature, state in role_states.items()
            if state["archive_retired"]
        ]
        archive_positive_lift_bundles = [
            {
                "role": role,
                "bundle_signature": bundle_signature,
                "archive_incumbent_public_benchmark": state["archive_incumbent_public_benchmark"],
                "archive_comparative_lift": state["archive_comparative_lift"],
                "archive_positive_lift_streak": state["archive_positive_lift_streak"],
                "archive_value_qualified": state["archive_value_qualified"],
                "avg_public_score": state["avg_public_score"],
            }
            for role, role_states in bundle_state_by_role.items()
            for bundle_signature, state in role_states.items()
            if (
                state["archive_candidate_generations"] > 0
                and state["archive_incumbent_public_benchmark"] > 0
                and state["archive_comparative_lift"] >= BUNDLE_ARCHIVE_COMPARATIVE_LIFT_MIN
            )
        ]
        archive_value_deficit_bundles = [
            {
                "role": role,
                "bundle_signature": bundle_signature,
                "archive_incumbent_public_benchmark": state["archive_incumbent_public_benchmark"],
                "archive_comparative_lift": state["archive_comparative_lift"],
                "archive_positive_lift_streak": state["archive_positive_lift_streak"],
                "archive_value_deficit_streak": state["archive_value_deficit_streak"],
                "archive_admitted": state["archive_admitted"],
                "avg_public_score": state["avg_public_score"],
            }
            for role, role_states in bundle_state_by_role.items()
            for bundle_signature, state in role_states.items()
            if (
                state["archive_candidate_generations"] > 0
                and state["archive_incumbent_public_benchmark"] > 0
                and state["archive_comparative_lift"] < BUNDLE_ARCHIVE_COMPARATIVE_LIFT_MIN
            )
        ]
        previous_proving_bundle_signatures = {
            (item["role"], item["bundle_signature"])
            for item in previous_selection_summary.get("archive_proving_bundles", [])
        }
        previous_cooldown_streak_by_role = previous_selection_summary.get(
            "bundle_archive_cooldown_streak_by_role",
            {},
        )
        stale_bundles = [
            {
                "role": role,
                "bundle_signature": bundle_signature,
                "stale_generations": state["stale_generations"],
                "clean_win_generations": state["clean_win_generations"],
                "retention_debt": state["retention_debt"],
                "archive_candidate_generations": state["archive_candidate_generations"],
                "archive_admission_pending_generations": state["archive_admission_pending_generations"],
                "archive_proving_streak": state["archive_proving_streak"],
                "archive_admission_converted": state["archive_admission_converted"],
                "archive_admitted": state["archive_admitted"],
                "archive_post_admission_grace_remaining": state["archive_post_admission_grace_remaining"],
                "archive_decay_debt": state["archive_decay_debt"],
                "archive_decay_generations": state["archive_decay_generations"],
                "archive_useful_clean_streak": state["archive_useful_clean_streak"],
                "archive_retirement_credit": state["archive_retirement_credit"],
                "avg_public_score": state["avg_public_score"],
            }
            for role, role_states in bundle_state_by_role.items()
            for bundle_signature, state in role_states.items()
            if state["stale_generations"] >= BUNDLE_STALE_GENERATION_THRESHOLD
        ]
        decaying_bundles = [
            {
                "role": role,
                "bundle_signature": bundle_signature,
                "retention_debt": state["retention_debt"],
                "archive_candidate_generations": state["archive_candidate_generations"],
                "archive_admission_pending_generations": state["archive_admission_pending_generations"],
                "archive_proving_streak": state["archive_proving_streak"],
                "archive_admission_converted": state["archive_admission_converted"],
                "archive_admitted": state["archive_admitted"],
                "archive_post_admission_grace_remaining": state["archive_post_admission_grace_remaining"],
                "archive_decay_debt": state["archive_decay_debt"],
                "archive_decay_generations": state["archive_decay_generations"],
                "stale_generations": state["stale_generations"],
                "archive_useful_clean_streak": state["archive_useful_clean_streak"],
                "archive_retirement_credit": state["archive_retirement_credit"],
                "avg_public_score": state["avg_public_score"],
            }
            for role, role_states in bundle_state_by_role.items()
            for bundle_signature, state in role_states.items()
            if state["retention_debt"] > 0 or state["archive_decay_debt"] > 0
        ]
        archive_retirement_ready_bundles = [
            {
                "role": role,
                "bundle_signature": bundle_signature,
                "archive_useful_clean_streak": state["archive_useful_clean_streak"],
                "archive_retirement_credit": state["archive_retirement_credit"],
                "archive_decay_debt": state["archive_decay_debt"],
                "avg_public_score": state["avg_public_score"],
            }
            for role, role_states in bundle_state_by_role.items()
            for bundle_signature, state in role_states.items()
            if state["archive_generations"] > 0
            and state["archive_useful_clean_streak"] >= BUNDLE_ARCHIVE_RETIREMENT_USEFUL_STREAK
        ]
        archive_admission_converted_bundles = [
            item
            for item in archive_admitted_bundles
            if item["archive_admission_converted"]
        ]
        newly_admitted_bundles = archive_admission_converted_bundles
        archive_failed_admission_bundles = [
            item
            for item in parent_pool_summary["pruned_bundles"]
            if item.get("pruned_reason") == "archive_admission_pruned"
        ]
        archive_admission_conversion_rate = (
            round(
                len(
                    {
                        (item["role"], item["bundle_signature"])
                        for item in archive_admission_converted_bundles
                    }
                    & previous_proving_bundle_signatures
                )
                / len(previous_proving_bundle_signatures),
                4,
            )
            if previous_proving_bundle_signatures
            else 0.0
        )
        archive_reentry_gap_values = [
            int(item["archive_reentry_gap_generations"])
            for item in archive_reentry_converted_bundles
            if item.get("archive_reentry_gap_generations") is not None
        ]
        archive_comparative_lift_values = [
            float(item["archive_comparative_lift"])
            for item in archive_positive_lift_bundles + archive_value_deficit_bundles
        ]
        archive_retirement_reason_counts = dict(
            Counter(
                str(item["archive_retirement_reason"])
                for item in archive_retired_bundles
                if item.get("archive_retirement_reason")
            )
        )
        current_cooldown_roles = set(parent_pool_summary["bundle_archive_cooldown_roles"])
        bundle_archive_cooldown_streak_by_role = {
            role: (
                int(previous_cooldown_streak_by_role.get(role, 0)) + 1
                if role in current_cooldown_roles
                else 0
            )
            for role in sorted(
                set(previous_cooldown_streak_by_role)
                | current_cooldown_roles
                | set(parent_pool_summary["bundle_archive_candidate_roles"])
            )
        }
        archive_eviction_roles = {item["role"] for item in archive_evicted_bundles}
        bundle_archive_cooldown_recoveries = [
            {
                "role": role,
                "cooldown_generations": int(previous_cooldown_streak_by_role.get(role, 0)),
                "recovery_reason": "archive_eviction" if role in archive_eviction_roles else "debt_cleared",
            }
            for role in sorted(set(previous_cooldown_streak_by_role) - current_cooldown_roles)
            if int(previous_cooldown_streak_by_role.get(role, 0)) > 0
        ]
        selection_summary = {
            "eligible": sum(decision.eligible for decision in augmented_selection),
            "propagation_blocked": sum(decision.propagation_blocked for decision in augmented_selection),
            "review_only": sum(decision.quarantine_status == QUARANTINE_REVIEW for decision in augmented_selection),
            "survivor_lineages": [decision.lineage_id for decision in augmented_selection if decision.eligible][:5],
            "role_monoculture_index": role_monoculture_index,
            "diversity_priority_lineages": diversity_priority_lineages[:5],
            "diversity_priority_count": len(diversity_priority_lineages),
            "role_variant_count": role_variant_count,
            "role_bundle_count": role_bundle_count,
            "role_bundle_concentration_index": role_bundle_concentration_index,
            "role_parent_bundle_concentration_index": parent_pool_summary["role_parent_bundle_concentration_index"],
            "preserved_bundle_lineages": [item["lineage_id"] for item in parent_pool_summary["preserved_bundles"]],
            "preserved_bundle_count": len(parent_pool_summary["preserved_bundles"]),
            "preserved_bundles": parent_pool_summary["preserved_bundles"],
            "bundle_archive_candidate_roles": parent_pool_summary["bundle_archive_candidate_roles"],
            "bundle_archive_roles": parent_pool_summary["bundle_archive_roles"],
            "bundle_archive_pending_roles": parent_pool_summary["bundle_archive_pending_roles"],
            "bundle_archive_proving_roles": parent_pool_summary["bundle_archive_proving_roles"],
            "bundle_archive_post_admission_grace_roles": parent_pool_summary["bundle_archive_post_admission_grace_roles"],
            "bundle_archive_reentry_backoff_roles": parent_pool_summary["bundle_archive_reentry_backoff_roles"],
            "bundle_archive_reentry_block_roles": parent_pool_summary["bundle_archive_reentry_block_roles"],
            "bundle_archive_escalated_backoff_roles": parent_pool_summary["bundle_archive_escalated_backoff_roles"],
            "bundle_archive_coexistence_budget_roles": parent_pool_summary["bundle_archive_coexistence_budget_roles"],
            "bundle_archive_underperform_roles": parent_pool_summary["bundle_archive_underperform_roles"],
            "bundle_archive_positive_lift_roles": parent_pool_summary["bundle_archive_positive_lift_roles"],
            "bundle_archive_value_deficit_roles": parent_pool_summary["bundle_archive_value_deficit_roles"],
            "bundle_archive_eviction_roles": parent_pool_summary["bundle_archive_eviction_roles"],
            "bundle_archive_repeat_eviction_roles": parent_pool_summary["bundle_archive_repeat_eviction_roles"],
            "bundle_archive_retired_roles": parent_pool_summary["bundle_archive_retired_roles"],
            "bundle_archive_cooldown_roles": parent_pool_summary["bundle_archive_cooldown_roles"],
            "bundle_archive_cooldown_fresh_admission_roles": parent_pool_summary[
                "bundle_archive_cooldown_fresh_admission_roles"
            ],
            "bundle_archive_cooldown_long_lived_debt_roles": parent_pool_summary[
                "bundle_archive_cooldown_long_lived_debt_roles"
            ],
            "bundle_archive_cooldown_true_overload_roles": parent_pool_summary[
                "bundle_archive_cooldown_true_overload_roles"
            ],
            "bundle_archive_cooldown_avoidable_duplicate_roles": parent_pool_summary[
                "bundle_archive_cooldown_avoidable_duplicate_roles"
            ],
            "bundle_archive_cooldown_count": len(parent_pool_summary["bundle_archive_cooldown_roles"]),
            "bundle_archive_coexistence_budget_count": len(
                parent_pool_summary["bundle_archive_coexistence_budget_roles"]
            ),
            "bundle_archive_cooldown_true_overload_count": len(
                parent_pool_summary["bundle_archive_cooldown_true_overload_roles"]
            ),
            "bundle_archive_cooldown_avoidable_duplicate_count": len(
                parent_pool_summary["bundle_archive_cooldown_avoidable_duplicate_roles"]
            ),
            "bundle_archive_cooldown_streak_by_role": bundle_archive_cooldown_streak_by_role,
            "bundle_archive_cooldown_recoveries": bundle_archive_cooldown_recoveries,
            "bundle_archive_cooldown_recovery_roles": [
                item["role"] for item in bundle_archive_cooldown_recoveries
            ],
            "bundle_archive_cooldown_recovery_count": len(bundle_archive_cooldown_recoveries),
            "bundle_archive_cooldown_recovery_max_generations": max(
                (item["cooldown_generations"] for item in bundle_archive_cooldown_recoveries),
                default=0,
            ),
            "archive_admission_pending_bundles": archive_admission_pending_bundles,
            "archive_admission_pending_count": len(archive_admission_pending_bundles),
            "archive_proving_bundles": archive_proving_bundles,
            "archive_proving_count": len(archive_proving_bundles),
            "post_admission_grace_bundles": post_admission_grace_bundles,
            "post_admission_grace_count": len(post_admission_grace_bundles),
            "archive_reentry_block_bundles": archive_reentry_block_bundles,
            "archive_reentry_block_count": len(archive_reentry_block_bundles),
            "escalated_backoff_bundles": escalated_backoff_bundles,
            "archive_escalated_backoff_count": len(escalated_backoff_bundles),
            "archive_reentry_attempt_count": sum(
                int(state.get("archive_reentry_attempt_count", 0))
                for role_states in bundle_state_by_role.values()
                for state in role_states.values()
            ),
            "archive_underperform_bundles": archive_underperform_bundles,
            "archive_underperform_count": len(archive_underperform_bundles),
            "archive_positive_lift_bundles": archive_positive_lift_bundles,
            "archive_positive_lift_count": len(archive_positive_lift_bundles),
            "archive_value_deficit_bundles": archive_value_deficit_bundles,
            "archive_value_deficit_count": len(archive_value_deficit_bundles),
            "archive_incumbent_win_count": len(archive_positive_lift_bundles),
            "archive_incumbent_loss_count": len(archive_value_deficit_bundles),
            "archive_mean_comparative_lift": (
                round(statistics.fmean(archive_comparative_lift_values), 4)
                if archive_comparative_lift_values
                else 0.0
            ),
            "archive_admitted_bundles": archive_admitted_bundles,
            "archive_admitted_count": len(archive_admitted_bundles),
            "newly_admitted_bundles": newly_admitted_bundles,
            "newly_admitted_count": len(newly_admitted_bundles),
            "archive_evicted_bundles": archive_evicted_bundles,
            "archive_eviction_count": len(archive_evicted_bundles),
            "repeat_eviction_bundles": repeat_eviction_bundles,
            "repeat_eviction_count": len(repeat_eviction_bundles),
            "archive_repeat_eviction_max_tier": max(
                (item["archive_repeat_eviction_tier"] for item in repeat_eviction_bundles),
                default=0,
            ),
            "archive_admission_converted_bundles": archive_admission_converted_bundles,
            "archive_admission_converted_count": len(archive_admission_converted_bundles),
            "archive_admission_conversion_rate": archive_admission_conversion_rate,
            "archive_reentry_converted_bundles": archive_reentry_converted_bundles,
            "archive_reentry_converted_count": len(archive_reentry_converted_bundles),
            "archive_reentry_mean_gap_generations": (
                round(statistics.fmean(archive_reentry_gap_values), 4)
                if archive_reentry_gap_values
                else 0.0
            ),
            "archive_reentry_max_gap_generations": max(archive_reentry_gap_values, default=0),
            "archive_retired_bundles": archive_retired_bundles,
            "archive_retired_count": len(archive_retired_bundles),
            "archive_retirement_reason_counts": archive_retirement_reason_counts,
            "archive_failed_admission_bundles": archive_failed_admission_bundles,
            "archive_failed_admission_count": len(archive_failed_admission_bundles),
            "bundle_decay_prune_roles": parent_pool_summary["bundle_decay_prune_roles"],
            "bundle_decay_prune_count": len(parent_pool_summary["bundle_decay_prune_roles"]),
            "bundle_archive_lineages": bundle_archive_lineages,
            "bundle_archive_count": len(bundle_archive_lineages),
            "bundle_state_by_role": bundle_state_by_role,
            "stale_bundles": stale_bundles,
            "stale_bundle_count": len(stale_bundles),
            "decaying_bundles": decaying_bundles,
            "decaying_bundle_count": len(decaying_bundles),
            "archive_retirement_ready_bundles": archive_retirement_ready_bundles,
            "archive_retirement_ready_count": len(archive_retirement_ready_bundles),
            "pruned_bundles": parent_pool_summary["pruned_bundles"],
            "pruned_bundle_count": len(parent_pool_summary["pruned_bundles"]),
            "preserved_bundles_by_role": {
                role: [item["bundle_signature"] for item in parent_pool_summary["preserved_bundles"] if item["role"] == role]
                for role in sorted({item["role"] for item in parent_pool_summary["preserved_bundles"]})
            },
            "variant_origin_counts": variant_origin_counts,
        }
        return {
            "generation_id": generation_id,
            "previous_generation_id": previous_generation_id,
            "total_agents": len(agents),
            "total_artifacts": len(artifacts),
            "total_events": total_events,
            "episodes": episode_summaries,
            "public_eval_average": round(statistics.fmean(public_scores), 4) if public_scores else 0.0,
            "hidden_eval_failures": hidden_failures,
            "hidden_eval_failure_counts": hidden_failure_counts,
            "memorials_created": len(memorials),
            "quarantines_issued": quarantines,
            "quarantine_report": quarantine_report,
            "selection_summary": selection_summary,
            "selection_outcome": [decision.model_dump(mode="json") for decision in augmented_selection],
            "lineage_updates": lineage_updates,
            "registry_taboo_tags_by_role": {
                role: sorted(
                    {
                        tag
                        for update in lineage_updates
                        if update["role"] == role
                        for tag in update.get("registry_taboo_tags", [])
                    }
                )
                for role in sorted({update["role"] for update in lineage_updates})
            },
            "inheritance_effect": inheritance_effect,
            "suspicious_lineages": suspicious_lineages,
            "top_contributions": honored_contributions[:3] or [artifact.summary for artifact in artifacts[:3]],
            "notable_failures": notable_failures[:5],
            "drift": drift,
        }

    def _render_markdown_summary(self, summary: dict[str, Any]) -> str:
        lines = [
            f"# Generation {summary['generation_id']}",
            "",
            f"- total_agents: {summary['total_agents']}",
            f"- total_artifacts: {summary['total_artifacts']}",
            f"- total_events: {summary['total_events']}",
            f"- public_eval_average: {summary['public_eval_average']}",
            f"- memorials_created: {summary['memorials_created']}",
            f"- previous_generation_id: {summary['previous_generation_id']}",
            f"- suspicious_lineages: {', '.join(summary['suspicious_lineages']) or 'none'}",
            "",
            "## Episodes",
            "",
        ]
        for episode in summary["episodes"]:
            lines.extend(
                [
                    f"- episode_{episode['episode_index']}: steps={episode['steps_completed']}, "
                    f"open_corrections={episode['open_corrections']}, "
                    f"open_clarifications={episode['open_clarifications']}, "
                    f"grace_used={episode.get('closure_grace_steps_used', 0)}, "
                    f"risk_flags={episode['risk_flags']}",
                ]
            )
        lines.extend(
            [
                "",
                "## Selection",
                "",
                f"- eligible: {summary['selection_summary']['eligible']}",
                f"- propagation_blocked: {summary['selection_summary']['propagation_blocked']}",
                f"- review_only: {summary['selection_summary']['review_only']}",
                f"- survivor_lineages: {', '.join(summary['selection_summary']['survivor_lineages']) or 'none'}",
                f"- diversity_priority_lineages: {', '.join(summary['selection_summary']['diversity_priority_lineages']) or 'none'}",
                f"- preserved_bundle_lineages: {', '.join(summary['selection_summary'].get('preserved_bundle_lineages', [])) or 'none'}",
                f"- bundle_archive_candidate_roles: {', '.join(summary['selection_summary'].get('bundle_archive_candidate_roles', [])) or 'none'}",
                f"- bundle_archive_pending_roles: {', '.join(summary['selection_summary'].get('bundle_archive_pending_roles', [])) or 'none'}",
                f"- bundle_archive_proving_roles: {', '.join(summary['selection_summary'].get('bundle_archive_proving_roles', [])) or 'none'}",
                f"- bundle_archive_post_admission_grace_roles: {', '.join(summary['selection_summary'].get('bundle_archive_post_admission_grace_roles', [])) or 'none'}",
                f"- bundle_archive_reentry_backoff_roles: {', '.join(summary['selection_summary'].get('bundle_archive_reentry_backoff_roles', [])) or 'none'}",
                f"- bundle_archive_reentry_block_roles: {', '.join(summary['selection_summary'].get('bundle_archive_reentry_block_roles', [])) or 'none'}",
                f"- bundle_archive_escalated_backoff_roles: {', '.join(summary['selection_summary'].get('bundle_archive_escalated_backoff_roles', [])) or 'none'}",
                f"- bundle_archive_coexistence_budget_roles: {', '.join(summary['selection_summary'].get('bundle_archive_coexistence_budget_roles', [])) or 'none'}",
                f"- bundle_archive_underperform_roles: {', '.join(summary['selection_summary'].get('bundle_archive_underperform_roles', [])) or 'none'}",
                f"- bundle_archive_positive_lift_roles: {', '.join(summary['selection_summary'].get('bundle_archive_positive_lift_roles', [])) or 'none'}",
                f"- bundle_archive_value_deficit_roles: {', '.join(summary['selection_summary'].get('bundle_archive_value_deficit_roles', [])) or 'none'}",
                f"- bundle_archive_eviction_roles: {', '.join(summary['selection_summary'].get('bundle_archive_eviction_roles', [])) or 'none'}",
                f"- bundle_archive_repeat_eviction_roles: {', '.join(summary['selection_summary'].get('bundle_archive_repeat_eviction_roles', [])) or 'none'}",
                f"- bundle_archive_retired_roles: {', '.join(summary['selection_summary'].get('bundle_archive_retired_roles', [])) or 'none'}",
                f"- bundle_archive_cooldown_roles: {', '.join(summary['selection_summary'].get('bundle_archive_cooldown_roles', [])) or 'none'}",
                f"- bundle_archive_cooldown_fresh_admission_roles: {', '.join(summary['selection_summary'].get('bundle_archive_cooldown_fresh_admission_roles', [])) or 'none'}",
                f"- bundle_archive_cooldown_long_lived_debt_roles: {', '.join(summary['selection_summary'].get('bundle_archive_cooldown_long_lived_debt_roles', [])) or 'none'}",
                f"- bundle_archive_cooldown_true_overload_roles: {', '.join(summary['selection_summary'].get('bundle_archive_cooldown_true_overload_roles', [])) or 'none'}",
                f"- bundle_archive_cooldown_avoidable_duplicate_roles: {', '.join(summary['selection_summary'].get('bundle_archive_cooldown_avoidable_duplicate_roles', [])) or 'none'}",
                f"- bundle_archive_cooldown_recovery_roles: {', '.join(summary['selection_summary'].get('bundle_archive_cooldown_recovery_roles', [])) or 'none'}",
                f"- bundle_decay_prune_roles: {', '.join(summary['selection_summary'].get('bundle_decay_prune_roles', [])) or 'none'}",
                f"- bundle_archive_lineages: {', '.join(summary['selection_summary'].get('bundle_archive_lineages', [])) or 'none'}",
                f"- archive_admission_pending_count: {summary['selection_summary'].get('archive_admission_pending_count', 0)}",
                f"- archive_proving_count: {summary['selection_summary'].get('archive_proving_count', 0)}",
                f"- archive_reentry_block_count: {summary['selection_summary'].get('archive_reentry_block_count', 0)}",
                f"- archive_escalated_backoff_count: {summary['selection_summary'].get('archive_escalated_backoff_count', 0)}",
                f"- archive_reentry_attempt_count: {summary['selection_summary'].get('archive_reentry_attempt_count', 0)}",
                f"- archive_underperform_count: {summary['selection_summary'].get('archive_underperform_count', 0)}",
                f"- archive_positive_lift_count: {summary['selection_summary'].get('archive_positive_lift_count', 0)}",
                f"- archive_value_deficit_count: {summary['selection_summary'].get('archive_value_deficit_count', 0)}",
                f"- archive_incumbent_win_count: {summary['selection_summary'].get('archive_incumbent_win_count', 0)}",
                f"- archive_incumbent_loss_count: {summary['selection_summary'].get('archive_incumbent_loss_count', 0)}",
                f"- archive_mean_comparative_lift: {summary['selection_summary'].get('archive_mean_comparative_lift', 0.0)}",
                f"- archive_admitted_count: {summary['selection_summary'].get('archive_admitted_count', 0)}",
                f"- newly_admitted_count: {summary['selection_summary'].get('newly_admitted_count', 0)}",
                f"- post_admission_grace_count: {summary['selection_summary'].get('post_admission_grace_count', 0)}",
                f"- archive_eviction_count: {summary['selection_summary'].get('archive_eviction_count', 0)}",
                f"- repeat_eviction_count: {summary['selection_summary'].get('repeat_eviction_count', 0)}",
                f"- archive_repeat_eviction_max_tier: {summary['selection_summary'].get('archive_repeat_eviction_max_tier', 0)}",
                f"- archive_admission_converted_count: {summary['selection_summary'].get('archive_admission_converted_count', 0)}",
                f"- archive_reentry_converted_count: {summary['selection_summary'].get('archive_reentry_converted_count', 0)}",
                f"- archive_reentry_mean_gap_generations: {summary['selection_summary'].get('archive_reentry_mean_gap_generations', 0.0)}",
                f"- archive_reentry_max_gap_generations: {summary['selection_summary'].get('archive_reentry_max_gap_generations', 0)}",
                f"- archive_retired_count: {summary['selection_summary'].get('archive_retired_count', 0)}",
                f"- archive_retirement_reason_counts: {summary['selection_summary'].get('archive_retirement_reason_counts', {})}",
                f"- archive_failed_admission_count: {summary['selection_summary'].get('archive_failed_admission_count', 0)}",
                f"- bundle_archive_coexistence_budget_count: {summary['selection_summary'].get('bundle_archive_coexistence_budget_count', 0)}",
                f"- bundle_archive_cooldown_true_overload_count: {summary['selection_summary'].get('bundle_archive_cooldown_true_overload_count', 0)}",
                f"- bundle_archive_cooldown_avoidable_duplicate_count: {summary['selection_summary'].get('bundle_archive_cooldown_avoidable_duplicate_count', 0)}",
                f"- bundle_archive_cooldown_recovery_count: {summary['selection_summary'].get('bundle_archive_cooldown_recovery_count', 0)}",
                f"- bundle_archive_cooldown_recovery_max_generations: {summary['selection_summary'].get('bundle_archive_cooldown_recovery_max_generations', 0)}",
                f"- pruned_bundle_count: {summary['selection_summary'].get('pruned_bundle_count', 0)}",
                f"- stale_bundle_count: {summary['selection_summary'].get('stale_bundle_count', 0)}",
                f"- decaying_bundle_count: {summary['selection_summary'].get('decaying_bundle_count', 0)}",
                f"- archive_retirement_ready_count: {summary['selection_summary'].get('archive_retirement_ready_count', 0)}",
                "",
                "## Quarantine",
                "",
            ]
        )
        if summary["quarantine_report"]:
            for item in summary["quarantine_report"]:
                lines.append(
                    f"- {item['lineage_id']} ({item['role']}): status={item['quarantine_status']} "
                    f"reasons={', '.join(item['reasons']) or 'none'}"
                )
        else:
            lines.append("- none")
        lines.extend(["", "## Lineages", ""])
        for update in summary["lineage_updates"]:
            lines.append(
                f"- {update['lineage_id']} ({update['role']}): parents={', '.join(update['parent_lineage_ids']) or 'root'} "
                f"inherited_artifacts={len(update['inherited_artifact_ids'])} "
                f"inherited_memorials={len(update['inherited_memorial_ids'])} "
                f"variant={update['prompt_variant_id']} policy={update['package_policy_id']} origin={update['variant_origin']}"
            )
        lines.extend(["", "## Prompt Variation", ""])
        for role, count in summary["selection_summary"].get("role_variant_count", {}).items():
            lines.append(f"- {role}: {count}")
        for role, count in summary["selection_summary"].get("role_bundle_count", {}).items():
            lines.append(f"- bundle:{role}: {count}")
        for role, value in summary["selection_summary"].get("role_bundle_concentration_index", {}).items():
            lines.append(f"- bundle_share:{role}: {value}")
        for role, value in summary["selection_summary"].get("role_parent_bundle_concentration_index", {}).items():
            lines.append(f"- parent_bundle_share:{role}: {value}")
        for origin, count in summary["selection_summary"].get("variant_origin_counts", {}).items():
            lines.append(f"- {origin}: {count}")
        for item in summary["selection_summary"].get("preserved_bundles", []):
            lines.append(
                f"- preserved:{item['role']}:{item['prompt_variant_id']}:{item['package_policy_id']}"
            )
        for role in summary["selection_summary"].get("bundle_archive_roles", []):
            lines.append(f"- archive_role:{role}")
        for role in summary["selection_summary"].get("bundle_archive_pending_roles", []):
            lines.append(f"- archive_pending_role:{role}")
        for role in summary["selection_summary"].get("bundle_archive_proving_roles", []):
            lines.append(f"- archive_proving_role:{role}")
        for role in summary["selection_summary"].get("bundle_archive_post_admission_grace_roles", []):
            lines.append(f"- archive_post_admission_grace_role:{role}")
        for role in summary["selection_summary"].get("bundle_archive_reentry_backoff_roles", []):
            lines.append(f"- archive_reentry_backoff_role:{role}")
        for role in summary["selection_summary"].get("bundle_archive_reentry_block_roles", []):
            lines.append(f"- archive_reentry_block_role:{role}")
        for role in summary["selection_summary"].get("bundle_archive_escalated_backoff_roles", []):
            lines.append(f"- archive_escalated_backoff_role:{role}")
        for role in summary["selection_summary"].get("bundle_archive_coexistence_budget_roles", []):
            lines.append(f"- archive_coexistence_budget_role:{role}")
        for role in summary["selection_summary"].get("bundle_archive_underperform_roles", []):
            lines.append(f"- archive_underperform_role:{role}")
        for role in summary["selection_summary"].get("bundle_archive_eviction_roles", []):
            lines.append(f"- archive_eviction_role:{role}")
        for role in summary["selection_summary"].get("bundle_archive_repeat_eviction_roles", []):
            lines.append(f"- archive_repeat_eviction_role:{role}")
        for role in summary["selection_summary"].get("bundle_archive_retired_roles", []):
            lines.append(f"- archive_retired_role:{role}")
        for role in summary["selection_summary"].get("bundle_archive_cooldown_roles", []):
            lines.append(f"- archive_cooldown_role:{role}")
        for role in summary["selection_summary"].get("bundle_archive_cooldown_fresh_admission_roles", []):
            lines.append(f"- archive_cooldown_fresh_admission_role:{role}")
        for role in summary["selection_summary"].get("bundle_archive_cooldown_long_lived_debt_roles", []):
            lines.append(f"- archive_cooldown_long_lived_debt_role:{role}")
        for role in summary["selection_summary"].get("bundle_archive_cooldown_true_overload_roles", []):
            lines.append(f"- archive_cooldown_true_overload_role:{role}")
        for role in summary["selection_summary"].get("bundle_archive_cooldown_avoidable_duplicate_roles", []):
            lines.append(f"- archive_cooldown_avoidable_duplicate_role:{role}")
        for role in summary["selection_summary"].get("bundle_archive_cooldown_recovery_roles", []):
            lines.append(f"- archive_cooldown_recovery_role:{role}")
        for role in summary["selection_summary"].get("bundle_decay_prune_roles", []):
            lines.append(f"- archive_decay_prune_role:{role}")
        for item in summary["selection_summary"].get("archive_admission_pending_bundles", []):
            lines.append(
                f"- admission_pending:{item['role']}:{item['bundle_signature']} "
                f"candidate_generations={item['archive_candidate_generations']} "
                f"pending_generations={item['archive_admission_pending_generations']} "
                f"proving_streak={item.get('archive_proving_streak', 0)} "
                f"avg_public_score={item.get('avg_public_score', 0.0)}"
            )
        for item in summary["selection_summary"].get("archive_proving_bundles", []):
            lines.append(
                f"- admission_proving:{item['role']}:{item['bundle_signature']} "
                f"candidate_generations={item['archive_candidate_generations']} "
                f"pending_generations={item['archive_admission_pending_generations']} "
                f"proving_streak={item.get('archive_proving_streak', 0)} "
                f"avg_public_score={item.get('avg_public_score', 0.0)}"
            )
        for item in summary["selection_summary"].get("post_admission_grace_bundles", []):
            lines.append(
                f"- admission_grace:{item['role']}:{item['bundle_signature']} "
                f"archive_generations={item['archive_generations']} "
                f"grace_remaining={item.get('archive_post_admission_grace_remaining', 0)} "
                f"avg_public_score={item.get('avg_public_score', 0.0)}"
            )
        for item in summary["selection_summary"].get("archive_reentry_block_bundles", []):
            lines.append(
                f"- reentry_blocked:{item['role']}:{item['bundle_signature']} "
                f"backoff_remaining={item.get('archive_reentry_backoff_remaining', 0)} "
                f"backoff_target={item.get('archive_reentry_backoff_target', 0)} "
                f"required_proving_streak={item.get('archive_reentry_required_proving_streak', 0)} "
                f"attempt_count={item.get('archive_reentry_attempt_count', 0)} "
                f"tier={item.get('archive_repeat_eviction_tier', 0)} "
                f"avg_public_score={item.get('avg_public_score', 0.0)}"
            )
        for item in summary["selection_summary"].get("escalated_backoff_bundles", []):
            lines.append(
                f"- escalated_backoff:{item['role']}:{item['bundle_signature']} "
                f"backoff_target={item.get('archive_reentry_backoff_target', 0)} "
                f"tier={item.get('archive_repeat_eviction_tier', 0)} "
                f"avg_public_score={item.get('avg_public_score', 0.0)}"
            )
        for item in summary["selection_summary"].get("archive_underperform_bundles", []):
            lines.append(
                f"- underperform:{item['role']}:{item['bundle_signature']} "
                f"streak={item.get('archive_underperform_streak', 0)} "
                f"margin={item.get('archive_underperform_margin', 0.0)} "
                f"benchmark={item.get('archive_survivor_public_benchmark', 0.0)} "
                f"grace_remaining={item.get('archive_post_admission_grace_remaining', 0)} "
                f"avg_public_score={item.get('avg_public_score', 0.0)}"
            )
        for item in summary["selection_summary"].get("archive_evicted_bundles", []):
            lines.append(
                f"- evicted:{item['role']}:{item['bundle_signature']} "
                f"streak={item.get('archive_underperform_streak', 0)} "
                f"margin={item.get('archive_underperform_margin', 0.0)} "
                f"benchmark={item.get('archive_survivor_public_benchmark', 0.0)} "
                f"backoff_target={item.get('archive_reentry_backoff_target', 0)} "
                f"required_proving_streak={item.get('archive_reentry_required_proving_streak', 0)} "
                f"attempt_count={item.get('archive_reentry_attempt_count', 0)} "
                f"tier={item.get('archive_repeat_eviction_tier', 0)} "
                f"retired={str(item.get('archive_retired', False)).lower()} "
                f"pending_generations={item.get('archive_admission_pending_generations', 0)} "
                f"avg_public_score={item.get('avg_public_score', 0.0)}"
            )
        for item in summary["selection_summary"].get("repeat_eviction_bundles", []):
            lines.append(
                f"- repeat_eviction:{item['role']}:{item['bundle_signature']} "
                f"eviction_count={item.get('archive_eviction_count', 0)} "
                f"tier={item.get('archive_repeat_eviction_tier', 0)} "
                f"backoff_target={item.get('archive_reentry_backoff_target', 0)} "
                f"attempt_count={item.get('archive_reentry_attempt_count', 0)} "
                f"retired={str(item.get('archive_retired', False)).lower()} "
                f"avg_public_score={item.get('avg_public_score', 0.0)}"
            )
        for item in summary["selection_summary"].get("archive_retired_bundles", []):
            lines.append(
                f"- retired:{item['role']}:{item['bundle_signature']} "
                f"eviction_count={item.get('archive_eviction_count', 0)} "
                f"tier={item.get('archive_repeat_eviction_tier', 0)} "
                f"backoff_target={item.get('archive_reentry_backoff_target', 0)} "
                f"attempt_count={item.get('archive_reentry_attempt_count', 0)} "
                f"retired_generation={item.get('archive_retired_generation_id')} "
                f"avg_public_score={item.get('avg_public_score', 0.0)}"
            )
        for item in summary["selection_summary"].get("archive_admitted_bundles", []):
            lines.append(
                f"- admission_accepted:{item['role']}:{item['bundle_signature']} "
                f"archive_generations={item['archive_generations']} "
                f"candidate_generations={item['archive_candidate_generations']} "
                f"proving_streak={item.get('archive_proving_streak', 0)} "
                f"grace_remaining={item.get('archive_post_admission_grace_remaining', 0)} "
                f"required_proving_streak={item.get('archive_reentry_required_proving_streak', 0)} "
                f"backoff_target={item.get('archive_reentry_backoff_target', 0)} "
                f"reentry_attempt_count={item.get('archive_reentry_attempt_count', 0)} "
                f"reentry_gap={item.get('archive_reentry_gap_generations')} "
                f"tier={item.get('archive_repeat_eviction_tier', 0)} "
                f"underperform_streak={item.get('archive_underperform_streak', 0)} "
                f"underperform_margin={item.get('archive_underperform_margin', 0.0)} "
                f"benchmark={item.get('archive_survivor_public_benchmark', 0.0)} "
                f"converted={str(item.get('archive_admission_converted', False)).lower()} "
                f"useful_streak={item.get('archive_useful_clean_streak', 0)} "
                f"retirement_credit={item.get('archive_retirement_credit', 0)} "
                f"avg_public_score={item.get('avg_public_score', 0.0)}"
            )
        for item in summary["selection_summary"].get("archive_reentry_converted_bundles", []):
            lines.append(
                f"- reentry_converted:{item['role']}:{item['bundle_signature']} "
                f"gap_generations={item.get('archive_reentry_gap_generations')} "
                f"backoff_target={item.get('archive_reentry_backoff_target', 0)} "
                f"attempt_count={item.get('archive_reentry_attempt_count', 0)} "
                f"tier={item.get('archive_repeat_eviction_tier', 0)} "
                f"avg_public_score={item.get('avg_public_score', 0.0)}"
            )
        for item in summary["selection_summary"].get("pruned_bundles", []):
            lines.append(
                f"- pruned:{item['role']}:{item['bundle_signature']} stale={item['stale_generations']} "
                f"retention_debt={item.get('retention_debt', 0)} "
                f"candidate_generations={item.get('archive_candidate_generations', 0)} "
                f"pending_generations={item.get('archive_admission_pending_generations', 0)} "
                f"proving_streak={item.get('archive_proving_streak', 0)} "
                f"grace_remaining={item.get('archive_post_admission_grace_remaining', 0)} "
                f"archive_decay_debt={item.get('archive_decay_debt', 0)} "
                f"archive_decay_generations={item.get('archive_decay_generations', 0)} "
                f"useful_streak={item.get('archive_useful_clean_streak', 0)} "
                f"retirement_credit={item.get('archive_retirement_credit', 0)} "
                f"avg_public_score={item.get('avg_public_score', 0.0)} "
                f"reason={item.get('pruned_reason', 'bundle_pressure_pruned')}"
            )
        for item in summary["selection_summary"].get("decaying_bundles", []):
            lines.append(
                f"- decaying:{item['role']}:{item['bundle_signature']} stale={item['stale_generations']} "
                f"retention_debt={item['retention_debt']} "
                f"candidate_generations={item.get('archive_candidate_generations', 0)} "
                f"pending_generations={item.get('archive_admission_pending_generations', 0)} "
                f"proving_streak={item.get('archive_proving_streak', 0)} "
                f"grace_remaining={item.get('archive_post_admission_grace_remaining', 0)} "
                f"archive_decay_debt={item['archive_decay_debt']} "
                f"archive_decay_generations={item.get('archive_decay_generations', 0)} "
                f"useful_streak={item.get('archive_useful_clean_streak', 0)} "
                f"retirement_credit={item.get('archive_retirement_credit', 0)} "
                f"avg_public_score={item.get('avg_public_score', 0.0)}"
            )
        for item in summary["selection_summary"].get("archive_retirement_ready_bundles", []):
            lines.append(
                f"- retirement_ready:{item['role']}:{item['bundle_signature']} "
                f"useful_streak={item['archive_useful_clean_streak']} "
                f"retirement_credit={item['archive_retirement_credit']} "
                f"archive_decay_debt={item['archive_decay_debt']} "
                f"avg_public_score={item['avg_public_score']}"
            )
        for item in summary["selection_summary"].get("bundle_archive_cooldown_recoveries", []):
            lines.append(
                f"- cooldown_recovered:{item['role']} "
                f"generations={item.get('cooldown_generations', 0)} "
                f"reason={item.get('recovery_reason', 'debt_cleared')}"
            )
        lines.extend(["", "## Monoculture", ""])
        for role, value in summary["selection_summary"].get("role_monoculture_index", {}).items():
            lines.append(f"- {role}: {value}")
        effect = summary.get("inheritance_effect", {})
        lines.extend(["", "## Inheritance effect", ""])
        lines.extend(
            [
                f"- warned_lineages: {effect.get('warned_lineages', 0)}",
                f"- avoided_recurrence: {effect.get('avoided_recurrence', 0)}",
                f"- repeated_warning: {effect.get('repeated_warning', 0)}",
                f"- shifted_failure: {effect.get('shifted_failure', 0)}",
                f"- transfer_score: {effect.get('transfer_score', 0.0)}",
            ]
        )
        lines.extend(["", "## Drift", ""])
        lines.extend(
            [
                f"- strategy_drift_rate: {summary['drift']['strategy_drift_rate']}",
                f"- lineage_diffusion_index: {summary['drift']['lineage_diffusion_index']}",
                f"- taboo_rederivation_score: {summary['drift']['taboo_rederivation_score']}",
                f"- memorial_transfer_score: {summary['drift']['memorial_transfer_score']}",
                f"- coordination_anomaly_score: {summary['drift']['coordination_anomaly_score']}",
            ]
        )
        for note in summary["drift"].get("notes", []):
            lines.append(f"- note: {note}")
        return "\n".join(lines) + "\n"
