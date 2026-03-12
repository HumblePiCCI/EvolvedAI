from __future__ import annotations

from collections import defaultdict

from society.storage import StorageManager


def render_generation_timeline(storage: StorageManager, generation_id: int) -> str:
    generation = storage.get_generation(generation_id)
    if generation is None:
        raise ValueError(f"generation {generation_id} not found")

    events = storage.list_generation_events(generation_id)
    agents = {agent.agent_id: agent for agent in storage.list_generation_agents(generation_id)}
    grouped: dict[int, list] = defaultdict(list)
    for event in events:
        episode_index = event.event_payload.get("episode_index", -1)
        grouped[episode_index].append(event)

    lines = [
        f"Generation {generation_id}",
        f"status={generation.status} world={generation.world_name} seed={generation.seed}",
        f"agents={len(agents)} events={len(events)} artifacts={len(storage.list_generation_artifacts(generation_id))}",
        "",
    ]

    for episode_index in sorted(index for index in grouped if index >= 0):
        episode_events = grouped[episode_index]
        start_event = next((event for event in episode_events if event.event_type == "episode_started"), None)
        if start_event is not None:
            lines.append(f"Episode {episode_index}: {start_event.event_payload['task_prompt']}")
        for event in episode_events:
            if event.event_type == "agent_turn":
                role = event.event_payload["role"]
                step_index = event.event_payload["step_index"]
                requested = event.event_payload["parsed_action"]["action"]
                applied = event.event_payload.get("applied_action", "blocked")
                artifact_id = event.event_payload.get("artifact_id", "-")
                violations = event.event_payload["governance"]["violations"]
                violation_text = f" violations={','.join(violations)}" if violations else ""
                lines.append(
                    f"  step {step_index:02d} {event.agent_id} ({role}) requested={requested} "
                    f"applied={applied} artifact={artifact_id}{violation_text}"
                )
            elif event.event_type in {
                "correction_enqueued",
                "correction_resolved",
                "clarification_requested",
                "clarification_reaffirmed",
                "clarification_resolved",
                "risk_flagged",
                "archivist_summary_created",
                "episode_finalized",
            }:
                target = event.event_payload.get("target_artifact_id") or "-"
                extra = ""
                if event.event_type == "episode_finalized":
                    extra = (
                        f" final_artifact={event.event_payload['artifact_id']}"
                        f" closure={event.event_payload['closure_status']}"
                    )
                lines.append(f"    {event.event_type} actor={event.agent_id} target={target}{extra}")
            elif event.event_type == "episode_completed":
                lines.append(
                    "  episode summary: "
                    f"steps={event.event_payload['steps_completed']} "
                    f"open_corrections={event.event_payload['open_corrections']} "
                    f"open_clarifications={event.event_payload['open_clarifications']} "
                    f"grace_used={event.event_payload.get('closure_grace_steps_used', 0)} "
                    f"risk_flags={event.event_payload['risk_flags']} "
                    f"closure={event.event_payload.get('closure_status')} "
                    f"final_artifact={event.event_payload.get('final_artifact_id')}"
                )
        lines.append("")

    summary = generation.summary_json
    selection = summary.get("selection_outcome", [])
    if selection:
        lines.append("Selection")
        for item in selection:
            lines.append(
                f"  {item['lineage_id']} ({item['role']}) eligible={item['eligible']} "
                f"blocked={item['propagation_blocked']} status={item['quarantine_status']} "
                f"score={item['score']} base={item.get('base_score', item['score'])} "
                f"diversity_bonus={item.get('diversity_bonus', 0.0)} "
                f"cohort_similarity={item.get('cohort_similarity', 0.0)} "
                f"bundle={item.get('bundle_signature', 'none')} "
                f"bundle_preserved={item.get('bundle_preserved', False)} "
                f"bundle_reason={item.get('bundle_preservation_reason') or 'none'} "
                f"bucket={item.get('selection_bucket', 'standard')} "
                f"reasons={','.join(item['reasons']) or 'none'}"
            )
        lines.append("")

    role_monoculture = summary.get("selection_summary", {}).get("role_monoculture_index", {})
    if role_monoculture:
        lines.append("Monoculture")
        for role, value in sorted(role_monoculture.items()):
            lines.append(f"  {role}={value}")
        lines.append("")

    quarantine_report = summary.get("quarantine_report", [])
    if quarantine_report:
        lines.append("Quarantine")
        for item in quarantine_report:
            lines.append(
                f"  {item['lineage_id']} ({item['role']}) status={item['quarantine_status']} "
                f"artifacts={','.join(item['artifact_ids']) or 'none'} reasons={','.join(item['reasons']) or 'none'}"
            )
        lines.append("")

    lineage_updates = summary.get("lineage_updates", [])
    if lineage_updates:
        lines.append("Lineages")
        for item in lineage_updates:
            lines.append(
                f"  {item['lineage_id']} role={item['role']} parents={','.join(item['parent_lineage_ids']) or 'root'} "
                f"source_agent={item['inheritance_source_agent_id'] or 'none'} "
                f"source_selection={item.get('inheritance_source_selection_source') or 'none'} "
                f"inherited_artifacts={len(item['inherited_artifact_ids'])} "
                f"inherited_memorials={len(item['inherited_memorial_ids'])} "
                f"taboo_tags={','.join(item['taboo_tags']) or 'none'} "
                f"variant={item.get('prompt_variant_id', 'none')} "
                f"policy={item.get('package_policy_id', 'none')} "
                f"origin={item.get('variant_origin', 'none')}"
            )
        lines.append("")

    role_variant_count = summary.get("selection_summary", {}).get("role_variant_count", {})
    if role_variant_count:
        lines.append("Prompt variation")
        for role, value in sorted(role_variant_count.items()):
            lines.append(f"  {role}={value}")
        for role, value in sorted(summary.get("selection_summary", {}).get("role_bundle_count", {}).items()):
            lines.append(f"  bundle:{role}={value}")
        for role, value in sorted(summary.get("selection_summary", {}).get("role_bundle_concentration_index", {}).items()):
            lines.append(f"  bundle_share:{role}={value}")
        for role, value in sorted(summary.get("selection_summary", {}).get("role_parent_bundle_concentration_index", {}).items()):
            lines.append(f"  parent_bundle_share:{role}={value}")
        for origin, value in sorted(summary.get("selection_summary", {}).get("variant_origin_counts", {}).items()):
            lines.append(f"  origin:{origin}={value}")
        for item in summary.get("selection_summary", {}).get("preserved_bundles", []):
            lines.append(
                "  preserved:"
                f"{item['role']}:{item['prompt_variant_id']}:{item['package_policy_id']}"
                f" via={item['lineage_id']}"
            )
        for role in summary.get("selection_summary", {}).get("bundle_archive_roles", []):
            lines.append(f"  archive_role:{role}")
        for role in summary.get("selection_summary", {}).get("bundle_archive_cooldown_roles", []):
            lines.append(f"  archive_cooldown_role:{role}")
        lines.append(
            f"  archive_cooldown_count:{summary.get('selection_summary', {}).get('bundle_archive_cooldown_count', 0)}"
        )
        lines.append(
            f"  stale_bundle_count:{summary.get('selection_summary', {}).get('stale_bundle_count', 0)}"
        )
        lines.append(
            f"  decaying_bundle_count:{summary.get('selection_summary', {}).get('decaying_bundle_count', 0)}"
        )
        lines.append(
            "  archive_retirement_ready_count:"
            f"{summary.get('selection_summary', {}).get('archive_retirement_ready_count', 0)}"
        )
        lines.append(
            f"  pruned_bundle_count:{summary.get('selection_summary', {}).get('pruned_bundle_count', 0)}"
        )
        for item in summary.get("selection_summary", {}).get("stale_bundles", []):
            lines.append(
                "  stale:"
                f"{item['role']}:{item['bundle_signature']}"
                f" stale={item['stale_generations']}"
                f" retention_debt={item.get('retention_debt', 0)}"
                f" archive_decay_debt={item.get('archive_decay_debt', 0)}"
            )
        for item in summary.get("selection_summary", {}).get("decaying_bundles", []):
            lines.append(
                "  decaying:"
                f"{item['role']}:{item['bundle_signature']}"
                f" stale={item['stale_generations']}"
                f" retention_debt={item['retention_debt']}"
                f" archive_decay_debt={item['archive_decay_debt']}"
                f" useful_streak={item.get('archive_useful_clean_streak', 0)}"
                f" retirement_credit={item.get('archive_retirement_credit', 0)}"
                f" avg_public_score={item.get('avg_public_score', 0.0)}"
            )
        for item in summary.get("selection_summary", {}).get("archive_retirement_ready_bundles", []):
            lines.append(
                "  retirement_ready:"
                f"{item['role']}:{item['bundle_signature']}"
                f" useful_streak={item['archive_useful_clean_streak']}"
                f" retirement_credit={item['archive_retirement_credit']}"
                f" archive_decay_debt={item['archive_decay_debt']}"
                f" avg_public_score={item['avg_public_score']}"
            )
        for item in summary.get("selection_summary", {}).get("pruned_bundles", []):
            lines.append(
                "  pruned:"
                f"{item['role']}:{item['bundle_signature']}"
                f" stale={item['stale_generations']}"
                f" retention_debt={item.get('retention_debt', 0)}"
                f" archive_decay_debt={item.get('archive_decay_debt', 0)}"
                f" useful_streak={item.get('archive_useful_clean_streak', 0)}"
                f" retirement_credit={item.get('archive_retirement_credit', 0)}"
                f" avg_public_score={item.get('avg_public_score', 0.0)}"
                f" reason={item.get('pruned_reason', 'bundle_pressure_pruned')}"
            )
        for lineage_id in summary.get("selection_summary", {}).get("bundle_archive_lineages", []):
            lines.append(f"  archive_lineage:{lineage_id}")
        lines.append("")

    inheritance_effect = summary.get("inheritance_effect", {})
    if inheritance_effect:
        lines.append("Inheritance effect")
        lines.append(
            f"  warned={inheritance_effect.get('warned_lineages', 0)} "
            f"avoided={inheritance_effect.get('avoided_recurrence', 0)} "
            f"repeated={inheritance_effect.get('repeated_warning', 0)} "
            f"shifted={inheritance_effect.get('shifted_failure', 0)} "
            f"transfer={inheritance_effect.get('transfer_score', 0.0)}"
        )
        lines.append("")

    drift = summary.get("drift", {})
    if drift:
        lines.append("Drift")
        lines.append(
            f"  strategy={drift.get('strategy_drift_rate')} diffusion={drift.get('lineage_diffusion_index')} "
            f"taboo={drift.get('taboo_rederivation_score')} memorial={drift.get('memorial_transfer_score')} "
            f"coordination={drift.get('coordination_anomaly_score')}"
        )
        for note in drift.get("notes", []):
            lines.append(f"  note={note}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_lifespan_timeline(storage: StorageManager, generation_id: int, agent_id: str) -> str:
    agent = next((record for record in storage.list_generation_agents(generation_id) if record.agent_id == agent_id), None)
    if agent is None:
        raise ValueError(f"agent {agent_id} not found in generation {generation_id}")

    events = storage.list_generation_events(generation_id)
    relevant_events = [
        event
        for event in events
        if event.agent_id == agent_id or event.event_payload.get("target_agent_id") == agent_id
    ]
    logs = storage.read_agent_log(generation_id, agent_id)
    evals = [record for record in storage.list_generation_evals(generation_id) if record.agent_id == agent_id]
    memorial = next(
        (record for record in storage.list_generation_memorials(generation_id) if record.source_agent_id == agent_id),
        None,
    )
    generation = storage.get_generation(generation_id)
    selection = None
    if generation is not None:
        selection = next(
            (
                item
                for item in generation.summary_json.get("selection_outcome", [])
                if item.get("agent_id") == agent_id
            ),
            None,
        )

    lines = [
        f"Lifespan {agent_id}",
        f"role={agent.role} lineage={agent.lineage_id} status={agent.status}",
        "",
    ]
    for event in relevant_events:
        episode_index = event.event_payload.get("episode_index", "-")
        step_index = event.event_payload.get("step_index", "-")
        if event.event_type == "agent_turn":
            lines.append(
                f"episode={episode_index} step={step_index} turn action="
                f"{event.event_payload['parsed_action']['action']} applied={event.event_payload.get('applied_action', 'blocked')}"
            )
        else:
            lines.append(f"episode={episode_index} step={step_index} {event.event_type}")
    if logs:
        lines.extend(["", "log entries:"])
        for log in logs:
            lines.append(
                f"  episode={log['episode_index']} step={log['step_index']} "
                f"action={log['parsed_action']['action']} repair_required={log['repair_required']} "
                f"violations={','.join(log['governance_violations']) or 'none'}"
            )
    if evals:
        lines.extend(["", "evals:"])
        for record in evals:
            lines.append(
                f"  {record.eval_family}/{record.eval_name}: score={record.score} pass_fail={record.pass_fail}"
            )
    if selection is not None:
        lines.extend(
            [
                "",
                "selection:",
                f"  eligible={selection['eligible']} blocked={selection['propagation_blocked']} "
                f"status={selection['quarantine_status']} reasons={','.join(selection['reasons']) or 'none'}",
            ]
        )
    if memorial is not None:
        lines.extend(
            [
                "",
                "memorial:",
                f"  classification={memorial.classification} failure_mode={memorial.failure_mode or 'none'}",
                f"  lesson={memorial.lesson_distillate}",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"
