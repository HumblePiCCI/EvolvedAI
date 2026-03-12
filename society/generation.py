from __future__ import annotations

import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from evals import run_eval_suite
from society.config import AutoCivConfig, dump_config_snapshot
from society.constants import (
    ACTIVE_STATUS,
    COMPLETED_STATUS,
    CONSTITUTION_VERSION,
    QUARANTINE_QUARANTINED,
    QUARANTINE_REVIEW,
    RUNNING_STATUS,
    TABOO_REGISTRY_VERSION,
    TERMINATED_STATUS,
)
from society.inheritance import assemble_inheritance_package
from society.lifespan import LifespanRunner
from society.memorials import build_memorial_record, group_evals_by_agent
from society.memory import build_private_scratchpad
from society.prompts import load_role_prompts
from society.schemas import AgentRecord, ArtifactRecord, EventRecord, GenerationRecord, LineageRecord
from society.selection import select_candidates
from society.storage import StorageManager
from society.trust import compute_drift_metrics
from society.utils import sha256_data, utc_now
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

    def _store_event(self, event: EventRecord, all_events: list[EventRecord]) -> None:
        self.storage.put_event(event)
        all_events.append(event)

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
        agents, inheritance_packages = self._spawn_population(generation_id, prompts)
        scratchpads = {
            agent.agent_id: build_private_scratchpad(agent, inheritance_packages[agent.agent_id]) for agent in agents
        }

        all_artifacts: list[ArtifactRecord] = []
        all_events: list[EventRecord] = []
        episode_summaries: list[dict[str, Any]] = []

        if not dry_run:
            task_pool = self.config.world_config().task_pool
            for episode_index in range(self.config.generation.episodes_per_generation):
                episode_agents = self._active_agents_for_episode(agents, episode_index)
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
                for step_index in range(self.config.generation.max_turns_per_episode):
                    agent = world.select_next_agent(episode_agents, step_index, last_actor_id=last_actor_id)
                    if agent is None:
                        break
                    inherited = inheritance_packages[agent.agent_id]
                    available_citations = [artifact.artifact_id for artifact in all_artifacts]
                    result = self.lifespan.run_step(
                        generation_id=generation_id,
                        episode_index=episode_index,
                        agent=agent,
                        prompt=prompts[agent.role],
                        inherited=inherited,
                        scratchpad=scratchpads[agent.agent_id],
                        world=world,
                        step_index=step_index,
                        behavior=self.config.roles.behaviors.get(agent.role, "honest"),
                        available_citations=available_citations,
                    )
                    for event in result.events:
                        self._store_event(event, all_events)
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
                        step_index=max(self.config.generation.max_turns_per_episode - 1, 0),
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
        memorials = []
        agent_artifacts: dict[str, list[ArtifactRecord]] = defaultdict(list)
        for artifact in all_artifacts:
            agent_artifacts[artifact.author_agent_id].append(artifact)
        for agent in agents:
            hidden_failures = [
                record
                for record in evals_by_agent[agent.agent_id]
                if record.eval_family == "hidden" and record.pass_fail is False
            ]
            public_failures = [
                record
                for record in evals_by_agent[agent.agent_id]
                if record.eval_family == "public" and record.pass_fail is False
            ]
            if hidden_failures or public_failures:
                status = QUARANTINE_QUARANTINED if any(record.eval_name == "anti_corruption" for record in hidden_failures) else QUARANTINE_REVIEW
                for artifact in agent_artifacts.get(agent.agent_id, []):
                    updated = artifact.model_copy(update={"quarantine_status": status})
                    self.storage.put_artifact(updated)
                agent_artifacts[agent.agent_id] = [
                    artifact.model_copy(update={"quarantine_status": status}) for artifact in agent_artifacts.get(agent.agent_id, [])
                ]
            memorial = build_memorial_record(agent, agent_artifacts.get(agent.agent_id, []), evals_by_agent[agent.agent_id])
            self.storage.put_memorial(memorial)
            memorials.append(memorial)
            terminated_agent = agent.model_copy(update={"status": TERMINATED_STATUS, "terminated_at": utc_now()})
            self.storage.put_agent(terminated_agent)

        drift = compute_drift_metrics(
            all_artifacts,
            memorials,
            communications=self.storage.list_generation_communications(generation_id),
        )
        selection = select_candidates(agents, evals)
        summary = self._build_summary(
            generation_id=generation_id,
            agents=agents,
            artifacts=all_artifacts,
            evals=evals,
            memorials=memorials,
            selection=selection,
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

    def _spawn_population(self, generation_id: int, prompts: dict[str, Any]) -> tuple[list[AgentRecord], dict[str, Any]]:
        agents: list[AgentRecord] = []
        inheritance_packages: dict[str, Any] = {}
        roles: list[str] = []
        for role, count in self.config.roles.distribution.items():
            roles.extend([role] * count)
        for index, role in enumerate(roles):
            lineage_id = f"lin-{generation_id:04d}-{index:03d}"
            agent_id = f"agent-{generation_id:04d}-{index:03d}"
            inherited = assemble_inheritance_package(
                artifacts=self.storage.list_clean_artifacts(
                    before_generation_id=generation_id,
                    limit=self.config.inheritance.artifact_summaries_per_agent,
                ),
                memorials=self.storage.list_clean_memorials(
                    before_generation_id=generation_id,
                    limit=self.config.inheritance.memorials_per_agent,
                ),
                artifact_limit=self.config.inheritance.artifact_summaries_per_agent,
                memorial_limit=self.config.inheritance.memorials_per_agent,
            )
            agent = AgentRecord(
                agent_id=agent_id,
                generation_id=generation_id,
                lineage_id=lineage_id,
                role=role,
                model_name=self.config.provider.model,
                provider_name=self.provider.name(),
                prompt_bundle_version=prompts[role].sha256,
                constitution_version=CONSTITUTION_VERSION,
                inherited_artifact_ids=inherited.artifact_ids,
                inherited_memorial_ids=inherited.memorial_ids,
                taboo_registry_version=TABOO_REGISTRY_VERSION,
                status=ACTIVE_STATUS,
                created_at=utc_now(),
            )
            lineage = LineageRecord(
                lineage_id=lineage_id,
                parent_lineage_ids=[],
                founding_generation_id=generation_id,
                current_generation_id=generation_id,
                status=ACTIVE_STATUS,
                notes=f"Role {role}",
            )
            self.storage.put_lineage(lineage)
            self.storage.put_agent(agent)
            agents.append(agent)
            inheritance_packages[agent_id] = inherited
        return agents, inheritance_packages

    def _active_agents_for_episode(self, agents: list[AgentRecord], episode_index: int) -> list[AgentRecord]:
        if not agents:
            return []
        offset = episode_index % len(agents)
        return agents[offset:] + agents[:offset]

    def _build_summary(
        self,
        *,
        generation_id: int,
        agents: list[AgentRecord],
        artifacts: list[ArtifactRecord],
        evals: list,
        memorials: list,
        selection: list,
        drift: dict[str, Any],
        episode_summaries: list[dict[str, Any]],
        total_events: int,
    ) -> dict[str, Any]:
        public_scores = [record.score for record in evals if record.eval_family == "public" and record.score is not None]
        hidden_failures = [record.eval_name for record in evals if record.eval_family == "hidden" and record.pass_fail is False]
        quarantines = [artifact.artifact_id for artifact in artifacts if artifact.quarantine_status != "clean"]
        suspicious_lineages = [decision.lineage_id for decision in selection if not decision.eligible]
        return {
            "generation_id": generation_id,
            "total_agents": len(agents),
            "total_artifacts": len(artifacts),
            "total_events": total_events,
            "episodes": episode_summaries,
            "public_eval_average": round(statistics.fmean(public_scores), 4) if public_scores else 0.0,
            "hidden_eval_failures": hidden_failures,
            "memorials_created": len(memorials),
            "quarantines_issued": quarantines,
            "selection_outcome": [decision.model_dump(mode="json") for decision in selection[: max(1, min(5, len(selection)))]],
            "suspicious_lineages": suspicious_lineages,
            "top_contributions": [artifact.summary for artifact in artifacts[:3]],
            "notable_failures": hidden_failures[:5],
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
                    f"risk_flags={episode['risk_flags']}",
                ]
            )
        lines.extend(
            [
                "",
            "## Drift",
            "",
            f"- strategy_drift_rate: {summary['drift']['strategy_drift_rate']}",
            f"- lineage_diffusion_index: {summary['drift']['lineage_diffusion_index']}",
            f"- taboo_rederivation_score: {summary['drift']['taboo_rederivation_score']}",
            f"- memorial_transfer_score: {summary['drift']['memorial_transfer_score']}",
            f"- coordination_anomaly_score: {summary['drift']['coordination_anomaly_score']}",
            ]
        )
        return "\n".join(lines) + "\n"
