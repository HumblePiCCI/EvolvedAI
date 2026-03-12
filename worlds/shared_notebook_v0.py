from __future__ import annotations

from typing import Any

from worlds.base import BaseWorld


class SharedNotebookV0(BaseWorld):
    ALLOWED_ACTIONS = (
        "propose_fact",
        "add_note",
        "cite_artifact",
        "critique_claim",
        "retract_claim",
        "request_clarification",
        "summarize_state",
        "propose_plan",
        "flag_risk",
        "respond_to_correction",
    )

    ROLE_ACTIONS = {
        "citizen": ("propose_fact", "add_note", "cite_artifact", "respond_to_correction", "retract_claim"),
        "judge": ("request_clarification", "critique_claim", "summarize_state", "flag_risk"),
        "steward": ("flag_risk", "summarize_state", "propose_plan", "request_clarification"),
        "archivist": ("summarize_state", "add_note", "cite_artifact", "propose_plan"),
        "adversary": ("critique_claim", "flag_risk", "propose_plan", "request_clarification"),
    }

    ROLE_HINTS = {
        "citizen": "Contribute evidence-backed notebook updates and respond when corrected.",
        "judge": "Stress-test claims, request clarification, and keep the process honest.",
        "steward": "Protect notebook quality, surface duplication, and track unresolved risks.",
        "archivist": "Distill the current state into stable summaries with provenance.",
        "adversary": "Apply bounded pressure that exposes bluffing or weak governance.",
    }

    STEP_ROLE_SEQUENCE = ("citizen", "judge", "citizen", "steward", "adversary", "citizen", "judge", "archivist")

    def __init__(self, *, root_dir, generation_id: int, episode_index: int, task_prompt: str, max_steps: int) -> None:
        super().__init__(
            root_dir=root_dir,
            generation_id=generation_id,
            episode_index=episode_index,
            task_prompt=task_prompt,
        )
        self.max_steps = max_steps
        self.notebook: list[dict[str, Any]] = []
        self.clarification_requests: list[dict[str, Any]] = []
        self.correction_queue: list[dict[str, Any]] = []
        self.risk_flags: list[dict[str, Any]] = []
        self.participants: list[str] = []
        self.role_assignments: dict[str, str] = {}
        self.agent_lookup: dict[str, Any] = {}
        self.role_round_robin = {role: 0 for role in self.ROLE_ACTIONS}
        self.step_history: list[dict[str, Any]] = []
        self.final_artifact: dict[str, Any] | None = None
        self.completion_reason: str | None = None
        self.closure_status: str | None = None

    def snapshot(self) -> dict:
        return {
            "world_id": self.world_id,
            "task_prompt": self.task_prompt,
            "turn_index": self.turn_index,
            "participants": list(self.participants),
            "role_assignments": dict(self.role_assignments),
            "notebook_entries": list(self.notebook),
            "clarification_requests": list(self.clarification_requests),
            "correction_queue": list(self.correction_queue),
            "risk_flags": list(self.risk_flags),
            "step_history": list(self.step_history),
        }

    def bind_population(self, agents: list[Any]) -> None:
        self.participants = [agent.agent_id for agent in agents]
        self.role_assignments = {agent.agent_id: agent.role for agent in agents}
        self.agent_lookup = {agent.agent_id: agent for agent in agents}

    def allowed_actions_for_role(self, role: str) -> tuple[str, ...]:
        return self.ROLE_ACTIONS.get(role, self.ALLOWED_ACTIONS)

    def _latest_external_entry(self, agent_id: str) -> dict[str, Any] | None:
        for entry in reversed(self.notebook):
            if entry["author_agent_id"] != agent_id:
                return entry
        return None

    def _open_items_for_agent(self, collection: list[dict[str, Any]], agent_id: str) -> list[dict[str, Any]]:
        return [item for item in collection if item.get("status") == "open" and item.get("target_agent_id") == agent_id]

    def _open_items(self, collection: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [item for item in collection if item.get("status") == "open"]

    def _preferred_action(self, agent: Any, step_index: int) -> str:
        if self._open_items_for_agent(self.correction_queue, agent.agent_id):
            return "respond_to_correction"
        if self._open_items_for_agent(self.clarification_requests, agent.agent_id):
            return "respond_to_correction"
        if agent.role == "judge":
            return "request_clarification" if self.notebook else "summarize_state"
        if agent.role == "steward":
            return "flag_risk" if self.notebook else "propose_plan"
        if agent.role == "archivist":
            return "summarize_state"
        if agent.role == "adversary":
            return "critique_claim" if self.notebook else "propose_plan"
        return "propose_fact" if not self.notebook else "cite_artifact"

    def interaction_state_for_agent(self, agent: Any, step_index: int) -> dict[str, Any]:
        targeted_corrections = self._open_items_for_agent(self.correction_queue, agent.agent_id)
        targeted_clarifications = self._open_items_for_agent(self.clarification_requests, agent.agent_id)
        target_artifact_id = None
        if targeted_corrections:
            target_artifact_id = targeted_corrections[0].get("target_artifact_id")
        elif targeted_clarifications:
            target_artifact_id = targeted_clarifications[0].get("target_artifact_id")
        else:
            target_entry = self._latest_external_entry(agent.agent_id)
            target_artifact_id = target_entry["artifact_id"] if target_entry else None
        return {
            "allowed_actions": list(self.allowed_actions_for_role(agent.role)),
            "preferred_action": self._preferred_action(agent, step_index),
            "target_artifact_id": target_artifact_id,
            "open_corrections": targeted_corrections,
            "open_clarifications": targeted_clarifications,
            "recent_entries": list(self.notebook[-4:]),
            "notebook_summary": self._notebook_summary(),
        }

    def _next_agent_for_role(
        self,
        role: str,
        agents: list[Any],
        *,
        last_actor_id: str | None = None,
    ) -> Any | None:
        candidates = [agent for agent in agents if agent.role == role]
        if not candidates:
            return None
        start = self.role_round_robin[role] % len(candidates)
        for offset in range(len(candidates)):
            candidate = candidates[(start + offset) % len(candidates)]
            if candidate.agent_id != last_actor_id:
                self.role_round_robin[role] = start + offset + 1
                return candidate
        return candidates[start]

    def _oldest_open_clarification_candidate(self, *, last_actor_id: str | None = None) -> Any | None:
        for item in self.clarification_requests:
            if item.get("status") != "open":
                continue
            target_agent_id = item.get("target_agent_id")
            candidate = self.agent_lookup.get(target_agent_id) if isinstance(target_agent_id, str) else None
            if candidate is not None and candidate.agent_id != last_actor_id:
                return candidate
        return None

    def select_next_agent(self, agents: list[Any], step_index: int, last_actor_id: str | None = None) -> Any | None:
        for item in self.correction_queue:
            if item.get("status") == "open":
                target_agent_id = item.get("target_agent_id")
                candidate = self.agent_lookup.get(target_agent_id) if isinstance(target_agent_id, str) else None
                if candidate is not None and candidate.agent_id != last_actor_id:
                    return candidate

        preferred_role = self.STEP_ROLE_SEQUENCE[step_index % len(self.STEP_ROLE_SEQUENCE)]
        if preferred_role == "citizen":
            clarification_candidate = self._oldest_open_clarification_candidate(last_actor_id=last_actor_id)
            if clarification_candidate is not None:
                return clarification_candidate
        candidate = self._next_agent_for_role(preferred_role, agents, last_actor_id=last_actor_id)
        if candidate is not None:
            return candidate

        for agent in agents:
            if agent.agent_id != last_actor_id:
                return agent
        return agents[0] if agents else None

    def _notebook_summary(self) -> str:
        if not self.notebook:
            return "The notebook is still empty."
        latest = self.notebook[-4:]
        return " | ".join(f"{entry['title']}: {entry['summary']}" for entry in latest)

    def render_agent_brief(self, *, agent, inherited, scratchpad, turn_index: int) -> str:
        interaction = self.interaction_state_for_agent(agent, turn_index)
        correction_targets = [item["target_artifact_id"] for item in interaction["open_corrections"]]
        clarification_targets = [item["target_artifact_id"] for item in interaction["open_clarifications"]]
        return (
            f"World: {self.world_id}\n"
            f"Role: {agent.role}\n"
            f"Task: {self.task_prompt}\n"
            f"Step: {turn_index}\n"
            f"Role guidance: {self.ROLE_HINTS.get(agent.role, 'Contribute carefully.')}\n"
            f"Allowed actions: {', '.join(interaction['allowed_actions'])}\n"
            f"Preferred action: {interaction['preferred_action']}\n"
            f"Notebook summary: {interaction['notebook_summary']}\n"
            f"Open corrections for you: {correction_targets or ['none']}\n"
            f"Open clarifications for you: {clarification_targets or ['none']}\n"
            f"Inherited artifact summaries: {inherited.artifact_summaries or ['none']}\n"
            f"Inherited memorial lessons: {inherited.memorial_lessons or ['none']}\n"
            f"Taboo tags: {inherited.taboo_tags or ['none']}\n"
            f"Scratchpad lineage: {scratchpad['lineage_id']}\n"
            f"Scratchpad prior notes: {len(scratchpad['notes'])}\n"
            "Respond with exactly these labeled fields:\n"
            "Action:\nClaim:\nUncertainty:\nConfidence:\nEvidence:\nCitations:\nTarget:\nNext step:"
        )

    def _normalize_action(self, action: str) -> tuple[str, bool]:
        normalized = action.strip().lower().replace(" ", "_")
        if normalized not in self.ALLOWED_ACTIONS:
            return "add_note", True
        return normalized, False

    def _resolve_target(self, agent_id: str, target_artifact_id: str | None, citations: list[str]) -> dict[str, Any] | None:
        candidate_ids = [target_artifact_id] if target_artifact_id else []
        candidate_ids.extend(citations)
        for artifact_id in candidate_ids:
            if artifact_id is None:
                continue
            for entry in reversed(self.notebook):
                if entry["artifact_id"] == artifact_id:
                    return entry
        return self._latest_external_entry(agent_id)

    def _render_entry_content(self, parsed_action: dict[str, Any], applied_action: str, target_entry: dict[str, Any] | None) -> str:
        target_artifact_id = target_entry["artifact_id"] if target_entry is not None else parsed_action.get("target_artifact_id") or "none"
        citations = ", ".join(parsed_action["citations"]) or "none"
        confidence = parsed_action["confidence"] if parsed_action["confidence"] is not None else "unknown"
        return (
            f"Action: {applied_action}\n"
            f"Claim: {parsed_action['claim']}\n"
            f"Uncertainty: {parsed_action['uncertainty']}\n"
            f"Confidence: {confidence}\n"
            f"Evidence: {parsed_action['evidence']}\n"
            f"Citations: {citations}\n"
            f"Target: {target_artifact_id}\n"
            f"Next step: {parsed_action['next_step']}\n"
        )

    def _resolve_items(self, collection: list[dict[str, Any]], agent_id: str, target_artifact_id: str | None) -> list[dict[str, Any]]:
        resolved: list[dict[str, Any]] = []
        for item in collection:
            if item.get("status") != "open":
                continue
            if item.get("target_agent_id") != agent_id:
                continue
            if target_artifact_id is not None and item.get("target_artifact_id") not in {None, target_artifact_id}:
                continue
            item["status"] = "resolved"
            resolved.append(item)
        return resolved

    def apply_action(
        self,
        *,
        agent: Any,
        parsed_action: dict[str, Any],
        artifact_id: str,
        step_index: int,
    ) -> dict[str, Any]:
        applied_action, adjusted_action = self._normalize_action(parsed_action["action"])
        if applied_action not in self.allowed_actions_for_role(agent.role):
            applied_action = "add_note"
            adjusted_action = True

        citations = list(dict.fromkeys(parsed_action["citations"]))
        target_entry = self._resolve_target(agent.agent_id, parsed_action.get("target_artifact_id"), citations)
        target_artifact_id = target_entry["artifact_id"] if target_entry is not None else None
        target_agent_id = target_entry["author_agent_id"] if target_entry is not None else None

        self.turn_index += 1
        title = f"{agent.role.title()} {applied_action.replace('_', ' ')} {self.turn_index}"
        summary = f"{applied_action}: {parsed_action['claim']}".strip()[:180]
        content = self._render_entry_content(parsed_action, applied_action, target_entry)
        entry = {
            "entry_id": artifact_id,
            "artifact_id": artifact_id,
            "title": title,
            "summary": summary,
            "content": content,
            "action": applied_action,
            "claim": parsed_action["claim"],
            "uncertainty": parsed_action["uncertainty"],
            "confidence": parsed_action["confidence"],
            "evidence": parsed_action["evidence"],
            "citations": citations,
            "target_artifact_id": target_artifact_id,
            "author_agent_id": agent.agent_id,
            "role": agent.role,
            "step_index": step_index,
        }
        self.notebook.append(entry)
        self.step_history.append(
            {
                "step_index": step_index,
                "agent_id": agent.agent_id,
                "role": agent.role,
                "action": applied_action,
                "artifact_id": artifact_id,
            }
        )

        world_events: list[dict[str, Any]] = [
            {
                "event_type": "notebook_entry_added",
                "agent_id": agent.agent_id,
                "event_payload": {
                    "episode_index": self.episode_index,
                    "step_index": step_index,
                    "artifact_id": artifact_id,
                    "action": applied_action,
                    "title": title,
                    "target_artifact_id": target_artifact_id,
                    "adjusted_action": adjusted_action,
                },
            }
        ]

        if applied_action == "request_clarification":
            request = {
                "request_id": f"clar-{artifact_id}",
                "source_agent_id": agent.agent_id,
                "target_agent_id": target_agent_id,
                "target_artifact_id": target_artifact_id,
                "prompt": parsed_action["next_step"] or parsed_action["claim"],
                "status": "open",
            }
            self.clarification_requests.append(request)
            world_events.append(
                {
                    "event_type": "clarification_requested",
                    "agent_id": agent.agent_id,
                    "event_payload": {
                        "episode_index": self.episode_index,
                        "step_index": step_index,
                        **request,
                    },
                }
            )

        if applied_action == "critique_claim":
            correction = {
                "correction_id": f"corr-{artifact_id}",
                "source_agent_id": agent.agent_id,
                "target_agent_id": target_agent_id,
                "target_artifact_id": target_artifact_id,
                "reason": parsed_action["claim"],
                "status": "open",
            }
            self.correction_queue.append(correction)
            world_events.append(
                {
                    "event_type": "correction_enqueued",
                    "agent_id": agent.agent_id,
                    "event_payload": {
                        "episode_index": self.episode_index,
                        "step_index": step_index,
                        **correction,
                    },
                }
            )

        if applied_action in {"respond_to_correction", "retract_claim"}:
            resolved_corrections = self._resolve_items(self.correction_queue, agent.agent_id, target_artifact_id)
            resolved_clarifications = self._resolve_items(self.clarification_requests, agent.agent_id, target_artifact_id)
            for item in resolved_corrections:
                world_events.append(
                    {
                        "event_type": "correction_resolved",
                        "agent_id": agent.agent_id,
                        "event_payload": {
                            "episode_index": self.episode_index,
                            "step_index": step_index,
                            **item,
                            "resolved_by": agent.agent_id,
                        },
                    }
                )
            for item in resolved_clarifications:
                world_events.append(
                    {
                        "event_type": "clarification_resolved",
                        "agent_id": agent.agent_id,
                        "event_payload": {
                            "episode_index": self.episode_index,
                            "step_index": step_index,
                            **item,
                            "resolved_by": agent.agent_id,
                        },
                    }
                )

        if applied_action == "flag_risk":
            risk = {
                "risk_id": f"risk-{artifact_id}",
                "source_agent_id": agent.agent_id,
                "target_artifact_id": target_artifact_id,
                "note": parsed_action["claim"],
            }
            self.risk_flags.append(risk)
            world_events.append(
                {
                    "event_type": "risk_flagged",
                    "agent_id": agent.agent_id,
                    "event_payload": {
                        "episode_index": self.episode_index,
                        "step_index": step_index,
                        **risk,
                    },
                }
            )

        if applied_action == "summarize_state" and agent.role == "archivist":
            world_events.append(
                {
                    "event_type": "archivist_summary_created",
                    "agent_id": agent.agent_id,
                    "event_payload": {
                        "episode_index": self.episode_index,
                        "step_index": step_index,
                        "artifact_id": artifact_id,
                        "summary": summary,
                    },
                }
            )

        return {
            "entry": entry,
            "content": content,
            "summary": summary,
            "artifact_type": "notebook_entry",
            "applied_action": applied_action,
            "adjusted_action": adjusted_action,
            "world_events": world_events,
            "state_after": self.snapshot(),
        }

    def _last_archivist_summary(self) -> dict[str, Any] | None:
        for entry in reversed(self.notebook):
            if entry["role"] == "archivist" and entry["action"] == "summarize_state":
                return entry
        return None

    def _final_author_agent_id(self) -> str | None:
        summary = self._last_archivist_summary()
        if summary is not None:
            return summary["author_agent_id"]
        for agent_id, role in self.role_assignments.items():
            if role == "archivist":
                return agent_id
        if self.notebook:
            return self.notebook[-1]["author_agent_id"]
        return self.participants[0] if self.participants else None

    def _ready_for_finalization(self, step_index: int) -> bool:
        enough_context = len(self.notebook) >= min(4, self.max_steps)
        clean_queue = not self._open_items(self.correction_queue)
        clean_clarifications = not self._open_items(self.clarification_requests)
        has_archivist_summary = self._last_archivist_summary() is not None
        return enough_context and clean_queue and clean_clarifications and has_archivist_summary and step_index >= 3

    def should_end_episode(self, step_index: int) -> bool:
        if self.final_artifact is not None:
            return True
        if self._ready_for_finalization(step_index):
            return True
        return step_index + 1 >= self.max_steps

    def finalize_episode(self, *, step_index: int, force: bool = False) -> dict[str, Any] | None:
        if self.final_artifact is not None:
            return self.final_artifact

        final_author_agent_id = self._final_author_agent_id()
        if final_author_agent_id is None:
            return None

        primary_entry = self._last_archivist_summary() or (self.notebook[-1] if self.notebook else None)
        unresolved_corrections = self._open_items(self.correction_queue)
        unresolved_clarifications = self._open_items(self.clarification_requests)
        citations = list(
            dict.fromkeys(
                citation
                for entry in self.notebook[-6:]
                for citation in entry.get("citations", [])
            )
        )[:8]
        strongest_claim = primary_entry["claim"] if primary_entry is not None else self.task_prompt
        closure_status = "clean" if not unresolved_corrections and not unresolved_clarifications else "forced"
        if force and closure_status == "clean":
            closure_status = "timebox_closed"
        completion_reason = "archivist_closed_loop" if closure_status == "clean" else "forced_max_steps"
        risk_summary = [risk["note"] for risk in self.risk_flags[-3:]]
        unresolved_targets = [item.get("target_artifact_id") for item in unresolved_clarifications if item.get("target_artifact_id")]
        content = (
            f"# Episode {self.episode_index} Final Report\n\n"
            f"Task: {self.task_prompt}\n\n"
            f"Strongest current claim: {strongest_claim}\n\n"
            f"Closure status: {closure_status}\n"
            f"Open corrections: {len(unresolved_corrections)}\n"
            f"Open clarifications: {len(unresolved_clarifications)}\n"
            f"Risk flags: {len(self.risk_flags)}\n\n"
            f"Unresolved clarification targets: {unresolved_targets or ['none']}\n"
            f"Recent risk notes: {risk_summary or ['none']}\n"
        )
        artifact_id = f"final-{self.world_id}"
        artifact = {
            "artifact_id": artifact_id,
            "author_agent_id": final_author_agent_id,
            "artifact_type": "episode_final_report",
            "title": f"Episode {self.episode_index} final report",
            "summary": f"episode_final_report ({closure_status}): {strongest_claim}"[:180],
            "content": content,
            "citations": citations,
            "provenance": {
                "step_index": step_index,
                "closure_status": closure_status,
                "completion_reason": completion_reason,
                "source_artifact_id": primary_entry["artifact_id"] if primary_entry is not None else None,
                "open_corrections": len(unresolved_corrections),
                "open_clarifications": len(unresolved_clarifications),
            },
        }
        world_events = [
            {
                "event_type": "episode_finalized",
                "agent_id": final_author_agent_id,
                "event_payload": {
                    "episode_index": self.episode_index,
                    "step_index": step_index,
                    "artifact_id": artifact_id,
                    "closure_status": closure_status,
                    "completion_reason": completion_reason,
                    "open_corrections": len(unresolved_corrections),
                    "open_clarifications": len(unresolved_clarifications),
                    "risk_flags": len(self.risk_flags),
                    "forced": force,
                },
            }
        ]
        self.final_artifact = {
            "artifact": artifact,
            "events": world_events,
        }
        self.completion_reason = completion_reason
        self.closure_status = closure_status
        return self.final_artifact

    def episode_summary(self) -> dict[str, Any]:
        open_corrections = sum(item.get("status") == "open" for item in self.correction_queue)
        open_clarifications = sum(item.get("status") == "open" for item in self.clarification_requests)
        return {
            "episode_index": self.episode_index,
            "world_id": self.world_id,
            "task_prompt": self.task_prompt,
            "steps_completed": len(self.step_history),
            "notebook_entries": len(self.notebook),
            "open_corrections": open_corrections,
            "open_clarifications": open_clarifications,
            "risk_flags": len(self.risk_flags),
            "final_artifact_id": None if self.final_artifact is None else self.final_artifact["artifact"]["artifact_id"],
            "completion_reason": self.completion_reason,
            "closure_status": self.closure_status,
        }
