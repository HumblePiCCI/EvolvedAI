from __future__ import annotations

from society.schemas import AgentRecord
from worlds.shared_notebook_v0 import SharedNotebookV0


def _agent(agent_id: str, role: str) -> AgentRecord:
    return AgentRecord(
        agent_id=agent_id,
        generation_id=1,
        lineage_id=f"lin-{agent_id}",
        role=role,
        model_name="mock",
        provider_name="mock",
        prompt_bundle_version="prompt",
        constitution_version="phase-0.5-draft",
        taboo_registry_version="v0",
        status="active",
    )


def test_shared_notebook_correction_lifecycle(tmp_path) -> None:
    citizen = _agent("agent-citizen", "citizen")
    judge = _agent("agent-judge", "judge")
    world = SharedNotebookV0(
        root_dir=tmp_path / "data",
        generation_id=1,
        episode_index=0,
        task_prompt="test task",
        max_steps=5,
    )
    world.bind_population([citizen, judge])

    first = world.apply_action(
        agent=citizen,
        parsed_action={
            "action": "propose_fact",
            "claim": "A bounded claim.",
            "uncertainty": "medium",
            "confidence": 0.6,
            "evidence": "public note",
            "citations": [],
            "target_artifact_id": None,
            "next_step": "wait for review",
        },
        artifact_id="art-1",
        step_index=0,
    )
    second = world.apply_action(
        agent=judge,
        parsed_action={
            "action": "critique_claim",
            "claim": "This claim needs tighter support.",
            "uncertainty": "medium",
            "confidence": 0.6,
            "evidence": "review",
            "citations": ["art-1"],
            "target_artifact_id": "art-1",
            "next_step": "request correction",
        },
        artifact_id="art-2",
        step_index=1,
    )
    third = world.apply_action(
        agent=citizen,
        parsed_action={
            "action": "respond_to_correction",
            "claim": "I am narrowing the claim.",
            "uncertainty": "explicit",
            "confidence": 0.5,
            "evidence": "correction response",
            "citations": ["art-1"],
            "target_artifact_id": "art-1",
            "next_step": "updated note",
        },
        artifact_id="art-3",
        step_index=2,
    )

    assert first["entry"]["artifact_id"] == "art-1"
    assert any(event["event_type"] == "correction_enqueued" for event in second["world_events"])
    assert any(event["event_type"] == "correction_resolved" for event in third["world_events"])
    assert world.episode_summary()["open_corrections"] == 0


def test_shared_notebook_force_finalizes_episode(tmp_path) -> None:
    citizen = _agent("agent-citizen", "citizen")
    judge = _agent("agent-judge", "judge")
    world = SharedNotebookV0(
        root_dir=tmp_path / "data",
        generation_id=1,
        episode_index=0,
        task_prompt="test task",
        max_steps=5,
    )
    world.bind_population([citizen, judge])
    world.apply_action(
        agent=citizen,
        parsed_action={
            "action": "propose_fact",
            "claim": "A bounded claim.",
            "uncertainty": "medium",
            "confidence": 0.6,
            "evidence": "public note",
            "citations": [],
            "target_artifact_id": None,
            "next_step": "wait for review",
        },
        artifact_id="art-1",
        step_index=0,
    )
    finalization = world.finalize_episode(step_index=4, force=True)
    assert finalization is not None
    assert finalization["artifact"]["artifact_type"] == "episode_final_report"
    assert world.episode_summary()["final_artifact_id"] == finalization["artifact"]["artifact_id"]


def test_steward_can_resolve_targeted_correction(tmp_path) -> None:
    citizen = _agent("agent-citizen", "citizen")
    judge = _agent("agent-judge", "judge")
    steward = _agent("agent-steward", "steward")
    world = SharedNotebookV0(
        root_dir=tmp_path / "data",
        generation_id=1,
        episode_index=0,
        task_prompt="test task",
        max_steps=6,
    )
    world.bind_population([citizen, judge, steward])

    world.apply_action(
        agent=steward,
        parsed_action={
            "action": "flag_risk",
            "claim": "The current notebook plan could create duplication.",
            "uncertainty": "medium",
            "confidence": 0.55,
            "evidence": "initial stewardship review",
            "citations": [],
            "target_artifact_id": None,
            "next_step": "tighten the plan",
        },
        artifact_id="art-steward",
        step_index=0,
    )
    critique = world.apply_action(
        agent=judge,
        parsed_action={
            "action": "critique_claim",
            "claim": "This stewardship note needs a direct correction response.",
            "uncertainty": "medium",
            "confidence": 0.6,
            "evidence": "review",
            "citations": ["art-steward"],
            "target_artifact_id": "art-steward",
            "next_step": "respond to the correction",
        },
        artifact_id="art-judge",
        step_index=1,
    )
    response = world.apply_action(
        agent=steward,
        parsed_action={
            "action": "respond_to_correction",
            "claim": "I am narrowing the stewardship note and addressing the correction directly.",
            "uncertainty": "explicit",
            "confidence": 0.44,
            "evidence": "correction response",
            "citations": ["art-steward"],
            "target_artifact_id": "art-steward",
            "next_step": "keep the corrected note narrow",
        },
        artifact_id="art-steward-fix",
        step_index=2,
    )

    assert any(event["event_type"] == "correction_enqueued" for event in critique["world_events"])
    assert any(event["event_type"] == "correction_resolved" for event in response["world_events"])
    assert world.episode_summary()["open_corrections"] == 0


def test_citizen_slot_is_not_stolen_by_steward_correction(tmp_path) -> None:
    citizen_a = _agent("agent-citizen-a", "citizen")
    citizen_b = _agent("agent-citizen-b", "citizen")
    judge = _agent("agent-judge", "judge")
    steward = _agent("agent-steward", "steward")
    adversary = _agent("agent-adversary", "adversary")
    archivist = _agent("agent-archivist", "archivist")
    agents = [citizen_a, citizen_b, judge, steward, archivist, adversary]
    world = SharedNotebookV0(
        root_dir=tmp_path / "data",
        generation_id=1,
        episode_index=0,
        task_prompt="test task",
        max_steps=10,
    )
    world.bind_population(agents)

    world.apply_action(
        agent=citizen_a,
        parsed_action={
            "action": "propose_fact",
            "claim": "A bounded claim.",
            "uncertainty": "medium",
            "confidence": 0.6,
            "evidence": "public note",
            "citations": [],
            "target_artifact_id": None,
            "next_step": "wait for review",
        },
        artifact_id="art-citizen-a",
        step_index=0,
    )
    world.apply_action(
        agent=judge,
        parsed_action={
            "action": "request_clarification",
            "claim": "Clarify the bounded claim.",
            "uncertainty": "medium",
            "confidence": 0.6,
            "evidence": "review",
            "citations": ["art-citizen-a"],
            "target_artifact_id": "art-citizen-a",
            "next_step": "answer directly",
        },
        artifact_id="art-judge",
        step_index=1,
    )
    world.apply_action(
        agent=citizen_a,
        parsed_action={
            "action": "respond_to_correction",
            "claim": "I am narrowing the claim.",
            "uncertainty": "explicit",
            "confidence": 0.5,
            "evidence": "correction response",
            "citations": ["art-citizen-a"],
            "target_artifact_id": "art-citizen-a",
            "next_step": "updated note",
        },
        artifact_id="art-citizen-a-fix",
        step_index=2,
    )
    world.apply_action(
        agent=steward,
        parsed_action={
            "action": "flag_risk",
            "claim": "A stewardship risk note.",
            "uncertainty": "medium",
            "confidence": 0.55,
            "evidence": "steward review",
            "citations": [],
            "target_artifact_id": None,
            "next_step": "watch the notebook",
        },
        artifact_id="art-steward",
        step_index=3,
    )
    world.apply_action(
        agent=adversary,
        parsed_action={
            "action": "critique_claim",
            "claim": "The stewardship note needs a direct response.",
            "uncertainty": "medium",
            "confidence": 0.6,
            "evidence": "pressure test",
            "citations": ["art-steward"],
            "target_artifact_id": "art-steward",
            "next_step": "respond to the correction",
        },
        artifact_id="art-adversary",
        step_index=4,
    )

    citizen_slot = world.select_next_agent(agents, step_index=5, last_actor_id=adversary.agent_id)
    assert citizen_slot is not None
    assert citizen_slot.role == "citizen"

    later_steward_slot = world.select_next_agent(agents, step_index=8, last_actor_id=archivist.agent_id)
    assert later_steward_slot is not None
    assert later_steward_slot.role == "steward"


def test_under_engaged_citizen_keeps_next_slot_despite_open_clarification(tmp_path) -> None:
    citizen_a = _agent("agent-citizen-a", "citizen")
    citizen_b = _agent("agent-citizen-b", "citizen")
    judge = _agent("agent-judge", "judge")
    world = SharedNotebookV0(
        root_dir=tmp_path / "data",
        generation_id=1,
        episode_index=0,
        task_prompt="test task",
        max_steps=6,
    )
    world.bind_population([citizen_a, citizen_b, judge])

    world.apply_action(
        agent=citizen_a,
        parsed_action={
            "action": "propose_fact",
            "claim": "A bounded claim.",
            "uncertainty": "medium",
            "confidence": 0.6,
            "evidence": "public note",
            "citations": [],
            "target_artifact_id": None,
            "next_step": "wait for review",
        },
        artifact_id="art-citizen-a",
        step_index=0,
    )
    world.apply_action(
        agent=judge,
        parsed_action={
            "action": "request_clarification",
            "claim": "Clarify the bounded claim.",
            "uncertainty": "medium",
            "confidence": 0.6,
            "evidence": "review",
            "citations": ["art-citizen-a"],
            "target_artifact_id": "art-citizen-a",
            "next_step": "answer directly",
        },
        artifact_id="art-judge",
        step_index=1,
    )

    next_citizen = world.select_next_agent(
        [citizen_a, citizen_b, judge],
        step_index=2,
        last_actor_id=judge.agent_id,
    )

    assert next_citizen is not None
    assert next_citizen.agent_id == citizen_b.agent_id
