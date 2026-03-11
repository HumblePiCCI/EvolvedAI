from __future__ import annotations

import importlib
from collections import defaultdict

from society.schemas import EvalRecord
from society.utils import short_hash, utc_now


def _run_eval_module(*, family: str, name: str, generation_id: int, agent, artifacts, events, all_artifacts, all_events) -> EvalRecord:
    module = importlib.import_module(f"evals.{family}.{name}")
    result = module.evaluate(
        agent=agent,
        artifacts=artifacts,
        events=events,
        all_artifacts=all_artifacts,
        all_events=all_events,
    )
    return EvalRecord(
        eval_id=f"eval-{generation_id:04d}-{short_hash(agent.agent_id + family + name)}",
        generation_id=generation_id,
        agent_id=agent.agent_id,
        eval_family=family,
        eval_name=name,
        visible_to_agent=(family == "public"),
        score=result.get("score"),
        pass_fail=result.get("pass_fail"),
        details_json=result.get("details", {}),
        created_at=utc_now(),
    )


def run_eval_suite(*, config, generation_id: int, agents, artifacts, events) -> list[EvalRecord]:
    artifacts_by_agent = defaultdict(list)
    events_by_agent = defaultdict(list)
    for artifact in artifacts:
        artifacts_by_agent[artifact.author_agent_id].append(artifact)
    for event in events:
        if event.agent_id is not None:
            events_by_agent[event.agent_id].append(event)

    results: list[EvalRecord] = []
    for agent in agents:
        agent_artifacts = artifacts_by_agent[agent.agent_id]
        agent_events = events_by_agent[agent.agent_id]
        for name in config.evals.public:
            results.append(
                _run_eval_module(
                    family="public",
                    name=name,
                    generation_id=generation_id,
                    agent=agent,
                    artifacts=agent_artifacts,
                    events=agent_events,
                    all_artifacts=artifacts,
                    all_events=events,
                )
            )
        for name in config.evals.hidden:
            results.append(
                _run_eval_module(
                    family="hidden",
                    name=name,
                    generation_id=generation_id,
                    agent=agent,
                    artifacts=agent_artifacts,
                    events=agent_events,
                    all_artifacts=artifacts,
                    all_events=events,
                )
            )
    return results

