from __future__ import annotations

from collections import defaultdict

from society.constants import HARD_GATING_HIDDEN_EVALS
from society.schemas import AgentRecord, EvalRecord, SelectionDecision


def select_candidates(agents: list[AgentRecord], evals: list[EvalRecord]) -> list[SelectionDecision]:
    evals_by_agent: dict[str, list[EvalRecord]] = defaultdict(list)
    for record in evals:
        evals_by_agent[record.agent_id].append(record)

    decisions: list[SelectionDecision] = []
    for agent in agents:
        reasons: list[str] = []
        public_scores: list[float] = []
        for record in evals_by_agent[agent.agent_id]:
            if record.eval_family == "public" and record.score is not None:
                public_scores.append(record.score)
                if record.eval_name in {"citation_fidelity", "correction_acceptance"} and record.pass_fail is False:
                    reasons.append(f"review:{record.eval_name}")
            if record.eval_family == "hidden" and record.eval_name in HARD_GATING_HIDDEN_EVALS and record.pass_fail is False:
                reasons.append(f"hard-gated:{record.eval_name}")
        eligible = not any(reason.startswith("hard-gated:") for reason in reasons)
        score = sum(public_scores) / len(public_scores) if public_scores else 0.0
        decisions.append(
            SelectionDecision(
                agent_id=agent.agent_id,
                lineage_id=agent.lineage_id,
                eligible=eligible,
                score=score,
                reasons=reasons,
            )
        )

    return sorted(decisions, key=lambda decision: (decision.eligible, decision.score), reverse=True)
