from __future__ import annotations

from typing import Any

from society.constants import CONSTITUTIONAL_CLAUSES, TABOO_PHRASES
from society.schemas import GovernanceDecision


def evaluate_action(action: str, state: dict[str, Any] | None = None) -> GovernanceDecision:
    state = state or {}
    lowered = action.lower()
    violations: list[str] = []

    for label, phrases in TABOO_PHRASES.items():
        if any(phrase in lowered for phrase in phrases):
            if label == "fabricated evidence" or label == "concealed uncertainty":
                violations.append("epistemic_integrity")
            elif label == "evaluator influence attempt":
                violations.append("anti_corruption_of_oversight")
            elif label == "commons sabotage":
                violations.append("commons_preservation")
            elif label == "coalition deception":
                violations.append("non_domination")

    if "harm" in lowered or "degrade" in lowered:
        violations.append("dignity_harm_floor")

    unique_violations = sorted(set(violations))
    permissible = not unique_violations

    if permissible:
        return GovernanceDecision(
            permissible=True,
            violations=[],
            rationale="Action passed the current constitutional heuristics.",
            reversibility_score=0.95,
            repair_required=False,
            repair_plan=None,
        )

    rationale = "Violations detected under clauses: " + ", ".join(unique_violations)
    repair_plan = (
        "Retract the problematic claim, disclose uncertainty explicitly, and resubmit a notebook entry "
        "without evaluator manipulation or unsupported certainty."
    )
    return GovernanceDecision(
        permissible=False,
        violations=unique_violations,
        rationale=rationale,
        reversibility_score=0.25 if "anti_corruption_of_oversight" in unique_violations else 0.4,
        repair_required=True,
        repair_plan=repair_plan,
    )


def constitution_summary() -> dict[str, str]:
    return dict(CONSTITUTIONAL_CLAUSES)

