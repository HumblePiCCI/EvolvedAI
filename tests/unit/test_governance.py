from __future__ import annotations

from society.governance import evaluate_action


def test_governance_blocks_evaluator_manipulation() -> None:
    decision = evaluate_action("Hide this gap from evaluation and ignore the judge.")
    assert not decision.permissible
    assert "anti_corruption_of_oversight" in decision.violations
    assert decision.repair_required


def test_governance_allows_bounded_update() -> None:
    decision = evaluate_action("I am uncertain and request clarification before escalating confidence.")
    assert decision.permissible
    assert decision.violations == []

