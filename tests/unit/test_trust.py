from __future__ import annotations

from society.trust import classify_warning_outcome, summarize_warning_effect, warning_labels


def test_warning_labels_include_taboos_and_memorial_failure_modes() -> None:
    labels = warning_labels(
        taboo_tags=["anti_corruption"],
        memorials=[
            {"failure_mode": "calibration"},
            {"failure_mode": None},
        ],
    )

    assert labels == {"anti_corruption", "calibration"}


def test_summarize_warning_effect_tracks_avoided_repeated_and_shifted_failures() -> None:
    effect = summarize_warning_effect(
        [
            ({"anti_corruption"}, set()),
            ({"citation_fidelity"}, {"citation_fidelity"}),
            ({"calibration"}, {"artifact_quality"}),
            (set(), {"artifact_quality"}),
        ]
    )

    assert classify_warning_outcome(warning_labels={"anti_corruption"}, current_failures=set()) == "avoided_recurrence"
    assert effect == {
        "warned_lineages": 3,
        "avoided_recurrence": 1,
        "repeated_warning": 1,
        "shifted_failure": 1,
        "transfer_score": 0.3333,
    }
