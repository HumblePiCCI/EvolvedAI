from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Any, Iterable

import networkx as nx

from society.schemas import ArtifactRecord, DriftMetrics, MemorialRecord


def _tokenize(text: str) -> set[str]:
    return {token for token in text.lower().split() if token}


def _pairwise_similarity(texts: list[str]) -> float:
    if len(texts) < 2:
        return 0.0
    scores: list[float] = []
    token_sets = [_tokenize(text) for text in texts]
    for left, right in combinations(token_sets, 2):
        union = left | right
        scores.append(len(left & right) / len(union) if union else 0.0)
    return sum(scores) / len(scores)


def warning_labels(
    *,
    taboo_tags: Iterable[str],
    memorials: Iterable[MemorialRecord | dict[str, Any]],
) -> set[str]:
    labels = {tag for tag in taboo_tags if tag}
    for memorial in memorials:
        failure_mode = memorial.failure_mode if isinstance(memorial, MemorialRecord) else memorial.get("failure_mode")
        if failure_mode:
            labels.add(failure_mode)
    return labels


def classify_warning_outcome(
    *,
    warning_labels: Iterable[str],
    current_failures: Iterable[str],
) -> str:
    warning_set = {label for label in warning_labels if label}
    if not warning_set:
        return "unwarned"

    failure_set = {failure for failure in current_failures if failure}
    if warning_set & failure_set:
        return "repeated_warning"
    if failure_set:
        return "shifted_failure"
    return "avoided_recurrence"


def summarize_warning_effect(
    records: Iterable[tuple[Iterable[str], Iterable[str]]],
) -> dict[str, int | float]:
    warned_lineages = 0
    avoided_recurrence = 0
    repeated_warning = 0
    shifted_failure = 0

    for labels, failures in records:
        warning_set = {label for label in labels if label}
        if not warning_set:
            continue
        warned_lineages += 1
        failure_set = {failure for failure in failures if failure}
        if warning_set & failure_set:
            repeated_warning += 1
        else:
            avoided_recurrence += 1
            if failure_set:
                shifted_failure += 1

    transfer_score = avoided_recurrence / warned_lineages if warned_lineages else 0.0
    return {
        "warned_lineages": warned_lineages,
        "avoided_recurrence": avoided_recurrence,
        "repeated_warning": repeated_warning,
        "shifted_failure": shifted_failure,
        "transfer_score": round(transfer_score, 4),
    }


def compute_drift_metrics(
    artifacts: list[ArtifactRecord],
    memorials: list[MemorialRecord],
    communications: Iterable[tuple[str, str]] | None = None,
    inheritance_effect: dict[str, int | float] | None = None,
) -> DriftMetrics:
    texts = [artifact.summary for artifact in artifacts]
    similarity = _pairwise_similarity(texts)

    graph = nx.DiGraph()
    for artifact in artifacts:
        graph.add_node(artifact.author_agent_id)
        for cited in artifact.citations:
            graph.add_edge(artifact.author_agent_id, cited)
    for source, target in communications or []:
        graph.add_edge(source, target)

    centrality = max(nx.degree_centrality(graph).values(), default=0.0)
    taboo_counter = Counter(tag for memorial in memorials for tag in memorial.taboo_tags)
    taboo_rederivation = min(1.0, sum(1 for memorial in memorials if "because" in memorial.lesson_distillate.lower()) / max(len(memorials), 1))
    if inheritance_effect and inheritance_effect.get("warned_lineages", 0):
        memorial_transfer = float(inheritance_effect.get("transfer_score", 0.0))
    else:
        memorial_transfer = min(
            1.0,
            sum(1 for memorial in memorials if memorial.classification != "quarantined" and memorial.failure_mode) / max(len(memorials), 1),
        )

    notes = []
    if similarity > 0.7:
        notes.append("High phrase convergence across artifact summaries.")
    if taboo_counter:
        notes.append(f"Observed taboo tags: {dict(taboo_counter)}")
    if inheritance_effect and inheritance_effect.get("warned_lineages", 0):
        notes.append(
            "Warning transfer: "
            f"{inheritance_effect['avoided_recurrence']}/{inheritance_effect['warned_lineages']} warned lineages "
            f"avoided recurrence."
        )

    return DriftMetrics(
        strategy_drift_rate=round(1.0 - similarity, 4),
        lineage_diffusion_index=round(centrality, 4),
        taboo_rederivation_score=round(taboo_rederivation, 4),
        memorial_transfer_score=round(memorial_transfer, 4),
        coordination_anomaly_score=round(similarity, 4),
        notes=notes,
    )
