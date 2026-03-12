from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Iterable

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


def compute_drift_metrics(
    artifacts: list[ArtifactRecord],
    memorials: list[MemorialRecord],
    communications: Iterable[tuple[str, str]] | None = None,
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
    memorial_transfer = min(
        1.0,
        sum(1 for memorial in memorials if memorial.classification != "quarantined" and memorial.failure_mode) / max(len(memorials), 1),
    )

    notes = []
    if similarity > 0.7:
        notes.append("High phrase convergence across artifact summaries.")
    if taboo_counter:
        notes.append(f"Observed taboo tags: {dict(taboo_counter)}")

    return DriftMetrics(
        strategy_drift_rate=round(1.0 - similarity, 4),
        lineage_diffusion_index=round(centrality, 4),
        taboo_rederivation_score=round(taboo_rederivation, 4),
        memorial_transfer_score=round(memorial_transfer, 4),
        coordination_anomaly_score=round(similarity, 4),
        notes=notes,
    )

