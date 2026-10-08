"""Transparent retrieval evaluation (macro Recall@k and MRR@k)."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class QueryCase:
    query: str
    relevant_ids: frozenset[str]

    def __post_init__(self) -> None:
        if not self.relevant_ids:
            raise ValueError("evaluation queries must include at least one relevant document")


@dataclass(frozen=True)
class EvaluationSummary:
    queries: int
    k: int
    recall_at_k: float
    mrr_at_k: float


def evaluate_rankings(
    cases: Sequence[QueryCase],
    rankings: Sequence[Sequence[str]],
    *,
    k: int = 5,
) -> EvaluationSummary:
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer")
    if len(cases) != len(rankings):
        raise ValueError("one ranking must be supplied per query")
    if not cases:
        return EvaluationSummary(0, k, 0.0, 0.0)
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    for case, ranking in zip(cases, rankings, strict=True):
        seen: set[str] = set()
        distinct: list[str] = []
        for item in ranking:
            if item not in seen:
                distinct.append(item)
                seen.add(item)
        top = distinct[:k]
        recalls.append(len(case.relevant_ids.intersection(top)) / len(case.relevant_ids))
        rank = next(
            (position for position, value in enumerate(top, 1) if value in case.relevant_ids),
            None,
        )
        reciprocal_ranks.append(1.0 / rank if rank is not None else 0.0)
    return EvaluationSummary(
        len(cases),
        k,
        sum(recalls) / len(recalls),
        sum(reciprocal_ranks) / len(reciprocal_ranks),
    )
