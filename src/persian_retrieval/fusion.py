"""Deterministic weighted reciprocal rank fusion (RRF)."""
from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from typing import TypeVar

IdT = TypeVar("IdT", bound=Hashable)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[IdT]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[IdT, float]]:
    """Fuse ordered rankings; a duplicate ID in one ranking is counted once.

    Stable tie-breaking uses the earliest appearance in the supplied rankings.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer")
    if weights is not None and len(weights) != len(rankings):
        raise ValueError("weights and rankings must have the same length")
    chosen = list(weights) if weights is not None else [1.0] * len(rankings)
    if any(not math.isfinite(weight) or weight < 0 for weight in chosen):
        raise ValueError("weights must be finite and nonnegative")
    score: dict[IdT, float] = {}
    first_seen: dict[IdT, int] = {}
    for ranking, weight in zip(rankings, chosen, strict=True):
        if weight == 0:
            continue
        unique: set[IdT] = set()
        for item in ranking:
            if item in unique:
                continue
            unique.add(item)
            position = len(unique)
            if item not in first_seen:
                first_seen[item] = len(first_seen)
            score[item] = score.get(item, 0.0) + weight / (k + position)
    return sorted(score.items(), key=lambda pair: (-pair[1], first_seen[pair[0]]))
