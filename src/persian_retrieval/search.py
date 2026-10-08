"""Small dependency-free BM25 lexical index.

The index is deliberately in-memory. Pair it with an external vector or SQL
retriever and fuse ranked identifiers with reciprocal_rank_fusion.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .normalization import lexical_terms


@dataclass(frozen=True)
class SearchHit:
    document_id: str
    score: float


class LexicalIndex:
    def __init__(
        self,
        documents: Mapping[str, str],
        *,
        stopwords: Iterable[str] | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 <= 0 or not 0 <= b <= 1 or not math.isfinite(k1):
            raise ValueError("invalid BM25 parameters")
        self.k1, self.b = k1, b
        self.stopwords = tuple(stopwords) if stopwords is not None else None
        self._counts: dict[str, Counter[str]] = {}
        self._df: Counter[str] = Counter()
        total_terms = 0
        for doc_id, content in documents.items():
            if not doc_id:
                raise ValueError("document IDs cannot be empty")
            terms = lexical_terms(content, stopwords=self.stopwords, deduplicate=False)
            counts = Counter(terms)
            self._counts[doc_id] = counts
            self._df.update(counts.keys())
            total_terms += sum(counts.values())
        self._average_length = total_terms / len(self._counts) if self._counts else 0

    def search(self, query: str, *, limit: int = 10) -> list[SearchHit]:
        if limit < 1:
            raise ValueError("limit must be positive")
        terms = lexical_terms(query, stopwords=self.stopwords)
        if not terms or not self._counts:
            return []
        n = len(self._counts)
        ranked: list[SearchHit] = []
        for doc_id, counts in self._counts.items():
            doc_length = sum(counts.values())
            score = 0.0
            for term in terms:
                tf = counts[term]
                if not tf:
                    continue
                df = self._df[term]
                idf = math.log1p((n - df + 0.5) / (df + 0.5))
                norm = 1 - self.b + self.b * doc_length / max(1e-9, self._average_length)
                score += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
            if score > 0:
                ranked.append(SearchHit(doc_id, score))
        return sorted(ranked, key=lambda item: (-item.score, item.document_id))[:limit]
