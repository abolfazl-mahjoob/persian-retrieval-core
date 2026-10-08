"""Dependency-light in-memory BM25 with an inverted index.

This is intentionally a small-corpus retrieval component, not an on-disk
full-text search engine or a distributed index.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .normalization import lexical_terms


@dataclass(frozen=True)
class SearchHit:
    document_id: str
    score: float


class LexicalIndex:
    """Immutable index; query work scales with postings, not all documents."""

    def __init__(
        self,
        documents: Mapping[str, str],
        *,
        stopwords: Iterable[str] | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not math.isfinite(k1) or k1 <= 0 or not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("invalid BM25 parameters")
        self.k1, self.b = k1, b
        self.stopwords = tuple(stopwords) if stopwords is not None else None
        self._doc_lengths: dict[str, int] = {}
        self._postings: dict[str, list[tuple[str, int]]] = defaultdict(list)
        total_terms = 0

        for doc_id, content in documents.items():
            if not isinstance(doc_id, str) or not doc_id:
                raise ValueError("document IDs must be nonempty strings")
            if not isinstance(content, str):
                raise TypeError("document content must be a string")
            terms = lexical_terms(content, stopwords=self.stopwords, deduplicate=False)
            counts = Counter(terms)
            doc_length = sum(counts.values())
            self._doc_lengths[doc_id] = doc_length
            total_terms += doc_length
            for term, count in counts.items():
                self._postings[term].append((doc_id, count))

        self._document_count = len(self._doc_lengths)
        self._average_length = (
            total_terms / self._document_count if self._document_count else 0.0
        )

    @property
    def document_count(self) -> int:
        return self._document_count

    @property
    def vocabulary_size(self) -> int:
        return len(self._postings)

    def search(self, query: str, *, limit: int = 10) -> list[SearchHit]:
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("limit must be a positive integer")
        terms = lexical_terms(query, stopwords=self.stopwords)
        if not terms or not self._document_count:
            return []

        scores: dict[str, float] = defaultdict(float)
        n = self._document_count
        for term in terms:
            postings = self._postings.get(term, ())
            if not postings:
                continue
            df = len(postings)
            idf = math.log1p((n - df + 0.5) / (df + 0.5))
            for doc_id, tf in postings:
                doc_length = self._doc_lengths[doc_id]
                norm = 1 - self.b + self.b * doc_length / max(1e-9, self._average_length)
                scores[doc_id] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)

        return [
            SearchHit(doc_id, score)
            for doc_id, score in sorted(scores.items(), key=lambda row: (-row[1], row[0]))[
                :limit
            ]
        ]
