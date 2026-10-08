"""Cross-check indexed BM25 with a deliberately simple reference implementation."""
from __future__ import annotations

import math
import random
from collections import Counter

import pytest

from persian_retrieval import LexicalIndex, lexical_terms


def slow_reference(
    documents: dict[str, str], query: str, k1: float, b: float
) -> list[tuple[str, float]]:
    counts = {
        key: Counter(lexical_terms(value, deduplicate=False))
        for key, value in documents.items()
    }
    n = len(documents)
    average = sum(sum(counter.values()) for counter in counts.values()) / max(n, 1)
    qterms = lexical_terms(query)
    rows: list[tuple[str, float]] = []
    for doc_id, counter in counts.items():
        length = sum(counter.values())
        score = 0.0
        for term in qterms:
            if not counter[term]:
                continue
            df = sum(1 for candidate in counts.values() if term in candidate)
            idf = math.log1p((n - df + 0.5) / (df + 0.5))
            norm = 1 - b + b * length / max(average, 1e-9)
            tf = counter[term]
            score += idf * tf * (k1 + 1) / (tf + k1 * norm)
        if score > 0:
            rows.append((doc_id, score))
    return sorted(rows, key=lambda pair: (-pair[1], pair[0]))


@pytest.mark.parametrize("k1,b", [(0.5, 0.0), (1.2, 0.5), (1.5, 0.75), (2.0, 1.0)])
def test_inverted_index_matches_reference_equation(k1: float, b: float) -> None:
    rng = random.Random(2026)
    vocab = [
        "Python", "پایتون", "آموزش", "پرداخت", "سفارش", "ایمیل",
        "ثبت‌نام", "دانشجو", "درگاه", "محصول", "ارسال", "مرجوعی",
    ]
    documents = {
        str(i): " ".join(rng.choices(vocab, k=rng.randint(0, 45)))
        for i in range(55)
    }
    index = LexicalIndex(documents, k1=k1, b=b)
    assert index.document_count == len(documents)
    assert index.vocabulary_size > 0
    for _ in range(25):
        query = " ".join(rng.choices(vocab, k=rng.randint(1, 5)))
        expected = slow_reference(documents, query, k1, b)
        actual = index.search(query, limit=1000)
        assert [x.document_id for x in actual] == [item[0] for item in expected]
        assert [x.score for x in actual] == pytest.approx(
            [item[1] for item in expected], rel=1e-12, abs=1e-12
        )


def test_index_keeps_zero_results_for_unrelated_term() -> None:
    index = LexicalIndex({"one": "سلام", "two": "پایتون"})
    assert index.search("سیاره") == []
