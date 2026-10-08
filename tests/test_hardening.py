"""Adversarial, boundary and metamorphic regression tests."""
from __future__ import annotations

import math
import random
import string

import pytest

from persian_retrieval import (
    LexicalIndex,
    QueryCase,
    chunk_text,
    evaluate_rankings,
    lexical_terms,
    normalize_persian,
    reciprocal_rank_fusion,
)


def test_rrf_compacts_duplicate_ranks_before_scoring() -> None:
    # The second *unique* result should have rank 2, not rank 4.
    result = dict(reciprocal_rank_fusion([["a", "a", "a", "b"]]))
    assert result["b"] == pytest.approx(1 / 62)
    assert result["a"] == pytest.approx(1 / 61)


def test_rrf_duplicate_results_cannot_change_scores() -> None:
    original = reciprocal_rank_fusion([["a", "b", "c"], ["c", "a"]])
    with_duplicates = reciprocal_rank_fusion(
        [["a", "a", "b", "b", "c"], ["c", "c", "a"]]
    )
    assert original == with_duplicates


def test_rrf_ties_preserve_first_seen_order() -> None:
    result = reciprocal_rank_fusion([["z", "b"], ["b", "z"]])
    assert [key for key, _ in result] == ["z", "b"]


def test_rrf_empty_and_zero_weight() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([["one"]], weights=[0.0]) == []


@pytest.mark.parametrize("k", [-1, 0, True, 1.5])
def test_rrf_rejects_bad_k(k: object) -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=k)  # type: ignore[arg-type]


def test_rrf_rejects_weight_mismatch() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])


def test_query_matching_is_case_insensitive_for_latin() -> None:
    index = LexicalIndex({"a": "Python Django FastAPI"})
    assert [hit.document_id for hit in index.search("python fastapi")] == ["a"]
    assert lexical_terms("PYTHON Python python") == ["python"]


def test_custom_stopword_matching_uses_normalized_case() -> None:
    assert lexical_terms("PYTHON فارسی", stopwords=["python"]) == ["فارسی"]


def test_unicode_normalization_is_idempotent() -> None:
    cases = ["قیمت كتاب", "A  B", "دوره‌ ي  خوب", "عدد ۱۲۳", "كِتاب", ""]
    for raw in cases:
        normalized = normalize_persian(raw)
        assert normalize_persian(normalized) == normalized


@pytest.mark.parametrize(
    "text", ["\u200c", "!!!", "و در به", "123", "یک", "قیمت‌طلا"]
)
def test_normalization_and_tokenization_are_deterministic(text: str) -> None:
    assert lexical_terms(text) == lexical_terms(text)
    assert normalize_persian(text) == normalize_persian(text)


def test_chunking_rejects_nan_and_infinite_overlap() -> None:
    for overlap in [math.nan, math.inf, -math.inf]:
        with pytest.raises(ValueError):
            chunk_text("متنی برای قطعه بندی", overlap_ratio=overlap)


@pytest.mark.parametrize("tokens", [True, 15.3, -5, 0])
def test_chunking_rejects_noninteger_or_too_small_budget(tokens: object) -> None:
    with pytest.raises(ValueError):
        chunk_text("متنی برای قطعه بندی", target_tokens=tokens)  # type: ignore[arg-type]


def test_heading_longer_than_context_budget_fails_explicitly() -> None:
    with pytest.raises(ValueError, match="heading prefix"):
        chunk_text(
            "# " + "بسیارطولانی" * 80 + "\nیک نمونه متنی.",
            target_tokens=20,
        )


def test_nested_headings_are_real_semantic_boundaries() -> None:
    text = (
        "# دوره\n## ثبت نام\nثبت نام فقط با شماره همراه است.\n"
        "### پرداخت\nشهریه از درگاه پرداخت می‌شود.\n"
        "## استرداد\nبازگشت وجه تابع قوانین است."
    )
    chunks = chunk_text(text, title="آموزش", target_tokens=75)
    assert [c.heading_path for c in chunks] == [
        ("دوره", "ثبت نام"),
        ("دوره", "ثبت نام", "پرداخت"),
        ("دوره", "استرداد"),
    ]


def test_fuzz_chunks_never_exceed_approximate_budget() -> None:
    rng = random.Random(20261008)
    words = ["پشتیبانی", "خرید", "مرجوعی", "آموزش", "پرداخت", "داده", "python"]
    for _ in range(200):
        target = rng.randint(16, 80)
        overlap = rng.choice([0.0, 0.1, 0.25, 0.4])
        raw = " ".join(rng.choice(words) for _ in range(rng.randint(1, 120)))
        chunks = chunk_text(raw, title="تست", target_tokens=target, overlap_ratio=overlap)
        assert chunks
        assert all(c.estimated_tokens <= target for c in chunks)
        assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_giant_unbroken_token_is_bounded() -> None:
    chunks = chunk_text("abcdef" * 1000, target_tokens=16)
    assert len(chunks) > 10
    assert all(c.estimated_tokens <= 16 for c in chunks)


def test_bm25_snapshot_not_affected_by_mutating_input() -> None:
    data = {"one": "مرجوعی کالا"}
    index = LexicalIndex(data)
    data["one"] = "موجودی محصول"
    assert [hit.document_id for hit in index.search("مرجوعی")] == ["one"]


def test_bm25_does_not_score_unrelated_documents() -> None:
    corpus = {str(i): f"موضوع{i} داده" for i in range(100)}
    index = LexicalIndex(corpus)
    result = index.search("ناشناختهکامل")
    assert result == []
    assert index.document_count == 100


def test_bm25_no_fake_matches_for_empty_document() -> None:
    index = LexicalIndex({"a": "", "b": "راهنمای پایتون"})
    assert [hit.document_id for hit in index.search("پایتون")] == ["b"]


@pytest.mark.parametrize("limit", [True, 0, -2, 2.1])
def test_bm25_invalid_limit(limit: object) -> None:
    with pytest.raises(ValueError):
        LexicalIndex({"a": "سلام"}).search("سلام", limit=limit)  # type: ignore[arg-type]


def test_bm25_rejects_invalid_documents() -> None:
    with pytest.raises(ValueError):
        LexicalIndex({"": "content"})
    with pytest.raises(TypeError):
        LexicalIndex({"a": None})  # type: ignore[dict-item]


def test_evaluation_duplicate_predictions_do_not_change_recall() -> None:
    case = [QueryCase("q", frozenset({"a", "b"}))]
    distinct = evaluate_rankings(case, [["a", "b"]], k=2)
    repeats = evaluate_rankings(case, [["a", "a", "b", "b"]], k=2)
    assert distinct == repeats


def test_evaluation_k_outside_ranked_list_is_bounded() -> None:
    result = evaluate_rankings(
        [QueryCase("q", frozenset({"a", "b", "c"}))],
        [["a", "b"]],
        k=100,
    )
    assert result.recall_at_k == pytest.approx(2 / 3)
    assert result.mrr_at_k == 1.0


def test_query_without_answer_has_zero_recall() -> None:
    summary = evaluate_rankings(
        [QueryCase("q", frozenset({"a"}))],
        [["b", "c"]],
        k=2,
    )
    assert summary.recall_at_k == summary.mrr_at_k == 0.0


def test_randomized_rrf_scores_are_finite_and_nonnegative() -> None:
    rng = random.Random(521)
    ids = list(string.ascii_lowercase)
    for _ in range(100):
        rankings = [rng.sample(ids, rng.randint(0, 20)) for _ in range(3)]
        scores = reciprocal_rank_fusion(rankings)
        assert all(math.isfinite(score) and score >= 0 for _, score in scores)
        assert len({doc_id for doc_id, _ in scores}) == len(scores)
        assert all(scores[i][1] >= scores[i + 1][1] for i in range(len(scores) - 1))
