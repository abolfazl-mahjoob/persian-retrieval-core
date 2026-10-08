import pytest

from persian_retrieval import LexicalIndex

DOCS = {
    "shipping": "ارسال سفارش تهران دو روز کاری طول می‌کشد.",
    "returns": "بازگشت کالا و مرجوعی تا هفت روز امکان‌پذیر است.",
    "hours": "ساعت کاری پشتیبانی روزانه نه تا پنج است.",
}


def test_bm25_finds_expected_document() -> None:
    results = LexicalIndex(DOCS).search("سفارش تهران چه روزی ارسال میشه؟")
    assert results and results[0].document_id == "shipping"


def test_no_match_has_no_fabricated_result() -> None:
    assert LexicalIndex(DOCS).search("کهکشان فضایی فضانورد") == []


def test_empty_index_and_stopword_only_query() -> None:
    assert LexicalIndex({}).search("ساعت") == []
    assert LexicalIndex(DOCS).search("شما چنده؟") == []


def test_results_have_stable_order() -> None:
    index = LexicalIndex({"b": "پیاده‌سازی", "a": "پیاده‌سازی"})
    assert [row.document_id for row in index.search("پیاده‌سازی")] == ["a", "b"]


def test_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        LexicalIndex(DOCS, k1=0)
    with pytest.raises(ValueError):
        LexicalIndex(DOCS).search("سلام", limit=0)
