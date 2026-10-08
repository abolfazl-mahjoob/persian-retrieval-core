import pytest

from persian_retrieval import chunk_text


def test_heading_isolation_prevents_cross_topic_overlap() -> None:
    text = (
        "# سفارش\nثبت سفارش نیازمند شماره تماس است.\n"
        "# مرجوعی\nکالا تا هفت روز قابل بازگشت است."
    )
    chunks = chunk_text(text, "راهنما", target_tokens=30, overlap_ratio=0.2)
    assert len(chunks) == 2
    assert chunks[0].heading_path == ("سفارش",)
    assert chunks[1].heading_path == ("مرجوعی",)
    assert "مرجوعی" not in chunks[0].content
    assert "شماره تماس" not in chunks[1].content


def test_overlap_keeps_tail_in_same_heading() -> None:
    chunks = chunk_text(
        "# قوانین\n" + "این یک جمله آزمایشی است. " * 60,
        "راهنما",
        target_tokens=35,
        overlap_ratio=0.3,
    )
    assert len(chunks) > 2
    assert all(c.heading_path == ("قوانین",) for c in chunks)


def test_long_unpunctuated_text_is_bounded() -> None:
    chunks = chunk_text(" ".join(["نمونه"] * 200), target_tokens=20, overlap_ratio=0)
    assert len(chunks) > 4
    assert all(c.estimated_tokens <= 20 for c in chunks)


def test_empty_input() -> None:
    assert chunk_text("  \n ") == []
    assert chunk_text("# فقط عنوان") == []


@pytest.mark.parametrize("tokens,overlap", [(0, 0.1), (20, 1.0), (20, -0.1)])
def test_invalid_settings(tokens: int, overlap: float) -> None:
    with pytest.raises(ValueError):
        chunk_text("متن", target_tokens=tokens, overlap_ratio=overlap)
