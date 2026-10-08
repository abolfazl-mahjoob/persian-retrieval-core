from persian_retrieval import lexical_terms, normalize_persian


def test_persian_arabic_variants_and_digits() -> None:
    assert normalize_persian(" يَك كِتاب ۱۲۳ و ٤٥٦ ـ ") == "یک کتاب 123 و 456"


def test_preserve_and_split_zwnj_compounds() -> None:
    terms = lexical_terms("دوره‌ی پيشرفته")
    assert "دوره‌ی" in terms and "دوره" in terms and "پیشرفته" in terms


def test_strip_punctuation_and_conversational_stopwords() -> None:
    terms = lexical_terms("سلام، ساعت کاری شما چنده؟")
    assert "ساعت" in terms and "کاری" in terms
    assert "شما" not in terms and "چنده" not in terms


def test_repeated_terms_support_term_frequency() -> None:
    assert lexical_terms("موتور موتور", deduplicate=False) == ["موتور", "موتور"]
    assert lexical_terms("موتور موتور") == ["موتور"]


def test_custom_stopwords() -> None:
    assert lexical_terms("طلای سنگین", stopwords=["سنگین"]) == ["طلای"]


def test_no_hidden_normalization_of_identifiers() -> None:
    assert normalize_persian("A  B") == "A B"
