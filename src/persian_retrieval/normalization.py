"""Unicode normalization and Persian tokenization adapted from Hamkalam RAG."""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

_ARABIC_TO_PERSIAN = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ة": "ه"})
_DIGITS_TO_ASCII = str.maketrans(
    "۰۱۲۳۴۵۶۷۸۹" "٠١٢٣٤٥٦٧٨٩",
    "01234567890123456789",
)
_DIACRITICS = re.compile("[ؐ-ًؚ-ٰٟۖ-ۭ]")
_ZWNJ = "\u200c"
_SEPARATOR = re.compile(r"[^\w\u200c]+")
_WHITESPACE = re.compile(r"[ \t\r\f\v]+")
_DEFAULT_STOPWORDS = frozenset(
    """و در به از که را با این آن یا هم برای تا بر می نمی رو اگر ولی اما چون چرا
    چطور چگونه چقدر چند چه چی کی کجا کدام چنده چیه کجاست چیست شما من ما
    است هست بود شد شده خواهد باید نیست دارد دارم داری کردن کند کنم میشه
    می‌شه بهش لطفا لطفاً سلام ممنون الان امروز میخوام می‌خوام""".split()
)


def normalize_persian(value: str) -> str:
    """Normalize Arabic/Persian letter variants, digits, diacritics and whitespace.

    NFKC is intentional: use on *search text*, not cryptographic identifiers or passwords.
    """
    text = unicodedata.normalize("NFKC", value)
    text = text.translate(_ARABIC_TO_PERSIAN).translate(_DIGITS_TO_ASCII)
    text = _DIACRITICS.sub("", text).replace("ـ", "")
    text = text.replace("\u200d", _ZWNJ).replace("\ufeff", "")
    text = re.sub(r"\s*\u200c\s*", _ZWNJ, text)
    text = re.sub(r"\u200c+", _ZWNJ, text)
    text = _WHITESPACE.sub(" ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def lexical_terms(
    text: str,
    *,
    stopwords: Iterable[str] | None = None,
    drop_stopwords: bool = True,
    deduplicate: bool = True,
) -> list[str]:
    """Tokenize for lexical recall. Emit ZWNJ compound and searchable components.

    Call with deduplicate=False for document term frequencies.
    """
    ignored = _DEFAULT_STOPWORDS if stopwords is None else frozenset(
        normalize_persian(word) for word in stopwords
    )
    tokens: list[str] = []
    seen: set[str] = set()
    for raw in _SEPARATOR.split(normalize_persian(text)):
        token = raw.strip(_ZWNJ).strip()
        candidates = [token, *token.split(_ZWNJ)] if _ZWNJ in token else [token]
        for candidate in candidates:
            if len(candidate) < 2 or (candidate.isdigit() and len(candidate) < 3):
                continue
            if drop_stopwords and candidate in ignored:
                continue
            if deduplicate and candidate in seen:
                continue
            seen.add(candidate)
            tokens.append(candidate)
    return tokens
