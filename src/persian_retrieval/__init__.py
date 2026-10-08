"""Persian-focused retrieval primitives, extracted and adapted from Hamkalam."""
from .chunking import TextChunk, chunk_text
from .evaluation import EvaluationSummary, QueryCase, evaluate_rankings
from .fusion import reciprocal_rank_fusion
from .normalization import lexical_terms, normalize_persian
from .search import LexicalIndex, SearchHit

__all__ = [
    "TextChunk",
    "chunk_text",
    "EvaluationSummary",
    "QueryCase",
    "evaluate_rankings",
    "reciprocal_rank_fusion",
    "lexical_terms",
    "normalize_persian",
    "LexicalIndex",
    "SearchHit",
]
