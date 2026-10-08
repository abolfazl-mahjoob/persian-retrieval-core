# Persian Retrieval Core

**Small, dependency-light Persian retrieval primitives extracted and adapted from real product engineering work.**

[![Quality gate](https://github.com/abolfazl-mahjoob/persian-retrieval-core/actions/workflows/ci.yml/badge.svg)](https://github.com/abolfazl-mahjoob/persian-retrieval-core/actions/workflows/ci.yml)

This project offers the building blocks that sit *before* an LLM:
Persian text normalization, heading-aware chunking, in-memory lexical BM25
retrieval, deterministic reciprocal rank fusion (RRF), and reproducible
retrieval evaluation.

## Features

| Feature | Engineering focus |
|---|---|
| Persian normalization | Arabic/Persian letter variants, numerals, diacritics and ZWNJ |
| Heading-aware chunking | Markdown heading isolation, approximate token budgets and overlap |
| BM25 | In-memory lexical retrieval, deterministic ranking |
| RRF | Merge lexical and external vector result lists with optional weights |
| Evaluation | Macro Recall@k and MRR@k for labelled queries |
| Quality | pytest, Ruff, package build, Python 3.11–3.13, Docker test |

## Install & run

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python examples/basic.py
```

Verify using Docker:

```bash
docker build -t persian-retrieval-core:test .
docker run --rm persian-retrieval-core:test
```

## Usage

```python
from persian_retrieval import LexicalIndex, reciprocal_rank_fusion

documents = {
    "shipping": "ارسال سفارش به تهران یک تا دو روز کاری زمان می‌برد.",
    "returns": "مرجوعی کالا تا هفت روز امکان‌پذیر است.",
}
index = LexicalIndex(documents)
lexical_ids = [row.document_id for row in index.search("ارسال سفارش تهران")]
vector_ids = ["shipping", "returns"]  # ranked IDs from any external vector store
combined = reciprocal_rank_fusion([lexical_ids, vector_ids])
print(combined)
```

## Engineering boundaries

This is **not** an LLM wrapper, a hosted vector database, a complete RAG
service, or a guarantee against hallucinations. The BM25 search index is
in-memory. It includes **no** user data, credentials or product database code.

Adapted from Hamkalam's Persian search and retrieval modules. The original
product's SQL, tenant-specific model, embedding provider integration and
commercial logic are deliberately excluded.

Read the [design notes](docs/architecture.md) for provenance, limitations and
trade-offs.

## Status

**0.1.0 candidate.** Release only after CI, standalone tests and
licensing clearance. MIT license included for the standalone implementation.
