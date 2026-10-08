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
| Quality | independent BM25 reference, edge-case tests, branch coverage, strict mypy, Ruff, multi-Python CI, Docker |

## Install & run

```bash
python -m pip install -e '.[dev]'
python -m pytest --cov=persian_retrieval --cov-branch --cov-fail-under=90
python scripts/evaluate_fixture.py
python scripts/benchmark_index.py --documents 2000 --queries 100
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

## Quality verification

The test suite contains a deliberately independent reference implementation
of the BM25 scoring equation, randomized index-vs-reference comparisons,
adversarial RRF inputs, and bounded chunking regression tests. CI runs strict
typing, code analysis, branch coverage and Docker verification.

The included [Persian evaluation fixture](benchmarks/persian_helpdesk_smoke_v1.json)
has **40 hand-written synthetic documents and 40 hand-written labelled queries**.
It is a regression smoke test, **not a representative real-user benchmark**.
Perfect scores on this fixture do not establish generalization to typos,
synonyms, unseen domains, or production workloads. To reproduce:

```bash
python scripts/evaluate_fixture.py --min-recall3 0.75
python scripts/benchmark_index.py --documents 2000 --queries 100
```

Read [measurement methodology and limitations](docs/quality.md).
Query timing and traced build-memory figures are machine-specific and should
not be interpreted as guaranteed service performance.

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

**0.1.0 candidate.** The reusable primitives have automated verification; not
validated as a production-ready RAG/search service. Source-rights clearance for
code adapted from Hamkalam remains a prerequisite for downstream use. MIT
license covers this standalone repository, subject to those rights.

For implementation notes see [Architecture](docs/architecture.md).
