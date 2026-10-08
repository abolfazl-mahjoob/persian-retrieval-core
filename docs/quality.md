# Quality and measurement protocol

This document separates **verified software properties** from **search-quality
claims that require independent data**.

## Verified by automation

The quality gate runs on CPython 3.11, 3.12 and 3.13:

1. Ruff code diagnostics.
2. Strict mypy on the public package.
3. Unit and adversarial/metamorphic tests with **branch** coverage >= 90%.
4. Cross-check of optimized BM25 against an independent O(N) reference
   implementation across four BM25 parameter settings and randomized inputs.
5. Synthetic Persian retrieval fixture evaluation.
6. A small timing smoke test, basic usage example, wheel/sdist packaging and
   dependency check.
7. A separately built Python 3.12 Docker image that runs the complete pytest
   suite and includes its fixture.
8. A 2,000-document / 100-query seeded synthetic timing run, saved as a CI
   artifact, with a 14-day retention period.

Code coverage measures *execution of code paths*, not algorithm correctness.
The independent BM25 reference and invariant-based tests address different
failure modes; neither replaces a formal search evaluation.

## Dataset provenance and interpretation

The committed evaluation fixture contains 40 manually constructed Persian
helpdesk/documentation snippets and 40 associated queries. It includes two
out-of-domain negative queries. No client text, scraped personal information,
or Hamkalam customer database content is used.

The queries deliberately share domain terms with the labelled documents. This
makes the fixture useful to catch regressions in normalization and ranking,
but **not representative of unseen user queries**.

Run:

```bash
python scripts/evaluate_fixture.py
python scripts/evaluate_fixture.py --min-recall3 0.75
```

The report includes Recall@1/3/5, MRR@5, unsuccessful queries (if any),
and negative-query matches (if any). The quality threshold in CI applies
only to this synthetic smoke fixture.

A real deployment should build a separate, consented, held-out collection
containing varied spelling, half-spaces, typos, synonyms and colloquial
queries. Report per-topic metrics, query count, corpus size, confidence
intervals and failure examples; do not tune on the final test set.

## Local performance methodology

```bash
python scripts/benchmark_index.py --documents 2000 --queries 100 --seed 20261008
```

The benchmark creates seeded **synthetic** Persian and English-term
documents, then reports:
- construction wall time and peak tracemalloc bytes allocated during index
  construction (the input documents were allocated before tracing began)
- query p50/p95 and mean milliseconds after five warm-up queries
- corpus/query counts, vocabulary size, Python version and platform.

No multithreading, concurrency, persistence, realistic traffic mix, CPU
pinning, network I/O, or external vector search is covered. Because
runtime performance depends on hardware, Python release and input corpus,
there is **no universal latency SLA** here. GitHub Actions artifacts
are intended for inspection, **not** cross-run hardware benchmarks.

## Public API and deliberately missing features

- `normalize_persian`, `lexical_terms`: Unicode and heuristic tokenization.
- `chunk_text`: heading-aware approximate chunk budgets, not model tokens.
- `LexicalIndex`: in-memory BM25 inverted index, immutable after build.
- `reciprocal_rank_fusion`: deterministic ID-based fusion.
- `evaluate_rankings`: macro retrieval metrics; it does not score generated
  answers.

This project does **not** provide embeddings, source-level authorization,
multi-tenant isolation, OCR, generative LLM responses, prompt-injection
defenses or hallucination guarantees. Integrators need to enforce these
boundaries in their own systems.

## External engineering review checklist

- [ ] Reproduce the quality gate from a clean checkout.
- [ ] Verify source ownership and publication permissions.
- [ ] Run independent Persian corpus evaluation without reusing the
      development fixture for tuning.
- [ ] Measure throughput/latency under realistic query distributions.
- [ ] Profile memory on target machines and set workload limits.
- [ ] Evaluate Unicode/half-space variants and no-answer behavior.
- [ ] Add more corpus-specific stemming/synonym behavior only with
      separate evaluation and regression tests.
