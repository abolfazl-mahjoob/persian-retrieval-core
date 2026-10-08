"""Repeatable, *synthetic* labelled corpus evaluation; zero external APIs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from persian_retrieval import LexicalIndex, QueryCase, evaluate_rankings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("benchmarks/persian_helpdesk_smoke_v1.json"),
    )
    parser.add_argument("--min-recall3", type=float, default=0.0)
    args = parser.parse_args()
    corpus = json.loads(args.fixture.read_text(encoding="utf-8"))
    index = LexicalIndex(corpus["documents"])
    cases = [
        QueryCase(item["query"], frozenset(item["relevant_ids"]))
        for item in corpus["queries"]
    ]
    rankings = [
        [hit.document_id for hit in index.search(case.query, limit=index.document_count)]
        for case in cases
    ]
    k1 = evaluate_rankings(cases, rankings, k=1)
    k3 = evaluate_rankings(cases, rankings, k=3)
    k5 = evaluate_rankings(cases, rankings, k=5)
    misses = [
        {"query": case.query, "expected": sorted(case.relevant_ids), "ranked_top3": rank[:3]}
        for case, rank in zip(cases, rankings, strict=True)
        if not case.relevant_ids.intersection(rank[:3])
    ]
    negative_queries = corpus.get("negative_queries", [])
    negative_hits = {
        query: [hit.document_id for hit in index.search(query, limit=3)]
        for query in negative_queries
    }
    report = {
        "dataset": corpus["metadata"]["name"],
        "dataset_source": corpus["metadata"]["source"],
        "scope": corpus["metadata"]["scope"],
        "documents": index.document_count,
        "queries": len(cases),
        "recall_at_1": round(k1.recall_at_k, 4),
        "recall_at_3": round(k3.recall_at_k, 4),
        "recall_at_5": round(k5.recall_at_k, 4),
        "mrr_at_5": round(k5.mrr_at_k, 4),
        "misses_at_3": misses,
        "negative_query_hits": negative_hits,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if k3.recall_at_k < args.min_recall3:
        print("Recall@3 below synthetic smoke-test floor", flush=True)
        return 1
    if any(negative_hits.values()):
        print("Negative smoke query unexpectedly matched", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
