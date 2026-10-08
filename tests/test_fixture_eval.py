"""Ground truth is documented, synthetic, and evaluated without an external service."""
from __future__ import annotations

import json
from pathlib import Path

from persian_retrieval import LexicalIndex, QueryCase, evaluate_rankings

DATA = Path(__file__).resolve().parents[1] / "benchmarks/persian_helpdesk_smoke_v1.json"


def test_fixture_is_labelled_and_scoped() -> None:
    raw = json.loads(DATA.read_text(encoding="utf-8"))
    assert raw["metadata"]["scope"].startswith("lexical retrieval")
    ids = set(raw["documents"])
    assert len(ids) >= 30
    assert len(raw["queries"]) >= 30
    assert all(set(row["relevant_ids"]) <= ids for row in raw["queries"])


def test_smoke_recall_is_explicitly_measured() -> None:
    raw = json.loads(DATA.read_text(encoding="utf-8"))
    index = LexicalIndex(raw["documents"])
    cases = [QueryCase(row["query"], frozenset(row["relevant_ids"])) for row in raw["queries"]]
    rankings = [
        [hit.document_id for hit in index.search(row.query)]
        for row in cases
    ]
    outcome = evaluate_rankings(cases, rankings, k=3)
    assert outcome.queries == len(cases)
    assert outcome.recall_at_k >= 0.75


def test_out_of_corpus_query_does_not_invent_a_result() -> None:
    raw = json.loads(DATA.read_text(encoding="utf-8"))
    index = LexicalIndex(raw["documents"])
    for query in raw["negative_queries"]:
        assert index.search(query) == []
