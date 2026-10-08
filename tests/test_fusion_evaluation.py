import math

import pytest

from persian_retrieval import QueryCase, evaluate_rankings, reciprocal_rank_fusion


def test_rrf_promotes_consensus_and_breaks_ties_deterministically() -> None:
    ranking = reciprocal_rank_fusion([["a", "b"], ["b", "c"]])
    assert ranking[0][0] == "b"
    assert {row[0] for row in ranking} == {"a", "b", "c"}


def test_duplicate_within_a_list_counts_once() -> None:
    assert reciprocal_rank_fusion([["a", "a"]]) == [("a", 1 / 61)]


def test_weights_change_fused_order() -> None:
    rows = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], weights=[3, 1])
    assert rows[0][0] == "a"


@pytest.mark.parametrize("weights", [[-1], [math.nan], [math.inf]])
def test_reject_invalid_weights(weights: list[float]) -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], weights=weights)


def test_evaluate_macro_recall_and_mrr() -> None:
    cases = [QueryCase("q1", frozenset({"a", "b"})), QueryCase("q2", frozenset({"c"}))]
    result = evaluate_rankings(cases, [["a", "x"], ["x", "c"]], k=2)
    assert result.recall_at_k == pytest.approx(0.75)
    assert result.mrr_at_k == pytest.approx(0.75)


def test_empty_evaluation_and_invalid_cases() -> None:
    assert evaluate_rankings([], [], k=5).queries == 0
    with pytest.raises(ValueError):
        QueryCase("bad", frozenset())
    with pytest.raises(ValueError):
        evaluate_rankings([QueryCase("ok", frozenset({"a"}))], [])
