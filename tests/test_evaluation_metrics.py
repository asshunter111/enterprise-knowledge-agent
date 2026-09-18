import pytest

from evaluation.metrics import hit_at_k, mean_reciprocal_rank, paired_improvement


def test_retrieval_metrics_use_expected_denominator_and_rank():
    assert hit_at_k(["a", "b", "c"], "c", 3)
    assert not hit_at_k(["a", "b", "c"], "c", 2)
    assert mean_reciprocal_rank([(["a", "b"], "b"), (["x"], "missing")]) == 0.25


def test_paired_comparison_requires_same_cases():
    assert paired_improvement([False, True], [True, False]) == {
        "improved": 1,
        "regressed": 1,
        "unchanged": 0,
    }
    with pytest.raises(ValueError):
        paired_improvement([True], [])
