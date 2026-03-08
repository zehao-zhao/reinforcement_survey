import pytest
pytest.importorskip("numpy")
from recsysrl.eval.ranking_eval import evaluate_from_scores


def test_metrics_bounds():
    res = evaluate_from_scores([1.0, 0.1], [[0.2, 0.3], [0.4, 0.5]], ks=(1, 2))
    for v in res.values():
        assert 0.0 <= v <= 1.0
