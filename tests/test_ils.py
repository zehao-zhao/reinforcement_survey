import pytest
pytest.importorskip("numpy")
from recsysrl.eval.ils import ils_jaccard


def test_ils_bounds():
    v = ils_jaccard([{1,2}, {2,3}, {3,4}])
    assert 0.0 <= v <= 1.0
