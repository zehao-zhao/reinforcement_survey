import pytest
pytest.importorskip("numpy")

from recsysrl.rl.linucb import LinUCB
from recsysrl.rl.slateq import slateq_value


def test_rl_basic():
    m = LinUCB(3)
    score = m.score(__import__('numpy').eye(3))
    assert len(score) == 3
    assert slateq_value([1,2,3], [0,2]) == 4.0
