import pytest
np = pytest.importorskip("numpy")
from recsysrl.ope.estimators import ips, snips, dr


def test_ope_sanity():
    r = np.array([1.0, 0.0, 1.0])
    pe = np.array([0.5, 0.5, 0.5])
    pb = np.array([0.5, 0.5, 0.5])
    assert abs(ips(r, pe, pb) - r.mean()) < 1e-9
    assert abs(snips(r, pe, pb) - r.mean()) < 1e-9
    assert isinstance(dr(r, pe, pb, r, r), float)
