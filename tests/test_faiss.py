import pytest
np = pytest.importorskip("numpy")
from recsysrl.retrieval.faiss_index import VectorIndex


def test_index_search_shape():
    emb = np.eye(4, dtype=np.float32)
    q = np.eye(2, 4, dtype=np.float32)
    ix = VectorIndex(dim=4)
    ix.build(emb)
    vals, idx = ix.search(q, k=2)
    assert vals.shape == (2,2)
    assert idx.shape == (2,2)
