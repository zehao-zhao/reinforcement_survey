import numpy as np

try:
    import faiss
except Exception:  # optional dependency
    faiss = None


class VectorIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self._emb = None
        self._index = None

    def build(self, emb: np.ndarray):
        emb = emb.astype("float32")
        if faiss is None:
            self._emb = emb
        else:
            idx = faiss.IndexFlatIP(self.dim)
            idx.add(emb)
            self._index = idx

    def search(self, q: np.ndarray, k: int):
        q = q.astype("float32")
        if faiss is None:
            scores = q @ self._emb.T
            topk = np.argsort(-scores, axis=1)[:, :k]
            vals = np.take_along_axis(scores, topk, axis=1)
            return vals, topk
        return self._index.search(q, k)
