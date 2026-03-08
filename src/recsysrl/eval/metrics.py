import numpy as np


def precision_at_k(ranks, k):
    hits = (np.array(ranks) < k).astype(float)
    return hits.mean() / k


def recall_at_k(ranks, k):
    return (np.array(ranks) < k).mean()


def ndcg_at_k(ranks, k):
    r = np.array(ranks)
    gains = (r < k).astype(float)
    dcg = gains / np.log2(r + 2)
    return dcg.mean()
