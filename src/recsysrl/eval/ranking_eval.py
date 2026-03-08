from __future__ import annotations
import numpy as np
from .metrics import precision_at_k, recall_at_k, ndcg_at_k


def evaluate_from_scores(pos_scores, neg_scores, ks=(5, 10, 20)):
    ranks = []
    for p, ns in zip(pos_scores, neg_scores):
        ranks.append(int((np.array(ns) >= p).sum()))
    out = {}
    for k in ks:
        out[f"precision@{k}"] = float(precision_at_k(ranks, k))
        out[f"recall@{k}"] = float(recall_at_k(ranks, k))
        out[f"ndcg@{k}"] = float(ndcg_at_k(ranks, k))
    return out
