from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd


def build_candidates(train: pd.DataFrame, target: pd.DataFrame, n_items: int, n_neg: int = 99, seed: int = 123):
    rng = np.random.default_rng(seed)
    seen = train.groupby("user_id")["item_id"].apply(set).to_dict()
    rows = []
    for _, r in target.iterrows():
        u, pos = int(r.user_id), int(r.item_id)
        forbid = set(seen.get(u, set())) | {pos}
        pool = [i for i in range(n_items) if i not in forbid]
        neg = rng.choice(pool, size=min(n_neg, len(pool)), replace=False).tolist()
        rows.append({"user_id": u, "pos_item": pos, "neg_items": neg})
    return rows


def save_candidates(cands, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as f:
        json.dump(cands, f)
