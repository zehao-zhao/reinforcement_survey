from __future__ import annotations
import json
from pathlib import Path
import pandas as pd


CANON = ["user_id", "item_id", "timestamp", "label"]


def canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in CANON if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    out = df[CANON].copy()
    out = out.dropna(subset=["user_id", "item_id", "timestamp"]) 
    out["timestamp"] = out["timestamp"].astype("int64")
    out["label"] = out["label"].astype(float)
    return out


def apply_min_filter(df: pd.DataFrame, min_user=5, min_item=5) -> pd.DataFrame:
    uok = df["user_id"].value_counts()
    iok = df["item_id"].value_counts()
    return df[df["user_id"].isin(uok[uok >= min_user].index) & df["item_id"].isin(iok[iok >= min_item].index)].copy()


def remap_ids(df: pd.DataFrame, out_dir: str | Path) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    umap = {u: i for i, u in enumerate(sorted(df.user_id.unique()))}
    imap = {it: i for i, it in enumerate(sorted(df.item_id.unique()))}
    out = df.copy()
    out["user_id"] = out["user_id"].map(umap)
    out["item_id"] = out["item_id"].map(imap)
    with (out_dir / "mappings.json").open("w") as f:
        json.dump({"num_users": len(umap), "num_items": len(imap)}, f)
    return out
