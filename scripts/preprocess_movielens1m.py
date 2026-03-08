#!/usr/bin/env python
import argparse
import zipfile
from pathlib import Path
import pandas as pd
from recsysrl.data.preprocess import canonicalize, apply_min_filter, remap_ids
from recsysrl.data.splits import leave_one_out
from recsysrl.data.negative_sampling import build_candidates, save_candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_zip", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.raw_zip) as zf:
        with zf.open("ml-1m/ratings.dat") as f:
            raw = pd.read_csv(f, sep="::", engine="python", names=["user_id", "item_id", "label", "timestamp"])
    df = canonicalize(raw)
    df = apply_min_filter(df)
    df = remap_ids(df, out)
    tr, va, te = leave_one_out(df)
    tr.to_parquet(out / "train.parquet")
    va.to_parquet(out / "val.parquet")
    te.to_parquet(out / "test.parquet")
    n_items = int(df.item_id.max()) + 1
    save_candidates(build_candidates(tr, va, n_items, seed=args.seed), out / "val_candidates.json")
    save_candidates(build_candidates(tr, te, n_items, seed=args.seed + 1), out / "test_candidates.json")


if __name__ == "__main__":
    main()
