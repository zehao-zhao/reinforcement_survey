#!/usr/bin/env python
import argparse
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small_matrix_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--max_transitions", type=int, default=200000)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.small_matrix_csv)
    cols = {c.lower(): c for c in df.columns}
    u = cols.get("user_id", list(df.columns)[0])
    i = cols.get("item_id", list(df.columns)[1])
    r = cols.get("watch_ratio", list(df.columns)[2])
    x = df[[u, i, r]].rename(columns={u: "user_id", i: "item_id", r: "reward"})
    x["reward"] = x["reward"].clip(0, 1)
    x.to_parquet(out / "interactions.parquet")
    x.sample(min(args.max_transitions, len(x)), random_state=args.seed).to_parquet(out / "transitions.parquet")


if __name__ == "__main__":
    main()
