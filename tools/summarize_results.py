#!/usr/bin/env python
import argparse
import json
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()
    rows = []
    for p in Path(args.runs_dir).glob("*/metrics.jsonl"):
        vals = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        if vals:
            row = {"run": p.parent.name, **vals[-1]}
            rows.append(row)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(args.out_csv)


if __name__ == "__main__":
    main()
