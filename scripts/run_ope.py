#!/usr/bin/env python
import argparse
import json
import pandas as pd
from recsysrl.ope.estimators import ips, snips, dr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.input_csv)
    res = {
        "ips": ips(df.reward.values, df.pi_e.values, df.pi_b.values),
        "snips": snips(df.reward.values, df.pi_e.values, df.pi_b.values),
        "dr": dr(df.reward.values, df.pi_e.values, df.pi_b.values, df.q_hat.values, df.v_hat.values),
    }
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
