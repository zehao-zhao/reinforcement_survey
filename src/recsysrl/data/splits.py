from __future__ import annotations
import pandas as pd


def leave_one_out(df: pd.DataFrame):
    df = df.sort_values(["user_id", "timestamp"], kind="stable")
    train, val, test = [], [], []
    for _, g in df.groupby("user_id", sort=False):
        if len(g) < 3:
            train.append(g)
            continue
        train.append(g.iloc[:-2])
        val.append(g.iloc[-2:-1])
        test.append(g.iloc[-1:])
    c = [pd.concat(x).reset_index(drop=True) if x else pd.DataFrame(columns=df.columns) for x in (train, val, test)]
    return c[0], c[1], c[2]
