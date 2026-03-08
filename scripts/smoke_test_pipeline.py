#!/usr/bin/env python
import sys


def main():
    try:
        import tempfile
        import pandas as pd
        from recsysrl.data.preprocess import canonicalize, apply_min_filter, remap_ids
        from recsysrl.data.splits import leave_one_out
    except ModuleNotFoundError as e:
        print(f"smoke skipped: missing dependency ({e})")
        return 0

    df = pd.DataFrame({
        "user_id": [1,1,1,2,2,2],
        "item_id": [10,11,12,10,12,13],
        "timestamp": [1,2,3,1,2,3],
        "label": [1,1,1,1,1,1],
    })
    x = canonicalize(df)
    x = apply_min_filter(x, 1, 1)
    with tempfile.TemporaryDirectory() as td:
        y = remap_ids(x, td)
    tr, va, te = leave_one_out(y)
    assert len(tr) == 2 and len(va) == 2 and len(te) == 2
    print("smoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
