#!/usr/bin/env python
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from recsysrl.utils.config import load_config
from recsysrl.rl.drr.networks import Actor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--num_episodes", type=int, default=100)
    ap.add_argument("--horizon", type=int, default=50)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = Path(args.run_dir)
    with (run / "meta.json").open() as f:
        meta = json.load(f)
    actor = Actor(meta["n_users"], meta["n_items"])
    actor.load_state_dict(torch.load(run / "actor.pt", map_location="cpu"))
    actor.eval()

    df = pd.read_parquet(cfg["data"]["interactions_path"])
    table = {(int(r.user_id), int(r.item_id)): float(r.reward) for _, r in df.iterrows()}
    users = list(range(meta["n_users"]))
    total = []
    for ep in range(args.num_episodes):
        u = users[ep % len(users)]
        s = torch.zeros(1, meta["n_users"]); s[0, u] = 1
        ep_r = 0.0
        for _ in range(args.horizon):
            with torch.no_grad():
                logits = actor(s).numpy()[0]
            it = int(np.argmax(logits))
            ep_r += table.get((u, it), 0.0)
        total.append(ep_r)
    print(json.dumps({"avg_cumulative_reward": float(np.mean(total))}, indent=2))


if __name__ == "__main__":
    main()
