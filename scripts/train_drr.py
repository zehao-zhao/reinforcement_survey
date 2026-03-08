#!/usr/bin/env python
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from recsysrl.utils.config import load_config
from recsysrl.utils.seed import set_seed
from recsysrl.utils.logging import init_run_dir, append_jsonl
from recsysrl.rl.drr.ddpg import DDPG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 123))
    run = init_run_dir(name=cfg.get("name", "drr"))
    tr = pd.read_parquet(cfg["data"]["transitions_path"])
    n_users = int(tr.user_id.max()) + 1
    n_items = int(tr.item_id.max()) + 1
    state_dim = n_users
    action_dim = n_items
    agent = DDPG(state_dim, action_dim, lr=cfg["train"].get("lr", 1e-3))

    batch = []
    for _, r in tr.iterrows():
        s = np.zeros(state_dim, dtype=np.float32); s[int(r.user_id)] = 1
        a = np.zeros(action_dim, dtype=np.float32); a[int(r.item_id)] = 1
        ns = s.copy()
        batch.append((s, a, float(r.reward), ns, 0.0))
        if len(batch) >= cfg["train"].get("batch_size", 256):
            cl, al = agent.train_step(batch)
            append_jsonl(run / "metrics.jsonl", {"critic_loss": cl, "actor_loss": al})
            batch = []
    torch.save(agent.actor.state_dict(), run / "actor.pt")
    with (run / "meta.json").open("w") as f:
        json.dump({"n_users": n_users, "n_items": n_items}, f)
    print(run)


if __name__ == "__main__":
    main()
