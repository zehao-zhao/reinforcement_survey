#!/usr/bin/env python
import argparse
import json
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from recsysrl.utils.config import load_config
from recsysrl.utils.seed import set_seed
from recsysrl.utils.logging import init_run_dir, append_jsonl
from recsysrl.models.mf import MF
from recsysrl.models.ncf import NCF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 123))
    run = init_run_dir(name=cfg.get("name", "supervised"))
    (run / "config.yaml").write_text(Path(args.config).read_text())

    train = pd.read_parquet(cfg["data"]["train_path"])
    n_users = int(train.user_id.max()) + 1
    n_items = int(train.item_id.max()) + 1
    model_name = cfg["model"]["name"]
    model = MF(n_users, n_items, cfg["model"].get("dim", 64)) if model_name == "mf" else NCF(n_users, n_items, cfg["model"].get("dim", 64))
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"].get("lr", 1e-3))
    ds = TensorDataset(torch.tensor(train.user_id.values), torch.tensor(train.item_id.values), torch.tensor(train.label.values, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=cfg["train"].get("batch_size", 1024), shuffle=True)
    for ep in range(cfg["train"].get("epochs", 2)):
        total = 0.0
        for u, i, y in dl:
            pred = model(u, i)
            loss = torch.nn.functional.mse_loss(pred, y)
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.item())
        append_jsonl(run / "metrics.jsonl", {"epoch": ep, "train_loss": total / max(len(dl), 1)})
    torch.save(model.state_dict(), run / "model.pt")
    with (run / "meta.json").open("w") as f:
        json.dump({"model": model_name, "n_users": n_users, "n_items": n_items}, f)
    print(run)


if __name__ == "__main__":
    main()
