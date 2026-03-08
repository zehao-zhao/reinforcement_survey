#!/usr/bin/env python
import argparse
import json
from pathlib import Path
import pandas as pd
import torch
from recsysrl.utils.config import load_config
from recsysrl.models.mf import MF
from recsysrl.models.ncf import NCF
from recsysrl.eval.ranking_eval import evaluate_from_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = Path(args.run_dir)
    with (run / "meta.json").open() as f:
        meta = json.load(f)
    model = MF(meta["n_users"], meta["n_items"], cfg["model"].get("dim", 64)) if meta["model"] == "mf" else NCF(meta["n_users"], meta["n_items"], cfg["model"].get("dim", 64))
    model.load_state_dict(torch.load(run / "model.pt", map_location="cpu"))
    model.eval()
    cands = json.loads(Path(cfg["data"][f"{args.split}_candidates_path"]).read_text())
    pos_scores, neg_scores = [], []
    with torch.no_grad():
        for r in cands:
            u = torch.tensor([r["user_id"]])
            p = torch.tensor([r["pos_item"]])
            n = torch.tensor(r["neg_items"])
            pos_scores.append(float(model(u, p)[0]))
            neg_scores.append(model(u.repeat(len(n)), n).numpy())
    res = evaluate_from_scores(pos_scores, neg_scores)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
