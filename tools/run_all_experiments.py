#!/usr/bin/env python
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PY = Path("/Users/andy/.julia/conda/3/bin/python")


def run_cmd(args, capture_path: Optional[Path] = None) -> str:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    proc = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, env=env)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"Command failed: {' '.join(args)}")
    out = proc.stdout.strip()
    if capture_path is not None:
        capture_path.parent.mkdir(parents=True, exist_ok=True)
        capture_path.write_text(proc.stdout)
    return out


def make_synthetic_inputs() -> None:
    rng = np.random.default_rng(123)
    (ROOT / "data/processed/movielens1m").mkdir(parents=True, exist_ok=True)
    (ROOT / "data/processed/kuairec_small").mkdir(parents=True, exist_ok=True)
    (ROOT / "data/ope").mkdir(parents=True, exist_ok=True)
    (ROOT / "results").mkdir(parents=True, exist_ok=True)

    n_users, n_items = 30, 80
    rows = []
    for user_id in range(n_users):
        items = rng.choice(n_items, size=20, replace=False)
        timestamp = 1
        for item_id in items:
            rows.append((user_id, int(item_id), timestamp, 1.0))
            timestamp += 1

    interactions = pd.DataFrame(rows, columns=["user_id", "item_id", "timestamp", "label"])
    train_parts, val_parts, test_parts = [], [], []
    for _, frame in interactions.groupby("user_id"):
        frame = frame.sort_values("timestamp")
        train_parts.append(frame.iloc[:-2])
        val_parts.append(frame.iloc[-2:-1])
        test_parts.append(frame.iloc[-1:])

    train = pd.concat(train_parts, ignore_index=True)
    val = pd.concat(val_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)

    train.to_parquet(ROOT / "data/processed/movielens1m/train.parquet")
    val.to_parquet(ROOT / "data/processed/movielens1m/val.parquet")
    test.to_parquet(ROOT / "data/processed/movielens1m/test.parquet")

    user_train = train.groupby("user_id")["item_id"].apply(set).to_dict()
    all_items = set(range(n_items))

    def build_candidates(split_df: pd.DataFrame, seed: int) -> list[dict]:
        cand_rng = np.random.default_rng(seed)
        out = []
        for row in split_df.itertuples(index=False):
            pos_item = int(row.item_id)
            user_id = int(row.user_id)
            forbidden = set(user_train.get(user_id, set())) | {pos_item}
            pool = sorted(all_items - forbidden)
            neg_items = cand_rng.choice(pool, size=99, replace=(len(pool) < 99)).tolist()
            out.append(
                {
                    "user_id": user_id,
                    "pos_item": pos_item,
                    "neg_items": [int(item) for item in neg_items],
                }
            )
        return out

    (ROOT / "data/processed/movielens1m/val_candidates.json").write_text(json.dumps(build_candidates(val, 123)))
    (ROOT / "data/processed/movielens1m/test_candidates.json").write_text(json.dumps(build_candidates(test, 124)))

    n_users_k, n_items_k = 25, 60
    kuairec_rows = []
    for user_id in range(n_users_k):
        items = rng.choice(n_items_k, size=18, replace=False)
        for item_id in items:
            reward = float(rng.uniform(0.05, 1.0))
            kuairec_rows.append((user_id, int(item_id), reward))

    kuairec = pd.DataFrame(kuairec_rows, columns=["user_id", "item_id", "reward"])
    kuairec.to_parquet(ROOT / "data/processed/kuairec_small/interactions.parquet")
    kuairec.sample(min(400, len(kuairec)), random_state=123).to_parquet(
        ROOT / "data/processed/kuairec_small/transitions.parquet"
    )

    n_ope = 500
    ope_df = pd.DataFrame(
        {
            "reward": rng.uniform(0, 1, size=n_ope),
            "pi_e": rng.uniform(0.05, 1, size=n_ope),
            "pi_b": rng.uniform(0.05, 1, size=n_ope),
            "q_hat": rng.uniform(0, 1, size=n_ope),
            "v_hat": rng.uniform(0, 1, size=n_ope),
        }
    )
    ope_df.to_csv(ROOT / "data/ope/synthetic_ope.csv", index=False)


def main() -> None:
    make_synthetic_inputs()

    mf_run = run_cmd(
        [
            str(PY),
            "scripts/train_supervised.py",
            "--config",
            "configs/experiments/movielens1m_mf.yaml",
        ],
        capture_path=ROOT / "results/mf_run_dir.txt",
    ).splitlines()[-1]

    run_cmd(
        [
            str(PY),
            "scripts/eval_ranking.py",
            "--config",
            "configs/experiments/movielens1m_mf.yaml",
            "--split",
            "test",
            "--run_dir",
            mf_run,
        ],
        capture_path=ROOT / "results/movielens1m_mf_eval.json",
    )

    ncf_run = run_cmd(
        [
            str(PY),
            "scripts/train_supervised.py",
            "--config",
            "configs/experiments/movielens1m_ncf.yaml",
        ],
        capture_path=ROOT / "results/ncf_run_dir.txt",
    ).splitlines()[-1]

    run_cmd(
        [
            str(PY),
            "scripts/eval_ranking.py",
            "--config",
            "configs/experiments/movielens1m_ncf.yaml",
            "--split",
            "test",
            "--run_dir",
            ncf_run,
        ],
        capture_path=ROOT / "results/movielens1m_ncf_eval.json",
    )

    drr_run = run_cmd(
        [
            str(PY),
            "scripts/train_drr.py",
            "--config",
            "configs/experiments/kuairec_drr.yaml",
        ],
        capture_path=ROOT / "results/drr_run_dir.txt",
    ).splitlines()[-1]

    run_cmd(
        [
            str(PY),
            "scripts/eval_drr_kuairec.py",
            "--config",
            "configs/experiments/kuairec_drr.yaml",
            "--run_dir",
            drr_run,
        ],
        capture_path=ROOT / "results/kuairec_drr_eval.json",
    )

    run_cmd(
        [str(PY), "scripts/run_ope.py", "--input_csv", "data/ope/synthetic_ope.csv"],
        capture_path=ROOT / "results/ope_eval.json",
    )

    run_cmd(
        [
            str(PY),
            "tools/summarize_results.py",
            "--runs_dir",
            "runs",
            "--out_csv",
            "results/runs_summary.csv",
        ]
    )

    mf_eval = json.loads((ROOT / "results/movielens1m_mf_eval.json").read_text())
    ncf_eval = json.loads((ROOT / "results/movielens1m_ncf_eval.json").read_text())
    drr_eval = json.loads((ROOT / "results/kuairec_drr_eval.json").read_text())
    ope_eval = json.loads((ROOT / "results/ope_eval.json").read_text())

    lines = [
        "# Experiment Results",
        "",
        "## Runs",
        f"- MF: {mf_run}",
        f"- NCF: {ncf_run}",
        f"- DRR: {drr_run}",
        "",
        "## Ranking (MovieLens-like synthetic)",
        "| model | precision@10 | recall@10 | ndcg@10 |",
        "|---|---:|---:|---:|",
        f"| MF | {mf_eval.get('precision@10', float('nan')):.6f} | {mf_eval.get('recall@10', float('nan')):.6f} | {mf_eval.get('ndcg@10', float('nan')):.6f} |",
        f"| NCF | {ncf_eval.get('precision@10', float('nan')):.6f} | {ncf_eval.get('recall@10', float('nan')):.6f} | {ncf_eval.get('ndcg@10', float('nan')):.6f} |",
        "",
        "## RL (KuaRec-like synthetic)",
        f"- avg_cumulative_reward: {drr_eval.get('avg_cumulative_reward', float('nan')):.6f}",
        "",
        "## OPE (synthetic)",
        "| IPS | SNIPS | DR |",
        "|---:|---:|---:|",
        f"| {ope_eval.get('ips', float('nan')):.6f} | {ope_eval.get('snips', float('nan')):.6f} | {ope_eval.get('dr', float('nan')):.6f} |",
        "",
        "## Artifacts",
        "- results/runs_summary.csv",
        "- results/movielens1m_mf_eval.json",
        "- results/movielens1m_ncf_eval.json",
        "- results/kuairec_drr_eval.json",
        "- results/ope_eval.json",
    ]
    (ROOT / "results/experiment_results.md").write_text("\n".join(lines))

    print("results/experiment_results.md")


if __name__ == "__main__":
    main()
