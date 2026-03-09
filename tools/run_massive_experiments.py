#!/usr/bin/env python
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recsysrl.eval.ranking_eval import evaluate_from_scores
from recsysrl.models.popularity import PopularityModel
from recsysrl.ope.estimators import ips, snips, dr

PY = Path("/Users/andy/.julia/conda/3/bin/python")
BASE_OUT = ROOT / "results" / "massive"
CONFIG_OUT = BASE_OUT / "configs"
RAW_OUT = BASE_OUT / "raw"


def run_cmd(args: List[str]) -> str:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    proc = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, env=env)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"Command failed: {' '.join(args)}")
    return proc.stdout.strip()


def ensure_dirs() -> None:
    for path in [BASE_OUT, CONFIG_OUT, RAW_OUT, ROOT / "data/processed/movielens1m", ROOT / "data/processed/kuairec_small", ROOT / "data/ope"]:
        path.mkdir(parents=True, exist_ok=True)


def make_synthetic_inputs(seed: int = 123) -> None:
    rng = np.random.default_rng(seed)

    n_users, n_items = 50, 120
    rows = []
    for user_id in range(n_users):
        items = rng.choice(n_items, size=24, replace=False)
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

    def build_candidates(split_df: pd.DataFrame, cand_seed: int) -> list:
        cand_rng = np.random.default_rng(cand_seed)
        out = []
        for row in split_df.itertuples(index=False):
            pos_item = int(row.item_id)
            user_id = int(row.user_id)
            forbidden = set(user_train.get(user_id, set())) | {pos_item}
            pool = sorted(all_items - forbidden)
            neg_items = cand_rng.choice(pool, size=99, replace=(len(pool) < 99)).tolist()
            out.append({"user_id": user_id, "pos_item": pos_item, "neg_items": [int(x) for x in neg_items]})
        return out

    (ROOT / "data/processed/movielens1m/val_candidates.json").write_text(json.dumps(build_candidates(val, seed)))
    (ROOT / "data/processed/movielens1m/test_candidates.json").write_text(json.dumps(build_candidates(test, seed + 1)))

    n_users_k, n_items_k = 40, 90
    kuairec_rows = []
    for user_id in range(n_users_k):
        items = rng.choice(n_items_k, size=22, replace=False)
        for item_id in items:
            reward = float(rng.uniform(0.02, 1.0))
            kuairec_rows.append((user_id, int(item_id), reward))
    kuairec = pd.DataFrame(kuairec_rows, columns=["user_id", "item_id", "reward"])
    kuairec.to_parquet(ROOT / "data/processed/kuairec_small/interactions.parquet")
    kuairec.sample(min(1200, len(kuairec)), random_state=seed).to_parquet(ROOT / "data/processed/kuairec_small/transitions.parquet")


def write_supervised_config(name: str, model_name: str, dim: int, lr: float, epochs: int, seed: int) -> Path:
    cfg = {
        "name": name,
        "seed": int(seed),
        "data": {
            "train_path": "data/processed/movielens1m/train.parquet",
            "val_candidates_path": "data/processed/movielens1m/val_candidates.json",
            "test_candidates_path": "data/processed/movielens1m/test_candidates.json",
        },
        "model": {"name": model_name, "dim": int(dim)},
        "train": {"epochs": int(epochs), "lr": float(lr), "batch_size": 1024},
    }
    out = CONFIG_OUT / f"{name}.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def write_drr_config(name: str, lr: float, batch_size: int, seed: int) -> Path:
    cfg = {
        "name": name,
        "seed": int(seed),
        "data": {
            "transitions_path": "data/processed/kuairec_small/transitions.parquet",
            "interactions_path": "data/processed/kuairec_small/interactions.parquet",
        },
        "train": {"lr": float(lr), "batch_size": int(batch_size)},
    }
    out = CONFIG_OUT / f"{name}.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def run_popularity(split: str) -> Dict[str, float]:
    train = pd.read_parquet(ROOT / "data/processed/movielens1m/train.parquet")
    cand_path = ROOT / f"data/processed/movielens1m/{split}_candidates.json"
    cands = json.loads(cand_path.read_text())

    model = PopularityModel()
    model.fit(train.item_id.values)

    pos_scores, neg_scores = [], []
    for row in cands:
        user_id = int(row["user_id"])
        pos_item = int(row["pos_item"])
        neg_items = np.array(row["neg_items"], dtype=np.int64)
        pos_scores.append(float(model.score(user_id, np.array([pos_item], dtype=np.int64))[0]))
        neg_scores.append(model.score(user_id, neg_items))

    return evaluate_from_scores(pos_scores, neg_scores)


def supervised_sweep() -> pd.DataFrame:
    rows = []
    models = ["mf", "ncf"]
    dims = [16, 64]
    lrs = [1e-3, 5e-4]
    epochs_list = [2, 4]
    seeds = [11, 22, 33]

    pop_val = run_popularity("val")
    pop_test = run_popularity("test")
    (BASE_OUT / "popularity_val.json").write_text(json.dumps(pop_val, indent=2))
    (BASE_OUT / "popularity_test.json").write_text(json.dumps(pop_test, indent=2))

    for model in models:
        for dim in dims:
            for lr in lrs:
                for epochs in epochs_list:
                    for seed in seeds:
                        exp_name = f"{model}_d{dim}_lr{lr}_ep{epochs}_s{seed}"
                        cfg = write_supervised_config(exp_name, model, dim, lr, epochs, seed)
                        run_dir = run_cmd([str(PY), "scripts/train_supervised.py", "--config", str(cfg.relative_to(ROOT))]).splitlines()[-1]
                        val_out = run_cmd([
                            str(PY), "scripts/eval_ranking.py", "--config", str(cfg.relative_to(ROOT)), "--split", "val", "--run_dir", run_dir
                        ])
                        test_out = run_cmd([
                            str(PY), "scripts/eval_ranking.py", "--config", str(cfg.relative_to(ROOT)), "--split", "test", "--run_dir", run_dir
                        ])
                        val_metrics = json.loads(val_out)
                        test_metrics = json.loads(test_out)
                        (RAW_OUT / f"{exp_name}_val.json").write_text(json.dumps(val_metrics, indent=2))
                        (RAW_OUT / f"{exp_name}_test.json").write_text(json.dumps(test_metrics, indent=2))

                        row = {
                            "exp_name": exp_name,
                            "family": "ranking",
                            "model": model,
                            "dim": dim,
                            "lr": lr,
                            "epochs": epochs,
                            "seed": seed,
                            "run_dir": run_dir,
                        }
                        row.update({f"val_{k}": v for k, v in val_metrics.items()})
                        row.update({f"test_{k}": v for k, v in test_metrics.items()})
                        rows.append(row)

    pop_row_val = {"exp_name": "popularity_baseline", "family": "ranking", "model": "popularity", "split": "val", **pop_val}
    pop_row_test = {"exp_name": "popularity_baseline", "family": "ranking", "model": "popularity", "split": "test", **pop_test}
    pd.DataFrame([pop_row_val, pop_row_test]).to_csv(BASE_OUT / "popularity_metrics.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(BASE_OUT / "ranking_sweep_all_runs.csv", index=False)
    summary = (
        df.groupby(["model", "dim", "lr", "epochs"]) [[c for c in df.columns if c.startswith("test_")]]
        .agg(["mean", "std"]) 
        .reset_index()
    )
    summary.columns = ["_".join([str(x) for x in col if str(x) != ""]) for col in summary.columns.values]
    summary.to_csv(BASE_OUT / "ranking_sweep_summary.csv", index=False)
    return df


def drr_sweep() -> pd.DataFrame:
    rows = []
    lrs = [1e-3, 5e-4, 2e-3]
    batch_sizes = [64, 128, 256]
    seeds = [11, 22, 33, 44, 55]
    eval_grid = [(50, 20), (100, 50), (200, 100)]

    for lr in lrs:
        for batch_size in batch_sizes:
            for seed in seeds:
                exp_name = f"drr_lr{lr}_bs{batch_size}_s{seed}"
                cfg = write_drr_config(exp_name, lr, batch_size, seed)
                run_dir = run_cmd([str(PY), "scripts/train_drr.py", "--config", str(cfg.relative_to(ROOT))]).splitlines()[-1]

                for num_episodes, horizon in eval_grid:
                    out = run_cmd([
                        str(PY), "scripts/eval_drr_kuairec.py", "--config", str(cfg.relative_to(ROOT)), "--run_dir", run_dir,
                        "--num_episodes", str(num_episodes), "--horizon", str(horizon)
                    ])
                    metrics = json.loads(out)
                    raw_name = f"{exp_name}_ep{num_episodes}_hz{horizon}.json"
                    (RAW_OUT / raw_name).write_text(json.dumps(metrics, indent=2))

                    rows.append({
                        "exp_name": exp_name,
                        "family": "rl",
                        "method": "drr",
                        "lr": lr,
                        "batch_size": batch_size,
                        "seed": seed,
                        "num_episodes": num_episodes,
                        "horizon": horizon,
                        "run_dir": run_dir,
                        **metrics,
                    })

    df = pd.DataFrame(rows)
    df.to_csv(BASE_OUT / "drr_sweep_all_runs.csv", index=False)
    summary = (
        df.groupby(["lr", "batch_size", "num_episodes", "horizon"]) ["avg_cumulative_reward"]
        .agg(["mean", "std", "min", "max"]) 
        .reset_index()
    )
    summary.columns = ["_".join([str(x) for x in col if str(x) != ""]) for col in summary.columns.values]
    summary.to_csv(BASE_OUT / "drr_sweep_summary.csv", index=False)
    return df


def ope_sweep() -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    rows = []
    dataset_sizes = [500, 1000, 2500, 5000]
    mismatch_scales = [0.5, 1.0, 2.0]
    clips = [1e-6, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1]

    for n in dataset_sizes:
        for scale in mismatch_scales:
            reward = rng.uniform(0, 1, size=n)
            pi_b = np.clip(rng.beta(2, 5, size=n), 1e-4, 1)
            pi_e = np.clip(pi_b ** (1 / max(scale, 1e-6)), 1e-4, 1)
            q_hat = np.clip(reward + rng.normal(0, 0.05, size=n), 0, 1)
            v_hat = np.clip(rng.uniform(0, 1, size=n), 0, 1)

            csv_name = f"ope_n{n}_m{scale}.csv"
            pd.DataFrame({"reward": reward, "pi_e": pi_e, "pi_b": pi_b, "q_hat": q_hat, "v_hat": v_hat}).to_csv(ROOT / "data/ope" / csv_name, index=False)

            for clip in clips:
                est_ips = ips(reward, pi_e, pi_b, clip=clip)
                est_snips = snips(reward, pi_e, pi_b, clip=clip)
                est_dr = dr(reward, pi_e, pi_b, q_hat, v_hat, clip=clip)

                boot = []
                for _ in range(200):
                    idx = rng.integers(0, n, size=n)
                    boot.append([
                        ips(reward[idx], pi_e[idx], pi_b[idx], clip=clip),
                        snips(reward[idx], pi_e[idx], pi_b[idx], clip=clip),
                        dr(reward[idx], pi_e[idx], pi_b[idx], q_hat[idx], v_hat[idx], clip=clip),
                    ])
                boot_arr = np.array(boot)
                ci = {
                    "ips_low": float(np.quantile(boot_arr[:, 0], 0.025)),
                    "ips_high": float(np.quantile(boot_arr[:, 0], 0.975)),
                    "snips_low": float(np.quantile(boot_arr[:, 1], 0.025)),
                    "snips_high": float(np.quantile(boot_arr[:, 1], 0.975)),
                    "dr_low": float(np.quantile(boot_arr[:, 2], 0.025)),
                    "dr_high": float(np.quantile(boot_arr[:, 2], 0.975)),
                }

                record = {
                    "family": "ope",
                    "n": n,
                    "mismatch_scale": scale,
                    "clip": clip,
                    "ips": est_ips,
                    "snips": est_snips,
                    "dr": est_dr,
                    **ci,
                }
                rows.append(record)

    df = pd.DataFrame(rows)
    df.to_csv(BASE_OUT / "ope_clip_bootstrap_sweep.csv", index=False)
    df.to_json(BASE_OUT / "ope_clip_bootstrap_sweep.json", orient="records", indent=2)
    summary = df.groupby(["n", "mismatch_scale"])[["ips", "snips", "dr"]].agg(["mean", "std", "min", "max"]).reset_index()
    summary.columns = ["_".join([str(x) for x in col if str(x) != ""]) for col in summary.columns.values]
    summary.to_csv(BASE_OUT / "ope_summary.csv", index=False)
    return df


def write_experiment_notes(ranking_df: pd.DataFrame, drr_df: pd.DataFrame, ope_df: pd.DataFrame) -> None:
    design = {
        "what_are_you_doing": {
            "ranking_benchmarks": {
                "methods": ["popularity", "mf", "ncf"],
                "grid": {"dims": [16, 64], "lrs": [1e-3, 5e-4], "epochs": [2, 4], "seeds": [11, 22, 33]},
                "splits": ["val", "test"],
            },
            "rl_benchmarks": {
                "method": "drr",
                "grid": {"lrs": [1e-3, 5e-4, 2e-3], "batch_sizes": [64, 128, 256], "seeds": [11, 22, 33, 44, 55]},
                "eval_grid": [{"num_episodes": 50, "horizon": 20}, {"num_episodes": 100, "horizon": 50}, {"num_episodes": 200, "horizon": 100}],
            },
            "ope_benchmarks": {
                "estimators": ["IPS", "SNIPS", "DR"],
                "dataset_sizes": [500, 1000, 2500, 5000],
                "mismatch_scales": [0.5, 1.0, 2.0],
                "clips": [1e-6, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1],
                "bootstrap_samples": 200,
            },
        },
        "how_to_solve": {
            "pipeline": [
                "Generate/prepare benchmark-ready datasets",
                "Train ranking models across hyperparameter x seed sweeps",
                "Evaluate ranking on candidate pools",
                "Train and evaluate RL agent across sweep",
                "Run OPE diagnostics with clipping and bootstrap CIs",
                "Persist every run plus aggregate summaries",
            ],
            "notes": "Current run uses synthetic benchmark-compatible datasets because real datasets were not provided in workspace.",
        },
        "counts": {
            "ranking_runs": int(len(ranking_df)),
            "drr_eval_runs": int(len(drr_df)),
            "ope_runs": int(len(ope_df)),
        },
    }
    (BASE_OUT / "experiment_design_and_method.json").write_text(json.dumps(design, indent=2))

    lines = [
        "# Massive Experiment Design + Method",
        "",
        "## Experiment design (what)",
        "- Ranking: Popularity + MF + NCF benchmark matrix across seeds/hparams.",
        "- RL: DRR sweep across LR/batch-size/seeds with multiple episode/horizon settings.",
        "- OPE: IPS/SNIPS/DR over dataset-size and policy-mismatch benchmarks with clipping + bootstrap CIs.",
        "",
        "## Experiment method (how)",
        "1. Prepare synthetic benchmark-compatible data in data/processed and data/ope.",
        "2. Auto-generate config files per sweep point.",
        "3. Train/eval each point and write raw JSON outputs.",
        "4. Aggregate all runs into wide CSV/JSON summary artifacts.",
        "",
        "## Run counts",
        f"- Ranking runs: {len(ranking_df)}",
        f"- DRR eval runs: {len(drr_df)}",
        f"- OPE runs: {len(ope_df)}",
        "",
        "## Caveat",
        "- This sweep is extensive and broad, but still synthetic-data benchmarking, not publication-ready real-dataset evidence.",
    ]
    (BASE_OUT / "experiment_design_and_method.md").write_text("\n".join(lines))


def main() -> None:
    ensure_dirs()
    make_synthetic_inputs(seed=123)

    ranking_df = supervised_sweep()
    drr_df = drr_sweep()
    ope_df = ope_sweep()

    write_experiment_notes(ranking_df, drr_df, ope_df)

    all_rows = []
    if len(ranking_df):
        for row in ranking_df.to_dict(orient="records"):
            row["benchmark_family"] = "ranking"
            all_rows.append(row)
    if len(drr_df):
        for row in drr_df.to_dict(orient="records"):
            row["benchmark_family"] = "rl"
            all_rows.append(row)
    if len(ope_df):
        for row in ope_df.to_dict(orient="records"):
            row["benchmark_family"] = "ope"
            all_rows.append(row)

    pd.DataFrame(all_rows).to_csv(BASE_OUT / "all_results_unified.csv", index=False)
    pd.DataFrame(all_rows).to_json(BASE_OUT / "all_results_unified.json", orient="records", indent=2)

    manifest = {
        "base_dir": str(BASE_OUT),
        "key_files": [
            "ranking_sweep_all_runs.csv",
            "ranking_sweep_summary.csv",
            "popularity_metrics.csv",
            "drr_sweep_all_runs.csv",
            "drr_sweep_summary.csv",
            "ope_clip_bootstrap_sweep.csv",
            "ope_summary.csv",
            "all_results_unified.csv",
            "all_results_unified.json",
            "experiment_design_and_method.json",
            "experiment_design_and_method.md",
        ],
    }
    (BASE_OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(str(BASE_OUT / "manifest.json"))


if __name__ == "__main__":
    main()
