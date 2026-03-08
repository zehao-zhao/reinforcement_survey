# Experiment Results Assessment and Publication-Grade Redesign

## Verdict on Current Results

The current runs are **not sufficient for paper publication**.

Why:
- The reported outputs were generated with **synthetic inputs** rather than the real MovieLens-1M and KuaRec datasets.
- Publication claims about recommender/RL performance require results on **real, publicly verifiable data**.
- A single end-to-end execution without multi-seed uncertainty, stronger baselines, and significance testing is typically not considered robust.

## What Was Run (Current State)

- `movielens1m_mf` train + ranking eval
- `movielens1m_ncf` train + ranking eval
- `kuairec_drr` train + DRR eval
- OPE via `scripts/run_ope.py`
- Run aggregation via `tools/summarize_results.py`

Expected output artifacts:
- `runs_summary.csv`
- `movielens1m_mf_eval.json`
- `movielens1m_ncf_eval.json`
- `kuairec_drr_eval.json`
- `ope_eval.json`

## Re-Designed Experiments (Publication-Oriented)

## 1) Data and Protocol

- Use **real datasets**: MovieLens-1M and KuaRec (official splits if available).
- Freeze preprocessing and splitting with deterministic seeds and commit split manifests.
- Ensure no user/item leakage across train/val/test.
- Keep candidate generation protocol identical across all models.

## 2) Baselines and Methods

### Static ranking (MovieLens-1M)
- Popularity baseline
- MF (matrix factorization)
- NCF
- Optional stronger baseline: LightGCN / SASRec (if implemented)

### Sequential / RL (KuaRec)
- Behavior policy / logging policy baseline
- Supervised re-ranker baseline
- DRR agent
- Ablations of DRR (e.g., no state-history module, no reward shaping)

## 3) Metrics

### Ranking metrics
- NDCG@5/10/20
- Recall@5/10/20
- HitRate@10
- MRR
- Coverage / catalog diversity

### RL / online-proxy metrics
- Episode return
- Long-term reward proxy
- Session length or retention proxy (if available)

### OPE metrics
- IPS, SNIPS, DR (already supported)
- Add confidence intervals via bootstrap
- Include effective sample size diagnostics and propensity clipping sensitivity

## 4) Statistical Rigor

- Run each experiment with **at least 5 seeds** (preferably 10).
- Report mean ± std and 95% confidence intervals.
- Perform paired significance tests vs strongest non-RL baseline (e.g., paired t-test or bootstrap paired test).
- Include robustness checks: hyperparameter sensitivity and candidate pool size sensitivity.

## 5) Experiment Matrix

Minimum matrix:
- Datasets: 2 (MovieLens-1M, KuaRec)
- Models: 5–7 total (incl. baselines + proposed)
- Seeds: 5
- Metrics: ranking + RL + OPE

Approximate run count:
- Static ranking: 2 datasets x 3-4 models x 5 seeds
- RL/OPE block: 1 dataset x 2-3 models x 5 seeds + OPE diagnostics

## 6) Reproducibility Requirements

- Save per-run config, git commit hash, and random seeds.
- Store full metric traces (`metrics.jsonl`) and final summarized tables.
- Export publication tables directly from scripts (CSV/LaTeX).
- Provide exact commands used for every table/figure.

## 7) Acceptance Criteria for “Paper-Ready”

The study is paper-ready only when all are true:
- All headline results are from real datasets (not synthetic).
- Proposed method outperforms strong baselines on primary metrics with significance.
- OPE estimates are stable across estimators and clipping settings.
- Results are reproducible from a clean checkout using documented commands.

## Suggested Immediate Next Run Plan

1. Place real MovieLens-1M and KuaRec files under `data/`.
2. Re-run preprocessing scripts with fixed seeds.
3. Run MF, NCF, and DRR with 5 seeds each.
4. Run ranking eval and DRR eval for each seed.
5. Run OPE on real logged bandit/replay data.
6. Aggregate with `tools/summarize_results.py` and generate final table artifacts.
7. Add significance testing and confidence intervals script outputs.

---

In short: your current pipeline execution validates that the code path works end-to-end, but it is a **smoke-test quality result**, not yet publication evidence.
