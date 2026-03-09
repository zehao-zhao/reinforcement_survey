# Experiment Results

## Runs
- MF: runs/movielens1m_mf_20260308_230405
- NCF: runs/movielens1m_ncf_20260308_230408
- DRR: runs/kuairec_drr_20260308_230410

## Ranking (MovieLens-like synthetic)
| model | precision@10 | recall@10 | ndcg@10 |
|---|---:|---:|---:|
| MF | 0.010000 | 0.100000 | 0.033500 |
| NCF | 0.003333 | 0.033333 | 0.009635 |

## RL (KuaRec-like synthetic)
- avg_cumulative_reward: 8.735582

## OPE (synthetic)
| IPS | SNIPS | DR |
|---:|---:|---:|
| 0.831409 | 0.504590 | 0.542157 |

## Artifacts
- results/runs_summary.csv
- results/movielens1m_mf_eval.json
- results/movielens1m_ncf_eval.json
- results/kuairec_drr_eval.json
- results/ope_eval.json