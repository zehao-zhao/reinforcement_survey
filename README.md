# recsysrl_scaffold

Reproducible experiment scaffold for recommendation + RL + OPE.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python scripts/smoke_test_pipeline.py
```

## Core workflow

1. Preprocess data to canonical schema.
2. Build deterministic splits and fixed candidate pools.
3. Train models (supervised or RL).
4. Evaluate ranking / reward / OPE.
5. Summarize seed runs into paper-ready CSV.
