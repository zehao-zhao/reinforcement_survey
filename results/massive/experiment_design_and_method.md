# Massive Experiment Design + Method

## Experiment design (what)
- Ranking: Popularity + MF + NCF benchmark matrix across seeds/hparams.
- RL: DRR sweep across LR/batch-size/seeds with multiple episode/horizon settings.
- OPE: IPS/SNIPS/DR over dataset-size and policy-mismatch benchmarks with clipping + bootstrap CIs.

## Experiment method (how)
1. Prepare synthetic benchmark-compatible data in data/processed and data/ope.
2. Auto-generate config files per sweep point.
3. Train/eval each point and write raw JSON outputs.
4. Aggregate all runs into wide CSV/JSON summary artifacts.

## Run counts
- Ranking runs: 48
- DRR eval runs: 135
- OPE runs: 72

## Caveat
- This sweep is extensive and broad, but still synthetic-data benchmarking, not publication-ready real-dataset evidence.