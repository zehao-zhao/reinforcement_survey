import numpy as np


def mean_std(values):
    arr = np.array(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def paired_bootstrap_ci(a, b, n_boot=2000, alpha=0.05, seed=123):
    rng = np.random.default_rng(seed)
    a, b = np.array(a), np.array(b)
    n = len(a)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs.append((a[idx] - b[idx]).mean())
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
