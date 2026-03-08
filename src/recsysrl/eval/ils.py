import numpy as np


def ils_jaccard(list_of_sets):
    if len(list_of_sets) < 2:
        return 0.0
    sims = []
    for i in range(len(list_of_sets)):
        for j in range(i + 1, len(list_of_sets)):
            a, b = list_of_sets[i], list_of_sets[j]
            u = len(a | b)
            sims.append(0.0 if u == 0 else len(a & b) / u)
    return float(np.mean(sims))
