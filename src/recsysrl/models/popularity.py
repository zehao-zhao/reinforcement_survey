import numpy as np


class PopularityModel:
    def __init__(self):
        self.counts = None

    def fit(self, item_ids):
        n = int(np.max(item_ids)) + 1
        self.counts = np.bincount(item_ids, minlength=n).astype(float)

    def score(self, user_id, item_ids):
        return self.counts[item_ids]
