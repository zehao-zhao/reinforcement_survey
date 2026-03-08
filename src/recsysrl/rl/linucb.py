import numpy as np


class LinUCB:
    def __init__(self, d, alpha=1.0):
        self.A = np.eye(d)
        self.b = np.zeros((d,))
        self.alpha = alpha

    def score(self, x):
        Ainv = np.linalg.inv(self.A)
        theta = Ainv @ self.b
        mean = x @ theta
        unc = np.sqrt(np.sum((x @ Ainv) * x, axis=1))
        return mean + self.alpha * unc

    def update(self, x, r):
        self.A += np.outer(x, x)
        self.b += r * x
