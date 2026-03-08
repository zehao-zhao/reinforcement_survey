import numpy as np


def ips(reward, pi_e, pi_b, clip=1e-3):
    w = pi_e / np.maximum(pi_b, clip)
    return float(np.mean(w * reward))


def snips(reward, pi_e, pi_b, clip=1e-3):
    w = pi_e / np.maximum(pi_b, clip)
    return float(np.sum(w * reward) / np.maximum(np.sum(w), 1e-12))


def dr(reward, pi_e, pi_b, q_hat, v_hat, clip=1e-3):
    w = pi_e / np.maximum(pi_b, clip)
    return float(np.mean(v_hat + w * (reward - q_hat)))
