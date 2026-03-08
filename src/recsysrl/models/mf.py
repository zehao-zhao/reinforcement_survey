import torch
import torch.nn as nn


class MF(nn.Module):
    def __init__(self, n_users, n_items, dim=64):
        super().__init__()
        self.u = nn.Embedding(n_users, dim)
        self.i = nn.Embedding(n_items, dim)

    def forward(self, user, item):
        return (self.u(user) * self.i(item)).sum(-1)
