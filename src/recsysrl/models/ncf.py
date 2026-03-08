import torch
import torch.nn as nn


class NCF(nn.Module):
    def __init__(self, n_users, n_items, dim=64, hidden=128):
        super().__init__()
        self.u = nn.Embedding(n_users, dim)
        self.i = nn.Embedding(n_items, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, hidden), nn.ReLU(), nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1)
        )

    def forward(self, user, item):
        x = torch.cat([self.u(user), self.i(item)], -1)
        return self.mlp(x).squeeze(-1)
