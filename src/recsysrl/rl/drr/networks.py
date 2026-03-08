import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, action_dim), nn.Tanh())

    def forward(self, s):
        return self.net(s)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim + action_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, s, a):
        return self.net(torch.cat([s, a], -1)).squeeze(-1)
