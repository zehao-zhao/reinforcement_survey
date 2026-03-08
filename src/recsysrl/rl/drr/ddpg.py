import torch
import torch.nn.functional as F
from .networks import Actor, Critic


class DDPG:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.t_actor = Actor(state_dim, action_dim)
        self.t_critic = Critic(state_dim, action_dim)
        self.t_actor.load_state_dict(self.actor.state_dict())
        self.t_critic.load_state_dict(self.critic.state_dict())
        self.ao = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.co = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma = gamma

    def train_step(self, batch, tau=0.005):
        s, a, r, ns, d = [torch.tensor(x, dtype=torch.float32) for x in zip(*batch)]
        with torch.no_grad():
            na = self.t_actor(ns)
            y = r + self.gamma * (1 - d) * self.t_critic(ns, na)
        q = self.critic(s, a)
        closs = F.mse_loss(q, y)
        self.co.zero_grad(); closs.backward(); self.co.step()
        aloss = -self.critic(s, self.actor(s)).mean()
        self.ao.zero_grad(); aloss.backward(); self.ao.step()
        for p, tp in zip(self.actor.parameters(), self.t_actor.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)
        for p, tp in zip(self.critic.parameters(), self.t_critic.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)
        return float(closs.item()), float(aloss.item())
