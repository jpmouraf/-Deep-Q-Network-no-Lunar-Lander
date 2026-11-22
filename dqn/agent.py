"""Agente DQN: rede online, rede alvo, política ε-greedy e passo de otimização."""

import copy
import math

import numpy as np
import torch
import torch.nn.functional as F

from dqn.network import QNetwork


def epsilon_at(step, cfg):
    """Valor de ε no passo `step`, segundo o decaimento escolhido na config."""
    if cfg.eps_schedule == "linear":
        frac = min(step / cfg.eps_decay_steps, 1.0)
        return cfg.eps_start + frac * (cfg.eps_min - cfg.eps_start)
    if cfg.eps_schedule == "exponential":
        return cfg.eps_min + (cfg.eps_start - cfg.eps_min) * math.exp(-step / cfg.eps_decay_steps)
    raise ValueError(f"eps_schedule inválido: {cfg.eps_schedule}")


class DQNAgent:
    def __init__(self, obs_dim, n_actions, cfg, rng=None):
        self.cfg = cfg
        self.n_actions = n_actions
        self.device = torch.device(cfg.device)
        self.rng = rng if rng is not None else np.random.default_rng()

        self.online = QNetwork(
            obs_dim, n_actions, cfg.hidden_sizes).to(self.device)
        # A rede alvo começa como cópia exata da online e só muda em sync_target().
        self.target = copy.deepcopy(self.online)
        self.target.requires_grad_(False)

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)

    @torch.no_grad()
    def greedy_action(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
        return int(self.online(obs_t).argmax(dim=1).item())

    def act(self, obs, epsilon):
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))
        return self.greedy_action(obs)

    def update(self, batch):
        obs, actions, rewards, next_obs, terminated = (
            torch.as_tensor(x, device=self.device) for x in batch
        )

        q = self.online(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target(next_obs).max(dim=1).values
            y = rewards + self.cfg.gamma * (1.0 - terminated) * next_q

        loss = F.smooth_l1_loss(q, y)

        self.optimizer.zero_grad()
        loss.backward()
        if self.cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.online.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        return loss.item()

    def sync_target(self):
        """Hard update: copia os pesos da rede online para a rede alvo."""
        self.target.load_state_dict(self.online.state_dict())

    def save(self, path):
        torch.save({"online": self.online.state_dict(),
                   "config": self.cfg.to_dict()}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ckpt["online"])
        self.sync_target()
