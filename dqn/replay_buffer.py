"""Replay buffer circular implementado com arrays NumPy pré-alocados."""

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, rng=None):
        self.capacity = capacity
        self.rng = rng if rng is not None else np.random.default_rng()

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)

        self.pos = 0    # próxima posição de escrita
        self.size = 0   # quantidade de transições válidas

    def add(self, obs, action, reward, next_obs, terminated):
        self.obs[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_obs[self.pos] = next_obs
        self.terminated[self.pos] = float(terminated)

        # Escrita circular: quando enche, sobrescreve as transições mais antigas.
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = self.rng.integers(0, self.size, size=batch_size)
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.terminated[idx],
        )

    def __len__(self):
        return self.size
