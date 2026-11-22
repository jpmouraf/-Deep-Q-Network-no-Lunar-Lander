"""Avaliação de políticas.

Uso pela linha de comando (avaliar um modelo salvo):
    python -m dqn.evaluate results/base/seed_0/best_model.pt --episodes 100
    python -m dqn.evaluate results/base/seed_0/best_model.pt --episodes 3 --render
"""

import argparse

import gymnasium as gym
import numpy as np
import torch

EVAL_SEED_BASE = 10_000  # seeds de avaliação separadas das de treino


def evaluate_policy(policy_fn, env_id="LunarLander-v3", n_episodes=10,
                    seed_base=EVAL_SEED_BASE, render_mode=None):
    # Roda `n_episodes` episódios com a política dada e retorna as recompensas totais.
    env = gym.make(env_id, render_mode=render_mode)
    returns = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed_base + i)
        done, ep_return = False, 0.0
        while not done:
            obs, reward, terminated, truncated, _ = env.step(policy_fn(obs))
            ep_return += reward
            done = terminated or truncated
        returns.append(ep_return)
    env.close()
    return np.array(returns)


def main():
    from dqn.agent import DQNAgent
    from dqn.config import DQNConfig

    parser = argparse.ArgumentParser(description="Avalia um modelo DQN salvo.")
    parser.add_argument("model", help="caminho para o .pt")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--render", action="store_true",
                        help="mostra a janela do jogo")
    args = parser.parse_args()

    ckpt = torch.load(args.model, map_location="cpu")
    cfg = DQNConfig(**ckpt["config"])
    cfg.device = "cpu"
    env = gym.make(cfg.env_id)
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, cfg)
    agent.load(args.model)

    returns = evaluate_policy(agent.greedy_action, cfg.env_id, args.episodes,
                              render_mode="human" if args.render else None)
    print(f"{args.episodes} episódios: média = {returns.mean():.1f} ± {returns.std():.1f} "
          f"| min = {returns.min():.1f} | max = {returns.max():.1f} "
          f"| sucesso (>= 200) = {(returns >= 200).mean():.0%}")


if __name__ == "__main__":
    main()
