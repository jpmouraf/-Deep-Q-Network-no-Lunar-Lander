"""Loop de treinamento do DQN.

Uso:
    python -m dqn.train --config configs/base.json --seed 0
    python -m dqn.train --config configs/base.json --seed 0 --set lr=1e-3 --name lr_1e-3

Saídas em results/<name>/seed_<seed>/:
    config.json          hiperparâmetros usados
    eval.csv             avaliação periódica da política gulosa (ε = 0)
    train_episodes.csv   recompensa de cada episódio de treino (com exploração)
    best_model.pt        modelo com melhor média de avaliação
    final_model.pt       modelo ao fim do treino
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from dqn.agent import DQNAgent, epsilon_at
from dqn.config import DQNConfig
from dqn.evaluate import evaluate_policy
from dqn.replay_buffer import ReplayBuffer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(cfg, seed, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps({**cfg.to_dict(), "seed": seed}, indent=2), encoding="utf-8")

    set_seed(seed)
    # Rede pequena: 1 thread é mais rápido e permite vários treinos em paralelo.
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)

    env = gym.make(cfg.env_id)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = DQNAgent(obs_dim, n_actions, cfg, rng=rng)
    buffer = ReplayBuffer(cfg.buffer_size, obs_dim, rng=rng)

    eval_file = open(out_dir / "eval.csv", "w", newline="", encoding="utf-8")
    eval_log = csv.writer(eval_file)
    eval_log.writerow(["step", "mean_return", "std_return", "min_return", "max_return",
                       "success_rate", "epsilon", "mean_loss", "elapsed_s"])
    ep_file = open(out_dir / "train_episodes.csv", "w", newline="", encoding="utf-8")
    ep_log = csv.writer(ep_file)
    ep_log.writerow(["step", "episode", "return", "length", "epsilon"])

    obs, _ = env.reset(seed=seed)
    ep_return, ep_len, episode = 0.0, 0, 0
    losses = []
    best_mean = -np.inf
    start = time.time()

    for step in range(1, cfg.total_steps + 1):
        # 1) Interage com o ambiente usando ε-greedy.
        eps = epsilon_at(step, cfg)
        action = agent.act(obs, eps)
        next_obs, reward, terminated, truncated, _ = env.step(action)

        # 2) Armazena a transição. Truncamento (limite de 1000 passos) não é fim
        #    real do episódio, então o alvo ainda deve fazer bootstrap nesse caso.
        buffer.add(obs, action, reward, next_obs, terminated)
        obs = next_obs
        ep_return += reward
        ep_len += 1

        if terminated or truncated:
            episode += 1
            ep_log.writerow([step, episode, round(ep_return, 2), ep_len, round(eps, 4)])
            obs, _ = env.reset()
            ep_return, ep_len = 0.0, 0

        # 3) Aprende a partir de um minibatch do replay buffer.
        if step >= cfg.learning_starts and step % cfg.train_freq == 0:
            losses.append(agent.update(buffer.sample(cfg.batch_size)))

        # 4) Atualiza a rede alvo periodicamente.
        if step % cfg.target_update_freq == 0:
            agent.sync_target()

        # 5) Avalia a política gulosa periodicamente.
        if step % cfg.eval_freq == 0:
            returns = evaluate_policy(agent.greedy_action, cfg.env_id, cfg.eval_episodes)
            mean_loss = float(np.mean(losses)) if losses else float("nan")
            losses = []
            elapsed = time.time() - start
            eval_log.writerow([step, round(returns.mean(), 2), round(returns.std(), 2),
                               round(returns.min(), 2), round(returns.max(), 2),
                               round((returns >= 200).mean(), 2), round(eps, 4),
                               round(mean_loss, 5), round(elapsed, 1)])
            eval_file.flush()
            ep_file.flush()
            if returns.mean() > best_mean:
                best_mean = returns.mean()
                agent.save(out_dir / "best_model.pt")
            print(f"[{out_dir.parent.name} s{seed}] passo {step:>7} | eval {returns.mean():7.1f} "
                  f"± {returns.std():5.1f} | eps {eps:.3f} | loss {mean_loss:.4f} "
                  f"| ep {episode} | {elapsed:.0f}s", flush=True)

    agent.save(out_dir / "final_model.pt")
    eval_file.close()
    ep_file.close()
    env.close()
    return best_mean


def main():
    parser = argparse.ArgumentParser(description="Treina um agente DQN.")
    parser.add_argument("--config", default="configs/base.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default=None, help="nome do experimento (padrão: nome do config)")
    parser.add_argument("--set", nargs="*", default=[], metavar="CAMPO=VALOR",
                        help="sobrescreve hiperparâmetros, ex.: --set lr=1e-3 gamma=0.95")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    cfg = DQNConfig.from_json(args.config).apply_overrides(args.set)
    name = args.name or Path(args.config).stem
    out_dir = Path(args.results_dir) / name / f"seed_{args.seed}"
    best = train(cfg, args.seed, out_dir)
    print(f"Concluído. Melhor média de avaliação: {best:.1f}. Resultados em {out_dir}")


if __name__ == "__main__":
    main()
