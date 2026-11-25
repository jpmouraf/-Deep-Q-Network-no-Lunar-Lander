"""DQN do Stable Baselines3 com a configuração PADRÃO, para comparação (seção 3.3).

Usa o mesmo número de passos, as mesmas seeds e o MESMO protocolo de avaliação do
nosso DQN (dqn.evaluate.evaluate_policy: 10 episódios gulosos com seeds fixas a cada
10 mil passos). Os resultados são salvos no mesmo formato, em results/sb3_default/,
então o notebook de análise os lê automaticamente.

Uso (a partir da raiz do projeto):
    python -m experiments.sb3_baseline                     # seeds 0, 1 e 2 em paralelo
    python -m experiments.sb3_baseline --seeds 0 --steps 500000
"""

import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

EVAL_FREQ = 10_000
EVAL_EPISODES = 10
SOLVED = 200


def run_seed(seed, total_steps, out_root):
    # Imports dentro da função: cada processo filho carrega o seu próprio torch.
    import gymnasium as gym
    import torch
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import BaseCallback

    from dqn.evaluate import evaluate_policy

    torch.set_num_threads(1)
    out_dir = Path(out_root) / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make("LunarLander-v3")
    # Configuração padrão: nenhum hiperparâmetro é alterado.
    model = DQN("MlpPolicy", env, seed=seed, device="cpu", verbose=0)

    defaults = {
        "algo": "stable_baselines3.DQN (padrão)", "seed": seed, "total_steps": total_steps,
        "learning_rate": model.learning_rate, "gamma": model.gamma,
        "buffer_size": model.buffer_size, "learning_starts": model.learning_starts,
        "batch_size": model.batch_size, "train_freq": model.train_freq.frequency,
        "gradient_steps": model.gradient_steps,
        "target_update_interval": model.target_update_interval,
        "exploration_initial_eps": model.exploration_initial_eps,
        "exploration_final_eps": model.exploration_final_eps,
        "exploration_fraction": model.exploration_fraction,
        "max_grad_norm": model.max_grad_norm,
        "net_arch": str(model.policy.net_arch),
    }
    (out_dir / "config.json").write_text(json.dumps(defaults, indent=2), encoding="utf-8")

    class EvalAndLog(BaseCallback):
        """Avalia com o nosso protocolo e grava eval.csv / train_episodes.csv."""

        def _on_training_start(self):
            self.start = time.time()
            self.best = -np.inf
            self.episode = 0
            self.eval_f = open(out_dir / "eval.csv", "w", newline="", encoding="utf-8")
            self.eval_w = csv.writer(self.eval_f)
            self.eval_w.writerow(["step", "mean_return", "std_return", "min_return", "max_return",
                                  "success_rate", "epsilon", "mean_loss", "elapsed_s"])
            self.ep_f = open(out_dir / "train_episodes.csv", "w", newline="", encoding="utf-8")
            self.ep_w = csv.writer(self.ep_f)
            self.ep_w.writerow(["step", "episode", "return", "length", "epsilon"])

        def _on_step(self):
            step = self.num_timesteps
            for info in self.locals.get("infos", []):
                if "episode" in info:  # preenchido pelo wrapper Monitor do SB3
                    self.episode += 1
                    self.ep_w.writerow([step, self.episode, round(info["episode"]["r"], 2),
                                        info["episode"]["l"], round(self.model.exploration_rate, 4)])
            if step % EVAL_FREQ == 0:
                policy = lambda obs: int(self.model.predict(obs, deterministic=True)[0])
                r = evaluate_policy(policy, "LunarLander-v3", EVAL_EPISODES)
                elapsed = time.time() - self.start
                self.eval_w.writerow([step, round(r.mean(), 2), round(r.std(), 2),
                                      round(r.min(), 2), round(r.max(), 2),
                                      round((r >= SOLVED).mean(), 2),
                                      round(self.model.exploration_rate, 4), "", round(elapsed, 1)])
                self.eval_f.flush()
                self.ep_f.flush()
                if r.mean() > self.best:
                    self.best = r.mean()
                    self.model.save(out_dir / "best_model")
                print(f"[sb3 s{seed}] passo {step:>7} | eval {r.mean():7.1f} ± {r.std():5.1f} "
                      f"| eps {self.model.exploration_rate:.3f} | {elapsed:.0f}s", flush=True)
            return True

        def _on_training_end(self):
            self.eval_f.close()
            self.ep_f.close()

    model.learn(total_timesteps=total_steps, callback=EvalAndLog())
    model.save(out_dir / "final_model")
    env.close()
    return seed


def main():
    parser = argparse.ArgumentParser(description="DQN do Stable Baselines3 (config padrão).")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=500_000,
                        help="use o mesmo total_steps da config final do nosso DQN")
    parser.add_argument("--out", default="results/sb3_default")
    args = parser.parse_args()

    with ProcessPoolExecutor(max_workers=len(args.seeds)) as pool:
        for seed in pool.map(run_seed, args.seeds, [args.steps] * len(args.seeds),
                             [args.out] * len(args.seeds)):
            print(f"seed {seed} concluída", flush=True)


if __name__ == "__main__":
    main()
