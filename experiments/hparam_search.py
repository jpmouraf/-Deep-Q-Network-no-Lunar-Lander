"""Busca de hiperparâmetros: varia UM hiperparâmetro por vez em relação a configs/base.json.

Cada variação roda com várias seeds, e vários treinos rodam em paralelo (1 núcleo cada).
Treinos já concluídos (com final_model.pt) são pulados, então dá para interromper e
retomar a busca rodando o mesmo comando de novo.

Uso (a partir da raiz do projeto):
    python -m experiments.hparam_search                  # busca completa
    python -m experiments.hparam_search --workers 6      # limita os processos em paralelo
    python -m experiments.hparam_search --only lr_1e-3 gamma_0.95
    python -m experiments.hparam_search --list           # só mostra o que seria rodado
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# nome do experimento -> sobrescritas em relação à base (--set)
# Valores da base: lr=5e-4, gamma=0.99, buffer=100k, batch=64, target=1000,
# eps 1.0 -> 0.05 linear em 100k passos, 500k passos de treino.
EXPERIMENTS = {
    "base": [],
    # Taxa de aprendizado
    "lr_1e-4": ["lr=1e-4"],
    "lr_1e-3": ["lr=1e-3"],
    # Fator de desconto
    "gamma_0.95": ["gamma=0.95"],
    "gamma_0.999": ["gamma=0.999"],
    # Tamanho do replay buffer
    "buffer_10k": ["buffer_size=10000"],
    "buffer_500k": ["buffer_size=500000"],
    # Tamanho do batch
    "batch_32": ["batch_size=32"],
    "batch_128": ["batch_size=128"],
    # Frequência de atualização da rede alvo (em passos de ambiente)
    "target_100": ["target_update_freq=100"],
    "target_10k": ["target_update_freq=10000"],
    # Política ε-greedy: valor inicial, decaimento e valor mínimo
    "eps_start_0.5": ["eps_start=0.5"],
    "eps_decay_25k": ["eps_decay_steps=25000"],
    "eps_decay_250k": ["eps_decay_steps=250000"],
    "eps_exp_50k": ["eps_schedule=exponential", "eps_decay_steps=50000"],
    "eps_min_0.01": ["eps_min=0.01"],
    "eps_min_0.1": ["eps_min=0.1"],
    # Número de passos de treino
    "steps_1m": ["total_steps=1000000"],
}

SEEDS = [0, 1, 2]


def run_one(name, overrides, seed, results_dir):
    out_dir = Path(results_dir) / name / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "dqn.train", "--config", "configs/base.json",
           "--seed", str(seed), "--name", name, "--results-dir", str(results_dir)]
    if overrides:
        cmd += ["--set", *overrides]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    start = time.time()
    with open(out_dir / "train.log", "w", encoding="utf-8") as log:
        code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
    return name, seed, code, time.time() - start


def main():
    parser = argparse.ArgumentParser(description="Busca de hiperparâmetros em paralelo.")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    parser.add_argument("--only", nargs="*", help="roda só estes experimentos")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--list", action="store_true", help="só lista os treinos pendentes")
    args = parser.parse_args()

    names = args.only or list(EXPERIMENTS)
    unknown = set(names) - set(EXPERIMENTS)
    if unknown:
        sys.exit(f"Experimentos desconhecidos: {sorted(unknown)}")

    jobs, skipped = [], 0
    for name in names:
        for seed in args.seeds:
            if (Path(args.results_dir) / name / f"seed_{seed}" / "final_model.pt").exists():
                skipped += 1
            else:
                jobs.append((name, EXPERIMENTS[name], seed))
    # Treinos mais longos primeiro, para não sobrarem no fim rodando sozinhos.
    jobs.sort(key=lambda j: "total_steps=1000000" not in j[1])

    print(f"{len(jobs)} treinos pendentes, {skipped} já concluídos, {args.workers} em paralelo.")
    if args.list:
        for name, overrides, seed in jobs:
            print(f"  {name:<16} seed {seed}  {' '.join(overrides)}")
        return

    start, failures = time.time(), []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, n, o, s, args.results_dir) for n, o, s in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            name, seed, code, secs = fut.result()
            status = "ok" if code == 0 else f"ERRO (código {code})"
            if code != 0:
                failures.append(f"{name}/seed_{seed}")
            print(f"[{i}/{len(jobs)}] {name} seed {seed}: {status} em {secs/60:.1f} min "
                  f"| total {(time.time() - start)/60:.1f} min", flush=True)

    print(f"Busca concluída em {(time.time() - start)/60:.1f} min.")
    if failures:
        print("Falharam (veja train.log em cada pasta):", ", ".join(failures))
        sys.exit(1)


if __name__ == "__main__":
    main()
