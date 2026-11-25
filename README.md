# Deep Q-Network no Lunar Lander

## Requisitos

- Python **3.12** (testado com 3.12.3; 3.10 e 3.11 devem funcionar)
- Treino em CPU; GPU não é necessária

Dependências (`requirements.txt`):

| Pacote              | Versão       | Uso                                           |
| ------------------- | ------------ | --------------------------------------------- |
| gymnasium[box2d]    | 1.4.0        | ambiente Lunar Lander (inclui Box2D e pygame) |
| torch               | 2.15.0 (CPU) | redes neurais                                 |
| numpy               | 2.5.5        | replay buffer e operações numéricas           |
| stable-baselines3   | 2.9.2        | DQN de referência para comparação             |
| matplotlib          | 3.11.2       | gráficos                                      |
| pandas              | 3.1.1        | leitura dos resultados                        |
| notebook, ipykernel | 7.6.5, 7.4.0 | notebook de análise                           |

## Instalação

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # Linux/macOS

pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

O `--extra-index-url` é necessário porque o projeto usa a versão CPU do PyTorch.
Se a instalação do Box2D falhar, rode `pip install swig` e repita o comando.

Teste rápido (avalia o modelo final já treinado; deve mostrar média acima de 200):

```bash
python -m dqn.evaluate results/final/seed_0/best_model.pt --episodes 10
```

## Estrutura

```
dqn/                       implementação do DQN
  config.py                hiperparâmetros (dataclass, leitura de JSON, --set)
  network.py               rede Q: MLP 8 -> 128 -> 128 -> 4
  replay_buffer.py         replay buffer circular (NumPy)
  agent.py                 ε-greedy, passo de otimização, rede alvo
  train.py                 loop de treino com avaliação periódica
  evaluate.py              protocolo de avaliação (usado também para o SB3)
configs/
  base.json                configuração base
  final.json               configuração final (base + lr=1e-3, batch=128, eps_start=0.5)
experiments/
  hparam_search.py         busca de hiperparâmetros em paralelo
  sb3_baseline.py          DQN do Stable Baselines3 com configuração padrão
analysis.ipynb             gráficos e tabelas do relatório
results/                   resultados de todos os experimentos
```

Cada execução salva em `results/<experimento>/seed_<n>/`:

| Arquivo                            | Conteúdo                                                                       |
| ---------------------------------- | ------------------------------------------------------------------------------ |
| `config.json`                      | hiperparâmetros e seed usados                                                  |
| `eval.csv`                         | avaliação a cada 10 mil passos (10 episódios, ε = 0, seeds de avaliação fixas) |
| `train_episodes.csv`               | recompensa de cada episódio de treino                                          |
| `best_model.pt` / `final_model.pt` | melhor modelo e modelo ao fim do treino (`.zip` para o SB3)                    |
| `train.log`                        | saída do treino                                                                |

## Como reproduzir os resultados

Todos os comandos devem ser executados **na raiz do projeto**, com o ambiente virtual ativado.
Os tempos são aproximados, medidos num notebook com 12 núcleos lógicos.

### 1. Treinar um agente

```bash
python -m dqn.train --config configs/base.json --seed 0
```

Hiperparâmetros podem ser sobrescritos sem editar o JSON:

```bash
python -m dqn.train --seed 0 --set lr=1e-3 gamma=0.95 --name meu_experimento
```

Um treino de 500 mil passos leva ~5 minutos sozinho em 1 núcleo.

### 2. Busca de hiperparâmetros (seção 3.2)

```bash
python -m experiments.hparam_search
```

Roda a base e 17 variações (cada uma alterando **um** hiperparâmetro em relação à base),
com as seeds 0, 1 e 2: 54 treinos, 10 em paralelo por padrão (~1h45). Treinos já concluídos são
pulados, então o comando pode ser interrompido e retomado. Opções úteis:

```bash
python -m experiments.hparam_search --list              # mostra o que será rodado
python -m experiments.hparam_search --workers 4         # menos processos em paralelo
python -m experiments.hparam_search --only lr_1e-3      # só um experimento
```

Variações testadas (valores da base entre parênteses):

| Hiperparâmetro           | Valores                                                            |
| ------------------------ | ------------------------------------------------------------------ |
| Taxa de aprendizado      | 1e-4, 1e-3 (5e-4)                                                  |
| Fator de desconto γ      | 0.95, 0.999 (0.99)                                                 |
| Tamanho do replay buffer | 10 mil, 500 mil (100 mil)                                          |
| Tamanho do batch         | 32, 128 (64)                                                       |
| Atualização da rede alvo | a cada 100, 10 mil passos (1000)                                   |
| ε inicial                | 0.5 (1.0)                                                          |
| Decaimento do ε          | linear em 25 mil e 250 mil passos, exponencial (linear em 100 mil) |
| ε mínimo                 | 0.01, 0.1 (0.05)                                                   |
| Passos de treino         | 1 milhão (500 mil)                                                 |

### 3. Configuração final

```bash
python -m dqn.train --config configs/final.json --seed 0
python -m dqn.train --config configs/final.json --seed 1
python -m dqn.train --config configs/final.json --seed 2
```

### 4. Comparação com o Stable Baselines3 (seção 3.3)

```bash
python -m experiments.sb3_baseline
```

Treina o `DQN("MlpPolicy", env)` do Stable Baselines3 **sem alterar nenhum hiperparâmetro**,
com as seeds 0, 1 e 2 e 500 mil passos, avaliado com o mesmo protocolo do nosso DQN
(~30 min). Resultados em `results/sb3_default/`.

### 5. Gráficos e tabelas

```bash
jupyter notebook analysis.ipynb
```

Execute todas as células (_Run All_). O notebook lê todas as pastas em `results/`,
gera os gráficos e a tabela-resumo e salva tudo em `results/figures/`.
No VS Code, selecione o kernel do `.venv` ao abrir o notebook.

### 6. Assistir o agente

```bash
python -m dqn.evaluate results/final/seed_0/best_model.pt --episodes 3 --render
```

## Reprodutibilidade

- Cada treino usa uma seed (`--seed`) que fixa o Python, o NumPy, o PyTorch e o ambiente.
- A avaliação usa sempre os mesmos episódios (seeds 10000 a 10009), tanto para o
  nosso DQN quanto para o SB3.
- Cada treino roda com 1 thread do PyTorch (`torch.set_num_threads(1)`).
- Os resultados podem variar um pouco entre máquinas e sistemas operacionais por
  diferenças de ponto flutuante, mas as tendências se mantêm.
