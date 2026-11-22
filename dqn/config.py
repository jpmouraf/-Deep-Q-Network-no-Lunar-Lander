import json  # configuração dos hiperparametros
from dataclasses import asdict, dataclass, field, fields


@dataclass
class DQNConfig:
    # Ambiente
    env_id: str = "LunarLander-v3"

    # Rede neural
    hidden_sizes: list = field(default_factory=lambda: [128, 128])

    # Otimização
    lr: float = 5e-4                # taxa de aprendizado
    gamma: float = 0.99             # fator de desconto
    batch_size: int = 64
    grad_clip: float = 10.0         # norma máxima do gradiente

    # Replay buffer
    buffer_size: int = 100_000
    learning_starts: int = 5_000
    train_freq: int = 4

    # Rede alvo
    target_update_freq: int = 1_000  # copia a rede online para a alvo

    # Exploração ε-greedy
    eps_start: float = 1.0
    eps_min: float = 0.05
    eps_decay_steps: int = 100_000
    eps_schedule: str = "linear"    # "linear" ou "exponential"

    # Duração do treino e avaliação
    total_steps: int = 500_000
    eval_freq: int = 10_000         # avalia a política gulosa a cada N passos
    eval_episodes: int = 10

    device: str = "cpu"

    @classmethod
    def from_json(cls, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(
                f"Campos desconhecidos em {path}: {sorted(unknown)}")
        return cls(**data)

    def apply_overrides(self, overrides):
        types = {f.name: f.type for f in fields(self)}
        for item in overrides or []:
            key, _, raw = item.partition("=")
            if key not in types:
                raise ValueError(f"Hiperparâmetro desconhecido: {key}")
            current = getattr(self, key)
            if isinstance(current, list):
                value = json.loads(raw)
            elif isinstance(current, bool):
                value = raw.lower() in ("1", "true", "yes")
            elif isinstance(current, int):
                value = int(float(raw))
            elif isinstance(current, float):
                value = float(raw)
            else:
                value = raw
            setattr(self, key, value)
        return self

    def to_dict(self):
        return asdict(self)
