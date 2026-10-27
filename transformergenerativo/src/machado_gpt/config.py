"""Configurações centrais do projeto (caminhos, modelo base, livros, hiperparâmetros)."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

# Raiz do projeto (src/machado_gpt/config.py -> sobe 2 níveis). Pode ser sobrescrita.
ROOT = Path(os.environ.get("MACHADO_ROOT", Path(__file__).resolve().parents[2]))
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
CHECKPOINTS = ROOT / "checkpoints"
RESULTS = ROOT / "results"

# Pasta onde o treino salva o modelo final e o melhor checkpoint.
RUN_DIR = CHECKPOINTS / "machado-gpt2"
TUNED_DIR = RUN_DIR / "best"

# Modelo base: GPT-2 small (124M) pré-treinado em português (Wikipedia PT).
# Se o download do tokenizer falhar, tente a cópia "egonrp/gpt2-small-portuguese",
# que inclui o tokenizer.json.
BASE_MODEL = os.environ.get("MACHADO_BASE_MODEL", "pierreguillou/gpt2-small-portuguese")

# Livros de Machado de Assis (domínio público) no Project Gutenberg: nome -> ebook nº.
# Para usar mais obras, adicione o nº do ebook aqui OU coloque um .txt em data/raw/.
GUTENBERG_BOOKS: dict[str, int] = {
    "dom_casmurro": 55752,
    "memorias_posthumas_de_braz_cubas": 54829,
    "quincas_borba": 55682,
}

# Prompts usados na avaliação qualitativa e como exemplos na interface.
DEFAULT_PROMPTS: list[str] = [
    "Capitu olhou para mim e",
    "Era uma tarde de domingo quando",
    "Dom Casmurro, ao chegar em casa,",
    "A vida, meu caro leitor, é",
    "Bentinho, disse-me José Dias,",
    "O dinheiro não faz a felicidade, mas",
]


@dataclass
class TrainConfig:
    """Hiperparâmetros de treino. Os padrões foram pensados para uma RTX 3050 (4-8 GB)."""

    block_size: int = 256          # tamanho do contexto de treino (tokens)
    batch_size: int = 4            # micro-batch (reduza para 2 se der falta de memória)
    grad_accum_steps: int = 4      # batch efetivo = batch_size * grad_accum_steps
    epochs: int = 4
    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    max_grad_norm: float = 1.0
    eval_every: int = 20           # avalia a cada N passos do otimizador
    val_fraction: float = 0.1      # fração dos blocos reservada para validação
    holdout_book: str | None = "quincas_borba"  # livro inteiro fora do treino (teste); None = usa todos
    gradient_checkpointing: bool = True  # troca velocidade por memória
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)
