"""Métricas e comparação entre o GPT-2 original e o GPT-2 ajustado em Machado de Assis."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd
import torch

from .config import BASE_MODEL, DATA_PROCESSED, DEFAULT_PROMPTS, RESULTS, RUN_DIR, TUNED_DIR, TrainConfig
from .data import DataSplits, load_books, make_splits
from .generate import default_device, generate_text, load_model

WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

# Grafias de época presentes nas edições do Gutenberg (ex.: "elle", "annos", "quasi").
# Um modelo que absorveu o estilo de Machado deve usá-las bem mais que o GPT-2 original.
ARCHAIC_MARKERS = frozenset({
    "elle", "elles", "ella", "ellas", "cousa", "cousas", "annos", "quasi", "mezes",
    "poz", "veiu", "commigo", "comsigo", "tambem", "athe", "outr'ora", "mettido",
    "philosophia", "idéa", "idéas", "psychologia",
})


# ----------------------------------------------------------------------------
# Perplexidade
# ----------------------------------------------------------------------------
@torch.no_grad()
def mean_loss(model, blocks: torch.Tensor, batch_size: int = 8, amp_dtype: torch.dtype | None = None) -> float:
    """Cross-entropy média por token (em nats) sobre `blocks` [N, T]."""
    model.eval()
    device = next(model.parameters()).device
    total, count = 0.0, 0
    for i in range(0, len(blocks), batch_size):
        x = blocks[i : i + batch_size].to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            loss = model(input_ids=x, labels=x).loss
        n = x.shape[0] * (x.shape[1] - 1)       # o rótulo do 1º token não existe
        total += loss.float().item() * n
        count += n
    return total / count


def perplexity_table(models: dict[str, tuple], splits: DataSplits, batch_size: int = 8) -> pd.DataFrame:
    """Loss e perplexidade de cada modelo nos conjuntos de validação (e teste, se houver).

    Todos os modelos usam o MESMO tokenizer (o do GPT-2 PT), então os números são comparáveis.
    """
    rows = []
    for name, (model, _tok) in models.items():
        row = {"modelo": name}
        sets = {"validação": splits.val}
        if splits.test is not None:
            sets[f"teste ({splits.holdout_book})"] = splits.test
        for set_name, blocks in sets.items():
            loss = mean_loss(model, blocks, batch_size)
            row[f"loss {set_name}"] = round(loss, 4)
            row[f"ppl {set_name}"] = round(math.exp(loss), 2)
        rows.append(row)
    return pd.DataFrame(rows).set_index("modelo")


# ----------------------------------------------------------------------------
# Métricas lexicais dos textos gerados
# ----------------------------------------------------------------------------
def words(text: str) -> list[str]:
    return [w.lower() for w in WORD_RE.findall(text)]


def corpus_vocab(texts) -> set[str]:
    vocab: set[str] = set()
    for t in texts:
        vocab.update(words(t))
    return vocab


def load_corpus_vocab(processed_dir: Path = DATA_PROCESSED) -> set[str] | None:
    try:
        return corpus_vocab(load_books(processed_dir).values())
    except FileNotFoundError:
        return None


def text_metrics(text: str, vocab: set[str] | None = None) -> dict:
    """Métricas simples e interpretáveis de um texto gerado.

    - distinct1/distinct2: diversidade (proporção de unigramas/bigramas únicos)
    - repeat4: fração de 4-gramas repetidos (sinal de laços de repetição)
    - oov: % de palavras que NÃO aparecem em nenhum livro do corpus (grafia moderna,
      palavras inventadas ou vocabulário alheio ao Machado)
    - arcaicas_1k: grafias de época por 1000 palavras (proxy de estilo)
    """
    ws = words(text)
    n = len(ws)
    if n == 0:
        return {"palavras": 0, "distinct1": 0.0, "distinct2": 0.0, "repeat4": 0.0,
                "oov": float("nan"), "arcaicas_1k": 0.0}
    bigrams = list(zip(ws, ws[1:]))
    fourgrams = list(zip(ws, ws[1:], ws[2:], ws[3:]))
    out = {
        "palavras": n,
        "distinct1": round(len(set(ws)) / n, 3),
        "distinct2": round(len(set(bigrams)) / len(bigrams), 3) if bigrams else 0.0,
        "repeat4": round(1 - len(set(fourgrams)) / len(fourgrams), 3) if fourgrams else 0.0,
        "arcaicas_1k": round(1000 * sum(w in ARCHAIC_MARKERS for w in ws) / n, 1),
    }
    out["oov"] = round(sum(w not in vocab for w in ws) / n, 3) if vocab else float("nan")
    return out


def compare_models(
    models: dict[str, tuple],
    prompts: list[str] | None = None,
    gen_kwargs: dict | None = None,
    n_samples: int = 3,
    vocab: set[str] | None = None,
    seed: int = 123,
) -> pd.DataFrame:
    """Gera textos com todos os modelos (mesmos prompts e mesmas seeds) e calcula métricas."""
    prompts = prompts or DEFAULT_PROMPTS
    gen_kwargs = gen_kwargs or {}
    rows = []
    for prompt in prompts:
        for s in range(n_samples):
            for name, (model, tok) in models.items():
                cont = generate_text(model, tok, prompt, seed=seed + s, **gen_kwargs)
                rows.append({"modelo": name, "prompt": prompt, "amostra": s,
                             "texto": prompt + cont, **text_metrics(cont, vocab)})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["palavras", "distinct1", "distinct2", "repeat4", "oov", "arcaicas_1k"]
    return df.groupby("modelo")[cols].mean().round(3)


def temperature_sweep(model, tokenizer, prompt: str, temperatures=(0.3, 0.7, 1.0, 1.3),
                      seed: int = 7, **gen_kwargs) -> dict[float, str]:
    """Mesmo prompt e mesma seed com temperaturas diferentes."""
    return {t: prompt + generate_text(model, tokenizer, prompt, temperature=t, seed=seed, **gen_kwargs)
            for t in temperatures}


# ----------------------------------------------------------------------------
# Curvas de treino e reconstrução dos splits
# ----------------------------------------------------------------------------
def plot_loss_curves(history: list[dict] | pd.DataFrame, save_to: Path | None = None):
    """Curvas de loss (treino x validação) e perplexidade de validação ao longo dos passos."""
    import matplotlib.pyplot as plt

    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(df["step"], df["train_loss"], marker="o", label="treino")
    axes[0].plot(df["step"], df["val_loss"], marker="o", label="validação")
    axes[0].set(xlabel="passo do otimizador", ylabel="loss (nats/token)", title="Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(df["step"], df["val_ppl"], marker="o", color="tab:green")
    axes[1].set(xlabel="passo do otimizador", ylabel="perplexidade", title="Perplexidade (validação)")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    if save_to:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=150)
    return fig


def load_run_config(run_dir: Path = RUN_DIR) -> TrainConfig:
    """Lê a configuração usada no treino (para reconstruir exatamente os mesmos splits)."""
    path = Path(run_dir) / "train_meta.json"
    if not path.exists():
        return TrainConfig()
    meta = json.loads(path.read_text())
    return TrainConfig(**meta["config"])


def rebuild_splits(tokenizer, run_dir: Path = RUN_DIR) -> DataSplits:
    cfg = load_run_config(run_dir)
    return make_splits(load_books(), tokenizer, cfg.block_size, cfg.val_fraction,
                       cfg.holdout_book, cfg.seed)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Compara o GPT-2 original com o ajustado.")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    args = parser.parse_args()

    device = default_device()
    base = load_model(BASE_MODEL, device)
    tuned = load_model(str(TUNED_DIR), device)
    models = {"GPT-2 PT (original)": base, "GPT-2 PT + Machado": tuned}

    splits = rebuild_splits(base[1])
    print(splits.summary())
    ppl = perplexity_table(models, splits)
    print(ppl, "\n")

    df = compare_models(models, n_samples=args.samples, vocab=load_corpus_vocab(),
                        gen_kwargs={"max_new_tokens": args.max_new_tokens})
    summary = summarize(df)
    print(summary)

    RESULTS.mkdir(parents=True, exist_ok=True)
    ppl.to_csv(RESULTS / "perplexidade.csv")
    df.to_csv(RESULTS / "geracoes.csv", index=False)
    summary.to_csv(RESULTS / "metricas_lexicais.csv")
    print(f"\nResultados salvos em {RESULTS}")


if __name__ == "__main__":
    main()
