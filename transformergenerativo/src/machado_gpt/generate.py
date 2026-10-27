"""Carregamento de modelos e geração de texto com amostragem própria.

O laço de geração é implementado à mão (temperatura, top-k, top-p, penalidade de repetição,
cache de chaves/valores) para deixar explícito o que acontece em cada passo autorregressivo.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def default_dtype(device: torch.device) -> torch.dtype:
    """Precisão de inferência/avaliação: fp32.

    O GPT-2 small tem só 124M parâmetros (~0.5 GB em fp32), então não vale arriscar overflow
    do fp16 nem a perda de precisão do bf16 na perplexidade. Assim a tabela de avaliação
    fica consistente com a loss de validação medida durante o treino.
    """
    return torch.float32


def load_model(path_or_name: str, device: torch.device | None = None, dtype: torch.dtype | None = None):
    """Carrega (modelo, tokenizer) em modo de avaliação (fp32 por padrão)."""
    device = device or default_device()
    dtype = dtype or default_dtype(device)
    tokenizer = AutoTokenizer.from_pretrained(str(path_or_name))
    model = AutoModelForCausalLM.from_pretrained(str(path_or_name))
    model.to(device=device, dtype=dtype).eval()
    return model, tokenizer


def sample_next_token(
    logits: torch.Tensor,
    generated: torch.Tensor,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
    generator: torch.Generator | None = None,
) -> int:
    """Escolhe o próximo token a partir dos logits [vocab] do último passo."""
    logits = logits.float().clone()

    # 1) penaliza tokens já gerados (reduz repetição)
    if repetition_penalty != 1.0 and generated.numel() > 0:
        seen = torch.unique(generated)
        scores = logits[seen]
        logits[seen] = torch.where(scores > 0, scores / repetition_penalty, scores * repetition_penalty)

    # 2) temperatura <= 0 -> decodificação gulosa
    if temperature <= 0:
        return int(torch.argmax(logits))
    logits = logits / temperature

    # 3) top-k: mantém só os k tokens mais prováveis
    if top_k and 0 < top_k < logits.numel():
        kth = torch.topk(logits, top_k).values[-1]
        logits[logits < kth] = float("-inf")

    # 4) top-p (nucleus): mantém o menor conjunto cuja probabilidade acumulada passa de p
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        remove = (cum - probs) > top_p          # sempre mantém o 1º token
        sorted_logits[remove] = float("-inf")
        logits = torch.full_like(logits, float("-inf")).scatter(0, sorted_idx, sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator))


@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 120,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
    seed: int | None = None,
    stop_at_eos: bool = True,
) -> str:
    """Gera a continuação de `prompt` (retorna só o texto novo, sem o prompt)."""
    device = next(model.parameters()).device
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else model.config.eos_token_id

    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
    if input_ids.shape[1] == 0:                      # prompt vazio -> começa com <|endoftext|>
        input_ids = torch.tensor([[eos_id]])
    max_ctx = int(getattr(model.config, "n_positions", 1024))
    max_new_tokens = min(max_new_tokens, max_ctx - 1)
    keep = max_ctx - max_new_tokens                   # corta o início do prompt se não couber
    input_ids = input_ids[:, -keep:].to(device)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(int(seed))

    out = model(input_ids=input_ids, use_cache=True)
    past, logits = out.past_key_values, out.logits[0, -1]

    new_ids: list[int] = []
    for _ in range(max_new_tokens):
        generated = torch.tensor(new_ids, dtype=torch.long, device=device)
        token = sample_next_token(logits, generated, temperature, top_k, top_p,
                                  repetition_penalty, generator)
        if stop_at_eos and token == eos_id:
            break
        new_ids.append(token)
        out = model(input_ids=torch.tensor([[token]], device=device),
                    past_key_values=past, use_cache=True)
        past, logits = out.past_key_values, out.logits[0, -1]

    return tokenizer.decode(new_ids)
