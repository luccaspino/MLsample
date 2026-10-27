"""Laço de treino (fine-tuning) escrito à mão em PyTorch, sem usar o `Trainer` do Hugging Face."""

from __future__ import annotations

import gc
import json
import math
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from .config import BASE_MODEL, RUN_DIR, TrainConfig
from .data import load_books, make_splits, seed_everything
from .evaluate import mean_loss
from .generate import default_device


def pick_amp(device: torch.device) -> tuple[torch.dtype | None, bool]:
    """Escolhe a precisão mista: bf16 (sem scaler) se a GPU suporta; senão fp16 com GradScaler."""
    if device.type != "cuda":
        return None, False
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, False
    return torch.float16, True


def build_optimizer(model, cfg: TrainConfig) -> torch.optim.Optimizer:
    """AdamW sem weight decay em bias e LayerNorm."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or "ln_" in name or "bias" in name else decay).append(p)
    groups = [{"params": decay, "weight_decay": cfg.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.95))


def train(
    cfg: TrainConfig | None = None,
    base_model: str = BASE_MODEL,
    out_dir: Path = RUN_DIR,
    device: torch.device | None = None,
) -> dict:
    """Ajusta o GPT-2 nos livros de Machado. Salva o melhor checkpoint (menor loss de validação)."""
    cfg = cfg or TrainConfig()
    out_dir = Path(out_dir)
    best_dir = out_dir / "best"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or default_device()
    seed_everything(cfg.seed)

    # --- modelo, tokenizer e dados ------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model).to(device)
    if cfg.block_size > model.config.n_positions:
        raise ValueError(f"block_size={cfg.block_size} > contexto do modelo ({model.config.n_positions}).")
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False

    splits = make_splits(load_books(), tokenizer, cfg.block_size, cfg.val_fraction,
                         cfg.holdout_book, cfg.seed)
    print(splits.summary())
    n_train = len(splits.train)
    steps_per_epoch = (n_train // cfg.batch_size) // cfg.grad_accum_steps
    if steps_per_epoch == 0:
        raise ValueError("Poucos dados para este batch_size/grad_accum_steps; reduza-os.")
    total_steps = steps_per_epoch * cfg.epochs
    print(f"{total_steps} passos de otimizador ({steps_per_epoch}/época) | "
          f"batch efetivo = {cfg.batch_size * cfg.grad_accum_steps} blocos de {cfg.block_size} tokens")

    optimizer = build_optimizer(model, cfg)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(cfg.warmup_ratio * total_steps), total_steps)
    amp_dtype, use_scaler = pick_amp(device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    print(f"dispositivo: {device} | precisão mista: {amp_dtype or 'desligada'}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # --- avaliação inicial (= desempenho do modelo ORIGINAL) -------------------
    history: list[dict] = []
    t0 = time.time()

    def evaluate(step: int, epoch: int, train_loss: float | None) -> float:
        # validação em fp32 (sem autocast): mesmos números da tabela de perplexidade na avaliação
        val = mean_loss(model, splits.val, batch_size=8, amp_dtype=None)
        history.append({"step": step, "epoch": epoch, "train_loss": train_loss, "val_loss": val,
                        "val_ppl": math.exp(val), "lr": scheduler.get_last_lr()[0],
                        "elapsed_s": round(time.time() - t0, 1)})
        tr = "   -  " if train_loss is None else f"{train_loss:.4f}"
        print(f"  [passo {step:>4}] loss treino {tr} | loss val {val:.4f} | ppl val {math.exp(val):.2f}")
        return val

    evaluate(0, 0, None)
    best_val = float("inf")

    # --- laço de treino ----------------------------------------------------------
    g = torch.Generator().manual_seed(cfg.seed)
    step = 0
    window_loss, window_n = 0.0, 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = torch.randperm(n_train, generator=g)
        for s in tqdm(range(steps_per_epoch), desc=f"época {epoch}/{cfg.epochs}"):
            step_loss = 0.0
            for k in range(cfg.grad_accum_steps):
                start = (s * cfg.grad_accum_steps + k) * cfg.batch_size
                x = splits.train[perm[start : start + cfg.batch_size]].to(device)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                    loss = model(input_ids=x, labels=x).loss / cfg.grad_accum_steps
                scaler.scale(loss).backward()
                step_loss += loss.item()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            step += 1
            window_loss += step_loss
            window_n += 1
            if step % cfg.eval_every == 0 or step == total_steps:
                val = evaluate(step, epoch, window_loss / window_n)
                window_loss, window_n = 0.0, 0
                model.train()
                if val < best_val:
                    best_val = val
                    model.config.use_cache = True
                    model.save_pretrained(best_dir)
                    tokenizer.save_pretrained(best_dir)
                    model.config.use_cache = False

    # --- metadados ---------------------------------------------------------------
    meta = {
        "base_model": base_model,
        "config": cfg.to_dict(),
        "n_blocks": {"train": n_train, "val": len(splits.val),
                     "test": None if splits.test is None else len(splits.test)},
        "total_steps": total_steps,
        "best_val_loss": best_val,
        "best_val_ppl": math.exp(best_val),
        "baseline_val_loss": history[0]["val_loss"],
        "baseline_val_ppl": history[0]["val_ppl"],
        "train_time_s": round(time.time() - t0, 1),
        "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device.type == "cuda" else None,
        "device": str(device),
    }
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"\nPronto em {meta['train_time_s']}s | ppl val: {meta['baseline_val_ppl']:.1f} -> "
          f"{meta['best_val_ppl']:.1f} | melhor modelo em {best_dir}")

    del model, optimizer, scheduler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"history": history, "meta": meta, "best_dir": best_dir}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tuning do GPT-2 PT em Machado de Assis.")
    defaults = TrainConfig()
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--grad-accum", type=int, default=defaults.grad_accum_steps)
    parser.add_argument("--block-size", type=int, default=defaults.block_size)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--holdout", default=defaults.holdout_book, help="livro de teste ('none' para usar todos)")
    args = parser.parse_args()

    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, grad_accum_steps=args.grad_accum,
                      block_size=args.block_size, lr=args.lr,
                      holdout_book=None if str(args.holdout).lower() == "none" else args.holdout)
    train(cfg)


if __name__ == "__main__":
    main()
