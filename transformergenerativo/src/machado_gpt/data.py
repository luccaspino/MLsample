"""Download, limpeza e tokenização do corpus de Machado de Assis."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import requests
import torch

from .config import DATA_PROCESSED, DATA_RAW, GUTENBERG_BOOKS

GUTENBERG_URLS = (
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
    "https://www.gutenberg.org/ebooks/{id}.txt.utf-8",
)
_START_RE = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*", re.I)
_END_RE = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", re.I)
_NOISE_RE = re.compile(r"^(Produced by|Produzido por|E-text prepared by|Updated editions)", re.I)


# ----------------------------------------------------------------------------
# Download e limpeza
# ----------------------------------------------------------------------------
def read_text(path: Path) -> str:
    """Lê um arquivo de texto tentando UTF-8 e, se falhar, Latin-1."""
    raw = Path(path).read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def download_book(book_id: int, dest: Path) -> None:
    """Baixa o texto puro de um ebook do Project Gutenberg para `dest`."""
    headers = {"User-Agent": "machado-gpt-student-project/0.1"}
    errors = []
    for template in GUTENBERG_URLS:
        url = template.format(id=book_id)
        try:
            resp = requests.get(url, headers=headers, timeout=60)
            if resp.status_code == 200 and resp.content:
                try:
                    text = resp.content.decode("utf-8-sig")
                except UnicodeDecodeError:
                    text = resp.content.decode("latin-1")
                dest.write_text(text, encoding="utf-8")
                return
            errors.append(f"{url} -> HTTP {resp.status_code}")
        except requests.RequestException as exc:  # rede fora, bloqueio etc.
            errors.append(f"{url} -> {exc}")
    raise RuntimeError(
        f"Não consegui baixar o ebook {book_id}.\n  " + "\n  ".join(errors) +
        f"\nBaixe manualmente em https://www.gutenberg.org/ebooks/{book_id} "
        f"(Plain Text UTF-8) e salve como {dest}"
    )


def clean_gutenberg_text(raw: str) -> str:
    """Remove cabeçalho/rodapé do Gutenberg e reflui as linhas em parágrafos."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n").lstrip("﻿")
    m = _START_RE.search(text)
    if m:
        text = text[m.end():]
    m = _END_RE.search(text)
    if m:
        text = text[: m.start()]
    text = re.sub(r"\[(?:Illustration|Ilustra[cç][aã]o|Nota|Note)[^\]]*\]", " ", text, flags=re.I)

    paragraphs = []
    for block in re.split(r"\n\s*\n", text):
        para = re.sub(r"\s+", " ", " ".join(line.strip() for line in block.split("\n"))).strip()
        if para and not _NOISE_RE.match(para):
            paragraphs.append(para)
    return "\n\n".join(paragraphs)


def prepare_corpus(
    books: dict[str, int] | None = None,
    raw_dir: Path = DATA_RAW,
    processed_dir: Path = DATA_PROCESSED,
    download: bool = True,
    verbose: bool = True,
) -> dict[str, dict]:
    """Baixa (se faltar) e limpa os livros. Qualquer .txt colocado em `raw_dir` também é usado.

    Retorna estatísticas por livro: caracteres, palavras e parágrafos.
    """
    books = GUTENBERG_BOOKS if books is None else books
    raw_dir, processed_dir = Path(raw_dir), Path(processed_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    for name, book_id in books.items():
        raw_path = raw_dir / f"{name}.txt"
        if raw_path.exists():
            continue
        if not download:
            if verbose:
                print(f"[aviso] {raw_path.name} não existe e download=False; pulando.")
            continue
        if verbose:
            print(f"Baixando {name} (ebook {book_id})...")
        download_book(book_id, raw_path)

    stats: dict[str, dict] = {}
    for raw_path in sorted(raw_dir.glob("*.txt")):
        text = clean_gutenberg_text(read_text(raw_path))
        if not text:
            continue
        (processed_dir / raw_path.name).write_text(text, encoding="utf-8")
        stats[raw_path.stem] = {
            "chars": len(text),
            "words": len(text.split()),
            "paragraphs": text.count("\n\n") + 1,
        }
    if not stats:
        raise RuntimeError(f"Nenhum livro encontrado em {raw_dir}. Veja o README.")
    (processed_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    if verbose:
        for name, s in stats.items():
            print(f"  {name:<36} {s['chars']:>9,} caracteres | {s['words']:>8,} palavras")
    return stats


def load_books(processed_dir: Path = DATA_PROCESSED) -> dict[str, str]:
    """Carrega os livros já limpos: {nome: texto}."""
    files = sorted(Path(processed_dir).glob("*.txt"))
    if not files:
        raise FileNotFoundError(
            f"Nenhum texto em {processed_dir}. Rode prepare_corpus() (ou `uv run machado-prepare`)."
        )
    return {p.stem: p.read_text(encoding="utf-8") for p in files}


# ----------------------------------------------------------------------------
# Tokenização e divisão treino/validação/teste
# ----------------------------------------------------------------------------
def tokenize_text(text: str, tokenizer) -> list[int]:
    """Tokeniza um texto longo sem o aviso de 'sequence longer than max length'."""
    old = tokenizer.model_max_length
    tokenizer.model_max_length = int(1e12)
    try:
        return tokenizer(text, add_special_tokens=False)["input_ids"]
    finally:
        tokenizer.model_max_length = old


def build_blocks(texts: list[str], tokenizer, block_size: int) -> torch.Tensor:
    """Concatena os textos (separados por <|endoftext|>) e fatia em blocos [N, block_size]."""
    eos = tokenizer.eos_token_id
    ids: list[int] = []
    for text in texts:
        ids.extend(tokenize_text(text, tokenizer))
        ids.append(eos)
    n_blocks = len(ids) // block_size
    if n_blocks == 0:
        raise ValueError("Texto curto demais para o block_size escolhido.")
    return torch.tensor(ids[: n_blocks * block_size], dtype=torch.long).view(n_blocks, block_size)


@dataclass
class DataSplits:
    train: torch.Tensor
    val: torch.Tensor
    test: torch.Tensor | None       # livro inteiro fora do treino (se holdout_book)
    holdout_book: str | None
    n_tokens_train: int

    def summary(self) -> str:
        parts = [f"treino: {len(self.train)} blocos", f"validação: {len(self.val)} blocos"]
        if self.test is not None:
            parts.append(f"teste ({self.holdout_book}): {len(self.test)} blocos")
        return " | ".join(parts)


def make_splits(
    books: dict[str, str],
    tokenizer,
    block_size: int,
    val_fraction: float = 0.1,
    holdout_book: str | None = None,
    seed: int = 42,
) -> DataSplits:
    """Gera treino/validação (blocos aleatórios) e, opcionalmente, um livro inteiro de teste."""
    if holdout_book is not None and holdout_book not in books:
        print(f"[aviso] holdout_book='{holdout_book}' não encontrado; usando todos os livros.")
        holdout_book = None
    if holdout_book is not None and len(books) < 2:
        holdout_book = None

    train_texts = [t for name, t in books.items() if name != holdout_book]
    blocks = build_blocks(train_texts, tokenizer, block_size)

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(blocks), generator=g)
    n_val = max(1, int(len(blocks) * val_fraction))
    val = blocks[perm[:n_val]]
    train = blocks[perm[n_val:]]

    test = build_blocks([books[holdout_book]], tokenizer, block_size) if holdout_book else None
    return DataSplits(train, val, test, holdout_book, n_tokens_train=train.numel())


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Baixa e limpa os livros de Machado de Assis.")
    parser.add_argument("--no-download", action="store_true", help="usa só o que já está em data/raw")
    args = parser.parse_args()
    prepare_corpus(download=not args.no_download)


if __name__ == "__main__":
    main()
