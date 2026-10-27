# GPT-2 português × Machado de Assis

Fine-tuning de um GPT-2 small em português (`pierreguillou/gpt2-small-portuguese`, 124M parâmetros) no estilo de Machado de Assis, com avaliação comparativa e uma interface (Gradio) que recebe um prompt e mostra o modelo original e o ajustado lado a lado.

Projeto em **uv + VSCode + Jupyter**. O laço de treino e a amostragem de texto (temperatura, top-k, top-p, penalidade de repetição, cache KV) são escritos à mão em PyTorch; só o modelo/tokenizer vêm do Hugging Face.

## Estrutura

```
machado-gpt/
├── pyproject.toml            # dependências (uv); PyTorch com CUDA 12.6
├── notebooks/
│   └── experimento.ipynb     # roteiro completo: dados → treino → avaliação → demo
├── src/machado_gpt/
│   ├── config.py             # caminhos, modelo base, livros, hiperparâmetros (TrainConfig)
│   ├── data.py               # download/limpeza do Gutenberg, tokenização, splits
│   ├── train.py              # laço de treino (AdamW, cosine, bf16, grad. checkpointing)
│   ├── generate.py           # carregamento de modelos e amostragem própria
│   ├── evaluate.py           # perplexidade, métricas lexicais, comparação, gráficos
│   └── app.py                # interface Gradio (original × fine-tuned)
├── data/raw/                 # textos baixados (ou colocados por você)
├── data/processed/           # textos limpos
├── checkpoints/              # modelo ajustado (gerado pelo treino)
└── results/                  # CSVs e gráficos da avaliação
```

## Como rodar (VSCode + uv)

**1. Pré-requisitos:** driver NVIDIA atualizado (`nvidia-smi` deve funcionar), [uv](https://docs.astral.sh/uv/) e as extensões **Python** e **Jupyter** do VSCode.

```bash
# instalar o uv (se ainda não tiver)
# Windows (PowerShell):  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
# Linux/macOS:           curl -LsSf https://astral.sh/uv/install.sh | sh
```

**2. Instalar as dependências** (dentro da pasta do projeto):

```bash
uv sync
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # deve imprimir True
```

**3. Abrir no VSCode:** `code .` → abra `notebooks/experimento.ipynb` → *Select Kernel* → *Python Environments* → escolha o `.venv` do projeto.

**4. Executar o notebook de cima para baixo.** A seção 2 baixa os livros, a seção 3 treina (≈ 10 min numa RTX 3050) e as seções 4 e 5 avaliam e abrem a interface.

### Sem o notebook (linha de comando)

```bash
uv run machado-prepare     # baixa e limpa os livros
uv run machado-train       # fine-tuning (opções: --epochs, --batch-size, --grad-accum, --holdout none)
uv run machado-eval        # perplexidade + comparação; salva CSVs em results/
uv run machado-app         # abre a interface em http://127.0.0.1:7860
```

## O que o experimento mede

| Métrica | O que mostra |
|---|---|
| Perplexidade (validação e teste) | quanto o modelo "se surpreende" com o texto de Machado. O **teste** é um livro inteiro (`quincas_borba`) que o modelo nunca viu |
| `oov` | % de palavras geradas ausentes do vocabulário do corpus |
| `arcaicas_1k` | grafias de época (`elle`, `annos`, `quasi`...) por 1000 palavras, proxy de estilo |
| `repeat4`, `distinct1/2` | repetição e diversidade do texto gerado |
| Curvas de loss | treino × validação ao longo dos passos (detecta overfitting) |

Os dois modelos usam o **mesmo tokenizer**, então as perplexidades são diretamente comparáveis. O passo 0 da curva de validação é o desempenho do modelo original, antes de qualquer ajuste.

## Ajustes comuns

- **`CUDA out of memory`** (RTX 3050 com 4 GB): em `TrainConfig`, use `batch_size=2` e `grad_accum_steps=8`. Se ainda faltar memória, `block_size=128`.
- **Mais livros:** adicione o nº do ebook em `GUTENBERG_BOOKS` (`config.py`) ou coloque qualquer `.txt` em `data/raw/`.
- **Treino final com todos os livros:** `TrainConfig(holdout_book=None)`.
- **Download do Gutenberg falhou:** baixe o *Plain Text UTF-8* de cada ebook (nº em `config.py`) e salve em `data/raw/<nome>.txt`, por exemplo `data/raw/dom_casmurro.txt`.
- **`torch.cuda.is_available()` dá `False`:** confira `nvidia-smi` e se o `uv sync` instalou a build CUDA (`uv run python -c "import torch; print(torch.version.cuda)"` não pode ser `None`).
- **Erro ao carregar o modelo base (`.bin` / tokenizer):** use a cópia com `tokenizer.json`: `MACHADO_BASE_MODEL=egonrp/gpt2-small-portuguese` (variável de ambiente; no PowerShell: `$env:MACHADO_BASE_MODEL="egonrp/gpt2-small-portuguese"`).
- **Sem GPU / macOS:** apague as seções `[[tool.uv.index]]` e `[tool.uv.sources]` do `pyproject.toml` e rode `uv sync`. O treino funciona na CPU, mas é bem mais lento.

## Usando o Colab (extensão do VSCode)

O kernel do Colab roda em outra máquina e não enxerga a pasta local. Compacte o projeto, envie o `.zip` pelo Colab e rode:

```python
!unzip -q machado-gpt.zip
%cd machado-gpt
!pip install -q -e . --no-deps
!pip install -q transformers gradio
```

Depois execute o notebook normalmente (o torch já vem instalado no Colab).

## Observação sobre os dados

Os textos vêm do Project Gutenberg (domínio público). As edições preservam a grafia da época (`elle`, `annos`, `quasi`), o que ajuda a medir a absorção de estilo, mas também significa que o modelo ajustado mistura grafia antiga e moderna.
