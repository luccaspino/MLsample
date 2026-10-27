"""Interface Gradio: digite um prompt e compare o GPT-2 original com o ajustado em Machado."""

from __future__ import annotations

import random
from pathlib import Path

from .config import BASE_MODEL, DEFAULT_PROMPTS, TUNED_DIR
from .evaluate import load_corpus_vocab, text_metrics
from .generate import default_device, generate_text, load_model

BASE_LABEL = "GPT-2 português (original)"
TUNED_LABEL = "GPT-2 português + Machado de Assis (fine-tuned)"


def build_demo(base_model: str = BASE_MODEL, tuned_dir: Path = TUNED_DIR, device=None):
    """Carrega os dois modelos na GPU e devolve o app Gradio (ainda sem iniciar o servidor)."""
    import gradio as gr

    tuned_dir = Path(tuned_dir)
    if not (tuned_dir / "config.json").exists():
        raise FileNotFoundError(
            f"Modelo ajustado não encontrado em {tuned_dir}. Rode o treino antes "
            "(notebook, seção 3, ou `uv run machado-train`)."
        )
    device = device or default_device()
    base, tuned = load_model(base_model, device), load_model(str(tuned_dir), device)
    vocab = load_corpus_vocab()

    def stats_row(label: str, continuation: str) -> str:
        m = text_metrics(continuation, vocab)
        oov = "-" if m["oov"] != m["oov"] else f"{100 * m['oov']:.0f}%"
        return (f"| {label} | {m['palavras']} | {oov} | {m['arcaicas_1k']} | "
                f"{100 * m['repeat4']:.0f}% | {m['distinct2']:.2f} |")

    def run(prompt, max_new_tokens, temperature, top_k, top_p, rep_penalty, seed):
        prompt = prompt or ""
        seed = int(seed) if seed is not None and int(seed) >= 0 else random.randint(0, 2**31 - 1)
        kw = dict(max_new_tokens=int(max_new_tokens), temperature=float(temperature), top_k=int(top_k),
                  top_p=float(top_p), repetition_penalty=float(rep_penalty), seed=seed)
        cont_base = generate_text(*base, prompt, **kw)       # mesma seed nos dois modelos
        cont_tuned = generate_text(*tuned, prompt, **kw)
        table = ("| modelo | palavras | fora do vocabulário de Machado | grafias de época /1000 | "
                 "4-gramas repetidos | diversidade (bigramas) |\n|---|---|---|---|---|---|\n"
                 f"{stats_row('original', cont_base)}\n{stats_row('fine-tuned', cont_tuned)}\n\n"
                 f"*seed usada: {seed}*")
        return prompt + cont_base, prompt + cont_tuned, table

    with gr.Blocks(title="GPT-2 x Machado de Assis", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# GPT-2 português × Machado de Assis\n"
                    "Mesmo prompt, mesma seed e mesmos parâmetros nos dois modelos: a diferença que você "
                    "vê vem só do fine-tuning.")
        prompt = gr.Textbox(label="Prompt", value=DEFAULT_PROMPTS[0], lines=3)
        with gr.Accordion("Parâmetros de geração", open=False):
            with gr.Row():
                max_new = gr.Slider(20, 400, value=150, step=10, label="Máx. de tokens novos")
                temp = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="Temperatura (0 = gulosa)")
            with gr.Row():
                top_k = gr.Slider(0, 200, value=50, step=5, label="top-k (0 = desligado)")
                top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="top-p")
            with gr.Row():
                rep = gr.Slider(1.0, 1.5, value=1.1, step=0.05, label="Penalidade de repetição")
                seed = gr.Number(value=-1, precision=0, label="Seed (-1 = aleatória)")
        btn = gr.Button("Gerar nos dois modelos", variant="primary")
        with gr.Row():
            out_base = gr.Textbox(label=BASE_LABEL, lines=14, interactive=False)
            out_tuned = gr.Textbox(label=TUNED_LABEL, lines=14, interactive=False)
        stats = gr.Markdown()
        gr.Examples(examples=[[p] for p in DEFAULT_PROMPTS], inputs=[prompt], label="Exemplos de prompt")

        btn.click(run, [prompt, max_new, temp, top_k, top_p, rep, seed], [out_base, out_tuned, stats])
    return demo


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Demo comparativa (Gradio).")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="gera um link público temporário")
    args = parser.parse_args()
    build_demo().launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
