"""
Gradio app for the English → Nepali Transformer (Hugging Face Spaces).

Loads the slim inference checkpoint, rebuilds the model from its stored config,
and translates English text to Nepali with greedy decoding. Runs on CPU.
"""

import os

import torch
import gradio as gr

from transformer import Transformer
from data_pipeline import load_tokenizers, greedy_translate

CKPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "en_ne_inference.pth")
device = torch.device("cpu")

# ── Load tokenizers + model once at startup ───────────────────────────────────
sp_src, sp_tgt = load_tokenizers()

ckpt = torch.load(CKPT_PATH, map_location=device)
model = Transformer(
    **ckpt["config"],
    src_vocab_size=sp_src.get_piece_size(),
    tgt_vocab_size=sp_tgt.get_piece_size(),
)
model.load_state_dict(ckpt["model_state_dict"])
model.to(device).eval()

EPOCH = ckpt.get("epoch", "?")


def translate(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    return greedy_translate(model, sp_src, sp_tgt, text, device)


demo = gr.Interface(
    fn=translate,
    inputs=gr.Textbox(lines=3, label="English", placeholder="Type an English sentence…"),
    outputs=gr.Textbox(lines=3, label="नेपाली (Nepali)"),
    title="English → Nepali Translator",
    description=(
        "A 35M-parameter Transformer (Vaswani et al. 2017) trained **from scratch** on "
        "1M NLLB sentence pairs. Greedy decoding, CPU inference. "
        "Scores BLEU 13.78 / ChrF 41.22 on the FLORES-200 benchmark — best on short, "
        "everyday prose."
    ),
    examples=[
        ["The weather is very nice today."],
        ["I want to learn the Nepali language."],
        ["Education is the most powerful tool to change the world."],
        ["In remote locations, without cell phone coverage, a satellite phone may be your only option."],
    ],
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
