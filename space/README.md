---
title: English to Nepali Translator
emoji: 🌐
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# English → Nepali Translator

A 35M-parameter Transformer (Vaswani et al. 2017) trained **from scratch** in PyTorch
on 1M NLLB English–Nepali sentence pairs, served with Gradio on CPU.

- **Scores**: BLEU 13.78 · ChrF 41.22 on the FLORES-200 devtest benchmark
- **Decoding**: greedy
- **Best on**: short, everyday prose

Type an English sentence and get a Nepali translation. The model reuses the exact
tokenizers and decoding logic from evaluation, so output matches the benchmark.

Source code and the full training/evaluation pipeline:
[github.com/ankitpokhrel08](https://github.com/ankitpokhrel08)
