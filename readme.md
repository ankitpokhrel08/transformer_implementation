# English-to-Nepali Neural Machine Translation

A from-scratch Transformer (Vaswani et al. 2017) in PyTorch, being trained on the
NLLB English-Nepali mined corpus (~19.6M raw pairs) and evaluated on the standard
FLORES-200 benchmark.

> **Status**: active project. Started as Sanskrit→English (see *Chapter 1* below for
> that work and the verdict that motivated the pivot), now English→Nepali: a 35M-param
> transformer trained from scratch on 1M NLLB pairs and evaluated on FLORES-200.

## Why the pivot? — Project history

This repo started as Sanskrit→English translation and went through three phases.
Each one taught something concrete; the readme keeps the full story because the
journey *is* the project.

### Chapter 1 — Sanskrit→English, and the verdict

**v1 (character-level)**: a faithful paper implementation that tokenized at the
character level and padded every sequence to 200 positions. Ten epochs took ~18
hours on an Apple M2 Air and produced phonetic mush ("The sordhipful asni samself
wonsimed tir hoat"). Post-mortem: ~80% of attention compute was spent on padding
tokens, and character-level models must learn to *spell* 50,000 unique English
words before they can translate.

**v2 (BPE + bucketed batching)**: same architecture, rebuilt data pipeline —
SentencePiece BPE (8k vocab/side), batches bucketed by length and padded only to
the batch max, paper LR schedule, dev-set early stopping. Result: **12× faster
epochs** (~7 min), 2.6× more usable data (73,227 pairs), and fluent
source-conditioned English after ~80 minutes of training:

```
SKT : तस्य तद् वचनं श्रुत्वा रामस्य मुनिपुङ्गवः। आख्यातुं तत्समारेभे विशालायाः पुरातनम्॥
REF : Hearing those words of Rāma, that foremost of ascetics began to relate the history of Viśālā...
HYP : Hearing those words of the ascetic, the king of the celestials, viz., Rama, became highly pleased.
```

Held-out test scores (500 unseen sentences, greedy decoding, best-dev checkpoint):
**BLEU 2.08 · ChrF 21.44 · TER 106.50**.

**The verdict**: the architecture and pipeline were correct — the *dataset* had no
headroom. The Itihāsa corpus is epic verse (Rāmāyaṇa/Mahābhārata ślokas) paired
with free, literary Victorian-style translations: no word alignment to latch onto,
75k pairs total, and published from-scratch baselines on it also score single-digit
BLEU. In low-resource MT, **data is the model**. A from-scratch transformer on 73k
pairs of poetry cannot produce a resume-worthy translator no matter how clean the
code is — so the engineering moved to a language pair with 20× the data and a
standard benchmark: English→Nepali (NLLB mined corpus, ~1.6M quality-filtered
pairs, FLORES-200 evaluation).

### Chapter 2 — English→Nepali (current)

- **Data**: [NLLB](https://opus.nlpl.eu/legacy/NLLB.php) mined English-Nepali bitext
  (~19.6M raw pairs). Cleaned by script check, length ratio, and dedupe, then the
  **top 1,000,000 pairs by LASER mining score** are kept (`data_nepali/prepare_data.py`).
  Evaluated on [FLORES-200](https://github.com/facebookresearch/flores) — dev (997)
  for validation, devtest (1,012) for the final number. A standard, citable benchmark.
- **Pipeline**: same as v2 — SentencePiece BPE (16k vocab/side), length-bucketed
  batching, warmup + inverse-sqrt LR, label smoothing 0.1, best-dev checkpointing.
- **Training**: 6 epochs on an Apple M2 Air (MPS), ~2 hours/epoch. Dev loss fell
  monotonically 9.69 → ~4.55 with no overfitting.

See `EVALUATION_GUIDE.txt` for a plain-English explanation of every metric and graph.

## Architecture

- **Type**: Encoder-Decoder Transformer (Vaswani et al. 2017), implemented from scratch
- **Size**: d_model 384, FFN hidden 1536, 8 heads, 4 encoder + 4 decoder layers
  (**35.0M parameters**)
- **Tokenization**: SentencePiece BPE, 16,000 pieces per language, byte fallback (no UNK)
- **Sequence length**: BPE pairs capped at 96 tokens (positional capacity 128);
  median sentence is ~28 tokens
- **Components**: multi-head self/cross attention, sinusoidal positional encodings,
  layer norm, position-wise FFN — see `transformer_core/transformer.py` and the
  step-by-step notebooks in `transformer_core/components/`
- **Training**: Adam (betas 0.9/0.98), 4000-step warmup + inverse-sqrt decay,
  label smoothing 0.1, gradient clipping (max-norm 1.0), Xavier init, batch size 64
- **Hardware**: Apple Silicon (MPS), CUDA, or CPU — auto-detected

## Results (English→Nepali)

Final corpus scores on FLORES-200 devtest (1,012 unseen sentences, greedy decoding,
best-dev checkpoint at epoch 5/step 92k). BLEU uses the `flores200` (Devanagari-aware)
tokenizer.

| Metric      | English→Nepali | Sanskrit→English (Ch. 1) |
| ----------- | -------------- | ------------------------ |
| Corpus BLEU | **13.78**      | 2.08                     |
| Corpus ChrF | **41.22**      | 21.44                    |
| Corpus TER  | **82.46**      | 106.50                   |

> For a from-scratch model on a low-resource pair, **ChrF is the fairer judge** —
> Nepali's rich inflection makes BLEU's exact-word matching read low. See the guide.

Double-digit BLEU and 41 ChrF from a 35M-parameter model trained from scratch on a
laptop is a solid, honest result. Large pretrained systems (NLLB, Google) score
higher because they pretrain on billions of sentences across 200 languages — but
this run proves the from-scratch pipeline genuinely works once the data has headroom.

The model handles everyday prose well; its best translations are near-human:

```
EN  : In remote locations, without cell phone coverage, a satellite phone may be your only option.
REF : दुर्गम स्थानहरूमा, सेल फोनको कभरेज बिना, उपग्रह फोन तपाईंको एक मात्र विकल्प हुन सक्छ।
HYP : रिमोट स्थानहरूमा, सेल फोन कभरेज बिना, एक उपग्रह फोन तपाईंको मात्र विकल्प हुन सक्छ।
      (BLEU 55.7 · ChrF 77.9)
```

Common failure modes: dropping rare compound terms ("machine learning" → "मेशिन"),
occasional clause repetition on long inputs, and borrowing English loanwords
("रिमोट") where a native term exists. All expected for the data scale and greedy
decoding; beam search and more epochs would close part of the gap.

## Quick Start

```bash
pip install -r requirements.txt

# 1. Get the data (raw NLLB dump + FLORES-200 benchmark)
cd data_nepali/raw
curl -L -o en-ne.txt.zip https://object.pouta.csc.fi/OPUS-NLLB/v1/moses/en-ne.txt.zip
unzip en-ne.txt.zip
curl -L https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz | tar xz
cd ..
python3 prepare_data.py --top_n 1000000   # → train.en/.ne, dev.*, test.*
cd ../transformer_core

# 2. Train tokenizers, then the model, then evaluate
python3 train_tokenizer.py   # once
python3 training.py          # resumes from checkpoint automatically
python3 evaluation.py        # BLEU/ChrF/TER + graphs on FLORES-200 devtest
```

## Lessons learned (the short version)

1. **Tokenization is leverage**: char → BPE cut sequence length 4×, epoch time 12×,
   and turned gibberish into grammar.
2. **Never pad to a global max**: bucket by length and pad per batch.
3. **Data beats architecture**: the same pipeline that floundered on 73k poetry
   pairs (BLEU ~2) produces fluent output on 1M everyday-prose pairs.
4. **Always hold out a real test set** and benchmark against published numbers —
   a metric only means something in context.

## Better Approach — Fine-tuning

The most effective solution for low-resource translation remains fine-tuning a
pretrained multilingual model rather than training from scratch. A related project
demonstrates this:

**[Fine-tuning Llama-2-7b-chat-hf with QLoRA](https://github.com/ankitpokhrel08/llm-projects/tree/main/finetuning_Llama-2-7b-chat-hf)**

Fine-tunes `meta-llama/Llama-2-7b-chat-hf` using QLoRA on a custom Bhagavad Gita
dataset. Pretrained language understanding + parameter-efficient fine-tuning gives
superior quality at a fraction of the compute.
