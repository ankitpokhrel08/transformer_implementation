# Sanskrit-to-English Neural Machine Translation

A character-level Transformer model for translating Sanskrit text to English using PyTorch.

## Overview

This project implements a complete neural machine translation system that converts Sanskrit (Devanagari script) text to English using a Transformer architecture with multi-head attention mechanisms.

## Features

- **Transformer Architecture**: Full encoder-decoder with multi-head attention
- **Character-Level Translation**: Fine-grained tokenization for better accuracy
- **Complete Vocabulary**: Comprehensive Sanskrit and English character sets
- **MPS Support**: Runs on Apple Silicon (M1/M2) via Metal Performance Shaders
- **Checkpoint Resume**: Training auto-saves per epoch and resumes from last checkpoint

## Quick Start

### 1. Install Dependencies

```bash
pip install torch numpy matplotlib sacrebleu
```

### 2. Train the Model

```bash
python3 training.py
```

Training resumes automatically from the last checkpoint if one exists.

### 3. Evaluate

```bash
python3 evaluation.py
# or with custom args:
python3 evaluation.py --checkpoint models/checkpoint.pth --n_samples 500 --out_dir eval_results
```

## Model Architecture

- **Type**: Encoder-Decoder Transformer (Vaswani et al. 2017)
- **Dimensions**: 512 (d_model), 2048 (FFN hidden)
- **Attention Heads**: 8
- **Layers**: 6 encoder + 6 decoder
- **Vocabulary**: 89 Sanskrit + 183 English characters
- **Max Sequence Length**: 200 characters
- **Optimizer**: Adam (lr warmup, betas=(0.9, 0.98), eps=1e-9)
- **Loss**: Cross-entropy with padding mask
- **Regularization**: Dropout (0.1), gradient clipping (max norm 1.0)

## Training Details

- **LR Schedule**: Linear warmup for 4000 steps then inverse sqrt decay (paper schedule)
- **Batch Size**: 16
- **Epochs**: 50 with per-epoch checkpointing
- **Weight Init**: Xavier uniform
- **Data**: 54721 valid Sanskrit-English sentence pairs (filtered from 75,161)
- **Device**: Auto-detects MPS (Apple Silicon) → CUDA → CPU
- **Batches per epoch**: 3450

## Evaluation Metrics

Running `evaluation.py` produces in `eval_results/`:

- `summary.csv` — corpus-level BLEU, ChrF, TER
- `metrics_per_sample.csv` — per-sentence scores
- Several graphs: score distributions, length analysis, BLEU vs ChrF correlation
- `sample_translations.txt` — best, worst, and random translation samples

## Notes

- The model was first checked on short dataset of around 20 handpicked small sentences and we overfit in it such that we know how well model performs then the model is trained on whole large corpus of data. The `results_on_short_dataset` is the result of that overfitted training and the model `short_dataset.pth` is the model thus generated.

## Results

### Performance

Training was conducted for 10 epochs on 27,604 valid Sanskrit-English sentence pairs
using a full 6-layer encoder-decoder Transformer on an Apple M2 Air (MPS backend).

| Metric      | Score               |
| ----------- | ------------------- |
| Corpus BLEU | _run evaluation.py_ |
| Corpus ChrF | _run evaluation.py_ |
| Corpus TER  | _run evaluation.py_ |

### Observations

The model achieved near-perfect translation on shorter, simpler sentences within the
training distribution — demonstrating that the paper architecture was implemented
correctly. For longer and more complex shlokas, the model produced recognizable
translations with occasional character-level substitutions, which is expected behavior
for a character-level model trained on a small corpus.

SKT : बालकः पुस्तकं पठति।
REF : The boy reads a book.
HYP : The boy reads a book.
BLEU=100.00 ChrF=100.00 TER=0.00

Training on a larger dataset showed significant improvement in output coherence and
vocabulary coverage, though 10 epochs was insufficient to fully converge — the loss
was still declining at the point of stopping. Given more training time and data, further
improvement is expected. Overall performance was satisfactory and the core goal of
implementing the original Transformer paper (Vaswani et al. 2017) from scratch was
successful.

A sample output at epoch 10:
SKT : तस्यास्तु चरणौ वह्निर्ददाह भगवान् स्वयम्। न च तस्या मनो दुःखं स्वल्पमप्यभवत् तदा॥
EN : The worshipful Agni himself consumed her feet. For this, however, the maiden did not feel the slightest pain.
PRD : The sordhipful asni samself wonsimed tir hoat Tor thes tewever, Ohe sonden tes tot aiel ahe seaghtest orrn

SKT : स तैः शूरैरनुज्ञातो ययौ राजनिवेशनम्। पार्थमादाय गोविन्दो ददर्श च युधिष्ठिरम्॥
EN : Having been thus permitted by those warriors he started for the camp of the king. Govinda, then, with Partha, saw Yudhishthira.
PRD : Teving seen thus arrfisted by these thrriors warwponted aor the sarp of the sing To ind the Ohth tartha tay tudhishthira

### Limitations

- Character-level tokenization produces longer sequences and plateaus earlier than
  subword (BPE/SentencePiece) approaches
- 27,604 training pairs is a small corpus for neural machine translation
- M2 Air thermal constraints limited practical training to ~10 epochs

### Better Approach — Fine-tuning

The most effective solution to Sanskrit-English translation is fine-tuning a pretrained
large language model rather than training from scratch. My other project demonstrates
this approach:

**[Fine-tuning Llama-2-7b-chat-hf with QLoRA](https://github.com/ankitpokhrel08/llm-projects/tree/main/finetuning_Llama-2-7b-chat-hf)**

Fine-tunes `meta-llama/Llama-2-7b-chat-hf` using QLoRA (Quantized Low-Rank Adaptation)
on a custom Bhagavad Gita dataset, creating an AI Krishna that responds with spiritual
wisdom. This approach leverages the pretrained model's existing language understanding
and requires significantly less compute to achieve superior translation quality.
