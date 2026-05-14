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
- **Data**: 27,604 valid Sanskrit-English sentence pairs (filtered from 75,161)
- **Device**: Auto-detects MPS (Apple Silicon) → CUDA → CPU

## Evaluation Metrics

Running `evaluation.py` produces in `eval_results/`:

- `summary.csv` — corpus-level BLEU, ChrF, TER
- `metrics_per_sample.csv` — per-sentence scores
- Several graphs: score distributions, length analysis, BLEU vs ChrF correlation
- `sample_translations.txt` — best, worst, and random translation samples
