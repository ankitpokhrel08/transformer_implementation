# Learning Log — How this project got here

This repo did not start as English→Nepali. It began as **Sanskrit→English** and went
through three phases before pivoting. Each phase taught something concrete, and the
journey *is* the project — that's why the story is kept here even though the current
README only documents the final system.

## Chapter 1 — Sanskrit→English, and the verdict

### v1 — character-level (the faithful paper implementation)

A by-the-book Transformer that tokenized at the **character level** and padded every
sequence to 200 positions. Ten epochs took ~18 hours on an Apple M2 Air and produced
phonetic mush:

```
"The sordhipful asni samself wonsimed tir hoat"
```

**Post-mortem:**
- ~80% of attention compute was spent on **padding tokens**.
- A character-level model must learn to *spell* 50,000 unique English words before it
  can even begin to translate. Far too much burden on a tiny model.

### v2 — BPE + bucketed batching

Same architecture, **rebuilt data pipeline**:
- SentencePiece BPE (8k vocab/side) instead of characters.
- Batches bucketed by length and padded only to the batch max (not a global 200).
- Paper LR schedule (warmup + inverse-sqrt), dev-set early stopping.

**Result:**
- **12× faster epochs** (~7 min vs ~18 hours for 10 epochs).
- 2.6× more usable data (73,227 pairs).
- Fluent, source-conditioned English after ~80 minutes of training:

```
SKT : तस्य तद् वचनं श्रुत्वा रामस्य मुनिपुङ्गवः। आख्यातुं तत्समारेभे विशालायाः पुरातनम्॥
REF : Hearing those words of Rāma, that foremost of ascetics began to relate the history of Viśālā...
HYP : Hearing those words of the ascetic, the king of the celestials, viz., Rama, became highly pleased.
```

Held-out test scores (500 unseen sentences, greedy decoding, best-dev checkpoint):
**BLEU 2.08 · ChrF 21.44 · TER 106.50**.

### The verdict

The architecture and pipeline were **correct** — the *dataset* had no headroom.

The Itihāsa corpus is epic verse (Rāmāyaṇa / Mahābhārata ślokas) paired with free,
literary Victorian-style translations: no word alignment to latch onto, only ~75k
pairs total, and published from-scratch baselines on it also score single-digit BLEU.

In low-resource MT, **data is the model**. A from-scratch transformer on 73k pairs of
poetry cannot produce a resume-worthy translator no matter how clean the code is. So
the engineering moved to a language pair with ~20× the data and a standard benchmark.

## Chapter 2 — The pivot to English→Nepali

- **Why**: ~20× more data and a standard, citable benchmark (FLORES-200).
- **Data**: NLLB mined English–Nepali bitext (~19.6M raw pairs), cleaned and filtered
  to the top 1,000,000 pairs by LASER mining score.
- **Pipeline**: same proven v2 recipe — SentencePiece BPE (bumped to 16k vocab/side),
  length-bucketed batching, warmup + inverse-sqrt LR, label smoothing 0.1, best-dev
  checkpointing.
- **Outcome**: the same pipeline that floundered on 73k poetry pairs (BLEU ~2) produced
  fluent output and **BLEU 13.78 / ChrF 41.22** on 1M everyday-prose pairs.

Side-by-side, the two chapters:

| Metric      | English→Nepali (Ch. 2) | Sanskrit→English (Ch. 1) |
| ----------- | ---------------------- | ------------------------ |
| Corpus BLEU | **13.78**              | 2.08                     |
| Corpus ChrF | **41.22**              | 21.44                    |
| Corpus TER  | **82.46**              | 106.50                   |

The full English→Nepali system is documented in **[readme.md](readme.md)**.

## Lessons learned

1. **Tokenization is leverage**: char → BPE cut sequence length ~4×, epoch time ~12×,
   and turned gibberish into grammar.
2. **Never pad to a global max**: bucket by length and pad per batch.
3. **Data beats architecture**: the same pipeline that floundered on 73k poetry pairs
   (BLEU ~2) produces fluent output on 1M everyday-prose pairs.
4. **Always hold out a real test set** and benchmark against published numbers — a
   metric only means something in context.
5. **Know when to pivot**: clean code on a dead-end dataset is still a dead end.
   Recognizing the data ceiling early saved the project.
