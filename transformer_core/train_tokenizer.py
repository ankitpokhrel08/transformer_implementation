"""
train_tokenizer.py — trains SentencePiece BPE tokenizers for English (source)
and Nepali (target).

Run once (re-run only if the training data changes):
    python3 train_tokenizer.py

Produces tokenizers/english_bpe.{model,vocab} and tokenizers/nepali_bpe.{model,vocab}.
Token ids: 0=<unk>, 1=<s> (BOS), 2=</s> (EOS), 3=<pad> — shared by both tokenizers.
"""

import os
import sentencepiece as spm

os.makedirs('tokenizers', exist_ok=True)

COMMON = dict(
    model_type='bpe',
    vocab_size=16000,
    unk_id=0, bos_id=1, eos_id=2, pad_id=3,
    byte_fallback=True,           # never emit <unk>: unknown chars become byte tokens
    normalization_rule_name='nmt_nfkc',
    input_sentence_size=1_000_000,
    shuffle_input_sentence=True,
)

spm.SentencePieceTrainer.train(
    input='../data_nepali/train.en',
    model_prefix='tokenizers/english_bpe',
    character_coverage=0.9999,
    **COMMON,
)

spm.SentencePieceTrainer.train(
    input='../data_nepali/train.ne',
    model_prefix='tokenizers/nepali_bpe',
    character_coverage=1.0,       # keep every Devanagari sign
    **COMMON,
)

print("Done — tokenizers/english_bpe.model, tokenizers/nepali_bpe.model")
