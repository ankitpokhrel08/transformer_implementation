"""
data_pipeline.py — BPE tokenization, bucketed batching, and attention masks
for the English → Nepali transformer.

Shared by training.py and evaluation.py so the tokenizers, special ids, and
mask logic can never drift apart.

Sequence layout per pair:
    encoder input : src tokens + </s>
    decoder input : <s> + tgt tokens
    labels        : tgt tokens + </s>
Batches are padded to the batch max length (not a global max), and batches are
formed from length-sorted buckets so similar lengths land together — the big
speed win over padding everything to 200.
"""

import os
import random

import sentencepiece as spm
import torch
from torch.utils.data import Dataset, Sampler

TOKENIZER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tokenizers')

# Set at tokenizer training time in train_tokenizer.py — same for both languages
UNK_ID, BOS_ID, EOS_ID, PAD_ID = 0, 1, 2, 3

NEG_INFTY = -1e9


def load_tokenizers():
    """Returns (source, target) tokenizers: English → Nepali."""
    sp_src = spm.SentencePieceProcessor(model_file=os.path.join(TOKENIZER_DIR, 'english_bpe.model'))
    sp_tgt = spm.SentencePieceProcessor(model_file=os.path.join(TOKENIZER_DIR, 'nepali_bpe.model'))
    return sp_src, sp_tgt


class TranslationDataset(Dataset):
    """Encodes sentence pairs to BPE ids once, dropping pairs longer than max_len tokens."""

    def __init__(self, src_sentences, tgt_sentences, sp_src, sp_tgt, max_len):
        self.pairs = []
        src_encoded = sp_src.encode(src_sentences)
        tgt_encoded = sp_tgt.encode(tgt_sentences)
        for src_ids, tgt_ids, src_text, tgt_text in zip(
                src_encoded, tgt_encoded, src_sentences, tgt_sentences):
            # +1 for </s> on source, +1 for <s>/</s> on target sides
            if len(src_ids) + 1 <= max_len and len(tgt_ids) + 1 <= max_len:
                self.pairs.append((src_ids, tgt_ids, src_text, tgt_text))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


class BucketBatchSampler(Sampler):
    """
    Groups indices of similar sequence length into batches.

    Each epoch: shuffle indices, sort by length within pools of
    (batch_size * pool_factor), slice into batches, shuffle batch order.
    Keeps batches length-homogeneous (little padding) while still random.
    """

    def __init__(self, lengths, batch_size, shuffle=True, pool_factor=100):
        self.lengths    = lengths
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.pool_size  = batch_size * pool_factor

    def _make_batches(self):
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            random.shuffle(indices)
        batches = []
        for pool_start in range(0, len(indices), self.pool_size):
            pool = indices[pool_start:pool_start + self.pool_size]
            pool.sort(key=lambda i: self.lengths[i])
            for i in range(0, len(pool), self.batch_size):
                batches.append(pool[i:i + self.batch_size])
        if self.shuffle:
            random.shuffle(batches)
        return batches

    def __iter__(self):
        return iter(self._make_batches())

    def __len__(self):
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


def make_masks(src_ids, dec_input_ids):
    """
    src_ids       : (B, S) encoder input ids
    dec_input_ids : (B, T) decoder input ids
    Returns float masks (NEG_INFTY at blocked positions):
        enc_mask (B, S, S), dec_mask (B, T, T), cross_mask (B, T, S)
    """
    src_pad = src_ids.eq(PAD_ID)         # (B, S)
    tgt_pad = dec_input_ids.eq(PAD_ID)   # (B, T)
    T = dec_input_ids.size(1)

    causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=src_ids.device), diagonal=1)

    enc_bool   = src_pad.unsqueeze(1) | src_pad.unsqueeze(2)
    dec_bool   = tgt_pad.unsqueeze(1) | tgt_pad.unsqueeze(2) | causal
    cross_bool = src_pad.unsqueeze(1) | tgt_pad.unsqueeze(2)

    zero = torch.tensor(0.0)
    neg  = torch.tensor(NEG_INFTY)
    return (torch.where(enc_bool,   neg, zero),
            torch.where(dec_bool,   neg, zero),
            torch.where(cross_bool, neg, zero))


def _pad(sequences, length):
    return torch.tensor(
        [seq + [PAD_ID] * (length - len(seq)) for seq in sequences],
        dtype=torch.long,
    )


def collate_batch(batch):
    """
    batch: list of (src_ids, tgt_ids, src_text, tgt_text)
    Returns dict of padded id tensors + masks (CPU; move to device in the loop).
    """
    src_seqs  = [ids + [EOS_ID] for ids, _, _, _ in batch]
    dec_in    = [[BOS_ID] + ids for _, ids, _, _ in batch]
    labels    = [ids + [EOS_ID] for _, ids, _, _ in batch]
    src_texts = [s for _, _, s, _ in batch]
    tgt_texts = [t for _, _, _, t in batch]

    S = max(len(s) for s in src_seqs)
    T = max(len(t) for t in dec_in)

    src_ids       = _pad(src_seqs, S)
    dec_input_ids = _pad(dec_in,   T)
    label_ids     = _pad(labels,   T)

    enc_mask, dec_mask, cross_mask = make_masks(src_ids, dec_input_ids)

    return {
        'src_ids'      : src_ids,
        'dec_input_ids': dec_input_ids,
        'label_ids'    : label_ids,
        'enc_mask'     : enc_mask,
        'dec_mask'     : dec_mask,
        'cross_mask'   : cross_mask,
        'src_texts'    : src_texts,
        'tgt_texts'    : tgt_texts,
    }


def greedy_translate(model, sp_src, sp_tgt, sentence, device, max_len=96, max_src_tokens=128):
    """Greedy autoregressive decode: source → target. Encoder runs once.

    Sources longer than max_src_tokens (the positional-encoding capacity)
    are truncated — the model never saw anything that long in training anyway.
    """
    model.eval()
    with torch.no_grad():
        src_ids = sp_src.encode(sentence)[:max_src_tokens - 1] + [EOS_ID]
        src = torch.tensor([src_ids], dtype=torch.long, device=device)
        out_ids = [BOS_ID]
        encoded = model.encoder(src, None)   # no mask needed: single unpadded sentence
        for _ in range(max_len):
            tgt = torch.tensor([out_ids], dtype=torch.long, device=device)
            T   = tgt.size(1)
            causal = torch.where(
                torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1),
                torch.tensor(NEG_INFTY, device=device), torch.tensor(0.0, device=device),
            ).unsqueeze(0)
            dec   = model.decoder(encoded, tgt, causal, None)
            logits = model.linear(dec)
            next_id = logits[0, -1].argmax().item()
            if next_id == EOS_ID:
                break
            out_ids.append(next_id)
    return sp_tgt.decode(out_ids[1:])
