"""
evaluation.py  —  Sanskrit → English Transformer
Loads saved checkpoint, runs evaluation, saves metrics as CSV and graphs as PNG.

Usage:
    python3 evaluation.py
    python3 evaluation.py --checkpoint models/checkpoint.pth --n_samples 500
"""

import os
import argparse
import csv
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

try:
    from sacrebleu.metrics import BLEU, CHRF, TER
except ImportError:
    raise SystemExit("pip install sacrebleu")

from transformer import Transformer

# ─────────────────────────────────────────────
# 0. ARGS
# ─────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint',  default='models/checkpoint.pth')
parser.add_argument('--n_samples',   type=int, default=500)
parser.add_argument('--out_dir',     default='eval_results')
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# ─────────────────────────────────────────────
# 1. DEVICE
# ─────────────────────────────────────────────

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"Device: {device}")

# ─────────────────────────────────────────────
# 2. VOCAB (copy from training.py)
# ─────────────────────────────────────────────

START_TOKEN   = '<START>'
PADDING_TOKEN = '<PADDING>'
END_TOKEN     = '<END>'

sanskrit_vocabulary = [
    START_TOKEN, ' ', '!', '"', "'", '(', ')', ',', '-', '.', '?', ':', ';',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ऋ', 'ॠ', 'ऌ', 'ॡ', 'ए', 'ऐ', 'ओ', 'औ',
    'क', 'ख', 'ग', 'घ', 'ङ',
    'च', 'छ', 'ज', 'झ', 'ञ',
    'ट', 'ठ', 'ड', 'ढ', 'ण',
    'त', 'थ', 'द', 'ध', 'न',
    'प', 'फ', 'ब', 'भ', 'म',
    'य', 'र', 'ल', 'व',
    'श', 'ष', 'स', 'ह',
    'ा', 'ि', 'ी', 'ु', 'ू', 'ृ', 'ॄ', 'े', 'ै', 'ो', 'ौ',
    'ं', 'ः', 'ँ', '्',
    '।', '॥',
    PADDING_TOKEN, END_TOKEN
]

english_vocabulary = [
    START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+',
    ',', '-', '.', '/',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    ':', ';', '<', '=', '>', '?', '@',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
    'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    '_',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
    'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    '{', '|', '}', '~', '·', 'º',
    'á', 'â', 'ã', 'ä', 'å', 'ç', 'é', 'î', 'ñ', 'ú', 'ü',
    'ă', 'ć', 'ę', 'ı', 'ļ', 'ł', 'ņ',
    'Ś', 'ś', 'Ş', 'ş', 'Š', 'š', 'ţ', 'ſ', 'ș', 'ț', 'ə',
    'ā', 'ī', 'ū', 'ṛ', 'ṝ', 'ḷ', 'ḹ', 'ṅ', 'ṭ', 'ḍ', 'ṇ', 'ṣ',
    'Ā', 'Ī', 'Ū', 'Ṛ', 'Ṝ', 'Ḷ', 'Ḹ', 'Ṅ', 'Ṭ', 'Ḍ', 'Ṇ', 'Ṣ',
    'о', 'О',
    'ả', 'ặ', 'ị',
    'ं', 'उ', 'ए', 'क', 'च', 'त', 'द', 'ध', 'न', 'भ', 'म', 'र', 'ल', 'व', 'श', 'स',
    'ा', 'ि', 'ु', 'ै', 'ो', '्', '।', '॥',
    '–', '—', '\u2018', '\u2019', '\u201c', '\u201d',
    PADDING_TOKEN, END_TOKEN
]

index_to_sanskrit = {k: v for k, v in enumerate(sanskrit_vocabulary)}
sanskrit_to_index = {v: k for k, v in enumerate(sanskrit_vocabulary)}
index_to_english  = {k: v for k, v in enumerate(english_vocabulary)}
english_to_index  = {v: k for k, v in enumerate(english_vocabulary)}

# ─────────────────────────────────────────────
# 3. HYPERPARAMETERS (must match training)
# ─────────────────────────────────────────────

max_sequence_length = 200
d_model             = 512
ffn_hidden          = 2048
num_heads           = 8
drop_prob           = 0.1
num_layers          = 6
NEG_INFTY           = -1e9

#Evaluation on Short Sentences (short.en/sn):
# max_sequence_length = 50
# d_model             = 128
# batch_size          = 8
# ffn_hidden          = 512
# num_heads           = 4
# drop_prob           = 0
# num_layers          = 6
# NEG_INFTY           = -1e9

tgt_vocab_size      = len(english_vocabulary)

# ─────────────────────────────────────────────
# 4. LOAD MODEL
# ─────────────────────────────────────────────

transformer = Transformer(
    d_model             = d_model,
    ffn_hidden          = ffn_hidden,
    num_heads           = num_heads,
    drop_prob           = drop_prob,
    num_layers          = num_layers,
    max_sequence_length = max_sequence_length,
    kn_vocab_size       = tgt_vocab_size,
    english_to_index    = english_to_index,
    sanskrit_to_index   = sanskrit_to_index,
    START_TOKEN         = START_TOKEN,
    END_TOKEN           = END_TOKEN,
    PADDING_TOKEN       = PADDING_TOKEN,
)

if not os.path.exists(args.checkpoint):
    raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

ckpt = torch.load(args.checkpoint, map_location=device)
transformer.load_state_dict(ckpt['model_state_dict'])
transformer.to(device)
transformer.eval()
trained_epoch = ckpt['epoch']
trained_loss  = ckpt['loss']
print(f"Loaded checkpoint — epoch {trained_epoch}, loss {trained_loss:.4f}")

# ─────────────────────────────────────────────
# 5. LOAD DATA & FILTER (same logic as training)
# ─────────────────────────────────────────────

english_file  = '../data/short.en'
sanskrit_file = '../data/short.sn'

with open(english_file,  'r', encoding='utf-8') as f:
    english_sentences = [l.rstrip('\n') for l in f.readlines()]
with open(sanskrit_file, 'r', encoding='utf-8') as f:
    sanskrit_sentences = [l.rstrip('\n') for l in f.readlines()]

def is_valid_tokens(sentence, vocab):
    vocab_set = set(vocab)
    return all(ch in vocab_set for ch in sentence)

def is_valid_length(sentence, max_len):
    return len(sentence) < (max_len - 2)

valid_pairs = [
    (skt, eng)
    for skt, eng in zip(sanskrit_sentences, english_sentences)
    if is_valid_length(skt, max_sequence_length)
    and is_valid_length(eng, max_sequence_length)
    and is_valid_tokens(skt, sanskrit_vocabulary)
    and is_valid_tokens(eng, english_vocabulary)
]

# Use last N pairs as test set (these were seen during training,
# swap to a held-out file if you have one)
test_pairs = valid_pairs[-args.n_samples:]
print(f"Evaluating on {len(test_pairs)} pairs")

# ─────────────────────────────────────────────
# 6. MASK HELPER
# ─────────────────────────────────────────────

def create_masks(skt_batch, eng_batch):
    num_sentences   = len(skt_batch)
    look_ahead_mask = torch.triu(
        torch.full((max_sequence_length, max_sequence_length), True), diagonal=1
    )
    enc_pad   = torch.zeros(num_sentences, max_sequence_length, max_sequence_length, dtype=torch.bool)
    dec_self  = torch.zeros(num_sentences, max_sequence_length, max_sequence_length, dtype=torch.bool)
    dec_cross = torch.zeros(num_sentences, max_sequence_length, max_sequence_length, dtype=torch.bool)

    for i in range(num_sentences):
        skt_len     = len(skt_batch[i])
        eng_len     = len(eng_batch[i])
        skt_pad_idx = np.arange(skt_len + 1, max_sequence_length)
        eng_pad_idx = np.arange(eng_len + 1, max_sequence_length)
        enc_pad[i, :, skt_pad_idx]  = True
        enc_pad[i, skt_pad_idx, :]  = True
        dec_self[i, :, eng_pad_idx] = True
        dec_self[i, eng_pad_idx, :] = True
        dec_cross[i, :, skt_pad_idx] = True
        dec_cross[i, eng_pad_idx, :] = True

    enc_mask   = torch.where(enc_pad,                     NEG_INFTY, 0.0)
    dec_mask   = torch.where(look_ahead_mask | dec_self,  NEG_INFTY, 0.0)
    cross_mask = torch.where(dec_cross,                   NEG_INFTY, 0.0)
    return enc_mask, dec_mask, cross_mask

# ─────────────────────────────────────────────
# 7. TRANSLATE (greedy)
# ─────────────────────────────────────────────

def translate(skt_sentence):
    with torch.no_grad():
        skt_batch    = (skt_sentence,)
        eng_sentence = ""
        for _ in range(max_sequence_length):
            eng_batch  = (eng_sentence,)
            enc_mask, dec_mask, cross_mask = create_masks(skt_batch, eng_batch)
            preds = transformer(
                skt_batch, eng_batch,
                enc_mask.to(device), dec_mask.to(device), cross_mask.to(device),
                enc_start_token=False, enc_end_token=False,
                dec_start_token=True,  dec_end_token=False,
            )
            next_idx   = torch.argmax(preds[0][len(eng_sentence)]).item()
            next_token = index_to_english[next_idx]
            if next_token == END_TOKEN:
                break
            eng_sentence += next_token
    return eng_sentence

# ─────────────────────────────────────────────
# 8. RUN INFERENCE
# ─────────────────────────────────────────────

bleu_metric  = BLEU(effective_order=True)
chrf_metric  = CHRF()
ter_metric   = TER()

hypotheses   = []
references   = []
rows         = []   # for CSV

print("Running inference...")
for i, (skt, ref) in enumerate(test_pairs):
    pred = translate(skt)
    hypotheses.append(pred)
    references.append(ref)

    # sentence-level scores
    s_bleu = bleu_metric.sentence_score(pred, [ref]).score
    s_chrf = chrf_metric.sentence_score(pred, [ref]).score
    s_ter  = ter_metric.sentence_score(pred, [ref]).score

    rows.append({
        'index'       : i,
        'sanskrit'    : skt,
        'reference'   : ref,
        'hypothesis'  : pred,
        'bleu'        : round(s_bleu, 4),
        'chrf'        : round(s_chrf, 4),
        'ter'         : round(s_ter,  4),
        'ref_len'     : len(ref),
        'hyp_len'     : len(pred),
        'src_len'     : len(skt),
    })

    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(test_pairs)}")

# corpus-level
corpus_bleu = bleu_metric.corpus_score(hypotheses, [references]).score
corpus_chrf = chrf_metric.corpus_score(hypotheses, [references]).score
corpus_ter  = ter_metric.corpus_score(hypotheses,  [references]).score

print(f"\n{'='*45}")
print(f"  Corpus BLEU : {corpus_bleu:.2f}")
print(f"  Corpus ChrF : {corpus_chrf:.2f}")
print(f"  Corpus TER  : {corpus_ter:.2f}")
print(f"  Checkpoint  : epoch {trained_epoch}, loss {trained_loss:.4f}")
print(f"{'='*45}\n")

# ─────────────────────────────────────────────
# 9. SAVE METRICS CSV
# ─────────────────────────────────────────────

csv_path = os.path.join(args.out_dir, 'metrics_per_sample.csv')
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f"Saved per-sample metrics → {csv_path}")

summary_path = os.path.join(args.out_dir, 'summary.csv')
with open(summary_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['metric', 'value'])
    writer.writerow(['corpus_bleu',   round(corpus_bleu, 4)])
    writer.writerow(['corpus_chrf',   round(corpus_chrf, 4)])
    writer.writerow(['corpus_ter',    round(corpus_ter,  4)])
    writer.writerow(['trained_epoch', trained_epoch])
    writer.writerow(['trained_loss',  round(trained_loss, 4)])
    writer.writerow(['n_samples',     len(test_pairs)])
print(f"Saved summary            → {summary_path}")

# ─────────────────────────────────────────────
# 10. GRAPHS
# ─────────────────────────────────────────────

bleu_scores = [r['bleu'] for r in rows]
chrf_scores = [r['chrf'] for r in rows]
ter_scores  = [r['ter']  for r in rows]
src_lens    = [r['src_len'] for r in rows]
ref_lens    = [r['ref_len'] for r in rows]
hyp_lens    = [r['hyp_len'] for r in rows]

# ── Graph 1: Score distributions ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle(f'Score Distributions  (epoch {trained_epoch}, n={len(test_pairs)})', fontsize=13)

for ax, scores, label, color in zip(
    axes,
    [bleu_scores, chrf_scores, ter_scores],
    ['BLEU', 'ChrF', 'TER'],
    ['steelblue', 'darkorange', 'seagreen']
):
    ax.hist(scores, bins=30, color=color, edgecolor='white', alpha=0.85)
    ax.axvline(np.mean(scores), color='red', linestyle='--', linewidth=1.5, label=f'mean={np.mean(scores):.2f}')
    ax.set_title(label)
    ax.set_xlabel('Score')
    ax.set_ylabel('Count')
    ax.legend()

plt.tight_layout()
p = os.path.join(args.out_dir, 'graph_score_distributions.png')
plt.savefig(p, dpi=150)
plt.close()
print(f"Saved → {p}")

# ── Graph 2: Source length vs BLEU ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(src_lens, bleu_scores, alpha=0.4, c=chrf_scores, cmap='plasma', s=18)
plt.colorbar(sc, ax=ax, label='ChrF Score')
ax.set_xlabel('Source (Sanskrit) Length (chars)')
ax.set_ylabel('Sentence BLEU')
ax.set_title('Source Length vs BLEU')
plt.tight_layout()
p = os.path.join(args.out_dir, 'graph_srclen_vs_bleu.png')
plt.savefig(p, dpi=150)
plt.close()
print(f"Saved → {p}")

# ── Graph 3: Reference vs Hypothesis length ───────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(ref_lens, hyp_lens, alpha=0.35, s=15, color='steelblue')
max_len = max(max(ref_lens), max(hyp_lens))
ax.plot([0, max_len], [0, max_len], 'r--', linewidth=1.5, label='perfect length')
ax.set_xlabel('Reference Length (chars)')
ax.set_ylabel('Hypothesis Length (chars)')
ax.set_title('Reference vs Predicted Length')
ax.legend()
plt.tight_layout()
p = os.path.join(args.out_dir, 'graph_ref_vs_hyp_length.png')
plt.savefig(p, dpi=150)
plt.close()
print(f"Saved → {p}")

# ── Graph 4: BLEU bucketed by source length ───────────────────────────────────
buckets     = defaultdict(list)
bucket_size = 20
for slen, bscore in zip(src_lens, bleu_scores):
    bucket = (slen // bucket_size) * bucket_size
    buckets[bucket].append(bscore)

bucket_keys  = sorted(buckets.keys())
bucket_means = [np.mean(buckets[k]) for k in bucket_keys]
bucket_stds  = [np.std(buckets[k])  for k in bucket_keys]

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(
    [str(k) for k in bucket_keys], bucket_means,
    yerr=bucket_stds, color='steelblue', edgecolor='white',
    capsize=4, alpha=0.85
)
ax.set_xlabel('Source Length Bucket (chars)')
ax.set_ylabel('Mean BLEU')
ax.set_title('BLEU by Source Length Bucket')
plt.xticks(rotation=45)
plt.tight_layout()
p = os.path.join(args.out_dir, 'graph_bleu_by_srclen_bucket.png')
plt.savefig(p, dpi=150)
plt.close()
print(f"Saved → {p}")

# ── Graph 5: BLEU vs ChrF correlation ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(bleu_scores, chrf_scores, alpha=0.35, s=15, color='darkorange')
ax.set_xlabel('Sentence BLEU')
ax.set_ylabel('Sentence ChrF')
ax.set_title('BLEU vs ChrF Correlation')
corr = np.corrcoef(bleu_scores, chrf_scores)[0, 1]
ax.text(0.05, 0.92, f'r = {corr:.3f}', transform=ax.transAxes, fontsize=11)
plt.tight_layout()
p = os.path.join(args.out_dir, 'graph_bleu_vs_chrf.png')
plt.savefig(p, dpi=150)
plt.close()
print(f"Saved → {p}")

# ── Graph 6: Summary bar chart ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
metrics = ['BLEU', 'ChrF', 'TER']
values  = [corpus_bleu, corpus_chrf, corpus_ter]
colors  = ['steelblue', 'darkorange', 'seagreen']
bars = ax.bar(metrics, values, color=colors, edgecolor='white', width=0.5)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylim(0, max(values) * 1.2)
ax.set_ylabel('Score')
ax.set_title(f'Corpus-Level Metrics  (epoch {trained_epoch})')
plt.tight_layout()
p = os.path.join(args.out_dir, 'graph_corpus_summary.png')
plt.savefig(p, dpi=150)
plt.close()
print(f"Saved → {p}")

# ─────────────────────────────────────────────
# 11. SAMPLE TRANSLATIONS (printed + saved)
# ─────────────────────────────────────────────

sample_path = os.path.join(args.out_dir, 'sample_translations.txt')
with open(sample_path, 'w', encoding='utf-8') as f:
    f.write(f"Epoch {trained_epoch}  |  Loss {trained_loss:.4f}  |  "
            f"BLEU {corpus_bleu:.2f}  ChrF {corpus_chrf:.2f}  TER {corpus_ter:.2f}\n")
    f.write("="*70 + "\n\n")
    # top 10 by BLEU, bottom 10 by BLEU, 10 random
    sorted_rows = sorted(rows, key=lambda r: r['bleu'], reverse=True)
    for section, sample in [("TOP 10 (highest BLEU)", sorted_rows[:10]),
                              ("BOTTOM 10 (lowest BLEU)", sorted_rows[-10:]),
                              ("RANDOM 10", [rows[i] for i in np.random.choice(len(rows), 10, replace=False)])]:
        f.write(f"\n{'─'*70}\n{section}\n{'─'*70}\n")
        for r in sample:
            f.write(f"SKT : {r['sanskrit']}\n")
            f.write(f"REF : {r['reference']}\n")
            f.write(f"HYP : {r['hypothesis']}\n")
            f.write(f"     BLEU={r['bleu']:.2f}  ChrF={r['chrf']:.2f}  TER={r['ter']:.2f}\n\n")

print(f"Saved sample translations → {sample_path}")
print("\nDone. All outputs in:", args.out_dir)