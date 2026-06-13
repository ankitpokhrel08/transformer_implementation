"""
evaluation.py  —  English → Nepali Transformer
Translates the FLORES-200 devtest benchmark, saves metrics as CSV and graphs as PNG.

Usage:
    python3 evaluation.py
    python3 evaluation.py --checkpoint models/en_ne_best.pth --n_samples 1012
    python3 evaluation.py --src ../data_nepali/dev.en --ref ../data_nepali/dev.ne
"""

import os
import argparse
import csv
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

try:
    from sacrebleu.metrics import BLEU, CHRF, TER
except ImportError:
    raise SystemExit("pip install sacrebleu")

from transformer import Transformer
from data_pipeline import load_tokenizers, greedy_translate

# ─────────────────────────────────────────────
# 0. ARGS
# ─────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint',  default='models/en_ne_best.pth')
parser.add_argument('--src',         default='../data_nepali/test.en')
parser.add_argument('--ref',         default='../data_nepali/test.ne')
parser.add_argument('--n_samples',   type=int, default=1012)
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
# 2. LOAD CHECKPOINT & BUILD MODEL
# ─────────────────────────────────────────────

if not os.path.exists(args.checkpoint):
    raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

sp_src, sp_tgt = load_tokenizers()

ckpt   = torch.load(args.checkpoint, map_location=device)
config = ckpt['config']
print(f"Model config: {config}")

transformer = Transformer(
    **config,
    src_vocab_size=sp_src.get_piece_size(),
    tgt_vocab_size=sp_tgt.get_piece_size(),
)
transformer.load_state_dict(ckpt['model_state_dict'])
transformer.to(device)
transformer.eval()
trained_epoch = ckpt['epoch']
trained_loss  = ckpt.get('dev_loss', float('nan'))
print(f"Loaded checkpoint — epoch {trained_epoch}, dev loss {trained_loss:.4f}")

# ─────────────────────────────────────────────
# 3. LOAD TEST DATA
# ─────────────────────────────────────────────

with open(args.src, encoding='utf-8') as f:
    src_sentences = [l.rstrip('\n') for l in f]
with open(args.ref, encoding='utf-8') as f:
    ref_sentences = [l.rstrip('\n') for l in f]

test_pairs = list(zip(src_sentences, ref_sentences))[:args.n_samples]
print(f"Evaluating on {len(test_pairs)} pairs from {args.src}")

# ─────────────────────────────────────────────
# 4. RUN INFERENCE
# ─────────────────────────────────────────────

# flores200 spm tokenizer scores Devanagari properly; fall back if this
# sacrebleu build doesn't ship it
try:
    bleu_metric = BLEU(tokenize='flores200', effective_order=True)
    print("BLEU tokenizer: flores200 (spm)")
except Exception:
    bleu_metric = BLEU(effective_order=True)
    print("BLEU tokenizer: default 13a — Devanagari BLEU will read low; trust ChrF")
chrf_metric  = CHRF()
ter_metric   = TER()

hypotheses   = []
references   = []
rows         = []   # for CSV

print("Running inference...")
for i, (src, ref) in enumerate(test_pairs):
    pred = greedy_translate(transformer, sp_src, sp_tgt, src, device)
    hypotheses.append(pred)
    references.append(ref)

    s_bleu = bleu_metric.sentence_score(pred, [ref]).score
    s_chrf = chrf_metric.sentence_score(pred, [ref]).score
    s_ter  = ter_metric.sentence_score(pred, [ref]).score

    rows.append({
        'index'       : i,
        'source'      : src,
        'reference'   : ref,
        'hypothesis'  : pred,
        'bleu'        : round(s_bleu, 4),
        'chrf'        : round(s_chrf, 4),
        'ter'         : round(s_ter,  4),
        'ref_len'     : len(ref),
        'hyp_len'     : len(pred),
        'src_len'     : len(src),
    })

    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(test_pairs)}", flush=True)

corpus_bleu = bleu_metric.corpus_score(hypotheses, [references]).score
corpus_chrf = chrf_metric.corpus_score(hypotheses, [references]).score
corpus_ter  = ter_metric.corpus_score(hypotheses,  [references]).score

print(f"\n{'='*45}")
print(f"  Corpus BLEU : {corpus_bleu:.2f}")
print(f"  Corpus ChrF : {corpus_chrf:.2f}")
print(f"  Corpus TER  : {corpus_ter:.2f}")
print(f"  Checkpoint  : epoch {trained_epoch}, dev loss {trained_loss:.4f}")
print(f"{'='*45}\n")

# ─────────────────────────────────────────────
# 5. SAVE METRICS CSV
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
    writer.writerow(['dev_loss',      round(trained_loss, 4)])
    writer.writerow(['n_samples',     len(test_pairs)])
print(f"Saved summary            → {summary_path}")

# ─────────────────────────────────────────────
# 6. GRAPHS
# ─────────────────────────────────────────────

bleu_scores = np.array([r['bleu'] for r in rows])
chrf_scores = np.array([r['chrf'] for r in rows])
ter_scores  = np.array([r['ter']  for r in rows])
src_lens    = np.array([r['src_len'] for r in rows])
ref_lens    = np.array([r['ref_len'] for r in rows])
hyp_lens    = np.array([r['hyp_len'] for r in rows])

# ── Shared professional style ─────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi'        : 120,
    'savefig.dpi'       : 200,
    'savefig.bbox'      : 'tight',
    'font.family'       : 'DejaVu Sans',
    'font.size'         : 11,
    'axes.titlesize'    : 13,
    'axes.titleweight'  : 'bold',
    'axes.labelsize'    : 11,
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'axes.grid'         : True,
    'grid.color'        : '#dddddd',
    'grid.linewidth'    : 0.8,
    'axes.axisbelow'    : True,
    'legend.frameon'    : False,
})
# Colorblind-friendly palette (Wong)
C_BLEU, C_CHRF, C_TER, C_ACCENT = '#0072B2', '#E69F00', '#009E73', '#D55E00'
SUBTITLE = f'English → Nepali  ·  FLORES-200 devtest  ·  epoch {trained_epoch}  ·  n = {len(test_pairs)}'


def _binned_trend(x, y, n_bins=12):
    """Mean y within equal-width x bins, for an uncluttered trend over noisy scatter."""
    edges   = np.linspace(x.min(), x.max(), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means   = np.full(n_bins, np.nan)
    for i in range(n_bins):
        sel = (x >= edges[i]) & (x < edges[i + 1] if i < n_bins - 1 else x <= edges[i + 1])
        if sel.any():
            means[i] = y[sel].mean()
    ok = ~np.isnan(means)
    return centers[ok], means[ok]


# ── Graph 1: Score distributions ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
fig.suptitle('Sentence-Level Score Distributions', fontsize=15, fontweight='bold', y=1.02)
fig.text(0.5, 0.965, SUBTITLE, ha='center', fontsize=10, color='#555555')

for ax, scores, label, color, better in zip(
    axes,
    [bleu_scores, chrf_scores, ter_scores],
    ['BLEU', 'ChrF', 'TER'],
    [C_BLEU, C_CHRF, C_TER],
    ['↑ higher is better', '↑ higher is better', '↓ lower is better'],
):
    ax.hist(scores, bins=30, color=color, edgecolor='white', alpha=0.9)
    ax.axvline(scores.mean(),   color=C_ACCENT, linestyle='--', linewidth=1.6, label=f'mean = {scores.mean():.1f}')
    ax.axvline(np.median(scores), color='#333333', linestyle=':', linewidth=1.6, label=f'median = {np.median(scores):.1f}')
    ax.set_title(f'{label}   ({better})')
    ax.set_xlabel('Sentence score')
    ax.set_ylabel('Count')
    ax.legend(loc='upper right')

plt.tight_layout()
p = os.path.join(args.out_dir, 'graph_score_distributions.png')
plt.savefig(p)
plt.close()
print(f"Saved → {p}")

# ── Graph 2: Source length vs BLEU (scatter + binned trend) ───────────────────
fig, ax = plt.subplots(figsize=(8.5, 5.5))
sc = ax.scatter(src_lens, bleu_scores, alpha=0.45, c=chrf_scores, cmap='viridis',
                s=22, edgecolor='none')
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Sentence ChrF', rotation=270, labelpad=15)
tx, ty = _binned_trend(src_lens, bleu_scores)
ax.plot(tx, ty, color=C_ACCENT, linewidth=2.5, marker='o', markersize=5,
        label='binned mean BLEU')
ax.set_xlabel('Source (English) length [characters]')
ax.set_ylabel('Sentence BLEU')
ax.set_title('Does translation quality degrade with sentence length?')
ax.text(0.5, 1.015, SUBTITLE, transform=ax.transAxes, ha='center',
        fontsize=9.5, color='#555555')
ax.legend(loc='upper right')
plt.tight_layout()
p = os.path.join(args.out_dir, 'graph_srclen_vs_bleu.png')
plt.savefig(p)
plt.close()
print(f"Saved → {p}")

# ── Graph 3: Reference vs Hypothesis length ───────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6.5))
ax.scatter(ref_lens, hyp_lens, alpha=0.4, s=20, color=C_BLEU, edgecolor='none')
lim = max(ref_lens.max(), hyp_lens.max()) * 1.02
ax.plot([0, lim], [0, lim], '--', color='#333333', linewidth=1.5, label='ideal (equal length)')
# fitted slope through origin — >1 means the model over-generates, <1 under-generates
slope = float(np.dot(ref_lens, hyp_lens) / np.dot(ref_lens, ref_lens))
ax.plot([0, lim], [0, slope * lim], color=C_ACCENT, linewidth=2,
        label=f'fit: hyp ≈ {slope:.2f} × ref')
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_aspect('equal')
ax.set_xlabel('Reference length [characters]')
ax.set_ylabel('Hypothesis length [characters]')
ax.set_title('Length calibration: is output too short or too long?')
ax.legend(loc='upper left')
plt.tight_layout()
p = os.path.join(args.out_dir, 'graph_ref_vs_hyp_length.png')
plt.savefig(p)
plt.close()
print(f"Saved → {p}")

# ── Graph 4: BLEU by source-length bucket (mean ± standard error) ──────────────
buckets     = defaultdict(list)
bucket_size = 20
for slen, bscore in zip(src_lens, bleu_scores):
    buckets[(slen // bucket_size) * bucket_size].append(bscore)

bucket_keys  = sorted(buckets.keys())
bucket_means = [np.mean(buckets[k]) for k in bucket_keys]
# standard error of the mean (std/sqrt(n)) — honest uncertainty, not raw spread
bucket_sems  = [np.std(buckets[k]) / max(np.sqrt(len(buckets[k])), 1) for k in bucket_keys]
bucket_n     = [len(buckets[k]) for k in bucket_keys]
labels       = [f'{k}–{k+bucket_size}' for k in bucket_keys]

fig, ax = plt.subplots(figsize=(10, 5.5))
bars = ax.bar(labels, bucket_means, yerr=bucket_sems, color=C_BLEU,
              edgecolor='white', capsize=4, alpha=0.9, error_kw={'ecolor': '#555555'})
for bar, n in zip(bars, bucket_n):
    ax.text(bar.get_x() + bar.get_width() / 2, 0.5, f'n={n}',
            ha='center', va='bottom', fontsize=8, color='white', rotation=90)
ax.axhline(corpus_bleu, color=C_ACCENT, linestyle='--', linewidth=1.6,
           label=f'corpus BLEU = {corpus_bleu:.1f}')
ax.set_xlabel('Source length bucket [characters]')
ax.set_ylabel('Mean sentence BLEU  (± SEM)')
ax.set_title('BLEU by source-length bucket')
ax.text(0.5, 1.015, SUBTITLE, transform=ax.transAxes, ha='center',
        fontsize=9.5, color='#555555')
ax.legend(loc='upper right')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
p = os.path.join(args.out_dir, 'graph_bleu_by_srclen_bucket.png')
plt.savefig(p)
plt.close()
print(f"Saved → {p}")

# ── Graph 5: BLEU vs ChrF correlation ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6.5))
ax.scatter(bleu_scores, chrf_scores, alpha=0.4, s=20, color=C_CHRF, edgecolor='none')
corr = np.corrcoef(bleu_scores, chrf_scores)[0, 1]
m, b = np.polyfit(bleu_scores, chrf_scores, 1)
xs = np.array([bleu_scores.min(), bleu_scores.max()])
ax.plot(xs, m * xs + b, color=C_ACCENT, linewidth=2, label=f'linear fit (r = {corr:.3f})')
ax.set_xlabel('Sentence BLEU')
ax.set_ylabel('Sentence ChrF')
ax.set_title('Agreement between BLEU and ChrF')
ax.legend(loc='lower right')
plt.tight_layout()
p = os.path.join(args.out_dir, 'graph_bleu_vs_chrf.png')
plt.savefig(p)
plt.close()
print(f"Saved → {p}")

# ── Graph 6: Corpus-level summary ─────────────────────────────────────────────
# BLEU/ChrF (higher=better) and TER (lower=better) live on different scales and
# directions, so they get separate panels rather than one misleading axis.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.8),
                               gridspec_kw={'width_ratios': [2, 1]})
fig.suptitle('Corpus-Level Metrics', fontsize=15, fontweight='bold', y=1.02)
fig.text(0.5, 0.95, SUBTITLE, ha='center', fontsize=10, color='#555555')

bars1 = ax1.bar(['BLEU', 'ChrF'], [corpus_bleu, corpus_chrf],
                color=[C_BLEU, C_CHRF], edgecolor='white', width=0.55)
for bar, val in zip(bars1, [corpus_bleu, corpus_chrf]):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             f'{val:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax1.set_ylim(0, max(corpus_bleu, corpus_chrf) * 1.25)
ax1.set_ylabel('Score')
ax1.set_title('Quality  (↑ higher is better)', fontsize=12)

bar2 = ax2.bar(['TER'], [corpus_ter], color=C_TER, edgecolor='white', width=0.45)
ax2.text(bar2[0].get_x() + bar2[0].get_width() / 2, corpus_ter + 0.5,
         f'{corpus_ter:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax2.set_ylim(0, corpus_ter * 1.25)
ax2.set_ylabel('Error rate')
ax2.set_title('Error  (↓ lower is better)', fontsize=12)

plt.tight_layout()
p = os.path.join(args.out_dir, 'graph_corpus_summary.png')
plt.savefig(p)
plt.close()
print(f"Saved → {p}")

# ─────────────────────────────────────────────
# 7. SAMPLE TRANSLATIONS (printed + saved)
# ─────────────────────────────────────────────

sample_path = os.path.join(args.out_dir, 'sample_translations.txt')
with open(sample_path, 'w', encoding='utf-8') as f:
    f.write(f"Epoch {trained_epoch}  |  Dev loss {trained_loss:.4f}  |  "
            f"BLEU {corpus_bleu:.2f}  ChrF {corpus_chrf:.2f}  TER {corpus_ter:.2f}\n")
    f.write("="*70 + "\n\n")
    sorted_rows = sorted(rows, key=lambda r: r['bleu'], reverse=True)
    for section, sample in [("TOP 10 (highest BLEU)", sorted_rows[:10]),
                              ("BOTTOM 10 (lowest BLEU)", sorted_rows[-10:]),
                              ("RANDOM 10", [rows[i] for i in np.random.choice(len(rows), 10, replace=False)])]:
        f.write(f"\n{'─'*70}\n{section}\n{'─'*70}\n")
        for r in sample:
            f.write(f"EN  : {r['source']}\n")
            f.write(f"REF : {r['reference']}\n")
            f.write(f"HYP : {r['hypothesis']}\n")
            f.write(f"     BLEU={r['bleu']:.2f}  ChrF={r['chrf']:.2f}  TER={r['ter']:.2f}\n\n")

print(f"Saved sample translations → {sample_path}")
print("\nDone. All outputs in:", args.out_dir)
