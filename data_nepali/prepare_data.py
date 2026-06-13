"""
prepare_data.py — builds the English→Nepali training set from the OPUS NLLB dump.

    python3 prepare_data.py --top_n 1000000

Streams the 19.6M-pair corpus (never loads it into RAM):
  1. keep a bounded min-heap of the highest-LASER-score pairs
  2. clean: drop empties/extremes, wrong-script lines, bad length ratios
  3. dedupe, trim to top N, shuffle
  4. write train.en / train.ne here, and copy FLORES-200 dev/devtest
     as dev/test — the standard benchmark splits
"""

import argparse
import heapq
import os
import random
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--raw_dir', default='raw')
parser.add_argument('--top_n', type=int, default=1_000_000)
args = parser.parse_args()

EN_FILE     = os.path.join(args.raw_dir, 'NLLB.en-ne.en')
NE_FILE     = os.path.join(args.raw_dir, 'NLLB.en-ne.ne')
SCORES_FILE = os.path.join(args.raw_dir, 'NLLB.en-ne.scores')
FLORES      = os.path.join(args.raw_dir, 'flores200_dataset')

# Overshoot before dedupe so we still have top_n afterwards
HEAP_CAP = int(args.top_n * 1.2)


def devanagari_ratio(s):
    s = s[:100]
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    return sum('ऀ' <= c <= 'ॿ' for c in letters) / len(letters)


def latin_ratio(s):
    s = s[:100]
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    return sum(c.isascii() for c in letters) / len(letters)


heap = []   # min-heap of (score, en, ne)
seen_total = 0
dropped = {'empty/len': 0, 'script': 0, 'ratio': 0}

with open(EN_FILE, encoding='utf-8') as fe, \
     open(NE_FILE, encoding='utf-8') as fn, \
     open(SCORES_FILE) as fs:
    for en, ne, sc in zip(fe, fn, fs):
        seen_total += 1
        if seen_total % 2_000_000 == 0:
            print(f"  scanned {seen_total:,} | heap min score "
                  f"{heap[0][0]:.4f}" if heap else "", flush=True)
        sc = float(sc)
        # Cheapest gate first: most lines fail this once the heap is warm
        if len(heap) >= HEAP_CAP and sc <= heap[0][0]:
            continue
        en, ne = en.strip(), ne.strip()
        if not (5 <= len(en) <= 500 and 5 <= len(ne) <= 500):
            dropped['empty/len'] += 1
            continue
        r = len(en) / len(ne)
        if not (0.3 <= r <= 3.0):
            dropped['ratio'] += 1
            continue
        if devanagari_ratio(ne) < 0.5 or latin_ratio(en) < 0.5:
            dropped['script'] += 1
            continue
        if len(heap) < HEAP_CAP:
            heapq.heappush(heap, (sc, en, ne))
        else:
            heapq.heapreplace(heap, (sc, en, ne))

print(f"Scanned {seen_total:,} pairs | dropped at gates: {dropped}")
print(f"Candidates in heap: {len(heap):,} | score range "
      f"{heap[0][0]:.4f} .. {max(h[0] for h in heap):.4f}")

# Highest score first, dedupe exact pairs, trim to top_n
heap.sort(key=lambda t: t[0], reverse=True)
seen, pairs = set(), []
for sc, en, ne in heap:
    key = (en, ne)
    if key in seen:
        continue
    seen.add(key)
    pairs.append((en, ne))
    if len(pairs) >= args.top_n:
        break
print(f"After dedupe: kept {len(pairs):,}")

random.seed(42)
random.shuffle(pairs)   # de-correlate score order before epoch-level bucketing

with open('train.en', 'w', encoding='utf-8') as fe, \
     open('train.ne', 'w', encoding='utf-8') as fn:
    for en, ne in pairs:
        fe.write(en + '\n')
        fn.write(ne + '\n')

# FLORES-200 = standard dev/devtest benchmark splits
shutil.copy(os.path.join(FLORES, 'dev',     'eng_Latn.dev'),      'dev.en')
shutil.copy(os.path.join(FLORES, 'dev',     'npi_Deva.dev'),      'dev.ne')
shutil.copy(os.path.join(FLORES, 'devtest', 'eng_Latn.devtest'),  'test.en')
shutil.copy(os.path.join(FLORES, 'devtest', 'npi_Deva.devtest'),  'test.ne')

print("Wrote train.en/train.ne, dev.* (FLORES dev), test.* (FLORES devtest)")
