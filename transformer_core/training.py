
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformer import Transformer
from data_pipeline import (
    PAD_ID, load_tokenizers,
    TranslationDataset, BucketBatchSampler, collate_batch, greedy_translate,
)

# ─────────────────────────────────────────────
# 0. DEVICE
# ─────────────────────────────────────────────

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f"Using device: {device}")


# ─────────────────────────────────────────────
# 1. HYPERPARAMETERS
# ─────────────────────────────────────────────

# 1M NLLB pairs — one epoch here is ~14x the data of the whole Sanskrit run,
# so a handful of epochs is real training. Model scaled to stay laptop-sized.
max_sequence_length = 128    # positional-encoding capacity
max_tokens          = 96     # drop pairs longer than this many BPE tokens
batch_size          = 64
d_model             = 384
ffn_hidden          = 1536
num_heads           = 8
drop_prob           = 0.1
num_layers          = 4
warmup_steps        = 4000
label_smoothing     = 0.1
validate_every      = 4000   # mid-epoch dev check + checkpoint (epochs are long)

MODEL_CONFIG = {
    'd_model'            : d_model,
    'ffn_hidden'         : ffn_hidden,
    'num_heads'          : num_heads,
    'drop_prob'          : drop_prob,
    'num_layers'         : num_layers,
    'max_sequence_length': max_sequence_length,
}


# ─────────────────────────────────────────────
# 2. TOKENIZERS & DATA
# ─────────────────────────────────────────────

sp_src, sp_tgt = load_tokenizers()
src_vocab_size = sp_src.get_piece_size()
tgt_vocab_size = sp_tgt.get_piece_size()
print(f"Vocab sizes — English: {src_vocab_size}, Nepali: {tgt_vocab_size}")


def load_pairs(src_path, tgt_path):
    with open(src_path, encoding='utf-8') as f:
        src = [l.rstrip('\n') for l in f]
    with open(tgt_path, encoding='utf-8') as f:
        tgt = [l.rstrip('\n') for l in f]
    assert len(src) == len(tgt), f"Line count mismatch: {len(src)} vs {len(tgt)}"
    return src, tgt


train_src, train_tgt = load_pairs('../data_nepali/train.en', '../data_nepali/train.ne')
dev_src,   dev_tgt   = load_pairs('../data_nepali/dev.en',   '../data_nepali/dev.ne')

train_dataset = TranslationDataset(train_src, train_tgt, sp_src, sp_tgt, max_tokens)
dev_dataset   = TranslationDataset(dev_src,   dev_tgt,   sp_src, sp_tgt, max_tokens)
print(f"Train pairs: {len(train_dataset)} / {len(train_src)}  |  Dev pairs: {len(dev_dataset)} / {len(dev_src)}")

train_lengths = [len(tgt) for _, tgt, _, _ in train_dataset.pairs]
train_loader = DataLoader(
    train_dataset,
    batch_sampler=BucketBatchSampler(train_lengths, batch_size, shuffle=True),
    collate_fn=collate_batch,
)
dev_lengths = [len(tgt) for _, tgt, _, _ in dev_dataset.pairs]
dev_loader = DataLoader(
    dev_dataset,
    batch_sampler=BucketBatchSampler(dev_lengths, batch_size, shuffle=False),
    collate_fn=collate_batch,
)
print(f"Batches per epoch: {len(train_loader)}")


# ─────────────────────────────────────────────
# 3. MODEL
# ─────────────────────────────────────────────

transformer = Transformer(
    **MODEL_CONFIG,
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer.to(device)
n_params = sum(p.numel() for p in transformer.parameters())
print(f"Model parameters: {n_params/1e6:.1f}M")


# ─────────────────────────────────────────────
# 4. LOSS & OPTIMISER
# ─────────────────────────────────────────────

criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=label_smoothing)
optimiser = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)


def get_lr(step, warmup=warmup_steps):
    step = max(step, 1)
    return d_model**(-0.5) * min(step**(-0.5), step * warmup**(-1.5))


# ─────────────────────────────────────────────
# 5. CHECKPOINT HELPERS
# ─────────────────────────────────────────────

os.makedirs('models', exist_ok=True)
CHECKPOINT_PATH = 'models/en_ne_checkpoint.pth'   # latest, for resuming
BEST_PATH       = 'models/en_ne_best.pth'         # lowest dev loss


def save_checkpoint(path, epoch, global_step, dev_loss, best_dev_loss):
    torch.save({
        'epoch'               : epoch,
        'global_step'         : global_step,
        'config'              : MODEL_CONFIG,
        'model_state_dict'    : transformer.state_dict(),
        'optimizer_state_dict': optimiser.state_dict(),
        'dev_loss'            : dev_loss,
        'best_dev_loss'       : best_dev_loss,
    }, path)


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_PATH):
        print("No checkpoint found — starting from epoch 0.")
        return 0, 0, float('inf')
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    transformer.load_state_dict(ckpt['model_state_dict'])
    optimiser.load_state_dict(ckpt['optimizer_state_dict'])
    start = ckpt['epoch'] + 1
    print(f"Checkpoint loaded — resuming from epoch {start}, step {ckpt['global_step']}  "
          f"(dev loss {ckpt['dev_loss']:.4f})")
    return start, ckpt['global_step'], ckpt.get('best_dev_loss', ckpt['dev_loss'])


# ─────────────────────────────────────────────
# 6. EVALUATION HELPERS
# ─────────────────────────────────────────────

def evaluate_dev_loss():
    transformer.eval()
    total_loss, n_batches = 0.0, 0
    with torch.no_grad():
        for batch in dev_loader:
            preds = transformer(
                batch['src_ids'].to(device),
                batch['dec_input_ids'].to(device),
                batch['enc_mask'].to(device),
                batch['dec_mask'].to(device),
                batch['cross_mask'].to(device),
            )
            loss = criterion(
                preds.view(-1, tgt_vocab_size),
                batch['label_ids'].view(-1).to(device),
            )
            total_loss += loss.item()
            n_batches  += 1
    transformer.train()
    return total_loss / max(n_batches, 1)


EVAL_SENTENCES = [
    "The boy is reading a book.",
    "Nepal is a beautiful country with high mountains.",
]


def translate(sentence):
    result = greedy_translate(transformer, sp_src, sp_tgt, sentence, device, max_len=max_tokens)
    transformer.train()
    return result


# ─────────────────────────────────────────────
# 7. TRAINING LOOP
# ─────────────────────────────────────────────

def train(num_epochs: int = 6, resume: bool = True):
    start_epoch, global_step, best_dev_loss = load_checkpoint() if resume else (0, 0, float('inf'))
    transformer.train()

    def validate_and_checkpoint(epoch):
        nonlocal best_dev_loss
        dev_loss = evaluate_dev_loss()
        save_checkpoint(CHECKPOINT_PATH, epoch, global_step, dev_loss, best_dev_loss)
        marker = ""
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            save_checkpoint(BEST_PATH, epoch, global_step, dev_loss, best_dev_loss)
            marker = "  ← new best"
        print(f"  [ckpt] step {global_step}  dev loss: {dev_loss:.4f}{marker}", flush=True)
        return dev_loss

    for epoch in range(start_epoch, num_epochs):
        print(f"\n{'='*55}")
        print(f"  Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*55}", flush=True)

        epoch_loss = 0.0
        epoch_start = time.time()

        for batch_num, batch in enumerate(train_loader):
            global_step += 1
            lr = get_lr(global_step)
            for g in optimiser.param_groups:
                g['lr'] = lr

            optimiser.zero_grad()
            preds = transformer(
                batch['src_ids'].to(device),
                batch['dec_input_ids'].to(device),
                batch['enc_mask'].to(device),
                batch['dec_mask'].to(device),
                batch['cross_mask'].to(device),
            )
            loss = criterion(
                preds.view(-1, tgt_vocab_size),
                batch['label_ids'].view(-1).to(device),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0)
            optimiser.step()
            epoch_loss += loss.item()

            if batch_num % 500 == 0:
                elapsed = time.time() - epoch_start
                print(f"\n  Iter {batch_num:>5}/{len(train_loader)}  "
                      f"loss: {loss.item():.4f}  lr: {lr:.2e}  [{elapsed:.0f}s]")
                print(f"  EN  : {batch['src_texts'][0]}")
                print(f"  NE  : {batch['tgt_texts'][0]}")
                print(f"  TRN : {translate(batch['src_texts'][0])}")
                for s in EVAL_SENTENCES:
                    print(f"  EVAL: {s}  →  {translate(s)}")
                print(f"  {'─'*50}", flush=True)

            if global_step % validate_every == 0:
                validate_and_checkpoint(epoch)

        avg_loss  = epoch_loss / max(len(train_loader), 1)
        epoch_min = (time.time() - epoch_start) / 60
        print(f"\n  Epoch {epoch + 1} done in {epoch_min:.1f} min — train loss: {avg_loss:.4f}")
        validate_and_checkpoint(epoch)

    print("\n  [done] Training finished.")


# ─────────────────────────────────────────────
# 8. ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == '__main__':
    train(num_epochs=6, resume=True)

    test_sentences = [
        "The boy is reading a book.",
        "Nepal is a beautiful country with high mountains.",
        "I want to learn machine learning.",
        "The weather is very cold today.",
    ]
    print("\n" + "="*55)
    print("INFERENCE DEMO  (English → Nepali)")
    print("="*55)
    for s in test_sentences:
        print(f"  EN : {s}")
        print(f"  NE : {greedy_translate(transformer, sp_src, sp_tgt, s, device)}")
        print()
