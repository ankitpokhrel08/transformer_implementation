
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformer import Transformer

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

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


# ─────────────────────────────────────────────
# 1. SPECIAL TOKENS
# ─────────────────────────────────────────────

START_TOKEN   = '<START>'
PADDING_TOKEN = '<PADDING>'
END_TOKEN     = '<END>'


# ─────────────────────────────────────────────
# 2. VOCABULARIES
# ─────────────────────────────────────────────

sanskrit_vocabulary = [
    START_TOKEN, ' ', '!', '"', "'", '(', ')', ',', '-', '.', '?', ':', ';',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',

    # Independent vowels
    'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ऋ', 'ॠ', 'ऌ', 'ॡ', 'ए', 'ऐ', 'ओ', 'औ',

    # Consonants
    'क', 'ख', 'ग', 'घ', 'ङ',
    'च', 'छ', 'ज', 'झ', 'ञ',
    'ट', 'ठ', 'ड', 'ढ', 'ण',
    'त', 'थ', 'द', 'ध', 'न',
    'प', 'फ', 'ब', 'भ', 'म',
    'य', 'र', 'ल', 'व',
    'श', 'ष', 'स', 'ह',

    # Vowel signs (matras)
    'ा', 'ि', 'ी', 'ु', 'ू', 'ृ', 'ॄ', 'े', 'ै', 'ो', 'ौ',

    # Other signs
    'ं', 'ः', 'ँ', '्',   # anusvara, visarga, chandrabindu, virama
    '।', '॥',             # danda, double danda

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

    # Extended Latin / IAST transliteration
    'á', 'â', 'ã', 'ä', 'å', 'ç', 'é', 'î', 'ñ', 'ú', 'ü',
    'ă', 'ć', 'ę', 'ı', 'ļ', 'ł', 'ņ',
    'Ś', 'ś', 'Ş', 'ş', 'Š', 'š', 'ţ', 'ſ', 'ș', 'ț', 'ə',
    'ā', 'ī', 'ū', 'ṛ', 'ṝ', 'ḷ', 'ḹ', 'ṅ', 'ṭ', 'ḍ', 'ṇ', 'ṣ',
    'Ā', 'Ī', 'Ū', 'Ṛ', 'Ṝ', 'Ḷ', 'Ḹ', 'Ṅ', 'Ṭ', 'Ḍ', 'Ṇ', 'Ṣ',
    'о', 'О',     # Cyrillic (found in data)

    # Vietnamese
    'ả', 'ặ', 'ị',

    # Devanagari found in English translations
    'ं', 'उ', 'ए', 'क', 'च', 'त', 'द', 'ध', 'न', 'भ', 'म', 'र', 'ल', 'व', 'श', 'स',
    'ा', 'ि', 'ु', 'ै', 'ो', '्', '।', '॥',

    # Punctuation: en dash, em dash, curved quotes, curly apostrophe
    '–', '—', '\u2018', '\u2019', '\u201c', '\u201d',

    PADDING_TOKEN, END_TOKEN
]

# Build index maps
index_to_sanskrit = {k: v for k, v in enumerate(sanskrit_vocabulary)}
sanskrit_to_index = {v: k for k, v in enumerate(sanskrit_vocabulary)}
index_to_english  = {k: v for k, v in enumerate(english_vocabulary)}
english_to_index  = {v: k for k, v in enumerate(english_vocabulary)}


# ─────────────────────────────────────────────
# 3. LOAD & CLEAN DATA
# ─────────────────────────────────────────────

english_file  = '../data/train.en'
sanskrit_file = '../data/train.sn'

with open(english_file,  'r', encoding='utf-8') as f:
    english_sentences = f.readlines()
with open(sanskrit_file, 'r', encoding='utf-8') as f:
    sanskrit_sentences = f.readlines()

english_sentences  = [s.rstrip('\n') for s in english_sentences]
sanskrit_sentences = [s.rstrip('\n') for s in sanskrit_sentences]

assert len(english_sentences) == len(sanskrit_sentences), \
    f"Line count mismatch: {len(english_sentences)} English vs {len(sanskrit_sentences)} Sanskrit"

print(f"Total sentence pairs loaded: {len(english_sentences)}")


# ─────────────────────────────────────────────
# 4. HYPERPARAMETERS
# ─────────────────────────────────────────────

max_sequence_length = 256
d_model             = 512
batch_size          = 8
ffn_hidden          = 2048
num_heads           = 8
drop_prob           = 0.1
num_layers          = 6
NEG_INFTY           = -1e9

# FIX 2: src = Sanskrit (encoder input), tgt = English (decoder output)
src_vocab_size = len(sanskrit_vocabulary)
tgt_vocab_size = len(english_vocabulary)


# ─────────────────────────────────────────────
# 5. FILTER VALID SENTENCE PAIRS
# ─────────────────────────────────────────────

PERCENTILE = 97
print(f"{PERCENTILE}th percentile length — Sanskrit : "
      f"{np.percentile([len(x) for x in sanskrit_sentences], PERCENTILE):.0f}")
print(f"{PERCENTILE}th percentile length — English  : "
      f"{np.percentile([len(x) for x in english_sentences],  PERCENTILE):.0f}")


def is_valid_tokens(sentence: str, vocab: list) -> bool:
    """All characters in sentence must be in vocab."""
    vocab_set = set(vocab)
    return all(ch in vocab_set for ch in sentence)


def is_valid_length(sentence: str, max_len: int) -> bool:
    """Reserve 2 positions for <START> and <END>."""
    return len(sentence) < (max_len - 2)


valid_indices = []
for i, (skt, eng) in enumerate(zip(sanskrit_sentences, english_sentences)):
    if (is_valid_length(skt, max_sequence_length)
            and is_valid_length(eng, max_sequence_length)
            and is_valid_tokens(skt, sanskrit_vocabulary)
            and is_valid_tokens(eng, english_vocabulary)):
        valid_indices.append(i)

print(f"Total pairs      : {len(sanskrit_sentences)}")
print(f"Valid pairs      : {len(valid_indices)}")
print(f"Filtered out     : {len(sanskrit_sentences) - len(valid_indices)}")

english_sentences  = [english_sentences[i]  for i in valid_indices]
sanskrit_sentences = [sanskrit_sentences[i] for i in valid_indices]


# ─────────────────────────────────────────────
# 6. DATASET & DATALOADER
# ─────────────────────────────────────────────

class TextDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences):
        self.src = src_sentences   # Sanskrit (encoder input)
        self.tgt = tgt_sentences   # English  (decoder output)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]


# FIX 1: Sanskrit is source (encoder), English is target (decoder)
dataset      = TextDataset(sanskrit_sentences, english_sentences)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
print(f"Dataset size: {len(dataset)}  |  Batches per epoch: {len(train_loader)}")


# ─────────────────────────────────────────────
# 7. MODEL
# ─────────────────────────────────────────────

transformer = Transformer(
    d_model             = d_model,
    ffn_hidden          = ffn_hidden,
    num_heads           = num_heads,
    drop_prob           = drop_prob,
    num_layers          = num_layers,
    max_sequence_length = max_sequence_length,
    kn_vocab_size       = tgt_vocab_size,       # FIX 2: output vocab = English
    english_to_index    = english_to_index,     # decoder vocab
    sanskrit_to_index   = sanskrit_to_index,    # encoder vocab
    START_TOKEN         = START_TOKEN,
    END_TOKEN           = END_TOKEN,
    PADDING_TOKEN       = PADDING_TOKEN,
)
transformer.to(device)

# Xavier init for all weight matrices
for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)


# ─────────────────────────────────────────────
# 8. LOSS & OPTIMISER
# ─────────────────────────────────────────────

# FIX 3: Labels are English tokens, so ignore English padding index
criterion = nn.CrossEntropyLoss(
    ignore_index=english_to_index[PADDING_TOKEN],
    reduction='none'
)
optimiser = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)

def get_lr(step, d_model=512, warmup=4000):
    step = max(step, 1)
    return d_model**(-0.5) * min(step**(-0.5), step * warmup**(-1.5))


# ─────────────────────────────────────────────
# 9. MASK CREATION
# ─────────────────────────────────────────────

def create_masks(skt_batch, eng_batch):
    """
    skt_batch : list of Sanskrit strings  (encoder / source)
    eng_batch : list of English  strings  (decoder / target)

    Returns three float masks shaped (batch, seq, seq) ready to
    add to attention logits.  Masked positions receive NEG_INFTY.
    """
    num_sentences = len(skt_batch)

    # Causal (look-ahead) mask for decoder self-attention
    look_ahead_mask = torch.triu(
        torch.full((max_sequence_length, max_sequence_length), True), diagonal=1
    )  # (seq, seq)

    enc_pad   = torch.zeros(num_sentences, max_sequence_length, max_sequence_length, dtype=torch.bool)
    dec_self  = torch.zeros(num_sentences, max_sequence_length, max_sequence_length, dtype=torch.bool)
    dec_cross = torch.zeros(num_sentences, max_sequence_length, max_sequence_length, dtype=torch.bool)

    for i in range(num_sentences):
        skt_len = len(skt_batch[i])
        eng_len = len(eng_batch[i])

        # Padding positions (after real tokens + 1 for <END>)
        skt_pad_idx = np.arange(skt_len + 1, max_sequence_length)
        eng_pad_idx = np.arange(eng_len + 1, max_sequence_length)

        # Encoder self-attention: mask Sanskrit padding
        enc_pad[i, :, skt_pad_idx] = True
        enc_pad[i, skt_pad_idx, :] = True

        # Decoder self-attention: mask English padding + causal
        dec_self[i, :, eng_pad_idx] = True
        dec_self[i, eng_pad_idx, :] = True

        # Decoder cross-attention: Q=English, K/V=Sanskrit → mask Sanskrit padding cols
        dec_cross[i, :, skt_pad_idx] = True
        dec_cross[i, eng_pad_idx, :] = True

    encoder_self_attn_mask  = torch.where(enc_pad,                      NEG_INFTY, 0.0)
    decoder_self_attn_mask  = torch.where(look_ahead_mask | dec_self,   NEG_INFTY, 0.0)  # FIX 5: bitwise OR
    decoder_cross_attn_mask = torch.where(dec_cross,                    NEG_INFTY, 0.0)

    return encoder_self_attn_mask, decoder_self_attn_mask, decoder_cross_attn_mask


# ─────────────────────────────────────────────
# 10. CHECKPOINT HELPERS
# ─────────────────────────────────────────────

# FIX 9: Save checkpoints inside models/
os.makedirs('models', exist_ok=True)
CHECKPOINT_PATH = 'models/checkpoint.pth'
FINAL_MODEL_PATH = 'models/transformer_final.pth'


def save_checkpoint(epoch, loss_val):
    torch.save({
        'epoch'               : epoch,
        'model_state_dict'    : transformer.state_dict(),
        'optimizer_state_dict': optimiser.state_dict(),
        'loss'                : loss_val,
    }, CHECKPOINT_PATH)
    print(f"  [ckpt] Saved checkpoint → {CHECKPOINT_PATH}  (epoch {epoch}, loss {loss_val:.4f})")


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_PATH):
        print("No checkpoint found — starting from epoch 0.")
        return 0
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    transformer.load_state_dict(ckpt['model_state_dict'])
    optimiser.load_state_dict(ckpt['optimizer_state_dict'])
    start = ckpt['epoch'] + 1
    print(f"Checkpoint loaded — resuming from epoch {start}  (prev loss {ckpt['loss']:.4f})")
    return start


# ─────────────────────────────────────────────
# 11. INFERENCE (greedy decode)
# ─────────────────────────────────────────────

def translate(skt_sentence: str) -> str:
    """
    Greedy autoregressive decode: Sanskrit → English.
    FIX 8: was feeding English to encoder and building Sanskrit output.
    """
    transformer.eval()
    with torch.no_grad():
        skt_batch   = (skt_sentence,)
        eng_sentence = ""
        for _ in range(max_sequence_length):
            eng_batch = (eng_sentence,)
            # FIX 6: Sanskrit is source (first arg), English is target (second arg)
            enc_mask, dec_mask, cross_mask = create_masks(skt_batch, eng_batch)
            preds = transformer(
                skt_batch, eng_batch,
                enc_mask.to(device), dec_mask.to(device), cross_mask.to(device),
                enc_start_token=False, enc_end_token=False,
                dec_start_token=True,  dec_end_token=False,
            )
            # Predict next English token at the current position
            next_idx   = torch.argmax(preds[0][len(eng_sentence)]).item()
            # FIX 8: use index_to_english, not index_to_sanskrit
            next_token = index_to_english[next_idx]
            if next_token == END_TOKEN:
                break
            eng_sentence += next_token
    return eng_sentence


# ─────────────────────────────────────────────
# 12. TRAINING LOOP
# ─────────────────────────────────────────────

def train(num_epochs: int = 10, resume: bool = True):
    start_epoch = load_checkpoint() if resume else 0
    global_step = 0          

    transformer.train()
    transformer.to(device)

    for epoch in range(start_epoch, num_epochs):
        print(f"\n{'='*55}")
        print(f"  Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*55}")

        epoch_loss   = 0.0
        total_tokens = 0

        # FIX 1: skt_batch = Sanskrit (encoder), eng_batch = English (decoder)
        for batch_num, (skt_batch, eng_batch) in enumerate(train_loader):
            transformer.train()
            # ADD THESE 3 LINES:
            global_step += 1
            lr = get_lr(global_step)
            for g in optimiser.param_groups:
                g['lr'] = lr

            # FIX 6: Sanskrit first (source/encoder), English second (target/decoder)
            enc_mask, dec_mask, cross_mask = create_masks(skt_batch, eng_batch)

            optimiser.zero_grad()

            # Forward pass
            # Encoder input : Sanskrit  (no special tokens)
            # Decoder input : <START> + English  (teacher forcing, no <END>)
            # Decoder target: English + <END>    (no <START>)
            kn_preds = transformer(
                skt_batch, eng_batch,          # FIX 1: correct source/target order
                enc_mask.to(device), dec_mask.to(device), cross_mask.to(device),
                enc_start_token=False, enc_end_token=False,
                dec_start_token=True,  dec_end_token=True,
            )

            # FIX 4: Labels must be English tokens (decoder's vocabulary)
            labels = transformer.decoder.sentence_embedding.batch_tokenize(
                eng_batch, start_token=False, end_token=True
            )

            # FIX 3 & 5: use tgt_vocab_size (English) and English padding index
            loss = criterion(
                kn_preds.view(-1, tgt_vocab_size).to(device),
                labels.view(-1).to(device),
            )
            valid_mask = labels.view(-1) != english_to_index[PADDING_TOKEN]
            loss       = loss.sum() / valid_mask.sum()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0)
            optimiser.step()

            epoch_loss   += loss.item()
            total_tokens += valid_mask.sum().item()

            # ── Per-batch logging ──────────────────────────
            if batch_num % 100 == 0:
                print(f"\n  Iter {batch_num:>4}  loss: {loss.item():.4f}")
                print(f"  SKT : {skt_batch[0]}")
                print(f"  EN  : {eng_batch[0]}")

                # Greedy prediction from logits (training mode, no autoregress)
                pred_indices = torch.argmax(kn_preds[0], dim=1)
                pred_str = ""
                for idx in pred_indices:
                    # FIX 7: model outputs English indices
                    tok = index_to_english[idx.item()]
                    if tok == END_TOKEN:
                        break
                    pred_str += tok
                print(f"  PRD : {pred_str}")

                # Eval on a fixed Sanskrit sentence
                eval_result = translate("कर्म करो, फल की चिन्ता मत करो।")
                print(f"  EVAL: {eval_result}")
                print(f"  {'─'*50}")

        avg_loss = epoch_loss / max(len(train_loader), 1)
        print(f"\n  Epoch {epoch + 1} complete — avg loss: {avg_loss:.4f}")
        save_checkpoint(epoch, avg_loss)

    # Save final model weights separately
    torch.save(transformer.state_dict(), FINAL_MODEL_PATH)
    print(f"\n  [done] Final model saved → {FINAL_MODEL_PATH}")


# ─────────────────────────────────────────────
# 13. ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == '__main__':
    train(num_epochs=50, resume=True)

    # Quick inference demo — inputs are Sanskrit sentences
    test_sentences = [
        "कर्म करो, फल की चिन्ता मत करो।",
        "अहं ब्रह्मास्मि।",
        "सत्यमेव जयते।",
        "वसुधैव कुटुम्बकम्।",
    ]
    print("\n" + "="*55)
    print("INFERENCE DEMO  (Sanskrit → English)")
    print("="*55)
    for s in test_sentences:
        print(f"  SKT : {s}")
        print(f"  EN  : {translate(s)}")
        print()