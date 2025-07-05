# Sanskrit to English Translation using Transformer Architecture

A complete implementation of the Transformer architecture for neural machine translation from Sanskrit (Devanagari script) to English, built from scratch using PyTorch.

## 🏗️ Architecture Overview

This project implements the full Transformer architecture as described in "Attention Is All You Need" (Vaswani et al., 2017), including:

### Core Components

- **Multi-Head Self-Attention**: Scaled dot-product attention with multiple attention heads
- **Positional Encoding**: Sinusoidal position embeddings for sequence modeling
- **Feed-Forward Networks**: Position-wise fully connected layers
- **Layer Normalization**: Applied before each sub-layer (Pre-LN architecture)
- **Residual Connections**: Skip connections around each sub-layer

### Model Architecture

```
Input (Sanskrit) → Encoder → Context → Decoder → Output (English)
```

**Encoder Stack:**

- 1 layer (configurable) of Multi-Head Attention + FFN
- Input: Sanskrit character sequences (Devanagari script)
- Output: Contextualized representations

**Decoder Stack:**

- 1 layer (configurable) of Masked Multi-Head Attention + Cross-Attention + FFN
- Input: English character sequences (target)
- Output: English token probabilities

## 📊 Implementation Details

### Vocabulary & Tokenization

- **Character-level tokenization** for both Sanskrit and English
- **Sanskrit vocabulary**: 114 characters (Devanagari script + punctuation)
- **English vocabulary**: 157 characters (Latin alphabet + extended characters)
- **Special tokens**: `START_TOKEN`, `END_TOKEN`, `PADDING_TOKEN`

### Model Configuration

```python
d_model = 512          # Model dimension
num_heads = 8          # Attention heads
ffn_hidden = 2048      # Feed-forward hidden size
num_layers = 1         # Encoder/Decoder layers
max_seq_length = 200   # Maximum sequence length
dropout = 0.1          # Dropout rate
```

### Training Setup

- **Loss function**: CrossEntropyLoss with padding token masking
- **Optimizer**: Adam (lr=1e-4)
- **Device**: CUDA GPU support with automatic fallback to CPU
- **Batch size**: 30 sequences

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch numpy matplotlib
```

### Project Structure

```
sanskrit_to_english/
├── transformer/
│   ├── transformer.py           # Core transformer implementation
│   ├── final_transformer.ipynb  # Training and inference notebook
│   └── notes/                   # Architecture diagrams
├── data/
│   ├── dev.en                   # English sentences
│   └── dev.sn                   # Sanskrit sentences
└── readme.md
```

### Running the Model

1. **Load the notebook**: Open `transformer/final_transformer.ipynb`
2. **Data preparation**: The notebook automatically builds vocabularies from actual data
3. **Training**: Run the training loop (supports both CPU and GPU)
4. **Inference**: Use `translate_sanskrit_to_english()` function

```python
# Example usage
translation = translate_sanskrit_to_english("नमस्ते")
print(translation)  # Output: English translation
```

## 📈 Training Process

### Data Pipeline

1. **Vocabulary Construction**: Scan all characters in the dataset
2. **Sentence Filtering**: Remove sentences exceeding max length or containing unknown characters
3. **Masking**: Create attention masks for padding and look-ahead constraints
4. **Batching**: Group sentences into batches for efficient training

### Attention Masks

- **Encoder Self-Attention**: Padding mask only
- **Decoder Self-Attention**: Causal (look-ahead) + padding mask
- **Decoder Cross-Attention**: Padding mask for encoder outputs

## 🔍 Key Insights & Lessons Learned

### ⚠️ Limitations of Current Approach

#### 1. **Character-Level Tokenization Issues**

- **Morphological complexity**: Sanskrit has rich morphology that character-level tokenization fails to capture
- **Compound words**: Sanskrit compounds are not properly segmented
- **Contextual meaning**: Characters alone don't preserve semantic units

#### 2. **Single-Layer Architecture**

- **Limited capacity**: Only 1 encoder/decoder layer severely limits model expressiveness
- **Poor long-range dependencies**: Cannot capture complex linguistic relationships
- **Underfitting**: Model lacks capacity for the complexity of Sanskrit-English translation

#### 3. **Dataset Limitations**

- **Size**: Limited training data (~4,633 valid sentence pairs)
- **Domain**: Narrow domain coverage affects generalization
- **Quality**: Character-level alignment may not reflect proper translation units

### 🎯 Better Approaches

#### 1. **Subword Tokenization**

```python
# Recommended: Use BPE or SentencePiece
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
```

#### 2. **Increased Model Capacity**

```python
# Better configuration
num_layers = 6        # Standard transformer depth
d_model = 512         # Keep reasonable for computational efficiency
num_heads = 8         # Multi-head attention
```

#### 3. **Pre-trained Models**

- **mBERT**: Multilingual BERT with Sanskrit support
- **IndicBERT**: Specialized for Indic languages
- **mT5**: Multilingual T5 for translation tasks

#### 4. **Word-Level or Subword-Level Translation**

```python
# Better tokenization preserves meaning
sanskrit_word = "नमस्कार"  # Complete word unit
vs_chars = ["न", "म", "स्", "क", "ा", "र"]  # Character fragments
```

## 📊 Results & Performance

### Current Model Performance

- **Training**: Model learns to optimize character-level cross-entropy loss
- **Translation quality**: Limited due to character-level approach and shallow architecture
- **Speed**: Fast inference due to simple architecture

### Expected Improvements with Better Architecture

- **BLEU Score**: Would significantly improve with proper tokenization
- **Semantic accuracy**: Better with word/subword units
- **Fluency**: Improved with deeper models and larger datasets

## 🔧 Technical Implementation Highlights

### Custom Transformer Components

```python
class MultiHeadAttention(nn.Module):
    # Implements scaled dot-product attention

class PositionalEncoding(nn.Module):
    # Sinusoidal position embeddings

class SentenceEmbedding(nn.Module):
    # Character-level embedding + positional encoding
```

### GPU Optimization

- Automatic CUDA detection and memory management
- Efficient tensor operations on GPU
- Memory monitoring during training

### Masking Strategy

```python
def create_masks(src_batch, tgt_batch):
    # Creates all necessary attention masks
    # - Padding masks for variable-length sequences
    # - Causal masks for autoregressive generation
```

## 🚦 Future Improvements

1. **Tokenization**: Implement BPE/SentencePiece tokenization
2. **Architecture**: Increase to 6+ layers for better capacity
3. **Dataset**: Use larger, more diverse Sanskrit-English parallel corpora
4. **Evaluation**: Implement BLEU score and other translation metrics
5. **Beam Search**: Add beam search decoding for better translations
6. **Fine-tuning**: Start from pre-trained multilingual models

## 📚 References

- Vaswani, A., et al. (2017). "Attention Is All You Need"
- Devlin, J., et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers"
- Kenton, J. D. M. W. C., & Toutanova, L. K. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"

## 🎓 Learning Outcomes

This project demonstrates:

- **Complete Transformer implementation** from scratch
- **Understanding of attention mechanisms** and their importance
- **Practical challenges** in neural machine translation
- **Why modern approaches** use subword tokenization and pre-trained models
- **Importance of model capacity** and architectural choices

---

**Note**: This implementation serves as an educational example of transformer architecture. For production Sanskrit-English translation, consider using pre-trained multilingual models with proper subword tokenization.
