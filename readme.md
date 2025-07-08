# Sanskrit-to-English Neural Machine Translation 🕉️

A character-level Transformer model for translating English text to Sanskrit using PyTorch, with a beautiful Streamlit web interface.

## 🎯 Overview

This project implements a complete neural machine translation system that converts English text to Sanskrit (Devanagari script) using a Transformer architecture with multi-head attention mechanisms.

## ✨ Features

- **🧠 Transformer Architecture**: Full encoder-decoder with multi-head attention
- **📝 Character-Level Translation**: Fine-grained tokenization for better accuracy
- **🌐 Web Interface**: Dark-themed Streamlit app for real-time testing
- **� Complete Vocabulary**: Comprehensive Sanskrit character set with IAST support
- **⚡ GPU/MPS Support**: Optimized for CUDA and Apple Silicon (M1/M2/M3)
- **💿 Model Persistence**: Checkpoint saving and vocabulary export

## 🏗️ Project Structure

```
sanskrit_to_english/
├── 📊 data/
│   ├── dev.en                    # English sentences
│   ├── dev.sn                    # Sanskrit sentences
│   └── cleaning.ipynb            # Data preprocessing
├── 🤖 transformer/
│   ├── transformer.py            # Main model implementation
│   ├── final_transformer.ipynb   # Training notebook
│   ├── working_transformer.ipynb # Development notebook
│   └── checkpoint.pth            # Trained model weights
├── 📦 models/
│   └── sanskrit_vocabulary.pkl   # Exported vocabulary data
├── 🌐 app.py                     # Streamlit web interface
├── 📋 requirements.txt           # Python dependencies
└── 🚀 run_app.sh                # Quick start script
```

## 🚀 Quick Start

### 1. **Install Dependencies**

```bash
pip install torch numpy streamlit matplotlib jupyter
```

### 2. **Train the Model** (Optional)

```bash
cd transformer/
jupyter notebook final_transformer.ipynb
# Run all cells to train from scratch
```

### 3. **Launch Web App**

```bash
chmod +x run_app.sh
./run_app.sh
```

### 4. **Use the Model**

- Open `http://localhost:8501` in your browser
- Enter English text: _"Your right is to perform your duty only"_
- Get Sanskrit output: _कर्मण्येवाधिकारस्ते मा फलेषु कदाचन_

## 🧠 Model Architecture

- **Type**: Encoder-Decoder Transformer
- **Dimensions**: 512 (d_model), 2048 (FFN)
- **Attention Heads**: 8
- **Layers**: 1 (configurable)
- **Vocabulary**: 89 Sanskrit + 183 English characters
- **Max Length**: 200 characters
- **Training**: Adam optimizer, Cross-entropy loss

## 📊 Vocabulary Coverage

### Sanskrit (89 tokens):

- **Vowels**: अ आ इ ई उ ऊ ऋ ॠ ऌ ॡ ए ऐ ओ औ
- **Consonants**: All 33 Devanagari consonants
- **Diacritics**: ा ि ी ु ू ृ ॄ े ै ो ौ
- **Special**: ं ः ँ ् । ॥ (anusvara, visarga, virama, etc.)

### English (183 tokens):

- Standard ASCII + Extended Latin + IAST transliteration
- Unicode support for scholarly texts

## 🌐 Web Interface

The Streamlit app provides:

- **🌑 Dark Theme**: Elegant black background with Sanskrit colors
- **⚡ Real-time Translation**: Character-by-character generation
- **📊 Model Info**: Vocabulary sizes, parameters, device status
- **📖 Examples**: Built-in test sentences
- **📋 Copy Support**: Easy copying of Sanskrit output

### Sample Translations:

- _"I am here"_ → _अहम् अत्र अस्मि_
- _"Do work don't expect result"_ → _कर्म कुर्वन्तु फलं मा प्रत्याशयन्तु_

## 💻 Usage Examples

### Command Line (Python):

```python
from transformer import Transformer
import pickle

# Load vocabulary
with open('models/sanskrit_vocabulary.pkl', 'rb') as f:
    vocab_data = pickle.load(f)

# Initialize and load model
transformer = Transformer(...)  # with vocab_data parameters
checkpoint = torch.load('transformer/checkpoint.pth')
transformer.load_state_dict(checkpoint['model_state_dict'])

# Translate
result = translate("Your text here")
print(result)  # Sanskrit output
```

### Web Interface:

```bash
streamlit run app.py
```

## 🔧 Technical Details

- **Framework**: PyTorch 2.0+
- **Training Data**: English-Sanskrit parallel corpus
- **Tokenization**: Character-level with special tokens
- **Attention**: Multi-head self-attention + cross-attention
- **Masking**: Look-ahead and padding masks
- **Device Support**: CUDA, MPS (Apple Silicon), CPU

## � Training Process

1. **Data Preparation**: Filter valid sentence pairs
2. **Vocabulary Building**: Character-level tokenization
3. **Model Initialization**: Xavier uniform weights
4. **Training Loop**: 50+ epochs with checkpoint saving
5. **Evaluation**: Real-time translation testing

## 🎯 Performance

- **Training**: ~50 epochs on parallel corpus
- **Inference**: Real-time character generation
- **Memory**: Efficient checkpoint system
- **Accuracy**: Contextual Sanskrit generation

## �️ Customization

### Model Parameters:

```python
d_model = 512        # Model dimension
num_heads = 8        # Attention heads
num_layers = 1       # Transformer layers
max_length = 200     # Sequence length
```

### Training Configuration:

```python
batch_size = 30      # Training batch size
learning_rate = 1e-4 # Adam learning rate
epochs = 50          # Training epochs
```

## 🐛 Troubleshooting

### Common Issues:

- **Model not loading**: Check `checkpoint.pth` exists
- **Vocabulary error**: Ensure `sanskrit_vocabulary.pkl` is present
- **Import errors**: Install PyTorch and dependencies
- **CUDA/MPS**: Model auto-detects available device

### File Verification:

```bash
ls transformer/checkpoint.pth models/sanskrit_vocabulary.pkl
```

## 🔍 Key Insights & Limitations

### Current Approach:

- **Character-level tokenization**: Simple but limited for morphologically rich Sanskrit
- **Single-layer architecture**: Fast training but limited capacity
- **Real-time inference**: Good for demonstration and testing

### Potential Improvements:

- **Subword tokenization** (BPE/SentencePiece) for better semantic units
- **Deeper models** (6+ layers) for increased capacity
- **Pre-trained multilingual models** (mBERT, IndicBERT) for better performance
- **Larger datasets** for improved generalization

## 📝 Notes

- **Character-level**: Works with any input length
- **Sanskrit Script**: Outputs proper Devanagari
- **Extensible**: Easy to add more languages/features
- **Research**: Based on "Attention is All You Need" paper
- **Educational**: Great for understanding Transformer architecture

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add improvements
4. Test with various inputs
5. Submit pull request

## 📄 License

Open source - feel free to use for research and education.

---

**🕉️ Preserving Ancient Wisdom Through Modern Technology**

_Built with PyTorch, Streamlit, and dedication to Sanskrit literature_
