import streamlit as st
import pickle
import torch
import numpy as np
import os
import sys

# Add the transformer directory to the path to import the Transformer class
sys.path.append('./transformer')
from transformer import Transformer

# Set page config
st.set_page_config(
    page_title="Sanskrit Translation Model",
    page_icon="🕉️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling with dark theme
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background-color: #000000;
        color: #FFFFFF;
    }
    
    /* Main content area */
    .main > div {
        background-color: #000000;
        color: #FFFFFF;
    }
    
    /* Header styling */
    .main-header {
        font-size: 3rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Sanskrit text output */
    .sanskrit-text {
        font-size: 1.5rem;
        color: #FFD700;
        font-family: 'Noto Sans Devanagari', Arial, sans-serif;
        text-align: center;
        padding: 1rem;
        background-color: #1A1A1A;
        border-radius: 10px;
        border-left: 5px solid #FF6B35;
        border: 1px solid #333333;
    }
    
    /* Input text styling */
    .input-text {
        font-size: 1.1rem;
        color: #FFFFFF;
    }
    
    /* Info box with dark theme */
    .info-box {
        background-color: #1A1A1A;
        color: #FFFFFF;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #FF6B35;
        border: 1px solid #333333;
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #111111;
    }
    
    /* Text areas and inputs */
    .stTextArea textarea {
        background-color: #1A1A1A !important;
        color: #FFFFFF !important;
        border: 1px solid #333333 !important;
    }
    
    /* Buttons */
    .stButton button {
        background-color: #FF6B35 !important;
        color: #FFFFFF !important;
        border: none !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #1A1A1A !important;
        color: #FFFFFF !important;
    }
    
    /* Code blocks */
    .stCodeBlock {
        background-color: #1A1A1A !important;
        border: 1px solid #333333 !important;
    }
    
    /* Markdown text */
    .stMarkdown {
        color: #FFFFFF !important;
    }
    
    /* Override any white backgrounds */
    div[data-testid="stSidebar"] {
        background-color: #111111 !important;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background-color: #1A4D3A !important;
        color: #FFFFFF !important;
    }
    
    .stError {
        background-color: #4D1A1A !important;
        color: #FFFFFF !important;
    }
    
    .stWarning {
        background-color: #4D3A1A !important;
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_vocabulary():
    """Load vocabulary data from pickle file"""
    try:
        with open('./models/sanskrit_vocabulary.pkl', 'rb') as f:
            vocab_data = pickle.load(f)
        return vocab_data
    except FileNotFoundError:
        st.error("❌ Vocabulary file not found. Please ensure 'sanskrit_vocabulary.pkl' exists in the models directory.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading vocabulary: {str(e)}")
        return None

@st.cache_resource
def load_model(vocab_data):
    """Load the trained transformer model"""
    if vocab_data is None:
        return None
        
    try:
        # Extract model parameters
        model_params = vocab_data.get('model_params', {})
        d_model = model_params.get('d_model', 512)
        ffn_hidden = model_params.get('ffn_hidden', 2048)
        num_heads = model_params.get('num_heads', 8)
        drop_prob = model_params.get('drop_prob', 0.1)
        num_layers = model_params.get('num_layers', 1)
        max_sequence_length = model_params.get('max_sequence_length', 200)
        
        # Extract vocabularies and mappings
        sanskrit_vocab = vocab_data['sanskrit_vocabulary']
        english_to_index = vocab_data['english_to_index']
        sanskrit_to_index = vocab_data['sanskrit_to_index']
        special_tokens = vocab_data['special_tokens']
        
        START_TOKEN = special_tokens['START_TOKEN']
        END_TOKEN = special_tokens['END_TOKEN']
        PADDING_TOKEN = special_tokens['PADDING_TOKEN']
        
        kn_vocab_size = len(sanskrit_vocab)
        
        # Initialize the transformer model
        transformer = Transformer(
            d_model=d_model,
            ffn_hidden=ffn_hidden,
            num_heads=num_heads,
            drop_prob=drop_prob,
            num_layers=num_layers,
            max_sequence_length=max_sequence_length,
            kn_vocab_size=kn_vocab_size,
            english_to_index=english_to_index,
            sanskrit_to_index=sanskrit_to_index,
            START_TOKEN=START_TOKEN,
            END_TOKEN=END_TOKEN,
            PADDING_TOKEN=PADDING_TOKEN
        )
        
        # Load the trained weights
        checkpoint_path = "./transformer/checkpoint.pth"
        if os.path.exists(checkpoint_path):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint = torch.load(checkpoint_path, map_location=device)
            transformer.load_state_dict(checkpoint['model_state_dict'])
            transformer.eval()
            transformer.to(device)
            
            return transformer, device, vocab_data
        else:
            st.error("❌ Model checkpoint not found. Please ensure 'checkpoint.pth' exists in the transformer directory.")
            return None, None, None
            
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None, None

def create_masks(eng_batch, kn_batch, max_sequence_length):
    """Create attention masks for the transformer"""
    NEG_INFTY = -1e9
    num_sentences = len(eng_batch)
    look_ahead_mask = torch.full([max_sequence_length, max_sequence_length], True)
    look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)
    encoder_padding_mask = torch.full([num_sentences, max_sequence_length, max_sequence_length], False)
    decoder_padding_mask_self_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length], False)
    decoder_padding_mask_cross_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length], False)

    for idx in range(num_sentences):
        eng_sentence_length, kn_sentence_length = len(eng_batch[idx]), len(kn_batch[idx])
        eng_chars_to_padding_mask = np.arange(eng_sentence_length + 1, max_sequence_length)
        kn_chars_to_padding_mask = np.arange(kn_sentence_length + 1, max_sequence_length)
        encoder_padding_mask[idx, :, eng_chars_to_padding_mask] = True
        encoder_padding_mask[idx, eng_chars_to_padding_mask, :] = True
        decoder_padding_mask_self_attention[idx, :, kn_chars_to_padding_mask] = True
        decoder_padding_mask_self_attention[idx, kn_chars_to_padding_mask, :] = True
        decoder_padding_mask_cross_attention[idx, :, eng_chars_to_padding_mask] = True
        decoder_padding_mask_cross_attention[idx, kn_chars_to_padding_mask, :] = True

    encoder_self_attention_mask = torch.where(encoder_padding_mask, NEG_INFTY, 0)
    decoder_self_attention_mask = torch.where(look_ahead_mask + decoder_padding_mask_self_attention, NEG_INFTY, 0)
    decoder_cross_attention_mask = torch.where(decoder_padding_mask_cross_attention, NEG_INFTY, 0)
    return encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask

def translate_text(english_text, transformer, device, vocab_data):
    """Translate English text to Sanskrit"""
    if transformer is None or vocab_data is None:
        return "❌ Model not loaded properly"
    
    try:
        # Extract necessary data
        index_to_sanskrit = vocab_data['index_to_sanskrit']
        special_tokens = vocab_data['special_tokens']
        END_TOKEN = special_tokens['END_TOKEN']
        model_params = vocab_data.get('model_params', {})
        max_sequence_length = model_params.get('max_sequence_length', 200)
        
        # Prepare input
        eng_sentence = (english_text.lower(),)
        kn_sentence = ("",)
        
        with torch.no_grad():
            for word_counter in range(max_sequence_length):
                encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(
                    eng_sentence, kn_sentence, max_sequence_length
                )
                
                predictions = transformer(
                    eng_sentence,
                    kn_sentence,
                    encoder_self_attention_mask.to(device),
                    decoder_self_attention_mask.to(device),
                    decoder_cross_attention_mask.to(device),
                    enc_start_token=False,
                    enc_end_token=False,
                    dec_start_token=True,
                    dec_end_token=False
                )
                
                next_token_prob_distribution = predictions[0][word_counter]
                next_token_index = torch.argmax(next_token_prob_distribution).item()
                next_token = index_to_sanskrit[next_token_index]
                kn_sentence = (kn_sentence[0] + next_token,)
                
                if next_token == END_TOKEN:
                    break
        
        return kn_sentence[0]
    
    except Exception as e:
        return f"❌ Translation error: {str(e)}"

def main():
    # Header
    st.markdown('<h1 class="main-header">🕉️ Sanskrit Translation Model</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #CCCCCC;">English to Sanskrit Neural Machine Translation</p>', unsafe_allow_html=True)
    
    # Load vocabulary and model
    vocab_data = load_vocabulary()
    transformer, device, _ = load_model(vocab_data)
    
    # Sidebar with model information
    with st.sidebar:
        st.header("📊 Model Information")
        if vocab_data:
            st.success("✅ Vocabulary loaded successfully")
            st.write(f"**Sanskrit Vocabulary Size**: {len(vocab_data['sanskrit_vocabulary'])}")
            st.write(f"**English Vocabulary Size**: {len(vocab_data['english_vocabulary'])}")
            
            if 'model_params' in vocab_data:
                params = vocab_data['model_params']
                st.write("**Model Parameters:**")
                st.write(f"- Model Dimension: {params.get('d_model', 'N/A')}")
                st.write(f"- Attention Heads: {params.get('num_heads', 'N/A')}")
                st.write(f"- Layers: {params.get('num_layers', 'N/A')}")
                st.write(f"- Max Sequence Length: {params.get('max_sequence_length', 'N/A')}")
        else:
            st.error("❌ Vocabulary not loaded")
            
        if transformer is not None:
            st.success("✅ Model loaded successfully")
            st.write(f"**Device**: {device}")
        else:
            st.error("❌ Model not loaded")
            
        st.markdown("---")
        st.markdown("### 📝 Sample Inputs")
        st.markdown("""
        - Your right is to perform your duty only
        - I am here
        - Do work don't expect result
        - Why did they do this?
        - I am well
        """)
    
    # Main content area
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("### 🔤 English Input")
        english_input = st.text_area(
            "Enter English text to translate:",
            height=150,
            placeholder="Type your English text here...",
            help="Enter the English text you want to translate to Sanskrit"
        )
        
        # Translation button
        if st.button("🔄 Translate to Sanskrit", type="primary"):
            if english_input.strip():
                if transformer is not None and vocab_data is not None:
                    with st.spinner("Translating... 🔄"):
                        sanskrit_output = translate_text(english_input, transformer, device, vocab_data)
                    
                    st.markdown("### 🕉️ Sanskrit Translation")
                    if sanskrit_output.startswith("❌"):
                        st.error(sanskrit_output)
                    else:
                        st.markdown(f'<div class="sanskrit-text">{sanskrit_output}</div>', unsafe_allow_html=True)
                        
                        # Copy button
                        st.code(sanskrit_output, language=None)
                else:
                    st.error("❌ Model not loaded. Please check the model files.")
            else:
                st.warning("⚠️ Please enter some English text to translate.")
    
    with col2:
        st.markdown("### ℹ️ About This Model")
        st.markdown("""
        <div class="info-box">
        <p><strong>🎯 Purpose:</strong> This is a character-level neural machine translation model that translates English text to Sanskrit using a Transformer architecture.</p>
        
        <p><strong>🏗️ Architecture:</strong> Encoder-Decoder Transformer with multi-head attention mechanisms.</p>
        
        <p><strong>📚 Training:</strong> Trained on English-Sanskrit parallel corpus with character-level tokenization.</p>
        
        <p><strong>⚡ Features:</strong></p>
        <ul>
        <li>Character-level translation</li>
        <li>Attention-based neural architecture</li>
        <li>Support for Devanagari script</li>
        <li>Real-time translation</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        

if __name__ == "__main__":
    main()
