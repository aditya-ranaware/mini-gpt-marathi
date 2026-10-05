import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import json

# Page config
st.set_page_config(
    page_title="Marathi GPT",
    page_icon="🇮🇳",
    layout="centered"
)

st.title("🇮🇳 Marathi Mini GPT")
st.markdown("AI model that generates Marathi text — built from scratch using PyTorch")
st.divider()

# Load vocabulary
@st.cache_resource
def load_vocab():
    with open("data/vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)
    
    chars = vocab["chars"]
    char_to_idx = vocab["char_to_idx"]
    idx_to_char = {int(k): v for k, v in vocab["idx_to_char"].items()}
    vocab_size = vocab["vocab_size"]
    
    return chars, char_to_idx, idx_to_char, vocab_size

# Encode decode functions
def encode(text, char_to_idx):
    return [char_to_idx[ch] for ch in text if ch in char_to_idx]

def decode(indices, idx_to_char):
    return "".join([idx_to_char[idx] for idx in indices])

# Model architecture — same as training
class SingleHead(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        head_size = k.shape[-1]
        scores = q @ k.transpose(-2, -1) * (head_size ** -0.5)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        return weights @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([
            SingleHead(n_embd, head_size, block_size, dropout)
            for _ in range(n_head)
        ])
        self.proj    = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ff  = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
        super().__init__()
        self.block_size        = block_size
        self.token_embedding   = nn.Embedding(vocab_size, n_embd)
        self.position_embedding= nn.Embedding(block_size, n_embd)
        self.blocks            = nn.Sequential(*[
            TransformerBlock(n_embd, n_head, block_size, dropout)
            for _ in range(n_layer)
        ])
        self.ln_final = nn.LayerNorm(n_embd)
        self.lm_head  = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_final(x)
        return self.lm_head(x)

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_crop = idx[:, -self.block_size:]
            logits   = self(idx_crop)
            logits   = logits[:, -1, :]
            probs    = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx      = torch.cat([idx, idx_next], dim=1)
        return idx

# Load model
@st.cache_resource
def load_model():
    chars, char_to_idx, idx_to_char, vocab_size = load_vocab()

    with open("model/hyperparams.json", "r") as f:
        hp = json.load(f)

    model = MiniGPT(
        vocab_size = hp["vocab_size"],
        n_embd     = hp["n_embd"],
        n_head     = hp["n_head"],
        n_layer    = hp["n_layer"],
        block_size = hp["block_size"],
        dropout    = hp["dropout"]
    )

    model.load_state_dict(
        torch.load("model/mini_gpt_marathi.pth", map_location="cpu")
    )
    model.eval()
    return model

# Load everything
chars, char_to_idx, idx_to_char, vocab_size = load_vocab()
model = load_model()

# UI
st.subheader("Generate Marathi Text")

col1, col2 = st.columns(2)

with col1:
    starting_text = st.text_input(
        "Enter starting Marathi text",
        value="मराठी",
        help="Type any Marathi word to start generation"
    )

with col2:
    max_tokens = st.slider(
        "Characters to generate",
        min_value=100,
        max_value=500,
        value=200,
        step=50
    )

if st.button("Generate Marathi Text 🚀", type="primary"):
    with st.spinner("Generating Marathi text..."):
        # Encode starting text
        encoded = encode(starting_text, char_to_idx)

        if len(encoded) == 0:
            st.error("Please enter valid Marathi text")
        else:
            context = torch.tensor(encoded, dtype=torch.long).unsqueeze(0)

            with torch.no_grad():
                generated = model.generate(context, max_new_tokens=max_tokens)

            generated_text = decode(generated[0].tolist(), idx_to_char)

            st.success("Generated Successfully!")
            st.subheader("Generated Marathi Text —")
            st.write(generated_text)

            # Show stats
            st.divider()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Characters Generated", max_tokens)
            with col2:
                st.metric("Model Parameters", "4.88M")
            with col3:
                st.metric("Training Loss", "1.60")

# Sidebar info
with st.sidebar:
    st.title("About")
    st.markdown("""
    **Marathi Mini GPT**
    
    GPT language model built completely 
    from scratch using PyTorch.
    
    Trained on Marathi news articles 
    from AI4Bharat dataset.
    
    **Model Details —**
    - Parameters — 4.88M
    - Layers — 6
    - Attention Heads — 8
    - Embedding Size — 256
    - Training Steps — 5000
    - Final Loss — 1.60
    
    **Dataset —**
    - AI4Bharat Sangraha
    - 1000 Marathi articles
    - 2M characters
    """)
    
    st.divider()
    st.markdown("Built with PyTorch and Streamlit")