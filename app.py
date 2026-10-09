import html
import json

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

st.set_page_config(
    page_title="Marathi Mini GPT",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styles ────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Sans+Devanagari:wght@400;500&display=swap');

html, body, .stApp, .stApp * { font-family: 'Inter', 'Noto Sans Devanagari', sans-serif; }
#MainMenu, footer, header,
section[data-testid="stSidebar"], div[data-testid="stSidebarCollapsedControl"] { display: none !important; }

.stApp { background: #0d1117; }
.block-container { max-width: 1120px; padding: 1.75rem 1.5rem 1.5rem 1.5rem; }
div[data-testid="stVerticalBlock"] { gap: 1rem; }
div[data-testid="stHorizontalBlock"] { gap: 1rem; }

/* ── Shared box ── */
.box, div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #161b22; border: 1px solid #262d36 !important; border-radius: 12px;
}
.box { padding: 1.1rem 1.25rem; }
.box-title { font-size: .72rem !important; font-weight: 600; color: #8b949e !important; text-transform: uppercase;
             letter-spacing: 1px; margin: 0 0 .8rem 0 !important; line-height: 1.2; }

/* ── Header ── */
.hero { display: flex; align-items: flex-start; justify-content: space-between; gap: 1.5rem; }
.hero-title { font-size: 1.65rem !important; font-weight: 700; color: #f0f6fc !important; letter-spacing: -.3px;
              line-height: 1.2; margin: 0 0 .35rem 0; }
.hero-desc  { font-size: .9rem !important; color: #8b949e !important; line-height: 1.55; margin: 0; }
.hero-tag   { flex-shrink: 0; font-size: .72rem !important; font-weight: 500; color: #8b949e; border: 1px solid #30363d;
              border-radius: 999px; padding: .3rem .8rem; white-space: nowrap; }
.spacer { height: 1rem; }

/* ── Metric cards ── */
.metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; }
.metric  { background: #161b22; border: 1px solid #262d36; border-radius: 12px; padding: 1rem 1.25rem; }
.metric-val { font-size: 1.55rem !important; font-weight: 700; color: #f0f6fc !important; line-height: 1.2; margin: 0; }
.metric-lab { font-size: .68rem !important; font-weight: 500; color: #8b949e !important; text-transform: uppercase;
              letter-spacing: .8px; margin: .35rem 0 0 0; white-space: nowrap; }

/* ── Key / value grid ── */
.kv { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; }
.kv-val { font-size: 1rem !important; font-weight: 600; color: #f0f6fc !important; margin: 0; line-height: 1.3; }
.kv-lab { font-size: .72rem !important; color: #8b949e !important; margin: .25rem 0 0 0; }

/* ── Inputs ── */
.stTextInput label p, .stSlider label p { color: #c9d1d9 !important; font-size: .8rem !important; font-weight: 500 !important; }
.stTextInput input { background: #0d1117 !important; border: 1px solid #30363d !important; border-radius: 8px !important;
                     color: #f0f6fc !important; font-size: 1rem !important; padding: .6rem .85rem !important; }
.stTextInput input:focus { border-color: #388bfd !important; box-shadow: 0 0 0 3px rgba(56,139,253,.18) !important; }
div[data-testid="stSlider"] div[data-baseweb="slider"] { filter: hue-rotate(205deg) saturate(1.1); }
div[data-testid="stSlider"] [data-testid="stTickBarMin"],
div[data-testid="stSlider"] [data-testid="stTickBarMax"] { color: #6e7681 !important; font-size: .7rem !important; }

/* ── Button ── */
div[data-testid="stButton"] button {
    width: 100% !important; background: #238636; color: #fff; font-weight: 600; font-size: .92rem;
    padding: .65rem 1.25rem; border-radius: 8px; border: 1px solid #2ea043; transition: background .15s; }
div[data-testid="stButton"] button:hover { background: #2ea043; border-color: #3fb950; color: #fff; }
div[data-testid="stButton"] button:focus:not(:active) { border-color: #3fb950; color: #fff; box-shadow: 0 0 0 3px rgba(46,160,67,.25); }

/* ── Output ── */
.out-empty { min-height: 210px; padding: 1rem 1.1rem; color: #6e7681 !important; font-size: .88rem !important;
             line-height: 1.6; border: 1px dashed #30363d; border-radius: 8px; text-align: left; }
.out-text { font-family: 'Noto Sans Devanagari', 'Inter', sans-serif !important; font-size: 1.05rem !important;
            line-height: 1.9; color: #e6edf3 !important; white-space: pre-wrap; word-break: break-word; margin: 0; text-align: left; }
.out-meta { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: .9rem; }
.chip { font-size: .74rem !important; color: #8b949e !important; background: #0d1117; border: 1px solid #262d36;
        border-radius: 999px; padding: .25rem .7rem; }
.chip b { color: #f0f6fc; font-weight: 600; }

/* ── Responsive ── */
@media (max-width: 900px) {
  .metrics, .kv { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 768px) {
  .block-container { padding: 1rem .9rem 1.25rem .9rem; }
  .hero { flex-direction: column; gap: .75rem; }
  .hero-title { font-size: 1.4rem !important; }
  .metric-val { font-size: 1.3rem !important; }
  /* let the fixed-height panels grow with their content on small screens */
  div[data-testid="stVerticalBlockBorderWrapper"],
  div[data-testid="stVerticalBlockBorderWrapper"] > div { height: auto !important; max-height: none !important; overflow: visible !important; }
  .out-empty { min-height: 120px; }
}
</style>
""",
    unsafe_allow_html=True,
)


# ── Vocab ─────────────────────────────────────────────
@st.cache_resource
def load_vocab():
    with open("data/vocab.json", "r", encoding="utf-8") as f:
        v = json.load(f)
    return v["char_to_idx"], {int(k): c for k, c in v["idx_to_char"].items()}


def encode(text, char_to_idx):
    return [char_to_idx[c] for c in text if c in char_to_idx]


def decode(indices, idx_to_char):
    return "".join(idx_to_char[i] for i in indices)


# ── Model (attribute names match the trained checkpoint) ──
class SingleHead(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, T, _ = x.shape
        q, k, v = self.query(x), self.key(x), self.value(x)
        scores = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        return self.dropout(F.softmax(scores, dim=-1)) @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList(
            [SingleHead(n_embd, head_size, block_size, dropout) for _ in range(n_head)]
        )
        self.proj = nn.Linear(n_embd, n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ff = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        return x + self.ff(self.ln2(x))


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[TransformerBlock(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_final = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        _, T = idx.shape
        x = self.token_embedding(idx) + self.position_embedding(torch.arange(T, device=idx.device))
        return self.lm_head(self.ln_final(self.blocks(x)))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            logits = self(idx[:, -self.block_size:])[:, -1, :] / max(temperature, 1e-5)
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


@st.cache_resource
def load_model():
    with open("model/hyperparams.json") as f:
        hp = json.load(f)
    m = MiniGPT(
        hp["vocab_size"], hp["n_embd"], hp["n_head"],
        hp["n_layer"], hp["block_size"], hp["dropout"],
    )
    m.load_state_dict(torch.load("model/mini_gpt_marathi.pth", map_location="cpu"))
    m.eval()
    return m


char_to_idx, idx_to_char = load_vocab()
model = load_model()


def kv_box(title, items):
    cells = "".join(
        f'<div><p class="kv-val">{v}</p><p class="kv-lab">{k}</p></div>' for k, v in items
    )
    return f'<div class="box"><p class="box-title">{title}</p><div class="kv">{cells}</div></div>'


# ── 1. Header ─────────────────────────────────────────
st.markdown(
    '<div class="box"><div class="hero"><div>'
    '<p class="hero-title">Marathi Mini GPT</p>'
    '<p class="hero-desc">A small GPT language model trained from scratch on Marathi news articles. '
    "Enter a starting phrase and the model continues it.</p>"
    '</div><span class="hero-tag">Character-level · PyTorch</span></div></div>'
    '<div class="spacer"></div>',
    unsafe_allow_html=True,
)

# ── 2. Metrics ────────────────────────────────────────
metrics = [("4.88M", "Parameters"), ("1.60", "Train loss"), ("1.64", "Validation loss"), ("222", "Vocabulary")]
st.markdown(
    '<div class="metrics">'
    + "".join(
        f'<div class="metric"><p class="metric-val">{v}</p><p class="metric-lab">{k}</p></div>'
        for v, k in metrics
    )
    + '</div><div class="spacer"></div>',
    unsafe_allow_html=True,
)

# ── 3. Workspace: prompt | output ─────────────────────
PANEL_HEIGHT = 340
left, right = st.columns([5, 6], gap="small")

with left:
    with st.container(border=True, height=PANEL_HEIGHT):
        st.markdown('<p class="box-title">Prompt</p>', unsafe_allow_html=True)
        starting_text = st.text_input("Starting text", value="मराठी")
        c1, c2 = st.columns(2)
        with c1:
            max_tokens = st.slider("Length (characters)", 100, 500, 200, 50)
        with c2:
            temperature = st.slider("Temperature", 0.3, 1.5, 0.8, 0.1)
        try:
            generate = st.button("Generate", width="stretch")
        except TypeError:
            generate = st.button("Generate", use_container_width=True)

with right:
    with st.container(border=True, height=PANEL_HEIGHT):
        st.markdown('<p class="box-title">Output</p>', unsafe_allow_html=True)
        slot = st.empty()
        slot.markdown(
            '<div class="out-empty">Your generated Marathi text will appear here. '
            "Enter a starting phrase and press Generate.</div>",
            unsafe_allow_html=True,
        )

        if generate:
            if not starting_text.strip():
                slot.error("Please enter some Marathi text.")
            else:
                encoded = encode(starting_text, char_to_idx)
                if not encoded:
                    slot.error("None of the characters in your input are in the model's vocabulary.")
                else:
                    with st.spinner("Generating..."):
                        ctx = torch.tensor(encoded, dtype=torch.long).unsqueeze(0)
                        out = model.generate(ctx, max_new_tokens=max_tokens, temperature=temperature)
                        result = decode(out[0].tolist(), idx_to_char)
                    slot.markdown(
                        f'<p class="out-text">{html.escape(result)}</p>'
                        f'<div class="out-meta">'
                        f'<span class="chip"><b>{len(result)}</b> characters</span>'
                        f'<span class="chip"><b>{len(result.split())}</b> words</span>'
                        f'<span class="chip">temperature <b>{temperature:.1f}</b></span></div>',
                        unsafe_allow_html=True,
                    )

# ── 4. Info boxes ─────────────────────────────────────
a, b = st.columns(2, gap="small")
with a:
    st.markdown(
        kv_box("Architecture", [("Layers", "6"), ("Attention heads", "8"), ("Embedding size", "256"), ("Context", "128 chars")]),
        unsafe_allow_html=True,
    )
with b:
    st.markdown(
        kv_box("Training", [("Dataset", "AI4Bharat"), ("Articles", "1,000"), ("Characters", "2M"), ("Steps", "5,000")]),
        unsafe_allow_html=True,
    )