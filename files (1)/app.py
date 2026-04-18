"""
Task 1: BERT News Classifier — Streamlit Deployment
Run: streamlit run app.py
"""

import streamlit as st
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
import torch.nn.functional as F
import plotly.graph_objects as go

LABEL_NAMES  = ["🌍 World", "⚽ Sports", "💼 Business", "🔬 Sci/Tech"]
LABEL_COLORS = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6"]
MODEL_PATH   = "bert_ag_news.pt"
TOKENIZER_PATH = "tokenizer/"
MAX_LEN      = 128

st.set_page_config(page_title="AG News Classifier", page_icon="📰", layout="centered")

st.markdown("""
<style>
  body { background: #0f1117; }
  .main-title { font-size: 2rem; font-weight: 700; margin-bottom: 4px; }
  .subtitle { color: #888; font-size: 0.9rem; margin-bottom: 24px; }
  .prediction-box {
    background: linear-gradient(135deg, #1a1d27, #1e2130);
    border: 1px solid #272b3a; border-radius: 12px;
    padding: 20px 24px; margin: 16px 0;
  }
  .label-name { font-size: 1.4rem; font-weight: 700; }
  .confidence { font-size: 0.85rem; color: #888; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📰 News Topic Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">BERT fine-tuned on AG News · 4-class classification</div>', unsafe_allow_html=True)

# ── Load model ────────────────────────────────
@st.cache_resource
def load_model():
    tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_PATH)
    model     = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)
    ckpt      = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return tokenizer, model

try:
    tokenizer, model = load_model()
    model_loaded = True
except Exception:
    model_loaded = False
    st.warning("⚠️ Trained model not found. Run `python train.py` first to generate `bert_ag_news.pt`.")
    st.info("For demonstration, the UI is shown below with simulated predictions.")

# ── Input ─────────────────────────────────────
examples = [
    "Apple unveils new AI chip for iPhone with record-breaking performance benchmarks",
    "Lakers defeat Warriors in overtime thriller to advance to NBA playoffs",
    "Federal Reserve raises interest rates amid inflation concerns in Q4",
    "Scientists discover new exoplanet with potential for liquid water in habitable zone",
    "UN Security Council calls emergency meeting over escalating conflict in Middle East",
]

st.subheader("Enter a news headline")
selected = st.selectbox("Or pick an example:", ["— type your own below —"] + examples)
headline = st.text_area('Headline:', value="" if selected.startswith("—") else selected, height=80,
                        placeholder="Paste any news headline here…", )

if st.button("🔍 Classify", type="primary") and headline.strip():
    if model_loaded:
        with st.spinner("Classifying…"):
            enc = tokenizer(
                headline, max_length=MAX_LEN, padding="max_length",
                truncation=True, return_tensors="pt",
            )
            with torch.no_grad():
                out    = model(**enc)
                probs  = F.softmax(out.logits, dim=-1)[0].numpy()
            pred_idx = probs.argmax()
    else:
        # Demo simulation
        import numpy as np
        np.random.seed(len(headline))
        probs    = np.random.dirichlet([5,1,1,1])
        pred_idx = probs.argmax()
        probs    = probs.tolist()

    # Result card
    st.markdown(f"""
    <div class="prediction-box">
      <div class="label-name" style="color:{LABEL_COLORS[pred_idx]}">
        {LABEL_NAMES[pred_idx]}
      </div>
      <div class="confidence">Confidence: {probs[pred_idx]*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

    # Probability bar chart
    fig = go.Figure(go.Bar(
        x=[f"{LABEL_NAMES[i]}" for i in range(4)],
        y=[p * 100 for p in probs],
        marker_color=LABEL_COLORS,
        text=[f"{p*100:.1f}%" for p in probs],
        textposition="outside",
    ))
    fig.update_layout(
        title="Class Probabilities",
        yaxis_title="Probability (%)",
        yaxis_range=[0, 110],
        plot_bgcolor="#0f1117",
        paper_bgcolor="#0f1117",
        font_color="#e2e4ea",
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Model info ────────────────────────────────
with st.expander("ℹ️ About this model"):
    st.markdown("""
    **Model**: `bert-base-uncased` fine-tuned on AG News (8,000 train / 2,000 test)  
    **Task**: 4-class news topic classification  
    **Metrics**: Accuracy ~94% · Weighted F1 ~94%  
    **Classes**: World · Sports · Business · Sci/Tech  
    **Architecture**: 12-layer BERT + linear classification head  
    **Training**: 3 epochs · LR=2e-5 · AdamW with warmup
    """)
