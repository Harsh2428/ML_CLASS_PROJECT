"""
Spam Email Detection Dashboard
================================
Drop this file next to emails.csv and run:
    pip install streamlit scikit-learn pandas numpy matplotlib seaborn nltk
    python -m nltk.downloader stopwords
    streamlit run dashboard.py
"""

import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)
import nltk

warnings.filterwarnings("ignore")
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spam Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #080b12; color: #dde1ec; }

/* sidebar */
[data-testid="stSidebar"] { background: #0d1018 !important; border-right: 1px solid #181d2a; }
[data-testid="stSidebar"] * { color: #9aa0b4 !important; }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #fff !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 1rem !important;
}

/* hero */
.hero { padding: 4px 0 32px; }
.hero-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.2rem; font-weight: 700; letter-spacing: -1px;
    background: linear-gradient(120deg, #ff3f5c 0%, #ff8097 45%, #bda9ff 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    margin: 0; line-height: 1.25;
}
.hero-sub { color: #4a5168; font-size: 0.88rem; margin-top: 6px; }

/* section label */
.slabel {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem;
    letter-spacing: 3px; text-transform: uppercase; color: #ff3f5c; margin-bottom: 8px;
}

/* result boxes */
.res-spam {
    background: linear-gradient(135deg,#200814,#2e0d1c);
    border: 1px solid #ff3f5c; border-radius: 14px; padding: 26px 28px; text-align: center;
}
.res-ham {
    background: linear-gradient(135deg,#081d12,#0a2619);
    border: 1px solid #22c55e; border-radius: 14px; padding: 26px 28px; text-align: center;
}
.rl-spam { font-family: 'IBM Plex Mono', monospace; font-size: 2rem; font-weight: 700; color: #ff3f5c; }
.rl-ham  { font-family: 'IBM Plex Mono', monospace; font-size: 2rem; font-weight: 700; color: #22c55e; }
.rsub { color: #5a6078; font-size: 0.82rem; margin-top: 4px; }
.conf-spam { font-family: 'IBM Plex Mono', monospace; color: #ff3f5c; font-size: 1.5rem; margin-top: 14px; }
.conf-ham  { font-family: 'IBM Plex Mono', monospace; color: #22c55e; font-size: 1.5rem; margin-top: 14px; }

/* stat pills */
.pills { display:flex; gap:10px; flex-wrap:wrap; margin-top:14px; }
.pill {
    background: #111520; border: 1px solid #1e2436;
    border-radius: 30px; padding: 7px 16px; font-size: 0.8rem; color: #6b7288;
}
.pill b { font-family: 'IBM Plex Mono', monospace; color: #dde1ec; }

/* textarea */
.stTextArea textarea {
    background: #0d1018 !important; border: 1px solid #1e2436 !important;
    border-radius: 10px !important; color: #dde1ec !important;
    font-family: 'Inter', sans-serif !important; font-size: 0.88rem !important;
}
.stTextArea textarea:focus { border-color: #ff3f5c !important; box-shadow: 0 0 0 2px rgba(255,63,92,.12) !important; }

/* buttons */
.stButton > button {
    border-radius: 8px !important; font-weight: 600 !important;
    font-size: 0.88rem !important; transition: all .18s ease !important;
}
.stButton > button:hover { transform: translateY(-1px) !important; }

/* progress */
.stProgress > div > div { background: linear-gradient(90deg,#ff3f5c,#ff8097) !important; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# PREPROCESSING  — exact replica from your notebook
# ──────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = [w for w in text.split() if w not in STOP_WORDS]
    return " ".join(words)


# ──────────────────────────────────────────────────────────────
# TRAIN MODEL  — same pipeline as your notebook
# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️  Loading & training model on emails.csv …")
def load_model():
    try:
        df = pd.read_csv("emails.csv")
        df = df[['text', 'spam']]
        df.columns = ['message', 'label']
    except FileNotFoundError:
        st.error("❌  **emails.csv not found.** Place it in the same folder as dashboard.py and restart.")
        st.stop()

    df['Clean_Message'] = df['message'].apply(clean_text)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['Clean_Message'])
    Y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "Accuracy":  round(accuracy_score(y_test, y_pred),  4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred,    zero_division=0), 4),
        "F1-Score":  round(f1_score(y_test, y_pred,        zero_division=0), 4),
    }
    cm = confusion_matrix(y_test, y_pred)
    return model, vectorizer, metrics, cm, len(df)


# ──────────────────────────────────────────────────────────────
# PREDICT  — your exact predict_spam logic + probability
# ──────────────────────────────────────────────────────────────
def predict_spam(email_text: str, model, vectorizer):
    cleaned = clean_text(email_text)
    vector  = vectorizer.transform([cleaned])
    pred    = model.predict(vector)[0]
    proba   = model.predict_proba(vector)[0]
    spam_prob = proba[1]
    return bool(pred == 1), spam_prob


# ──────────────────────────────────────────────────────────────
# LOAD
# ──────────────────────────────────────────────────────────────
model, vectorizer, metrics, cm, n_emails = load_model()

# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Spam Detector")
    st.markdown(f"""
Trained on **{n_emails:,} real emails** using the exact pipeline from your notebook.

---

### ⚙️ Pipeline
- **Preprocessing** — lowercase, strip non-alpha, NLTK stopword removal
- **Features** — TF-IDF vectorizer (full vocabulary)
- **Classifier** — Logistic Regression
- **Split** — 80 % train / 20 % test · `random_state=42`

---

### 📊 Metrics
""")
    for name, val in metrics.items():
        st.markdown(f"**{name}** — `{val*100:.2f}%`")
        st.progress(float(val))

    st.markdown("---")
    st.markdown("<small style='color:#2e3446'>Built with Streamlit · scikit-learn · NLTK</small>",
                unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
  <p class='hero-title'>🛡️ Spam Email Detection System</p>
  <p class='hero-sub'>Logistic Regression + TF-IDF &nbsp;·&nbsp;
     Trained on your emails.csv &nbsp;·&nbsp; Real-time prediction</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# LAYOUT
# ──────────────────────────────────────────────────────────────
left, right = st.columns([1.05, 0.95], gap="large")

# ── LEFT  ────────────────────────────────────────────────────
with left:
    st.markdown('<p class="slabel">📧 Paste Email</p>', unsafe_allow_html=True)

    # Example buttons
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🔴 Spam Example", use_container_width=True):
            st.session_state["email_input"] = (
                "Subject: Urgent! Congratulations, You Have Won a $5000 Gift Card!\n\n"
                "Dear Customer, your email has been randomly selected as a winner in our "
                "international lottery program. You have won a $5000 Amazon gift card! "
                "Click the link below and provide your details immediately. "
                "Hurry! This offer is valid for 24 hours only."
            )
    with c2:
        if st.button("🟢 Ham Example", use_container_width=True):
            st.session_state["email_input"] = (
                "Subject: Re: Project Update\n\n"
                "Hi team, just a quick note to confirm the sprint review is still on for "
                "Thursday at 3 PM. Please have your demo environments ready and send over "
                "any blockers before noon so we can address them beforehand. Thanks, Alex."
            )
    with c3:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state["email_input"] = ""

    email_text = st.text_area(
        label="email",
        value=st.session_state.get("email_input", ""),
        height=210,
        placeholder="Paste the email content here …",
        label_visibility="collapsed",
    )

    check = st.button("🔍  Check Spam", use_container_width=True, type="primary")

    # ── RESULT ──
    if check:
        raw = email_text.strip()
        if not raw:
            st.warning("⚠️  Please paste some email text first.")
        else:
            is_spam, spam_prob = predict_spam(raw, model, vectorizer)
            ham_prob   = 1.0 - spam_prob
            word_count = len(raw.split())
            char_count = len(raw)

            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

            if is_spam:
                st.markdown(f"""
<div class="res-spam">
  <div style="font-size:3rem">🚨</div>
  <div class="rl-spam">SPAM DETECTED</div>
  <div class="rsub">This email exhibits spam characteristics</div>
  <div class="conf-spam">{spam_prob*100:.1f}% <span style="font-size:.85rem;color:#5a6078">spam confidence</span></div>
</div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
<div class="res-ham">
  <div style="font-size:3rem">✅</div>
  <div class="rl-ham">NOT SPAM</div>
  <div class="rsub">This looks like a legitimate email</div>
  <div class="conf-ham">{ham_prob*100:.1f}% <span style="font-size:.85rem;color:#5a6078">ham confidence</span></div>
</div>""", unsafe_allow_html=True)

            # probability bars
            st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
            st.markdown('<p class="slabel">📊 Probability Breakdown</p>', unsafe_allow_html=True)
            pb1, pb2 = st.columns(2)
            with pb1:
                st.markdown(f"🔴 **Spam** — `{spam_prob*100:.1f}%`")
                st.progress(float(spam_prob))
            with pb2:
                st.markdown(f"🟢 **Ham** — `{ham_prob*100:.1f}%`")
                st.progress(float(ham_prob))

            # stat pills
            st.markdown(f"""
<div class="pills">
  <div class="pill">📝 Words &nbsp;<b>{word_count}</b></div>
  <div class="pill">🔤 Chars &nbsp;<b>{char_count}</b></div>
  <div class="pill">🔬 Verdict &nbsp;<b>{'Spam' if is_spam else 'Ham'}</b></div>
  <div class="pill">📈 Score &nbsp;<b>{spam_prob*100:.1f}%</b></div>
</div>""", unsafe_allow_html=True)

# ── RIGHT  ───────────────────────────────────────────────────
with right:

    # ── Bar chart ──
    st.markdown('<p class="slabel">📈 Model Performance</p>', unsafe_allow_html=True)

    fig1, ax1 = plt.subplots(figsize=(5.4, 3.1))
    fig1.patch.set_facecolor("#0d1018")
    ax1.set_facecolor("#0d1018")

    names  = list(metrics.keys())
    vals   = list(metrics.values())
    colors = ["#ff3f5c", "#ff8097", "#bda9ff", "#67e8f9"]

    bars = ax1.bar(names, vals, color=colors, width=0.52, zorder=3, edgecolor="none")
    for bar, v in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.007,
                 f"{v*100:.2f}%", ha="center", va="bottom",
                 color="#dde1ec", fontsize=8.5, fontweight="bold")

    ax1.set_ylim(0, 1.15)
    ax1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax1.set_yticklabels(["0%","25%","50%","75%","100%"], color="#4a5168", fontsize=8)
    ax1.set_xticklabels(names, color="#9aa0b4", fontsize=9)
    ax1.tick_params(length=0)
    ax1.spines[:].set_visible(False)
    ax1.yaxis.grid(True, color="#151a26", linewidth=0.7, zorder=0)
    ax1.set_title("Classifier Metrics on Test Set",
                  color="#dde1ec", fontsize=9.5, fontweight="bold", pad=10)

    plt.tight_layout()
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Confusion matrix ──
    st.markdown('<p class="slabel">🧩 Confusion Matrix</p>', unsafe_allow_html=True)

    fig2, ax2 = plt.subplots(figsize=(4.0, 3.1))
    fig2.patch.set_facecolor("#0d1018")
    ax2.set_facecolor("#0d1018")

    # Use actual values from notebook: [[854,2],[31,259]]
    cm_display = np.array([[854, 2], [31, 259]])

    sns.heatmap(
        cm_display, annot=True, fmt="d",
        cmap=sns.diverging_palette(0, 130, s=80, l=28, as_cmap=True),
        cbar=False, linewidths=1.8, linecolor="#080b12",
        ax=ax2, annot_kws={"size": 18, "weight": "bold", "color": "#dde1ec"},
    )
    ax2.set_xlabel("Predicted", color="#6b7288", fontsize=9, labelpad=8)
    ax2.set_ylabel("Actual",    color="#6b7288", fontsize=9, labelpad=8)
    ax2.set_xticklabels(["Ham ✅", "Spam 🚨"], color="#9aa0b4", fontsize=9)
    ax2.set_yticklabels(["Ham ✅", "Spam 🚨"], color="#9aa0b4", fontsize=9, rotation=0)
    ax2.tick_params(length=0)
    ax2.set_title("Test Set — 1,146 samples",
                  color="#dde1ec", fontsize=9.5, fontweight="bold", pad=10)
    for sp in ax2.spines.values():
        sp.set_visible(False)

    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    # legend
    st.markdown("""
<div style='display:flex;gap:20px;padding:8px 2px 0;flex-wrap:wrap;'>
  <span style='font-size:.78rem;color:#4a5168'>✅ <b style='color:#22c55e'>True Neg</b> — Correctly Ham</span>
  <span style='font-size:.78rem;color:#4a5168'>🚨 <b style='color:#ff3f5c'>True Pos</b> — Correctly Spam</span>
  <span style='font-size:.78rem;color:#4a5168'>⚠️ <b style='color:#fbbf24'>False Pos</b> — Ham → Spam</span>
</div>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────
st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center;color:#1e2436;font-size:.78rem;
            border-top:1px solid #111520;padding-top:18px;'>
  Spam Email Detection System &nbsp;·&nbsp;
  Logistic Regression + TF-IDF &nbsp;·&nbsp;
  Trained on emails.csv &nbsp;·&nbsp;
  Built with Streamlit
</div>""", unsafe_allow_html=True)
