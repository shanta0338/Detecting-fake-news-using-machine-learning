"""
Fake News Prediction Comparator: Gradient Boosting vs Random Forest
Run with: streamlit run app.py
Requires: gb_best_pipeline.pkl, rf_best_pipeline.pkl
"""

import os
import pickle

import numpy as np
import spacy
import streamlit as st

# ---------------------------------------------------------------- page setup
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

GREEN, GREEN_BG = "#16a34a", "rgba(22,163,74,0.12)"
RED, RED_BG = "#dc2626", "rgba(220,38,38,0.12)"
AMBER, AMBER_BG = "#d97706", "rgba(217,119,6,0.12)"

st.markdown(
    f"""
<style>
.block-container {{ padding-top: 2rem; max-width: 1150px; }}
.hero {{
    padding: 1.6rem 1.8rem; border-radius: 18px; margin-bottom: 1.2rem;
    background: linear-gradient(135deg, #1e293b 0%, #334155 55%, #475569 100%);
    color: #f8fafc;
}}
.hero h1 {{ margin: 0; font-size: 2rem; font-weight: 800; letter-spacing: -0.02em; }}
.hero p {{ margin: .35rem 0 0; color: #cbd5e1; font-size: 1rem; }}
.card {{
    border-radius: 16px; padding: 1.3rem 1.4rem; height: 100%;
    border: 1px solid rgba(128,128,128,.25);
    border-left-width: 8px; border-left-style: solid;
}}
.card.real {{ background: {GREEN_BG}; border-left-color: {GREEN}; }}
.card.fake {{ background: {RED_BG};   border-left-color: {RED}; }}
.card.na   {{ background: {AMBER_BG}; border-left-color: {AMBER}; }}
.card .model {{ font-size: .85rem; opacity: .75; text-transform: uppercase; letter-spacing: .06em; }}
.card .verdict {{ font-size: 2rem; font-weight: 800; margin: .2rem 0 .1rem; }}
.card.real .verdict {{ color: {GREEN}; }}
.card.fake .verdict {{ color: {RED}; }}
.card.na .verdict {{ color: {AMBER}; }}
.card .conf {{ font-size: 1rem; opacity: .85; margin-bottom: .8rem; }}
.bar {{ display: flex; height: 14px; border-radius: 999px; overflow: hidden; background: rgba(128,128,128,.2); }}
.bar .r {{ background: {GREEN}; }}
.bar .f {{ background: {RED}; }}
.legend {{ display: flex; justify-content: space-between; font-size: .85rem; margin-top: .4rem; opacity: .9; }}
.banner {{ border-radius: 12px; padding: .9rem 1.2rem; font-weight: 600; margin: .4rem 0 1.1rem; }}
.banner.real {{ background: {GREEN_BG}; color: {GREEN}; border: 1px solid {GREEN}; }}
.banner.fake {{ background: {RED_BG}; color: {RED}; border: 1px solid {RED}; }}
.banner.mixed {{ background: {AMBER_BG}; color: {AMBER}; border: 1px solid {AMBER}; }}
.stButton > button[kind="primary"] {{
    border-radius: 12px; padding: .65rem 1.4rem; font-weight: 700; border: none;
    background: linear-gradient(135deg, #2563eb, #7c3aed);
}}
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------- model loading
@st.cache_resource
def load_nlp():
    try:
        return spacy.load("en_core_web_lg")
    except OSError:
        import en_core_web_lg
        return en_core_web_lg.load()


@st.cache_resource
def load_pipelines():
    models = {}
    for key, path in (("gb", "gb_best_pipeline.pkl"), ("rf", "rf_best_pipeline.pkl")):
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[key] = pickle.load(f)
        else:
            models[key] = None
    return models["gb"], models["rf"]


nlp = load_nlp()
gb_pipeline, rf_pipeline = load_pipelines()

# ------------------------------------------------------------------- samples
SAMPLES = {
    "Custom article": {"title": "", "text": ""},
    "Hillary Clinton FBI investigation (Fake)": {
        "title": "You Can Smell Hillary's Fear",
        "text": "Daniel Greenfield, a Shillman Journalism Fellow at the Freedom Center, is a New York writer focusing on radical Islam. In the final stretch of the election, Hillary Rodham Clinton has gone to war with the FBI. The FBI is under attack by everyone from Obama to CNN. Hillary's people have circulated a letter attacking Comey. You can smell the fear. Whatever she is afraid of, it lies buried in her emails.",
    },
    "Kerry to go to Paris (Real)": {
        "title": "Kerry to go to Paris in gesture of sympathy",
        "text": "U.S. Secretary of State John F. Kerry said Monday that he will stop in Paris later this week, amid criticism that no top American officials attended Sunday's unity march against terrorism. Kerry said he expects to arrive in Paris Thursday evening, as he heads home after a week abroad.",
    },
    "The Battle of New York primary (Real)": {
        "title": "The Battle of New York: Why This Primary Matters",
        "text": "It's primary day in New York and front-runners Hillary Clinton and Donald Trump are leading in the polls. Trump is now vowing to win enough delegates to clinch the Republican nomination and prevent a contested convention.",
    },
    "Paul Ryan Trump rally (Fake)": {
        "title": "Watch The Exact Moment Paul Ryan Committed Political Suicide",
        "text": "Paul Ryan re-re-re-reversed course and announced that he was back on the Trump Train after all. Suddenly, Ryan was appearing at a pro-Trump rally and boldly declaring that he already sent in his vote. It was a surreal moment.",
    },
}

# ------------------------------------------------------------------- sidebar
st.sidebar.title("⚙️ Settings")
threshold = st.sidebar.slider(
    "Decision threshold θ",
    0.10, 0.90, 0.50, 0.05,
    help="An article is classified REAL when P(Real) ≥ θ.",
)
sample_choice = st.sidebar.selectbox("Load a sample article", list(SAMPLES))
st.sidebar.markdown("---")
st.sidebar.markdown("**Models**")
st.sidebar.caption("🌲 Gradient Boosting · 90.19% CV")
st.sidebar.caption("🌴 Random Forest · 87.49% CV")
if gb_pipeline is None:
    st.sidebar.error("gb_best_pipeline.pkl not found")
if rf_pipeline is None:
    st.sidebar.error("rf_best_pipeline.pkl not found")

# -------------------------------------------------------------------- header
st.markdown(
    """
<div class="hero">
  <h1>📰 Fake News Detector</h1>
  <p>Compare Gradient Boosting and Random Forest predictions side by side.</p>
</div>
""",
    unsafe_allow_html=True,
)

sample = SAMPLES[sample_choice]
# A key per sample makes the inputs refresh when the selection changes.
title_input = st.text_input(
    "Article title", value=sample["title"], placeholder="Enter headline…",
    key=f"title_{sample_choice}",
)
article_text = st.text_area(
    "Article content", value=sample["text"], height=180,
    placeholder="Paste the full article text…", key=f"text_{sample_choice}",
)
run = st.button("Analyze article", type="primary")


# ----------------------------------------------------------------- inference
def predict(pipe, vec, theta):
    if pipe is None:
        return None
    if hasattr(pipe, "predict_proba"):
        probs = pipe.predict_proba(vec)[0]  # [P(Fake), P(Real)]
        p_fake, p_real = float(probs[0]), float(probs[1])
    else:
        p_real = 1.0 if pipe.predict(vec)[0] == 1 else 0.0
        p_fake = 1.0 - p_real
    return {"real": p_real >= theta, "p_real": p_real, "p_fake": p_fake}


def card(title, result):
    if result is None:
        return f"""<div class="card na"><div class="model">{title}</div>
        <div class="verdict">UNAVAILABLE</div><div class="conf">Model file missing</div></div>"""
    cls = "real" if result["real"] else "fake"
    label = "REAL NEWS" if result["real"] else "FAKE NEWS"
    conf = result["p_real"] if result["real"] else result["p_fake"]
    r, f = result["p_real"] * 100, result["p_fake"] * 100
    icon = "✅" if result["real"] else "🚫"
    return f"""
    <div class="card {cls}">
      <div class="model">{title}</div>
      <div class="verdict">{icon} {label}</div>
      <div class="conf">{conf*100:.1f}% confidence</div>
      <div class="bar"><div class="r" style="width:{r}%"></div><div class="f" style="width:{f}%"></div></div>
      <div class="legend"><span>Real {r:.1f}%</span><span>Fake {f:.1f}%</span></div>
    </div>"""


if run:
    combined = f"{title_input} {article_text}".strip()
    if not combined:
        st.warning("Please enter a title or some article text.")
    else:
        with st.spinner("Analyzing…"):
            vec = np.array(nlp(combined).vector).reshape(1, -1)
            gb = predict(gb_pipeline, vec, threshold)
            rf = predict(rf_pipeline, vec, threshold)

        if gb and rf:
            if gb["real"] == rf["real"]:
                cls = "real" if gb["real"] else "fake"
                verdict = "REAL" if gb["real"] else "FAKE"
                st.markdown(
                    f'<div class="banner {cls}">Both models agree: this article looks <b>{verdict}</b>.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="banner mixed">⚠️ The models disagree. Treat this result with caution.</div>',
                    unsafe_allow_html=True,
                )

        c1, c2 = st.columns(2, gap="large")
        c1.markdown(card("🌲 Gradient Boosting", gb), unsafe_allow_html=True)
        c2.markdown(card("🌴 Random Forest", rf), unsafe_allow_html=True)
        st.caption(
            "For research and learning only. Model predictions can be wrong, "
            "so verify news with trusted sources."
        )