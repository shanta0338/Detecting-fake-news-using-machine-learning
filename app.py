"""
📰 Fake News Prediction Comparator: Gradient Boosting vs Random Forest
Run with: streamlit run app.py
Requires: gb_best_pipeline.pkl, rf_best_pipeline.pkl
"""

import streamlit as st
import pickle
import numpy as np
import spacy
import os

# 1. Page Configuration
st.set_page_config(
    page_title="Fake News Model Comparator",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Cache spaCy model loading
@st.cache_resource
def load_nlp():
    try:
        return spacy.load("en_core_web_lg")
    except OSError:
        st.warning("Downloading spaCy en_core_web_lg model...")
        from spacy.cli import download
        download("en_core_web_lg")
        return spacy.load("en_core_web_lg")

nlp = load_nlp()

# 3. Load Trained Pipelines
@st.cache_resource
def load_pipelines():
    gb_path = "gb_best_pipeline.pkl"
    rf_path = "rf_best_pipeline.pkl"
    
    gb_model = None
    rf_model = None
    
    if os.path.exists(gb_path):
        with open(gb_path, "rb") as f:
            gb_model = pickle.load(f)
    else:
        st.sidebar.error(f"❌ '{gb_path}' not found in current folder.")
        
    if os.path.exists(rf_path):
        with open(rf_path, "rb") as f:
            rf_model = pickle.load(f)
    else:
        st.sidebar.error(f"❌ '{rf_path}' not found in current folder.")
        
    return gb_model, rf_model

gb_pipeline, rf_pipeline = load_pipelines()

# 4. Streamlit Sidebar
st.sidebar.title("🎛️ Pipeline Settings")
st.sidebar.markdown("**Loaded Models:**")
st.sidebar.info("• **Gradient Boosting:** 90.19% CV\n• **Random Forest:** 87.49% CV")

threshold = st.sidebar.slider(
    "Decision Threshold (P(Real) ≥ θ)",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05,
    help="Default is 0.50. Articles with P(Real) >= threshold are classified as REAL (1)."
)

sample_choice = st.sidebar.selectbox(
    "Load Sample Article",
    [
        "Custom Article",
        "Sample: Hillary Clinton FBI Investigation (Fake)",
        "Sample: Kerry to go to Paris (Real)",
        "Sample: The Battle of New York Primary (Real)",
        "Sample: Paul Ryan Trump Rally (Fake)",
    ]
)

samples_dict = {
    "Sample: Hillary Clinton FBI Investigation (Fake)": {
        "title": "You Can Smell Hillary's Fear",
        "text": "Daniel Greenfield, a Shillman Journalism Fellow at the Freedom Center, is a New York writer focusing on radical Islam. In the final stretch of the election, Hillary Rodham Clinton has gone to war with the FBI. The FBI is under attack by everyone from Obama to CNN. Hillary's people have circulated a letter attacking Comey. You can smell the fear. Whatever she is afraid of, it lies buried in her emails."
    },
    "Sample: Kerry to go to Paris (Real)": {
        "title": "Kerry to go to Paris in gesture of sympathy",
        "text": "U.S. Secretary of State John F. Kerry said Monday that he will stop in Paris later this week, amid criticism that no top American officials attended Sunday's unity march against terrorism. Kerry said he expects to arrive in Paris Thursday evening, as he heads home after a week abroad."
    },
    "Sample: The Battle of New York Primary (Real)": {
        "title": "The Battle of New York: Why This Primary Matters",
        "text": "It's primary day in New York and front-runners Hillary Clinton and Donald Trump are leading in the polls. Trump is now vowing to win enough delegates to clinch the Republican nomination and prevent a contested convention."
    },
    "Sample: Paul Ryan Trump Rally (Fake)": {
        "title": "Watch The Exact Moment Paul Ryan Committed Political Suicide",
        "text": "Paul Ryan re-re-re-reversed course and announced that he was back on the Trump Train after all. Suddenly, Ryan was appearing at a pro-Trump rally and boldly declaring that he already sent in his vote. It was a surreal moment."
    }
}

# 5. Main View Header
st.title("📰 Fake News Detection: Model Prediction Comparator")
st.caption("Side-by-side inference comparison using Gradient Boosting and Random Forest pipelines.")

# Input Form
title_input = ""
text_input = ""

if sample_choice in samples_dict:
    title_input = samples_dict[sample_choice]["title"]
    text_input = samples_dict[sample_choice]["text"]

col_title = st.text_input("News Article Title", value=title_input, placeholder="Enter headline...")
article_text = st.text_area("Article Content", value=text_input, height=180, placeholder="Paste full article text...")

if st.button("🚀 Compare Model Predictions", type="primary"):
    combined_text = f"{col_title} {article_text}".strip()
    
    if not combined_text:
        st.warning("Please provide article title or body content.")
    else:
        with st.spinner("Extracting 300-d spaCy word embeddings & running pipelines..."):
            # Step A: Vectorize text using spaCy large model (300 dimensions)
            doc = nlp(combined_text)
            vector_300d = np.array(doc.vector).reshape(1, -1)
            
            # Step B: Model Inferences
            def predict_with_pipeline(pipe, vec, theta):
                if pipe is None:
                    return "Model Unavailable", 0.5, 0.5
                
                # Predict probabilities
                if hasattr(pipe, "predict_proba"):
                    probs = pipe.predict_proba(vec)[0] # [P(Fake), P(Real)]
                    p_fake = float(probs[0])
                    p_real = float(probs[1])
                else:
                    pred = pipe.predict(vec)[0]
                    p_real = 1.0 if pred == 1 else 0.0
                    p_fake = 1.0 - p_real
                    
                label = "REAL (1)" if p_real >= theta else "FAKE (0)"
                return label, p_fake, p_real

            gb_label, gb_pfake, gb_preal = predict_with_pipeline(gb_pipeline, vector_300d, threshold)
            rf_label, rf_pfake, rf_preal = predict_with_pipeline(rf_pipeline, vector_300d, threshold)
            
        st.markdown("---")
        
        # Step C: Consensus Callout
        agreed = (gb_label == rf_label)
        if agreed:
            st.success(f"🤝 **Consensus Reached:** Both pipelines classified this article as **{gb_label}**.")
        else:
            st.warning(f"⚠️ **Model Divergence:** Gradient Boosting predicted **{gb_label}**, but Random Forest predicted **{rf_label}**.")

        # Step D: Side-by-side Columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌲 Gradient Boosting Pipeline")
            st.markdown(f"`gb_best_pipeline.pkl` *(CV Score: 90.19%)*")
            if "REAL" in gb_label:
                st.metric("Prediction", "REAL NEWS", delta=f"{gb_preal*100:.1f}% confidence")
            else:
                st.metric("Prediction", "FAKE NEWS", delta=f"-{gb_pfake*100:.1f}% confidence", delta_color="inverse")
                
            st.progress(gb_preal, text=f"P(Real): {gb_preal*100:.1f}% | P(Fake): {gb_pfake*100:.1f}%")
            st.caption("Parameters: n_estimators=200, lr=0.2, max_depth=5")

        with col2:
            st.subheader("🌴 Random Forest Pipeline")
            st.markdown(f"`rf_best_pipeline.pkl` *(CV Score: 87.49%)*")
            if "REAL" in rf_label:
                st.metric("Prediction", "REAL NEWS", delta=f"{rf_preal*100:.1f}% confidence")
            else:
                st.metric("Prediction", "FAKE NEWS", delta=f"-{rf_pfake*100:.1f}% confidence", delta_color="inverse")
                
            st.progress(rf_preal, text=f"P(Real): {rf_preal*100:.1f}% | P(Fake): {rf_pfake*100:.1f}%")
            st.caption("Parameters: n_estimators=200, max_depth=10, min_samples_split=2")
