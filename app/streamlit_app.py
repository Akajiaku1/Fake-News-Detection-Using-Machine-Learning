import streamlit as st
import joblib
import os

from src.features import clean_text

st.set_page_config(page_title="Fake News Detector", page_icon="📰")

ARTIFACT_DIR = "artifacts"
model = joblib.load(os.path.join(ARTIFACT_DIR, "model.joblib"))
vectorizer = joblib.load(os.path.join(ARTIFACT_DIR, "vectorizer.joblib"))
le = joblib.load(os.path.join(ARTIFACT_DIR, "label_encoder.joblib"))

st.title("📰 Fake News Detection (Classical ML)")
st.write("Paste an article title or body and get a prediction.")

user_text = st.text_area("Enter news text:", height=250)

if st.button("Predict"):
    if not user_text.strip():
        st.warning("Please provide some text.")
    else:
        cleaned = clean_text(user_text)
        X_vec = vectorizer.transform([cleaned])
        pred = model.predict(X_vec)[0]
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_vec)[0, 1]

        label = le.inverse_transform([pred])[0]
        st.markdown(f"**Predicted label:** `{label}`")
        if proba is not None:
            st.markdown(f"**Probability of being fake:** `{proba:.4f}`")
