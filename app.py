import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from src.pipelines import build_pipeline
from src.preprocess import ensure_nltk_resources

st.set_page_config(page_title="Fake News Detector")
st.title("Fake News Detector")

fake_file = st.file_uploader("Upload Fake.csv", type="csv")
real_file = st.file_uploader("Upload True.csv", type="csv")


@st.cache_resource
def load_model(f_csv, r_csv):
    ensure_nltk_resources()
    fake = pd.read_csv(f_csv).assign(label=0)
    real = pd.read_csv(r_csv).assign(label=1)
    df = pd.concat([
        fake.sample(min(1500, len(fake)), random_state=42),
        real.sample(min(1500, len(real)), random_state=42)
    ], ignore_index=True)
    df = df[["text", "label"]].dropna(subset=["text"]).drop_duplicates(subset=["text"])
    pipe = build_pipeline(
        LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42),
        combined=False
    )
    return pipe.fit(df["text"].values, df["label"].values)


if fake_file and real_file:
    model = load_model(fake_file, real_file)
    text = st.text_area("Paste a news article:")
    if st.button("Predict") and text.strip():
        pred = model.predict([text])[0]
        st.success("Real News (1)" if pred == 1 else "Fake News (0)")
else:
    st.info("Upload both Fake.csv and True.csv to train and predict.")