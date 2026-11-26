import streamlit as st
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Identification")
st.write("By **Israr Ahmad** | ML Project 2023 | TF-IDF + Logistic Regression")

texts = [
    "Government announces new education policy for universities",
    "Scientists discover new species in deep ocean research",
    "Breaking aliens landed in Peshawar and took over city shocking",
    "Miracle cure drinking this juice cures all diseases overnight doctors shocked",
    "Prime Minister inaugurates new motorway project in Pakistan",
    "You won 10 million lottery click this link to claim prize now",
    "University announces exam schedule for final semester",
    "Secret government hiding aliens in Area 51 leaked video proves",
    "Pakistan cricket team wins match by 5 wickets",
    "Share this message or you will have bad luck for 10 years",
]
labels = [0, 0, 1, 1, 0, 1, 0, 1, 0, 1]

def clean(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text

@st.cache_resource
def train():
    vec = TfidfVectorizer(stop_words='english', max_features=500)
    X = vec.fit_transform([clean(t) for t in texts])
    model = LogisticRegression()
    model.fit(X, labels)
    return vec, model

vec, model = train()
news = st.text_area("Apni News Yahan Likho:", height=120)

if st.button("Check - Real ya Fake?"):
    if not news.strip():
        st.warning("Pehle koi news likho!")
    else:
        X_new = vec.transform([clean(news)])
        pred = model.predict(X_new)[0]
        prob = model.predict_proba(X_new)[0]
        confidence = max(prob) * 100
        if pred == 1:
            st.error(f"🚨 FAKE NEWS LAG RAHI HAI - Confidence: {confidence:.1f}%")
        else:
            st.success(f"✅ REAL NEWS LAG RAHI HAI - Confidence: {confidence:.1f}%")
        st.progress(float(max(prob)))
        st.info("Note: Ye Demo Prototype hai. Real dataset par train karne se accuracy aur behtar hogi.")

st.markdown("---")
st.markdown("**How it works:** Text Cleaning -> TF-IDF -> Logistic Regression -> Prediction")
