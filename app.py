import streamlit as st
import joblib
import re
import numpy as np
from pathlib import Path

# ----------------------------------
# PAGE CONFIGURATION
# ----------------------------------

st.set_page_config(
    page_title="Kindle Sentiment Analyzer",
    page_icon="📚",
    layout="centered"
)

# ----------------------------------
# CUSTOM CSS
# ----------------------------------

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #10172a,
        #1e293b,
        #312e81
    );
    color: white;
}

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 17px;
    margin-bottom: 30px;
}

.result-card {
    background: rgba(255,255,255,0.10);
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.2);
    margin-top: 20px;
}

.positive {
    color: #4ade80;
    font-size: 32px;
    font-weight: bold;
}

.negative {
    color: #fb7185;
    font-size: 32px;
    font-weight: bold;
}

.rating {
    font-size: 35px;
    color: #facc15;
    font-weight: bold;
}


/* Review input box */
div[data-testid="stTextArea"] textarea {
    background-color: #ffffff !important;
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
    font-size: 16px !important;
    font-weight: 500 !important;
    border: 2px solid #6366f1 !important;
    border-radius: 12px !important;
}

/* Placeholder text */
div[data-testid="stTextArea"] textarea::placeholder {
    color: #6b7280 !important;
    opacity: 1 !important;
}

/* Text area label */
div[data-testid="stTextArea"] label {
    color: #ffffff !important;
    font-weight: 600 !important;
}


div.stButton > button {
    width: 100%;
    height: 50px;
    border-radius: 12px;
    background: #6366f1;
    color: white;
    font-size: 18px;
    font-weight: bold;
    border: none;
}

div.stButton > button:hover {
    background: #4f46e5;
    color: white;
}

textarea {
    border-radius: 12px !important;
}

.footer {
    text-align: center;
    color: #cbd5e1;
    margin-top: 45px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# ----------------------------------
# LOAD TRAINED MODELS
# ----------------------------------

BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_models():

    vectorizer = joblib.load(
        BASE_DIR / "models" / "tfidf_vectorizer.pkl"
    )

    model = joblib.load(
        BASE_DIR / "models" / "tfidf_model.pkl"
    )

    return vectorizer, model


try:
    tfidf_vectorizer, model = load_models()

except Exception as e:
    st.error(f"Unable to load trained models: {e}")
    st.stop()


# ----------------------------------
# TEXT PREPROCESSING
# ----------------------------------

def clean_text(text):

    text = text.lower()

    text = re.sub(r"<.*?>", " ", text)

    text = re.sub(r"[^a-z\s']", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ----------------------------------
# MAIN HEADER
# ----------------------------------

st.markdown(
    '<div class="main-title">📚 Kindle Review Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered Amazon Review Sentiment Analysis</div>',
    unsafe_allow_html=True
)

st.divider()

st.markdown("### ✍️ Enter Your Book Review")

st.write(
    "Write or paste your Amazon Kindle review below "
    "to discover its sentiment."
)


# ----------------------------------
# USER INPUT
# ----------------------------------

review = st.text_area(
    "Your Review",
    placeholder="Example: This book is amazing! I really enjoyed reading it...",
    height=150
)


# ----------------------------------
# SENTIMENT PREDICTION
# ----------------------------------

if st.button("🔍 Analyze Sentiment"):

    if not review.strip():

        st.warning("Please enter a review first.")

    else:

        cleaned_review = clean_text(review)

        if not cleaned_review:

            st.warning("Please enter a valid text review.")

        else:

            # Convert text into TF-IDF features
            features = tfidf_vectorizer.transform(
                [cleaned_review]
            )

            # Predict sentiment
            prediction = model.predict(features)[0]

            # Get probability
            probabilities = model.predict_proba(features)[0]

            positive_probability = probabilities[
                list(model.classes_).index(1)
            ]

            confidence = max(probabilities) * 100

            # Determine sentiment
            if prediction == 1:

                sentiment = "Positive"
                sentiment_class = "positive"
                emoji = "😊"

            else:

                sentiment = "Negative"
                sentiment_class = "negative"
                emoji = "😞"

            # Estimated rating from positive probability
            estimated_rating = int(
                np.clip(
                    np.rint(1 + 4 * positive_probability),
                    1,
                    5
                )
            )

            stars = "⭐" * estimated_rating

            # ----------------------------------
            # DISPLAY RESULTS
            # ----------------------------------

            st.markdown("---")

st.markdown("## 📊 Sentiment Analysis Result")

if prediction == 1:

    st.success("😊 Positive Sentiment Detected!")

    st.markdown("### 🌟 Customer Rating")

    st.markdown(
        f"# {'⭐' * estimated_rating} "
        f"({estimated_rating}/5)"
    )

    st.balloons()

else:

    st.error("😞 Negative Sentiment Detected!")

    st.markdown("### 🌟 Customer Rating")

    st.markdown(
        f"# {'⭐' * estimated_rating} "
        f"({estimated_rating}/5)"
    )

st.markdown("### 🎯 Prediction Confidence")

st.progress(float(confidence / 100))

st.write(f"**{confidence:.2f}%**")

st.info(
    "This rating is an estimated rating based on "
    "the sentiment model's probability, not an actual "
    "star-rating prediction."
)

st.markdown("---")



# ----------------------------------
# FOOTER
# ----------------------------------

st.markdown(
    """
    <div class="footer">
        Built with Python | Streamlit | NLP | Machine Learning
        <br><br>
        Amazon Kindle Review Sentiment Analysis Project
    </div>
    """,
    unsafe_allow_html=True
)
