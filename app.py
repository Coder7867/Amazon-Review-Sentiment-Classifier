
import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already downloaded
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

def clean_input(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def predict_sentiment(text):
    cleaned = clean_input(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return 'Positive' if prediction == 1 else 'Negative'

st.title("Amazon Review Sentiment Classifier")
user_input = st.text_area("Enter your Amazon review below:")

if st.button("Predict Sentiment"):
    result = predict_sentiment(user_input)
    st.write(f"Predicted Sentiment: {result}")
