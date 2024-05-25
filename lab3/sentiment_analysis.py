import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

st.title('Анализ тональности текста')

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

text = st.text_area(label='Введите текст (на английском языке)')

button = st.button('Анализ тональности')
if button:
    result = sia.polarity_scores(text)
    st.write(f"Positive: {result['pos']}")
    st.write(f"Negative: {result['neg']}")
    st.write(f"Neutral: {result['neu']}")
