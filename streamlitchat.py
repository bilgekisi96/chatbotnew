from tensorflow.keras.models import load_model
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np

model = load_model("moviesmodel.keras")

st.title("Movie Review Sentiment Analysis")

user_input = st.input("Ruh halinizi veya şu anki hislerinizi birkaç cümleyle ifade edin:: ")
#user_input = input("Ruh halinizi veya şu anki hislerinizi birkaç cümleyle ifade edin: ")
user_vector = vectorizer.transform([user_input]).toarray()  # Kullanıcı girişini vektörleştir

# 10. Film Tahmini
prediction = model.predict(user_vector)
predicted_label = np.argmax(prediction)
predicted_film = label_encoder.inverse_transform([predicted_label])[0]

st.write(f"Size uygun olabilecek en iyi film önerisi: {predicted_film}")
