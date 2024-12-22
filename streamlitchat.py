import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

model = load_model("moviesmodel.keras")
vectorizer = TfidfVectorizer()  
label_encoder = LabelEncoder()  

st.title("Movie Review Sentiment Analysis")


user_input = st.text_input("Ruh halinizi veya şu anki hislerinizi birkaç cümleyle ifade edin:")


if st.button("Tahmin Et"):
    if user_input.strip():  
        try:
            
            user_vector = vectorizer.transform([user_input]).toarray()
            
            # Tahmin yapma
            prediction = model.predict(user_vector)
            predicted_label = np.argmax(prediction)
            predicted_film = label_encoder.inverse_transform([predicted_label])[0]

            st.success(f"Size uygun olabilecek en iyi film önerisi: {predicted_film}")
        except Exception as e:
            st.error(f"Tahmin sirasinda bir hata oluştu: {str(e)}")
    else:
        st.warning("Lütfen bir metin girin!")

