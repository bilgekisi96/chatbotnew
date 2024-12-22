import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


# Model ve yardımcı nesneleri yükleme
model = load_model("moviesmodel.keras")
vectorizer = TfidfVectorizer()  # Gerçek model eğitiminde kullanılan TfidfVectorizer yüklenmeli
label_encoder = LabelEncoder()  # Gerçek model eğitiminde kullanılan LabelEncoder yüklenmeli

# Streamlit başlığı
st.title("Movie Review Sentiment Analysis")

# Kullanıcıdan giriş alma
user_input = st.text_input("Ruh halinizi veya şu anki hislerinizi birkaç cümleyle ifade edin:")

# Düğme ile tahmin çalıştırma
if st.button("Tahmin Et"):
    if user_input.strip():  # Kullanıcı bir şey girmişse
        try:
            # Kullanıcı girişini vektörleştir
            user_vector = vectorizer.transform([user_input]).toarray()
            
            # Tahmin yapma
            prediction = model.predict(user_vector)
            predicted_label = np.argmax(prediction)
            predicted_film = label_encoder.inverse_transform([predicted_label])[0]

            # Sonucu yazdırma
            st.success(f"Size uygun olabilecek en iyi film önerisi: {predicted_film}")
        except Exception as e:
            st.error(f"Tahmin sirasinda bir hata oluştu: {str(e)}")
    else:
        st.warning("Lütfen bir metin girin!")

