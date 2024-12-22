import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

# Başlık
st.title("Movie Review Sentiment Analysis")

# Kullanıcı giriş alanı
user_input = st.text_input("Ruh halinizi veya şu anki hislerinizi birkaç cümleyle ifade edin:")

# Tahmin işlemi için buton
if st.button("Tahmin Et"):
    try:
        # Veri yükleme
        df = pd.read_csv("Topmovies.csv")  # Film verisi
        if "name" not in df.columns or "tagline" not in df.columns:
            st.error("CSV dosyasında gerekli sütunlar bulunamadı!")
        else:
            # Film bilgilerini hazırlama
            filmer = [{"title": row["name"], "description": row["tagline"]} for _, row in df.iterrows() if pd.notna(row["tagline"])]
            
            # Film açıklamaları ve başlıkları
            film_descriptions = [film['description'] for film in filmer]
            film_titles = [film['title'] for film in filmer]

            # Vektörleştirici ve etiket kodlayıcı
            vectorizer = TfidfVectorizer()
            film_vectors = vectorizer.fit_transform(film_descriptions).toarray()

            label_encoder = LabelEncoder()
            film_labels = label_encoder.fit_transform(film_titles)

            # Kullanıcı girişini kontrol et
            if user_input.strip():
                # Kullanıcı girişini vektörleştir
                user_vector = vectorizer.transform([user_input]).toarray()

                # Modeli yükleme
                model = load_model("moviesmodel.keras")

                # Tahmin yapma
                prediction = model.predict(user_vector)
                predicted_label = np.argmax(prediction)
                predicted_film = label_encoder.inverse_transform([predicted_label])[0]

                # Tahmin sonucunu gösterme
                st.success(f"Size uygun olabilecek en iyi film önerisi: {predicted_film}")
            else:
                st.warning("Lütfen bir metin girin!")
    except FileNotFoundError as fnf_error:
        st.error("Gerekli dosyalar bulunamadı! Lütfen 'Topmovies.csv' ve 'moviesmodel.keras' dosyalarını kontrol edin.")
    except Exception as e:
        st.error(f"Tahmin sırasında bir hata oluştu: {str(e)}")


