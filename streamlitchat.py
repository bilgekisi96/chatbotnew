import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

  

st.title("Movie Review Sentiment Analysis")


user_input = st.text_input("Ruh halinizi veya şu anki hislerinizi birkaç cümleyle ifade edin:")

if st.button("Tahmin Et"):
    
    
    vectorizer = TfidfVectorizer()
    df = pd.read_csv("Topmovies.csv")
    filmer = [{"title":k} for k in df.name]
    t = 0
    for p in df.tagline:
        filmer[t]['description'] = p
        t+=1   
        
    films = filmer
    film_descriptions = [film['description'] for film in films]
    film_titles = [film['title'] for film in films]    


    if user_input:  
        try:
            model = load_model("moviesmodel.keras")
            film_vectors = vectorizer.fit_transform(film_descriptions).toarray()  # Film açıklamalarını vektörleştir
            user_vector = vectorizer.transform([user_input]).toarray()  # Kullanıcı girişini vektörleştir

            label_encoder = LabelEncoder()
            film_labels = label_encoder.fit_transform(film_titles)
            
            # Tahmin yapma
            prediction = model.predict(user_vector)
            predicted_label = np.argmax(prediction)
            predicted_film = label_encoder.inverse_transform([predicted_label])[0]

            st.success(f"Size uygun olabilecek en iyi film önerisi: {predicted_film}")
        except Exception as e:
            st.error(f"Tahmin sirasinda bir hata oluştu: {str(e)}")
    else:
        st.warning("Lütfen bir metin girin!")

