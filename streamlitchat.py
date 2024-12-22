import threading
from flask import Flask, request, jsonify
import tensorflow as tf
import streamlit as st
import requests

# Flask API
app = Flask(__name__)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=[1])
])
model.compile(optimizer='sgd', loss='mean_squared_error')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_value = data['input']
    prediction = model.predict([input_value]).tolist()
    return jsonify({"prediction": prediction})

def run_flask():
    app.run(debug=True, use_reloader=False, port=5000)

def run_streamlit():
    st.title("Streamlit ve Flask Entegrasyonu")
    input_value = st.number_input("Tahmin için bir değer girin:", step=1.0)
    if st.button("Tahmin Et"):
        response = requests.post("http://127.0.0.1:5000/predict", json={"input": [input_value]})
        prediction = response.json()["prediction"]
        st.write(f"Tahmin Sonucu: {prediction[0]}")

# Paralel çalıştırma
if __name__ == '__main__':
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()
    run_streamlit()