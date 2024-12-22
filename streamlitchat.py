import threading
from flask import Flask, request, jsonify
import tensorflow as tf
import os
import sys

# Flask API uygulaması
app = Flask(__name__)

# Basit bir TensorFlow modeli
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
    os.environ["STREAMLIT_SERVER_PORT"] = "8501"
    sys.argv = ["streamlit", "run", "streamlit_app.py"]
    

if __name__ == '__main__':
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()
    
    streamlit_thread = threading.Thread(target=run_streamlit)
    streamlit_thread.start()