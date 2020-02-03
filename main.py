import base64
import re
import requests
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request

app = Flask(__name__)

MODEL_PATH = "static/models/mnist_cnn.h5"

def parse_image(imgData):
    imgstr = re.search(b"base64,(.*)", imgData).group(1)
    img_decode = base64.decodebytes(imgstr)
    return img_decode

def preprocess_single_image(image_bytes):
    image = tf.image.decode_jpeg(image_bytes, channels=1)
    image = tf.image.resize(image, [28, 28])
    image = (255 - image) / 255.0  # normalize to [0,1] range
    image = tf.reshape(image, (1, 28, 28, 1))
    return image

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/upload/", methods=["POST"])
def upload_file():
    img_raw = parse_image(request.get_data())
    image = preprocess_single_image(img_raw)
    model = tf.keras.models.load_model(MODEL_PATH)
    y_proba = model.predict(image)
    prediction = np.argmax(y_proba, axis=1)
    return str(prediction)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)