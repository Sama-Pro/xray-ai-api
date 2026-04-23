from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import gdown

app = Flask(__name__)

# -----------------------------
# BASE PATH
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.json")
LABELS_PATH = os.path.join(BASE_DIR, "config", "labels.txt")
MODEL_PATH = os.path.join(BASE_DIR, "model", "xray_model.keras")

# -----------------------------
# LOAD CONFIG
# -----------------------------
with open(CONFIG_PATH) as f:
    config = json.load(f)

IMG_SIZE = tuple(config["img_size"])

with open(LABELS_PATH, "r") as f:
    labels = f.read().splitlines()

# -----------------------------
# DOWNLOAD MODEL (ONLY IF NEEDED)
# -----------------------------
FILE_ID = "1N_cB6Sgp1qb6RH_EBt_wAI-zyUcEc3AE"
URL = f"https://drive.google.com/uc?id={FILE_ID}"

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(URL, MODEL_PATH, quiet=False)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH)

# -----------------------------
# PREPROCESS
# -----------------------------
def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# -----------------------------
# ROUTES
# -----------------------------
@app.route("/")
def home():
    return "X-Ray AI API is running ✔"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["image"]
        img = Image.open(file).convert("RGB")

        processed = preprocess_image(img)
        pred = model.predict(processed)[0][0]

        # FIXED confidence
        confidence = pred if pred > 0.5 else (1 - pred)

        result = labels[1] if pred > 0.5 else labels[0]

        return jsonify({
            "prediction": result,
            "confidence": float(confidence)
        })

    except Exception as e:
        return jsonify({"error": str(e)})