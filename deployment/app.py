from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import gdown

app = Flask(__name__)

# -----------------------------
# SAFE BASE PATH
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_PATH = os.path.join(BASE_DIR, "../config/config.json")
LABELS_PATH = os.path.join(BASE_DIR, "../config/labels.txt")
MODEL_PATH = os.path.join(BASE_DIR, "../model/xray_model.keras")

# -----------------------------
# LOAD CONFIG
# -----------------------------
with open(CONFIG_PATH) as f:
    config = json.load(f)

IMG_SIZE = tuple(config["img_size"])

with open(LABELS_PATH, "r") as f:
    labels = f.read().splitlines()

# -----------------------------
# LOAD MODEL (FROM GOOGLE DRIVE)
# -----------------------------

MODEL_PATH = "model/xray_model.keras"

# Google Drive FILE ID (PUT YOUR REAL ID HERE)
FILE_ID = "1N_cB6Sgp1qb6RH_EBt_wAI-zyUcEc3AE"
URL = f"https://drive.google.com/uc?id={FILE_ID}"

# Create model folder
os.makedirs("model", exist_ok=True)

# Download model if not present
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(URL, MODEL_PATH, quiet=False)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# -----------------------------
# PREPROCESS FUNCTION
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

        prediction = model.predict(processed)[0][0]

        result = labels[1] if prediction > 0.5 else labels[0]

        return jsonify({
            "prediction": result,
            "confidence": float(prediction)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# -----------------------------
# NO app.run() FOR PRODUCTION 
# -----------------------------