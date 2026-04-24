from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import gdown

app = Flask(__name__)

# -----------------------------
# BASE DIRECTORY (deployment/)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Move OUT of deployment → access root folders
CONFIG_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "config", "config.json"))
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "model", "xray_model_clean.h5"))

# -----------------------------
# LOAD CONFIG
# -----------------------------
with open(CONFIG_PATH) as f:
    config = json.load(f)

IMG_SIZE = tuple(config["img_size"])
labels = config["classes"]

# -----------------------------
# GOOGLE DRIVE MODEL DOWNLOAD
# -----------------------------
FILE_ID = "11dolXa13dFeErsoRhn5nBl0w6GOyv8Jv" 
DOWNLOAD_URL = f"https://drive.google.com/uc?id={FILE_ID}"

# Ensure model folder exists
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Download model if not already present
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=False)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded successfully")

# -----------------------------
# IMAGE PREPROCESSING
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

        # Calculate confidence properly
        confidence = prediction if prediction > 0.5 else (1 - prediction)

        result = labels[1] if prediction > 0.5 else labels[0]

        return jsonify({
            "prediction": result,
            "confidence": float(confidence)
        })

    except Exception as e:
        return jsonify({"error": str(e)})