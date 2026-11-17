
import io
import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from flask import Flask, jsonify, render_template, request
from PIL import Image
from torchvision import models, transforms


# -----------------------------
# Flask configuration
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# -----------------------------
# Load trained model
# -----------------------------
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pth")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

CLASSES = checkpoint["classes"]
IMG_SIZE = checkpoint.get("img_size", 224)

model = models.mobilenet_v2(weights=None)

model.classifier[1] = nn.Linear(
    model.last_channel,
    len(CLASSES)
)

model.load_state_dict(checkpoint["state_dict"])

model = model.to(device)
model.eval()

print("Flower model loaded successfully!")
print("Number of categories:", len(CLASSES))


# -----------------------------
# Load flower names
# -----------------------------
JSON_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "102 flower",
    "cat_to_name.json"
)

with open(JSON_PATH, "r", encoding="utf-8") as f:
    cat_to_name = json.load(f)


# -----------------------------
# Image preprocessing
# -----------------------------
tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])


ALLOWED_EXT = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# -----------------------------
# Home page
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")


# -----------------------------
# Flower prediction
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    file = request.files.get("image")

    if file is None or file.filename == "":
        return jsonify({
            "error": "No image uploaded"
        }), 400

    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in ALLOWED_EXT:
        return jsonify({
            "error": "Unsupported image format"
        }), 400

    try:
        img = Image.open(
            io.BytesIO(file.read())
        ).convert("RGB")

    except Exception:
        return jsonify({
            "error": "Could not read image"
        }), 400

    # Predict
    with torch.no_grad():

        x = tf(img).unsqueeze(0).to(device)

        outputs = model(x)

        probs = F.softmax(outputs, dim=1).squeeze(0)

    top = torch.topk(
        probs,
        k=min(5, len(CLASSES))
    )

    results = []

    for j, idx in enumerate(top.indices.tolist()):

        class_id = CLASSES[idx]

        flower_name = cat_to_name.get(
            str(class_id),
            str(class_id)
        )

        confidence = round(
            float(top.values[j]) * 100,
            2
        )

        results.append({
            "label": flower_name,
            "confidence": confidence
        })

    return jsonify({
        "predictions": results
    })


# -----------------------------
# Run application
# -----------------------------
if __name__ == "__main__":

   app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )
