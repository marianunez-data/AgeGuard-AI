import io
import os
import time
import urllib.request
from pathlib import Path

import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image

# ── Download model if not present ────────────────────────────────
MODEL_DIR = Path(__file__).parent.parent / "models" / "onnx"
MODEL_PATH = MODEL_DIR / "ageguard_v1.onnx"
HF_MODEL_URL = "https://huggingface.co/spaces/marianunez-data/AgeGuard-AI/resolve/main/models/onnx/ageguard_v1.onnx"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def download_model():
    if not MODEL_PATH.exists():
        print("Downloading ONNX model from HuggingFace...")
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(HF_MODEL_URL, str(MODEL_PATH))
        print(f"Model downloaded: {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1e6:.1f} MB)")

download_model()

import onnxruntime as ort

# ── App ──────────────────────────────────────────────────────────
app = FastAPI(
    title="AgeGuard AI",
    description="Automated age verification API for retail compliance. Upload a face image to get age prediction and alert level.",
    version="1.0.0",
)

session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])

LEGAL_AGE = int(os.environ.get("LEGAL_AGE", 21))
ALERT_THRESHOLD = int(os.environ.get("ALERT_THRESHOLD", 25))

def preprocess(image: np.ndarray) -> np.ndarray:
    from PIL import Image as PILImage
    import cv2
    img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)

@app.get("/")
def root():
    return {
        "service": "AgeGuard AI",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Upload face image for age prediction",
            "/health": "GET - Service health check",
            "/config": "GET - Model configuration",
        }
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": session is not None,
        "model_size_mb": round(MODEL_PATH.stat().st_size / 1e6, 1),
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start = time.time()

    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_array = np.array(img)

    input_tensor = preprocess(img_array)
    pred = session.run(None, {"image": input_tensor})[0]
    age = int(round(float(pred.flatten()[0])))
    age = max(0, min(100, age))

    if age < LEGAL_AGE:
        alert_level = "RED"
        alert_action = "BLOCK — Predicted minor, deny sale"
    elif age < ALERT_THRESHOLD:
        alert_level = "YELLOW"
        alert_action = "VERIFY — Request physical ID"
    else:
        alert_level = "GREEN"
        alert_action = "APPROVED — Sale cleared"

    latency = round((time.time() - start) * 1000, 1)

    return JSONResponse(content={
        "predicted_age": age,
        "alert_level": alert_level,
        "action": alert_action,
        "needs_verification": age < ALERT_THRESHOLD,
        "latency_ms": latency,
    })

@app.get("/config")
def config():
    return {
        "model": "EfficientNetV2-S",
        "inference": "ONNX Runtime",
        "legal_age": LEGAL_AGE,
        "alert_threshold": ALERT_THRESHOLD,
        "image_size": 224,
    }
