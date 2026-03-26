# ══════════════════════════════════════════════════════════════════
#  src/api.py
#  FastAPI endpoint for AgeGuard AI production inference.
#  Receives image, returns age prediction + alert level.
# ══════════════════════════════════════════════════════════════════
import io
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image

from inference import AgePredictor

# ── Initialize ────────────────────────────────────────────────────
app = FastAPI(
    title="AgeGuard AI",
    description="Automated age verification for regulated retail",
    version="1.0.0",
)

MODEL_PATH = Path(__file__).parent.parent / "models" / "onnx" / "ageguard_v1.onnx"

predictor = AgePredictor(
    model_path=str(MODEL_PATH),
    alert_threshold=25,
    legal_age=21,
)


# ── Endpoints ─────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "healthy", "model": "ageguard_v1.onnx"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload a face image, receive age prediction + alert level.
    """
    start = time.time()

    # Read image
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_array = np.array(img)

    # Predict
    result = predictor.predict(img_array)

    # Add latency
    result["latency_ms"] = round((time.time() - start) * 1000, 1)

    return JSONResponse(content=result)


@app.get("/config")
def config():
    return {
        "legal_age": predictor.legal_age,
        "alert_threshold": predictor.alert_threshold,
        "model": "EfficientNetV2-S",
        "inference": "ONNX Runtime",
    }
