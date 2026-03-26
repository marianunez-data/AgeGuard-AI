# ══════════════════════════════════════════════════════════════════
#  app.py — Gradio Demo for HuggingFace Spaces
#  AgeGuard AI: Age Verification System
# ══════════════════════════════════════════════════════════════════
import gradio as gr
import numpy as np
import cv2
import onnxruntime as ort
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

MODEL_PATH = "models/onnx/ageguard_v1.onnx"
LEGAL_AGE = 21
ALERT_THRESHOLD = 25


# ── Load model ────────────────────────────────────────────────────
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])


# ── Prediction function ──────────────────────────────────────────
def predict_age(image):
    if image is None:
        return "No image provided", None

    # Resize to 224x224
    img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Normalize
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD

    # HWC to NCHW
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    # Inference
    pred = session.run(None, {"image": img})[0]
    age = float(pred.flatten()[0])
    age = max(0, min(100, age))

    # Alert logic
    if age < LEGAL_AGE:
        alert = "🔴 BLOCK — Predicted minor"
        color = "#FF0000"
    elif age < ALERT_THRESHOLD:
        alert = "🟡 VERIFY — Request ID"
        color = "#FFA500"
    else:
        alert = "🟢 PASS — Predicted adult"
        color = "#00AA00"

    # Format output
    result = f"""
    ## Predicted Age: {age:.1f} years

    ### Alert: {alert}

    ---

    **System Configuration:**
    - Legal age: {LEGAL_AGE}
    - Alert threshold: {ALERT_THRESHOLD}
    - RED = Predicted < {LEGAL_AGE} → BLOCK sale
    - YELLOW = Predicted {LEGAL_AGE}-{ALERT_THRESHOLD} → REQUEST ID
    - GREEN = Predicted > {ALERT_THRESHOLD} → APPROVE sale
    """

    return result


# ── Gradio Interface ──────────────────────────────────────────────
demo = gr.Interface(
    fn=predict_age,
    inputs=gr.Image(label="Upload Face Image", type="numpy"),
    outputs=gr.Markdown(label="Prediction Result"),
    title="🛡️ AgeGuard AI — Age Verification System",
    description=(
        "Upload a face image to estimate age and receive an alert level. "
        "The system uses EfficientNetV2-S trained on facial features to predict age "
        "with a Mean Absolute Error of ~5 years. "
        "Inference runs in ~17ms using ONNX Runtime."
    ),
    examples=[],
    theme=gr.themes.Soft(),
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
