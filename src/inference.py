# ══════════════════════════════════════════════════════════════════
#  src/inference.py
#  Production inference pipeline for AgeGuard AI.
#  Loads ONNX model, preprocesses image, returns age prediction
#  with confidence and alert status.
#
#  Usage:
#    predictor = AgePredictor('models/onnx/ageguard_v1.onnx')
#    result = predictor.predict('path/to/image.jpg')
# ══════════════════════════════════════════════════════════════════
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# ImageNet normalization (confirmed in EDA 2.3)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class AgePredictor:
    """
    Production-ready age predictor using ONNX Runtime.

    Args:
        model_path: path to .onnx model file
        alert_threshold: age below which an alert is triggered (default 25)
        legal_age: legal age for compliance (default 21)
        img_size: model input size (default 224)
    """

    def __init__(
        self,
        model_path: str,
        alert_threshold: int = 25,
        legal_age: int = 21,
        img_size: int = 224,
    ):
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self.alert_threshold = alert_threshold
        self.legal_age = legal_age
        self.img_size = img_size

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model input.
        Accepts BGR (OpenCV) or RGB numpy array.
        """
        # Resize
        img = cv2.resize(
            image, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA
        )

        # BGR to RGB if needed (OpenCV loads BGR)
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1] then apply ImageNet stats
        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD

        # HWC to NCHW
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        return img

    def predict(self, image) -> dict:
        """
        Predict age from image.

        Args:
            image: file path (str/Path), or numpy array (BGR/RGB)

        Returns:
            dict with: predicted_age, alert, alert_level, needs_verification
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                return {"error": f"Could not load image: {image}"}
        else:
            img = image

        # Preprocess
        input_tensor = self.preprocess(img)

        # Inference
        pred = self.session.run(None, {"image": input_tensor})[0]
        predicted_age = float(pred.flatten()[0])

        # Clamp to valid range
        predicted_age = max(0, min(100, predicted_age))

        # Alert logic
        if predicted_age < self.legal_age:
            alert_level = "RED"
            alert_msg = "BLOCK — Predicted minor"
        elif predicted_age < self.alert_threshold:
            alert_level = "YELLOW"
            alert_msg = "VERIFY — Request ID"
        else:
            alert_level = "GREEN"
            alert_msg = "PASS — Predicted adult"

        return {
            "predicted_age": round(predicted_age, 1),
            "alert_level": alert_level,
            "alert_message": alert_msg,
            "needs_verification": predicted_age < self.alert_threshold,
            "legal_age": self.legal_age,
            "alert_threshold": self.alert_threshold,
        }
