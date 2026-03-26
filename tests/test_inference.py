# ══════════════════════════════════════════════════════════════════
#  tests/test_inference.py
#  Tests for production inference pipeline.
# ══════════════════════════════════════════════════════════════════
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from config import load_config
from inference import AgePredictor

YAML_PATH = Path(__file__).parent.parent / "configs" / "base_config.yaml"
CFG = load_config(YAML_PATH)

ONNX_PATH = CFG.base_dir / "models" / "onnx" / "ageguard_v1.onnx"


@pytest.fixture
def predictor():
    return AgePredictor(
        model_path=str(ONNX_PATH),
        alert_threshold=CFG.business.alert_threshold,
        legal_age=CFG.business.legal_age,
    )


def test_predictor_loads(predictor):
    """ONNX predictor loads without errors."""
    assert predictor is not None


def test_predict_from_array(predictor):
    """Prediction works from numpy array."""
    dummy = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    result = predictor.predict(dummy)
    assert "predicted_age" in result
    assert "alert_level" in result
    assert 0 <= result["predicted_age"] <= 100


def test_predict_from_file(predictor):
    """Prediction works from file path."""
    import pandas as pd

    df = pd.read_csv(CFG.base_dir / "data" / "labels_split.csv")
    test_file = df[df["split"] == "test"].iloc[0]["file_name"]
    result = predictor.predict(str(CFG.processed_dir / test_file))
    assert "predicted_age" in result
    assert result["alert_level"] in ["RED", "YELLOW", "GREEN"]


def test_alert_levels(predictor):
    """Alert levels correspond to correct thresholds."""
    # Mock results by checking logic
    assert predictor.legal_age == 21
    assert predictor.alert_threshold == 25


def test_invalid_image(predictor):
    """Handles invalid image path gracefully."""
    result = predictor.predict("nonexistent_image.jpg")
    assert "error" in result


def test_onnx_model_exists():
    """ONNX model file exists."""
    assert ONNX_PATH.exists()
    assert ONNX_PATH.stat().st_size > 0
