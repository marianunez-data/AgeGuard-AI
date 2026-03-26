# 🛡️ AgeGuard AI — Automated Age Verification System

> Deep learning-powered age estimation for regulated retail environments. Real-time facial analysis with a three-tier alert system (RED/YELLOW/GREEN) for compliance at point of sale.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-green)
![Tests](https://img.shields.io/badge/Tests-21%2F21_passing-brightgreen)
![MAE](https://img.shields.io/badge/MAE-5.02_years-orange)
![Inference](https://img.shields.io/badge/Inference-17ms-blue)

### [Live Demo on HuggingFace](https://huggingface.co/spaces/marianunez-data/AgeGuard-AI)

---

## Business Problem

Retail businesses selling age-restricted products face regulatory fines when minors are not properly identified. Human verification is inconsistent. AgeGuard AI automates the first layer of age verification, alerting supervisors when a customer's predicted age falls below the safety threshold.

## System Architecture
```
Camera at POS --> Face Detection --> Age Estimation (ONNX, 17ms)
                                          |
                                +---------+---------+
                                |   Alert Engine     |
                                +---------+---------+
                                | RED    (< 21) BLOCK|
                                | YELLOW (21-25) ID  |
                                | GREEN  (> 25) PASS |
                                +--------------------+
```

## Key Results

| Metric | Value | Target |
|---|---|---|
| Test MAE (global) | 5.02 years | <= 5.0 |
| Test MAE (18-25 critical) | 4.34 years | <= 5.0 |
| False Accept Rate (t=25) | 12.3% | Minimized |
| False Reject Rate (t=25) | 20.3% | Acceptable |
| Inference latency | 16.7 ms | < 50 ms |
| Adult auto-approval | 79.7% | Maximized |
| Model size (ONNX) | 77.5 MB | Deployable |

## Project Structure
```
AgeGuard-AI/
├── artifacts/          # EDA summary, removal lists, split stats
├── configs/            # YAML config (Pydantic-validated)
├── data/               # Original + processed images (DVC tracked)
├── models/             # Checkpoints + ONNX (DVC tracked)
├── notebooks/          # Full pipeline notebook
├── reports/
│   ├── eda/            # 17+ EDA visualizations
│   ├── training/       # Training curves + summary
│   ├── evaluation/     # Test metrics + threshold analysis
│   └── explainability/ # GradCAM visualizations
├── src/
│   ├── config.py       # Pydantic config loader
│   ├── dataset.py      # PyTorch Dataset + transforms
│   ├── model.py        # EfficientNetV2-S regression head
│   ├── train.py        # Training loop with resume
│   ├── inference.py    # ONNX production inference
│   └── api.py          # FastAPI endpoint
├── tests/              # pytest (21/21 passing)
├── Dockerfile          # Production container
├── app.py              # Gradio demo (HuggingFace)
└── requirements.txt
```

## Quick Start
```bash
git clone https://github.com/marianunez-data/AgeGuard-AI.git
cd AgeGuard-AI
pip install -r requirements.txt
```

### Predict age from image
```python
from src.inference import AgePredictor

predictor = AgePredictor(model_path="models/onnx/ageguard_v1.onnx")
result = predictor.predict("path/to/face.jpg")
print(result)
# {'predicted_age': 22, 'alert_level': 'YELLOW', 'alert_message': 'VERIFY - Request ID'}
```

### Run tests
```bash
python -m pytest tests/ -v
```

## Pipeline Phases

| Phase | Description | Key Output |
|---|---|---|
| 1. Setup | Project structure, Pydantic config | config.py, base_config.yaml |
| 2. EDA | 11 audits: age, resolution, blur, duplicates, faces | eda_summary.json |
| 3. Preprocessing | Visual review, face crop, stratified split | 7,446 images 224x224 |
| 4. Training | EfficientNetV2-S, HuberLoss, 20 epochs | best_model.pth |
| 5. Evaluation | Test metrics, FAR/FRR threshold analysis | test_evaluation.json |
| 6. Explainability | GradCAM + ONNX export + live demo | ageguard_v1.onnx |

## Technical Decisions

| Decision | Rationale |
|---|---|
| EfficientNetV2-S | Balance accuracy (20.3M params) vs speed (17ms) |
| HuberLoss (d=5.0) | Robust to UTKFace mislabels; d aligned with MAE target |
| ImageNet normalization | Dataset within 0.05 delta (EDA validated) |
| Face crop + 40% margin | Median coverage was 0.168 |
| Alert threshold = 25 | FAR/FRR tradeoff optimization |
| ONNX export | 3x faster than 50ms target |

## Data Quality

- Manual visual review of 110 no-face + 41 duplicate pairs
- Multi-threshold analysis: 41.8% FRR at t=0.7, 53 images rescued
- Blur audit: Conservative Lap < 20, only 33 extreme cases removed
- Great Expectations: 2 validation gates (post-EDA + post-cleanup)
- pytest: 21/21 tests passing

## Tech Stack

PyTorch, EfficientNetV2-S, ONNX Runtime, OpenCV DNN, Great Expectations, Pydantic, pytest, Docker, FastAPI, Gradio, DVC

## Known Limitations

1. UTKFace mislabels - mitigated with Huber Loss
2. MAE degrades at 60+ (9.98 years) - irrelevant for compliance
3. Single dataset - production needs POS camera data
4. Face detector FRR 41.8% at t=0.7 - recommend RetinaFace
