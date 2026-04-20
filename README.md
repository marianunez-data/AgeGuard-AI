# AgeGuard AI — Automated Age Verification System

> Deep learning-powered age estimation for regulated retail environments. Real-time facial analysis with a three-tier alert system BLOCK/VERIFY/APPROVED for compliance at point of sale.

![CI — AgeGuard AI](https://github.com/marianunez-data/AgeGuard-AI/actions/workflows/ci.yml/badge.svg) ![MAE](https://img.shields.io/badge/MAE-5.02-blue) ![API](https://img.shields.io/badge/API-live-brightgreen) ![Dashboard](https://img.shields.io/badge/Dashboard-live-brightgreen)

### [Live Demo](https://huggingface.co/spaces/marianunez-data/AgeGuard-AI) | [Dashboard](https://ageguard-ai-dashboard.streamlit.app) | [API](https://ageguard-ai.onrender.com/docs)

---

## Business Problem

Retail businesses selling age-restricted products face regulatory fines averaging $10,000 per violation when minors are not properly identified. Human verification is inconsistent, employees make errors under pressure during rush hours, shift changes, and high-volume periods. AgeGuard AI automates the first layer of age verification, alerting supervisors when a customer's predicted age falls below the safety threshold.

## System Architecture
```mermaid
flowchart LR
    A[POS Camera] --> B[Face Detection\nOpenCV DNN SSD]
    B --> C[Face Crop\n224x224 + 40% margin]
    C --> D[Age Estimation\nEfficientNetV2-S\nONNX 17ms]
    D --> E{Predicted Age}
    E -->|"&lt; 21"| F[BLOCK\nDeny sale]
    E -->|"21-24"| G[VERIFY\nRequest ID]
    E -->|"&ge; 25"| H[APPROVED\nAuto-clear]
    F --> I[Supervisor\nDashboard]
    G --> I
    H --> J[Transaction\nCompleted]
    I --> J
```

## Key Results

| Metric                           | Value      | Target     |
| -------------------------------- | ---------- | ---------- |
| Test MAE (global)                | 5.02 years | <= 5.0     |
| Test MAE (18-25 critical)        | 4.34 years | <= 5.0     |
| False Accept Rate (threshold 25) | 12.3%      | Minimized  |
| False Reject Rate (threshold 25) | 20.3%      | Acceptable |
| Inference latency                | 17 ms      | < 50 ms    |
| Adult auto-approval              | 79.7%      | Maximized  |
| Model size (ONNX)                | 77.5 MB    | Deployable |

## Deployed Services

| Platform           | URL                                                                    | Purpose                               |
| ------------------ | ---------------------------------------------------------------------- | ------------------------------------- |
| HuggingFace Spaces | [Live Demo](https://huggingface.co/spaces/marianunez-data/AgeGuard-AI) | Interactive demo: upload a face image |
| Streamlit Cloud    | [Dashboard](https://ageguard-ai-dashboard.streamlit.app)               | Business intelligence dashboard       |
| Render             | [API](https://ageguard-ai.onrender.com/docs)                           | REST API (FastAPI + Swagger docs)     |

## Project Structure
```
AgeGuard-AI/
├── artifacts/              # EDA summary, removal lists, split stats
├── configs/
│   └── base_config.yaml    # All hyperparameters (Pydantic-validated)
├── data/                   # Original + processed images (DVC tracked)
├── models/                 # Checkpoints + ONNX + face detector (DVC tracked)
├── notebooks/
│   └── AgeGuard-AI.ipynb   # Full pipeline notebook (6 phases)
├── reports/
│   ├── eda/                # 17+ EDA visualizations + GE reports
│   ├── training/           # Training curves + summary JSON
│   ├── evaluation/         # Test metrics + threshold analysis + demo
│   └── explainability/     # GradCAM heatmaps
├── src/
│   ├── config.py           # Pydantic config loader
│   ├── dataset.py          # PyTorch Dataset + augmentation transforms
│   ├── model.py            # EfficientNetV2-S regression head
│   ├── train.py            # Training loop with crash recovery
│   ├── inference.py        # ONNX production inference + alert system
│   └── api.py              # FastAPI REST endpoint
├── tests/                  # pytest (21 local tests)
├── Dockerfile              # Production container
├── app.py                  # Gradio demo (HuggingFace Spaces)
├── streamlit_app.py        # BI dashboard (Streamlit Cloud)
└── requirements.txt
```

## Quick Start

### Installation
```bash
git clone https://github.com/marianunez-data/AgeGuard-AI.git
cd AgeGuard-AI
pip install -r requirements.txt
```

Model weights are hosted on HuggingFace and downloaded automatically when running the API.
Manual download: https://huggingface.co/spaces/marianunez-data/AgeGuard-AI/tree/main/models/onnx

### Predict age from image
```python
from src.inference import AgePredictor

predictor = AgePredictor(model_path="models/onnx/ageguard_v1.onnx")
result = predictor.predict("path/to/face.jpg")
# {'predicted_age': 22, 'alert_level': 'VERIFY', 'action': 'Request ID'}
```

### API call
```bash
curl -X POST https://ageguard-ai.onrender.com/predict -F "file=@face.jpg"
```

### Run tests
```bash
python -m pytest tests/ -v
# Test suite covering config, dataset, model, and inference
```

## Pipeline Phases

| Phase             | Description                                                               | Key Output                           |
| ----------------- | ------------------------------------------------------------------------- | ------------------------------------ |
| 1. Setup          | Project structure, Pydantic config, YAML                                  | config.py, base_config.yaml          |
| 2. EDA            | 11 audits: age distribution, resolution, blur, duplicates, face detection | eda_summary.json, evaluation reports |
| 3. Preprocessing  | Visual review, face crop, grayscale conversion, stratified split          | 7,446 clean images (224x224)         |
| 4. Training       | EfficientNetV2-S, HuberLoss(d=1.0), CosineAnnealing, AMP                  | best_model.pth, Val MAE 5.09         |
| 5. Evaluation     | Test metrics, per-band MAE, FAR/FRR threshold optimization                | test_evaluation.json                 |
| 6. Explainability | GradCAM visualization + ONNX export + live demo                           | ageguard_v1.onnx, GradCAM heatmaps   |

## Design Decisions

| Component        | Choice                           | Why                                                                                         |
| ---------------- | -------------------------------- | ------------------------------------------------------------------------------------------- |
| Architecture     | EfficientNetV2-S (20.34M params) | Balance between accuracy and speed; strong transfer learning from ImageNet                  |
| Loss function    | HuberLoss (delta=1.0)            | Robust to UTKFace mislabels identified in Phase 3                                           |
| Face detector    | OpenCV DNN SSD                   | Reliable, pretrained, no additional dependencies                                            |
| Alert threshold  | 25 years                         | Validated across 8 thresholds — reduces FAR from 28.9% (at 21) to 12.3% with acceptable FRR |
| Export format    | ONNX                             | Cross-platform deployment; ~17ms inference latency                                          |
| Normalization    | ImageNet stats                   | Dataset pixel distribution within 0.05 of ImageNet means (EDA validated)                    |
| Face crop margin | 40%                              | Empirical default — adds context around the face bounding box                               |
| Blur handling    | Remove only Lap < 20             | Preserves 18-25 band; removing all 1,385 flagged loses ~24% of critical band                |

## Data Quality Measures

- Manual visual review of 110 no-face candidates + 38 duplicate files (from 41 pHash pairs)
- Cascading threshold strategy: strict detection (0.7) + visual review + permissive crop (0.3) to maximize valid samples retained
- Blur audit with conservative threshold (Lap < 20): 33 extreme cases removed
- Great Expectations: 2 validation gates (post-EDA 8/8, post-cleanup 12/12)
- Stratified split verified: 18-25 band consistent at ~21% across train/val/test
- pytest test suite covering config validation, dataset integrity, model architecture, and inference pipeline

## Known Limitations

1. **UTKFace mislabels**: dataset contains incorrect age labels. Mitigated with HuberLoss but not eliminated.
2. **Age extremes**: MAE degrades at 60+ (9.98 years). Irrelevant for compliance since ID is not requested at that age.
3. **Single dataset**: trained only on UTKFace. Production would benefit from POS camera data for domain adaptation.
4. **Face detector**: OpenCV DNN SSD has significant FRR at threshold 0.7, mitigated by cascading strategy (strict flag + visual review + permissive crop).


## Future Improvements

- [ ] Face alignment with landmark detection (dlib/MTCNN) to normalize eye-line orientation (~3-7% MAE improvement)
- [ ] Hyperparameter optimization with Optuna on validation set
- [ ] Face detection pre-check before age prediction (input validation gate in /predict endpoint)
- [ ] Fine-tune on POS camera dataset for domain adaptation
- [ ] Demographic fairness analysis across age, gender, ethnicity

## Tech Stack

**Deep Learning:** PyTorch, EfficientNetV2-S, ONNX Runtime, torchvision

**Computer Vision:** OpenCV DNN (face detection + cropping), GradCAM

**Data:** pandas, numpy, scikit-learn, Pillow, imagehash

**Validation:** Great Expectations, Pydantic, pytest

**Deployment:** FastAPI, Gradio, Streamlit, Docker, HuggingFace Spaces

**MLOps:** DVC (data versioning), GitHub Actions (CI/CD), YAML config

## Author

**Maria Camila Gonzalez Nuñez**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-marianunez--data-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/marianunez-data)
