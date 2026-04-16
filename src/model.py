# ══════════════════════════════════════════════════════════════════
#  src/model.py
#  EfficientNetV2-S wrapper for age regression.
#  Replaces classifier head with single-output regression head.
#
#  Design decisions:
#    - Pretrained ImageNet weights (confirmed in EDA: delta < 0.05)
#    - Dropout from config (default 0.3)
#    - Single output neuron (regression, not classification)
# ══════════════════════════════════════════════════════════════════
import torch.nn as nn
from torchvision import models


def build_model(cfg) -> nn.Module:
    """
    Builds EfficientNetV2-S with regression head.

    Args:
        cfg: AgeGuardConfig instance

    Returns:
        nn.Module ready for training
    """
    # Load pretrained backbone
    weights = models.EfficientNet_V2_S_Weights.DEFAULT if cfg.model.pretrained else None
    model = models.efficientnet_v2_s(weights=weights)

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(p=cfg.model.dropout),
        nn.Linear(128, 1),
    )

    return model
