# ══════════════════════════════════════════════════════════════════
#  tests/test_dataset.py
#  Tests for dataset loading and transforms.
# ══════════════════════════════════════════════════════════════════
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from config import load_config
from dataset import AgeDataset

YAML_PATH = Path(__file__).parent.parent / "configs" / "base_config.yaml"
CFG = load_config(YAML_PATH)


def test_train_dataset_loads():
    """Train dataset loads without errors."""
    ds = AgeDataset(
        split_csv=CFG.base_dir / "data" / "labels_split.csv",
        split="train",
        img_size=224,
        processed_dir=CFG.processed_dir,
    )
    assert len(ds) > 0


def test_dataset_returns_correct_shapes():
    """Dataset returns (3, 224, 224) tensor and scalar age."""
    ds = AgeDataset(
        split_csv=CFG.base_dir / "data" / "labels_split.csv",
        split="test",
        img_size=224,
        processed_dir=CFG.processed_dir,
    )
    img, age = ds[0]
    assert img.shape == torch.Size([3, 224, 224])
    assert age.dtype == torch.float32
    assert 0 <= age.item() <= 100


def test_splits_are_exclusive():
    """No file appears in more than one split."""
    import pandas as pd

    df = pd.read_csv(CFG.base_dir / "data" / "labels_split.csv")
    train = set(df[df["split"] == "train"]["file_name"])
    val = set(df[df["split"] == "val"]["file_name"])
    test = set(df[df["split"] == "test"]["file_name"])
    assert len(train & val) == 0, "Leakage: train ∩ val"
    assert len(train & test) == 0, "Leakage: train ∩ test"
    assert len(val & test) == 0, "Leakage: val ∩ test"


def test_train_transforms_augment():
    """Train transforms produce different outputs for same image."""
    ds = AgeDataset(
        split_csv=CFG.base_dir / "data" / "labels_split.csv",
        split="train",
        img_size=224,
        processed_dir=CFG.processed_dir,
    )
    img1, _ = ds[0]
    img2, _ = ds[0]
    # Random augmentation should produce different tensors
    assert not torch.equal(img1, img2)


def test_val_transforms_deterministic():
    """Val transforms produce identical outputs for same image."""
    ds = AgeDataset(
        split_csv=CFG.base_dir / "data" / "labels_split.csv",
        split="val",
        img_size=224,
        processed_dir=CFG.processed_dir,
    )
    img1, _ = ds[0]
    img2, _ = ds[0]
    assert torch.equal(img1, img2)
