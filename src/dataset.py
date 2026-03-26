# ══════════════════════════════════════════════════════════════════
#  src/dataset.py
#  PyTorch Dataset & DataLoader factory for AgeGuard AI.
#  Reads labels_split.csv, applies transforms, returns
#  (image_tensor, age_float) pairs.
#
#  Design decisions (derived from EDA):
#    - ImageNet normalization (EDA 2.3: delta < 0.05)
#    - 224×224 input (EfficientNetV2-S requirement)
#    - Augmentation on-the-fly for train only (no data leakage)
#    - Grayscale already converted to RGB in Phase 3
# ══════════════════════════════════════════════════════════════════
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ── Constants ─────────────────────────────────────────────────────
# Confirmed in EDA 2.3: dataset mean is within 0.05 of ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ── Transforms ────────────────────────────────────────────────────
def get_transforms(split: str, img_size: int = 224) -> transforms.Compose:
    """
    Train: augmentation + normalize.
    Val/Test: resize + normalize only.

    Augmentation strategy:
      - HorizontalFlip: people face both directions in production
      - Rotation ±15°: cameras capture faces at slight angles
      - ColorJitter: lighting varies across store locations
      - RandomAffine: minor position shifts for robustness
      - RandomResizedCrop: scale variation in real-world footage
      - RandomGrayscale 2%: matches 0.76% grayscale found in dataset
    """
    if split == "train":
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
                ),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
                transforms.RandomGrayscale(p=0.02),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )


# ── Dataset ───────────────────────────────────────────────────────
class AgeDataset(Dataset):
    """
    PyTorch Dataset for age estimation.

    Args:
        split_csv     : path to labels_split.csv
        split         : 'train' | 'val' | 'test'
        img_size      : model input size (default 224)
        processed_dir : path to data/processed/
    """

    def __init__(
        self,
        split_csv: Path,
        split: str,
        img_size: int = 224,
        processed_dir: Path = None,
    ):
        df = pd.read_csv(split_csv)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.split = split
        self.transform = get_transforms(split, img_size)
        self.proc_dir = processed_dir

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # Resolve image path
        if self.proc_dir is not None:
            img_path = self.proc_dir / row["file_name"]
        else:
            img_path = Path(row["img_path"])

        # Load + ensure RGB (handles edge cases)
        img = Image.open(img_path).convert("RGB")

        # Apply transforms
        tensor = self.transform(img)

        # Age as float32 scalar (regression target)
        age = torch.tensor(float(row["real_age"]), dtype=torch.float32)

        return tensor, age


# ── DataLoader factory ────────────────────────────────────────────
def get_dataloaders(cfg) -> dict:
    """
    Creates train/val/test DataLoaders from AgeGuardConfig.

    Returns:
        dict with keys 'train', 'val', 'test' → DataLoader
    """
    split_csv = cfg.base_dir / "data" / "labels_split.csv"
    loaders = {}

    for split in ["train", "val", "test"]:
        dataset = AgeDataset(
            split_csv=split_csv,
            split=split,
            img_size=cfg.model.img_size,
            processed_dir=cfg.processed_dir,
        )

        loaders[split] = DataLoader(
            dataset,
            batch_size=cfg.training.batch_size,
            shuffle=(split == "train"),
            num_workers=cfg.training.num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
        )

        print(
            f"  {split:5s}: {len(dataset):5d} samples "
            f"| {len(loaders[split]):4d} batches "
            f"| batch_size={cfg.training.batch_size}"
        )

    return loaders
