"""
utils/preprocessing.py
=======================
Model-specific preprocessing (no single shared pipeline — each architecture
gets the transform it actually needs) and a custom flat-directory dataset
for the test set, since the images in ``test/`` are NOT organized into
per-class subfolders (so ``torchvision.datasets.ImageFolder`` cannot be
used).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils.model_loader import TEST_CSV, TEST_DIR

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transform(model_name: str) -> transforms.Compose:
    """
    Return the correct preprocessing pipeline for a given model.

      * Simple CNN         -> 64x64,  ImageNet-style normalization
      * Custom ResNet-18    -> 64x64,  ImageNet-style normalization
                               (matches the original training notebook)
      * Frozen / Fine-Tuned -> 224x224, standard ImageNet-compatible
        ResNet-18              preprocessing expected by torchvision ResNet-18
    """
    if model_name in ("Simple CNN", "Custom ResNet-18"):
        return transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    if model_name in ("Frozen ResNet-18", "Fine-Tuned ResNet-18"):
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    raise ValueError(f"Unknown model name: {model_name}")


def get_all_transforms() -> Dict[str, transforms.Compose]:
    """Convenience: one transform per model, keyed by model name."""
    from utils.model_loader import MODEL_FILES

    return {name: get_transform(name) for name in MODEL_FILES}


def preprocess_uploaded_image(image: Image.Image, model_name: str) -> torch.Tensor:
    """
    Turn a single uploaded PIL image into a model-ready batch tensor (1, C, H, W).
    A separate transformed tensor is created per model — the same resized
    tensor is never reused across models with different expected input sizes.
    """
    image = image.convert("RGB")
    transform = get_transform(model_name)
    tensor = transform(image)
    return tensor.unsqueeze(0)


class EuroSATTestDataset(Dataset):
    """
    Custom dataset for the flat ``test/`` directory + ``test.csv`` ground
    truth, since the images are not arranged into per-class folders.

    test.csv columns: img_id, label
    """

    def __init__(
        self,
        csv_path: Path = TEST_CSV,
        image_dir: Path = TEST_DIR,
        transform: Callable | None = None,
    ):
        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir)
        self.transform = transform

        if not self.csv_path.exists():
            raise FileNotFoundError(f"test.csv not found at {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)
        if "img_id" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError("test.csv must contain 'img_id' and 'label' columns.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_id = row["img_id"]
        label = int(row["label"])

        image_path = self.image_dir / img_id
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label, img_id

    def get_image_path(self, img_id: str) -> Path:
        return self.image_dir / img_id
