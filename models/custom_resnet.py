"""
Models 3 & 4 — Transfer-Learning ResNet-18 (Frozen / Fine-Tuned)
==================================================================
Both models use the exact same architecture:

    torchvision.models.resnet18

with its classification head replaced by ``Linear(512 -> 10)``.

They are NOT different architectures. The only difference between
Model 3 ("Frozen ResNet-18") and Model 4 ("Fine-Tuned ResNet-18") is
which parameters were updated during training:

    Model 3 — Frozen ResNet-18 — Transfer Learning
        backbone:   FROZEN     (requires_grad = False)
        classifier: TRAINABLE  (requires_grad = True)

    Model 4 — Fine-Tuned ResNet-18 — Transfer Learning
        backbone:   TRAINABLE  (requires_grad = True)
        classifier: TRAINABLE  (requires_grad = True)

Source notebook for both: Claude Sol.ipynb
"""

from typing import Literal

import torch
import torch.nn as nn
from torchvision.models import resnet18


TrainingMode = Literal["frozen", "fine_tuned"]


def build_transfer_resnet18(
    num_classes: int = 10,
    mode: TrainingMode = "frozen",
    pretrained: bool = True,
) -> nn.Module:
    """
    Build a torchvision ResNet-18 with a replaced classifier head, and set
    ``requires_grad`` on the backbone according to ``mode``.

    Note: when loading already-trained checkpoints for inference, ``pretrained``
    should be False (or irrelevant) since ``load_state_dict`` will overwrite the
    weights anyway. It exists mainly for architectural symmetry / potential
    from-scratch reconstruction.
    """
    weights = "IMAGENET1K_V1" if pretrained else None
    model = resnet18(weights=weights)

    # Replace the ImageNet 1000-way classifier with a 10-way EuroSAT classifier.
    in_features = model.fc.in_features  # 512 for ResNet-18
    model.fc = nn.Linear(in_features, num_classes)

    set_backbone_trainable(model, trainable=(mode == "fine_tuned"))

    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """
    Freeze or unfreeze every parameter EXCEPT the final classifier (``fc``).
    The classifier (``fc``) is always left trainable.
    """
    for name, param in model.named_parameters():
        if name.startswith("fc."):
            param.requires_grad = True
        else:
            param.requires_grad = trainable


def build_frozen_model(num_classes: int = 10) -> nn.Module:
    """Factory for Model 3 — Frozen ResNet-18 — Transfer Learning."""
    return build_transfer_resnet18(num_classes=num_classes, mode="frozen", pretrained=False)


def build_fine_tuned_model(num_classes: int = 10) -> nn.Module:
    """Factory for Model 4 — Fine-Tuned ResNet-18 — Transfer Learning."""
    return build_transfer_resnet18(num_classes=num_classes, mode="fine_tuned", pretrained=False)


def is_backbone_frozen(model: nn.Module) -> bool:
    """Inspect a loaded model's parameters to determine if the backbone is frozen."""
    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            if param.requires_grad:
                return False
    return True
