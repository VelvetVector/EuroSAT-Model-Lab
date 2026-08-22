"""
utils/model_loader.py
======================
Central place for:

  * Resolving all project paths relative to the repository root (so the
    app works identically locally and on Streamlit Community Cloud).
  * Reconstructing the correct architecture for each of the four models
    and loading its already-trained ``.pth`` weights.
  * Robust checkpoint-format handling (raw state_dict, ``state_dict`` key,
    or ``model_state_dict`` key).
  * Device selection (CUDA -> MPS -> CPU) that never *requires* an
    accelerator.
  * Dynamic parameter introspection (total / trainable / frozen counts)
    via ``named_parameters()`` — nothing is hard-coded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import streamlit as st
import torch
import torch.nn as nn

from models.cnn import build_model as build_simple_cnn
from models.custom_resnet import build_model as build_custom_resnet
from models.transfer_resnet import build_frozen_model, build_fine_tuned_model, is_backbone_frozen

# ---------------------------------------------------------------------------
# Paths — all resolved relative to this repository, never hard-coded/absolute.
# ---------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = APP_DIR / "models"
TEST_DIR = APP_DIR / "test"
TRAIN_DIR = APP_DIR / "train"
TEST_CSV = APP_DIR / "test.csv"

CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]
NUM_CLASSES = len(CLASS_NAMES)

# Which checkpoint file backs which model, and how to build/describe it.
MODEL_FILES: Dict[str, str] = {
    "Simple CNN": "model1.pth",
    "Custom ResNet-18": "model2.pth",
    "Frozen ResNet-18": "model3.pth",
    "Fine-Tuned ResNet-18": "model4.pth",
}

MODEL_LABELS: Dict[str, str] = {
    "Simple CNN": "Simple CNN",
    "Custom ResNet-18": "Custom ResNet-18 — Trained From Scratch",
    "Frozen ResNet-18": "Frozen ResNet-18 — Transfer Learning",
    "Fine-Tuned ResNet-18": "Fine-Tuned ResNet-18 — Transfer Learning",
}

MODEL_INPUT_SIZE: Dict[str, int] = {
    "Simple CNN": 64,
    "Custom ResNet-18": 64,
    "Frozen ResNet-18": 224,
    "Fine-Tuned ResNet-18": 224,
}

MODEL_DESCRIPTIONS: Dict[str, str] = {
    "Simple CNN": (
        "A four-stage convolutional network trained from scratch — the "
        "starting point of the learning journey. No residual connections."
    ),
    "Custom ResNet-18": (
        "A hand-implemented ResNet-18 (8 BasicBlocks, custom 3x3 stem) "
        "trained from scratch. Introduces residual/skip connections."
    ),
    "Frozen ResNet-18": (
        "torchvision ResNet-18 pretrained on ImageNet. The backbone is "
        "frozen; only a new 512->10 classifier head is trained."
    ),
    "Fine-Tuned ResNet-18": (
        "The same torchvision ResNet-18 architecture as the frozen model, "
        "but the entire backbone is fine-tuned on EuroSAT alongside the classifier."
    ),
}


class CheckpointLoadError(RuntimeError):
    """Raised when a checkpoint cannot be loaded into its architecture."""

    def __init__(self, model_name: str, checkpoint_path: Path, problem: str):
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.problem = problem
        message = (
            f"Failed to load checkpoint for '{model_name}'.\n"
            f"  Checkpoint path: {checkpoint_path}\n"
            f"  Problem: {problem}"
        )
        super().__init__(message)


@dataclass
class LoadedModel:
    name: str
    label: str
    model: nn.Module
    input_size: int
    description: str
    checkpoint_path: Path


def get_device() -> torch.device:
    """Select the best available device without ever requiring one."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_architecture(model_name: str) -> nn.Module:
    if model_name == "Simple CNN":
        return build_simple_cnn(num_classes=NUM_CLASSES)
    if model_name == "Custom ResNet-18":
        return build_custom_resnet(num_classes=NUM_CLASSES)
    if model_name == "Frozen ResNet-18":
        return build_frozen_model(num_classes=NUM_CLASSES)
    if model_name == "Fine-Tuned ResNet-18":
        return build_fine_tuned_model(num_classes=NUM_CLASSES)
    raise ValueError(f"Unknown model name: {model_name}")


def _extract_state_dict(checkpoint) -> Optional[dict]:
    """Handle common checkpoint container formats without altering the architecture."""
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            return checkpoint["state_dict"]
        if "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
            return checkpoint["model_state_dict"]
        # Heuristic: if the dict's values look like tensors, treat it as a raw state_dict.
        if checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint
        return checkpoint  # fall through, let load_state_dict raise a clear error
    return None


def load_single_model(model_name: str, device: Optional[torch.device] = None) -> LoadedModel:
    """
    Reconstruct the architecture for ``model_name``, load its ``.pth`` weights,
    and return it in eval mode on the given device.

    Raises CheckpointLoadError with a clear model name / path / problem if the
    checkpoint cannot be loaded. Never silently falls back to random weights.
    """
    if device is None:
        device = get_device()

    filename = MODEL_FILES.get(model_name)
    if filename is None:
        raise CheckpointLoadError(model_name, MODEL_DIR, f"No checkpoint mapping for '{model_name}'.")

    checkpoint_path = MODEL_DIR / filename

    if not checkpoint_path.exists():
        raise CheckpointLoadError(
            model_name,
            checkpoint_path,
            "Checkpoint file not found. Place the .pth file directly inside models/.",
        )

    architecture = _build_architecture(model_name)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception:
        # weights_only=True can reject some legitimate but non-pure-tensor
        # checkpoints (e.g. dicts containing ints/strings alongside tensors).
        # Fall back to a normal load — this project only ever loads
        # first-party checkpoints the user trained themselves.
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except Exception as exc:  # pragma: no cover - surfaced to the UI
            raise CheckpointLoadError(model_name, checkpoint_path, f"torch.load failed: {exc}") from exc

    state_dict = _extract_state_dict(checkpoint)
    if not isinstance(state_dict, dict):
        raise CheckpointLoadError(
            model_name,
            checkpoint_path,
            "Checkpoint did not contain a recognizable state_dict "
            "(expected a raw state_dict, or a dict with a 'state_dict' / "
            "'model_state_dict' key).",
        )

    try:
        missing, unexpected = architecture.load_state_dict(state_dict, strict=False)
    except Exception as exc:
        raise CheckpointLoadError(model_name, checkpoint_path, f"load_state_dict failed: {exc}") from exc

    if missing or unexpected:
        problem = (
            f"load_state_dict completed with mismatches — "
            f"missing keys: {list(missing)[:5]}{'...' if len(missing) > 5 else ''}, "
            f"unexpected keys: {list(unexpected)[:5]}{'...' if len(unexpected) > 5 else ''}"
        )
        raise CheckpointLoadError(model_name, checkpoint_path, problem)

    architecture.to(device)
    architecture.eval()

    return LoadedModel(
        name=model_name,
        label=MODEL_LABELS[model_name],
        model=architecture,
        input_size=MODEL_INPUT_SIZE[model_name],
        description=MODEL_DESCRIPTIONS[model_name],
        checkpoint_path=checkpoint_path,
    )


@st.cache_resource(show_spinner="Loading trained models...")
def load_all_models() -> Dict[str, LoadedModel]:
    """
    Load all four models once and cache them as Streamlit resources so they
    are not reloaded on every interaction/rerun.
    """
    device = get_device()
    loaded: Dict[str, LoadedModel] = {}
    errors = []
    for model_name in MODEL_FILES:
        try:
            loaded[model_name] = load_single_model(model_name, device=device)
        except CheckpointLoadError as exc:
            errors.append(str(exc))
    if errors:
        # Surface every loading problem at once rather than failing on the first.
        raise CheckpointLoadError(
            "one or more models",
            MODEL_DIR,
            "\n\n".join(errors),
        )
    return loaded


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Dynamically compute total / trainable / frozen parameter counts."""
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    frozen = total - trainable
    return {"total": total, "trainable": trainable, "frozen": frozen}


def count_layer_types(model: nn.Module) -> Dict[str, int]:
    """Count Conv2d / BatchNorm2d / Linear layers etc. via named_modules()."""
    counts: Dict[str, int] = {}
    for _, module in model.named_modules():
        cls_name = module.__class__.__name__
        if cls_name in ("Conv2d", "BatchNorm2d", "Linear", "MaxPool2d", "ReLU", "Dropout"):
            counts[cls_name] = counts.get(cls_name, 0) + 1
    return counts


def count_residual_blocks(model: nn.Module) -> int:
    """Count BasicBlock instances (0 for non-residual architectures)."""
    count = 0
    for _, module in model.named_modules():
        if module.__class__.__name__ == "BasicBlock":
            count += 1
    return count


def model_size_mb(model: nn.Module) -> float:
    """Approximate on-disk / in-memory size of a model's parameters, in MB."""
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    total_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
    return total_bytes / (1024 ** 2)


def backbone_frozen_status(model_name: str, model: nn.Module) -> Optional[bool]:
    """Return True/False for transfer-learning models, None for from-scratch models."""
    if model_name in ("Frozen ResNet-18", "Fine-Tuned ResNet-18"):
        return is_backbone_frozen(model)
    return None
