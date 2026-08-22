"""
utils/evaluation.py
====================
Runs the four already-trained models over the same EuroSAT test set
(read from test.csv + test/) and derives every metric dynamically:
confusion matrices, per-class precision/recall/F1, wrong-prediction
analysis, and measured (never hard-coded) inference timing.

All four models are evaluated against the SAME images in the SAME
order, so confusion matrices and error analyses line up img-for-img.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

from utils.model_loader import (
    CLASS_NAMES,
    NUM_CLASSES,
    get_device,
    load_all_models,
)
from utils.preprocessing import EuroSATTestDataset, get_transform


@dataclass
class EvaluationResult:
    model_name: str
    y_true: np.ndarray
    y_pred: np.ndarray
    y_conf: np.ndarray  # confidence of the predicted class
    img_ids: List[str]
    probs: np.ndarray  # full (N, num_classes) softmax probabilities
    total_inference_seconds: float
    num_images: int

    @property
    def avg_inference_ms_per_image(self) -> float:
        if self.num_images == 0:
            return 0.0
        return (self.total_inference_seconds / self.num_images) * 1000.0

    @property
    def accuracy(self) -> float:
        if self.num_images == 0:
            return 0.0
        return float(np.mean(self.y_true == self.y_pred))


@st.cache_data(show_spinner="Running inference on the test set...")
def evaluate_model_on_test_set(model_name: str, batch_size: int = 64) -> EvaluationResult:
    """
    Evaluate a single already-trained model on the full test set.
    Cached by model_name so repeated navigation doesn't re-run inference.
    """
    models_by_name = load_all_models()
    loaded = models_by_name[model_name]
    device = get_device()

    transform = get_transform(model_name)
    dataset = EuroSATTestDataset(transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_true: List[int] = []
    all_pred: List[int] = []
    all_conf: List[float] = []
    all_probs: List[np.ndarray] = []
    all_ids: List[str] = []

    total_seconds = 0.0

    model = loaded.model
    model.eval()
    with torch.no_grad():
        for images, labels, img_ids in loader:
            images = images.to(device)

            start = time.perf_counter()
            logits = model(images)
            if device.type in ("cuda", "mps"):
                # Ensure kernels have actually completed before stopping the timer.
                if device.type == "cuda":
                    torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            total_seconds += elapsed

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            confs = probs.max(axis=1)

            all_true.extend(labels.numpy().tolist())
            all_pred.extend(preds.tolist())
            all_conf.extend(confs.tolist())
            all_probs.append(probs)
            all_ids.extend(list(img_ids))

    return EvaluationResult(
        model_name=model_name,
        y_true=np.array(all_true),
        y_pred=np.array(all_pred),
        y_conf=np.array(all_conf),
        img_ids=all_ids,
        probs=np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, NUM_CLASSES)),
        total_inference_seconds=total_seconds,
        num_images=len(all_ids),
    )


def compute_confusion_matrix(result: EvaluationResult) -> np.ndarray:
    return confusion_matrix(result.y_true, result.y_pred, labels=list(range(NUM_CLASSES)))


def compute_per_class_metrics(result: EvaluationResult) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(
        result.y_true,
        result.y_pred,
        labels=list(range(NUM_CLASSES)),
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "Class": CLASS_NAMES,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Support": support,
        }
    )


def get_wrong_predictions(result: EvaluationResult) -> pd.DataFrame:
    """Return a DataFrame of every misclassified test image, with confidence."""
    wrong_mask = result.y_true != result.y_pred
    idxs = np.where(wrong_mask)[0]

    rows = []
    for i in idxs:
        rows.append(
            {
                "img_id": result.img_ids[i],
                "true_label": int(result.y_true[i]),
                "true_class": CLASS_NAMES[int(result.y_true[i])],
                "pred_label": int(result.y_pred[i]),
                "pred_class": CLASS_NAMES[int(result.y_pred[i])],
                "confidence": float(result.y_conf[i]),
            }
        )
    return pd.DataFrame(rows)


def overall_summary(result: EvaluationResult) -> Dict[str, float]:
    return {
        "accuracy": result.accuracy,
        "num_images": result.num_images,
        "num_correct": int(np.sum(result.y_true == result.y_pred)),
        "num_wrong": int(np.sum(result.y_true != result.y_pred)),
        "avg_inference_ms_per_image": result.avg_inference_ms_per_image,
        "total_inference_seconds": result.total_inference_seconds,
    }


def time_single_image_inference(model: torch.nn.Module, tensor: torch.Tensor, device: torch.device) -> float:
    """Measure wall-clock inference time (ms) for a single preprocessed image tensor."""
    model.eval()
    tensor = tensor.to(device)
    with torch.no_grad():
        start = time.perf_counter()
        logits = model(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return elapsed_ms, probs
