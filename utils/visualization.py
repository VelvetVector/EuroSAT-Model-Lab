"""
utils/visualization.py
=======================
Plotly-based visualizations used throughout the app: architecture block
diagrams, parameter breakdowns, confusion matrices, per-class comparisons,
and efficiency charts. Kept dependency-light (plotly + matplotlib only).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.model_loader import CLASS_NAMES

FROZEN_COLOR = "#7f8c8d"
TRAINABLE_COLOR = "#2e86de"
RESIDUAL_COLOR = "#e67e22"
CONV_COLOR = "#2e86de"
CLASSIFIER_COLOR = "#27ae60"


# ---------------------------------------------------------------------------
# Architecture block diagrams
# ---------------------------------------------------------------------------

def _draw_stack_diagram(blocks: List[dict], title: str) -> go.Figure:
    """
    Draw a simple top-to-bottom stack of labeled blocks. Each block dict:
        {"label": str, "color": str, "sub": str (optional)}
    """
    n = len(blocks)
    box_h = 1.0
    gap = 0.35
    total_h = n * box_h + (n - 1) * gap

    fig = go.Figure()
    y = total_h

    for block in blocks:
        y0 = y - box_h
        fig.add_shape(
            type="rect",
            x0=0,
            x1=6,
            y0=y0,
            y1=y,
            line=dict(color="rgba(0,0,0,0.25)", width=1.5),
            fillcolor=block.get("color", CONV_COLOR),
            opacity=0.85,
        )
        label = block["label"]
        sub = block.get("sub", "")
        text = f"<b>{label}</b>" + (f"<br><span style='font-size:11px'>{sub}</span>" if sub else "")
        fig.add_annotation(
            x=3,
            y=(y0 + y) / 2,
            text=text,
            showarrow=False,
            font=dict(color="white", size=13),
            align="center",
        )
        if y0 > 0:
            fig.add_annotation(
                x=3,
                y=y0 - gap / 2,
                text="↓",
                showarrow=False,
                font=dict(size=16, color="rgba(0,0,0,0.5)"),
            )
        y = y0 - gap

    fig.update_xaxes(visible=False, range=[-0.5, 6.5])
    fig.update_yaxes(visible=False, range=[-0.5, total_h + 0.5])
    fig.update_layout(
        title=title,
        height=max(320, 70 * n),
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def simple_cnn_diagram() -> go.Figure:
    blocks = [
        {"label": "Input", "sub": "3 × 64 × 64", "color": "#95a5a6"},
        {"label": "Conv 3→32 + BN + ReLU + MaxPool", "color": CONV_COLOR},
        {"label": "Conv 32→64 + BN + ReLU + MaxPool", "color": CONV_COLOR},
        {"label": "Conv 64→128 + BN + ReLU + MaxPool", "color": CONV_COLOR},
        {"label": "Conv 128→256 + BN + ReLU + MaxPool", "color": CONV_COLOR},
        {"label": "AdaptiveAvgPool(1×1) + Flatten", "color": "#8e44ad"},
        {"label": "Linear 256→128 + ReLU + Dropout", "color": CLASSIFIER_COLOR},
        {"label": "Linear 128→10", "color": CLASSIFIER_COLOR},
    ]
    return _draw_stack_diagram(blocks, "Simple CNN — Architecture")


def custom_resnet_diagram() -> go.Figure:
    blocks = [
        {"label": "Input", "sub": "3 × 64 × 64", "color": "#95a5a6"},
        {"label": "Custom Stem", "sub": "Conv 3×3, 3→64 + BN + ReLU", "color": CONV_COLOR},
        {"label": "Stage 1 — 2× BasicBlock", "sub": "64 channels", "color": RESIDUAL_COLOR},
        {"label": "Stage 2 — 2× BasicBlock", "sub": "128 channels", "color": RESIDUAL_COLOR},
        {"label": "Stage 3 — 2× BasicBlock", "sub": "256 channels", "color": RESIDUAL_COLOR},
        {"label": "Stage 4 — 2× BasicBlock", "sub": "512 channels", "color": RESIDUAL_COLOR},
        {"label": "AdaptiveAvgPool(1×1) + Flatten", "color": "#8e44ad"},
        {"label": "Linear 512→10", "color": CLASSIFIER_COLOR},
    ]
    return _draw_stack_diagram(blocks, "Custom ResNet-18 — Architecture (8 BasicBlocks)")


def transfer_resnet_diagram(frozen: bool) -> go.Figure:
    backbone_color = FROZEN_COLOR if frozen else TRAINABLE_COLOR
    backbone_tag = "FROZEN" if frozen else "TRAINABLE"
    blocks = [
        {"label": "Input", "sub": "3 × 224 × 224", "color": "#95a5a6"},
        {
            "label": f"torchvision ResNet-18 Backbone — {backbone_tag}",
            "sub": "ImageNet-pretrained conv stem + 4 residual stages",
            "color": backbone_color,
        },
        {"label": "512 Features", "color": "#8e44ad"},
        {"label": "Linear 512→10 — TRAINABLE", "color": TRAINABLE_COLOR},
    ]
    title = "Frozen ResNet-18" if frozen else "Fine-Tuned ResNet-18"
    return _draw_stack_diagram(blocks, f"{title} — Architecture (same backbone as its counterpart)")


def basic_block_diagram() -> go.Figure:
    """Illustrate a single residual BasicBlock with its skip connection."""
    fig = go.Figure()

    steps = [
        ("Input", 0.5),
        ("Conv 3×3 + BN + ReLU", 1.6),
        ("Conv 3×3 + BN", 2.7),
    ]
    for label, x in steps:
        fig.add_shape(
            type="rect", x0=x - 0.45, x1=x + 0.45, y0=0.4, y1=1.0,
            fillcolor=RESIDUAL_COLOR, opacity=0.85, line=dict(color="rgba(0,0,0,0.3)"),
        )
        fig.add_annotation(x=x, y=0.7, text=f"<b>{label}</b>", showarrow=False,
                            font=dict(size=10, color="white"))

    fig.add_shape(type="line", x0=0.95, x1=1.15, y0=0.7, y1=0.7,
                  line=dict(color="black", width=2))
    fig.add_shape(type="line", x0=2.05, x1=2.25, y0=0.7, y1=0.7,
                  line=dict(color="black", width=2))

    # Skip connection arc
    fig.add_shape(
        type="path",
        path="M 0.5,1.0 C 0.5,1.7 2.7,1.7 2.7,1.0",
        line=dict(color="#c0392b", width=2, dash="dash"),
    )
    fig.add_annotation(x=1.6, y=1.55, text="identity / projection skip connection",
                        showarrow=False, font=dict(size=10, color="#c0392b"))

    fig.add_shape(type="circle", x0=3.15, x1=3.45, y0=0.55, y1=0.85,
                  fillcolor="white", line=dict(color="black"))
    fig.add_annotation(x=3.3, y=0.7, text="+", showarrow=False, font=dict(size=16))
    fig.add_annotation(x=3.9, y=0.7, text="<b>ReLU</b>", showarrow=False, font=dict(size=11))

    fig.add_shape(type="line", x0=2.7, x1=3.15, y0=0.7, y1=0.7, line=dict(color="black", width=2))
    fig.add_shape(type="line", x0=0.5, x1=0.5, y0=1.0, y1=1.0)

    fig.update_xaxes(visible=False, range=[0, 4.3])
    fig.update_yaxes(visible=False, range=[0.2, 1.9])
    fig.update_layout(
        title="BasicBlock — Residual / Skip Connection",
        height=260,
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ---------------------------------------------------------------------------
# Parameter charts
# ---------------------------------------------------------------------------

def parameter_breakdown_chart(counts: Dict[str, int]) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Bar(
                x=["Trainable", "Frozen"],
                y=[counts["trainable"], counts["frozen"]],
                marker_color=[TRAINABLE_COLOR, FROZEN_COLOR],
                text=[f"{counts['trainable']:,}", f"{counts['frozen']:,}"],
                textposition="auto",
            )
        ]
    )
    fig.update_layout(
        title=f"Parameters — Total: {counts['total']:,}",
        yaxis_title="Parameter count",
        height=350,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def parameter_comparison_chart(rows: List[dict]) -> go.Figure:
    """rows: [{"model": str, "trainable": int, "frozen": int}, ...]"""
    models = [r["model"] for r in rows]
    fig = go.Figure()
    fig.add_bar(name="Trainable", x=models, y=[r["trainable"] for r in rows], marker_color=TRAINABLE_COLOR)
    fig.add_bar(name="Frozen", x=models, y=[r["frozen"] for r in rows], marker_color=FROZEN_COLOR)
    fig.update_layout(
        barmode="stack",
        title="Trainable vs. Frozen Parameters Across Models",
        yaxis_title="Parameter count",
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


# ---------------------------------------------------------------------------
# Confusion matrix / per-class metrics
# ---------------------------------------------------------------------------

def confusion_matrix_heatmap(cm: np.ndarray, title: str) -> go.Figure:
    fig = px.imshow(
        cm,
        x=CLASS_NAMES,
        y=CLASS_NAMES,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="True", color="Count"),
        text_auto=True,
        aspect="auto",
    )
    fig.update_layout(title=title, height=550, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def per_class_metric_chart(df: pd.DataFrame, metric: str) -> go.Figure:
    fig = px.bar(
        df,
        x="Class",
        y=metric,
        color=metric,
        color_continuous_scale="Blues",
        range_y=[0, 1],
    )
    fig.update_layout(title=f"Per-Class {metric}", height=400, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def multi_model_metric_chart(combined_df: pd.DataFrame, metric: str) -> go.Figure:
    """combined_df has columns: Class, Model, and the metric column."""
    fig = px.bar(
        combined_df,
        x="Class",
        y=metric,
        color="Model",
        barmode="group",
        range_y=[0, 1],
    )
    fig.update_layout(title=f"Per-Class {metric} — Model Comparison", height=450,
                       margin=dict(l=10, r=10, t=50, b=10))
    return fig


# ---------------------------------------------------------------------------
# Efficiency / comparison charts
# ---------------------------------------------------------------------------

def efficiency_scatter(rows: List[dict]) -> go.Figure:
    """rows: [{"model": str, "accuracy": float, "inference_ms": float, "params": int}, ...]"""
    df = pd.DataFrame(rows)
    fig = px.scatter(
        df,
        x="inference_ms",
        y="accuracy",
        size="params",
        color="model",
        text="model",
        labels={"inference_ms": "Avg inference time (ms/image)", "accuracy": "Test accuracy"},
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(title="Accuracy vs. Inference Time (bubble size = parameter count)",
                       height=450, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def accuracy_bar_chart(rows: List[dict]) -> go.Figure:
    """rows: [{"model": str, "accuracy": float}, ...]"""
    df = pd.DataFrame(rows)
    fig = px.bar(df, x="model", y="accuracy", color="model", range_y=[0, 1], text="accuracy")
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(title="Test Accuracy by Model", height=400, margin=dict(l=10, r=10, t=50, b=10),
                       showlegend=False)
    return fig


def learning_journey_diagram() -> go.Figure:
    steps = [
        {"label": "STEP 1 — Simple CNN", "sub": "Learn convolutional feature extraction", "color": CONV_COLOR},
        {"label": "STEP 2 — Custom ResNet-18", "sub": "Learn residual connections", "color": RESIDUAL_COLOR},
        {"label": "STEP 3 — Frozen ResNet-18", "sub": "Reuse pretrained visual representations", "color": FROZEN_COLOR},
        {"label": "STEP 4 — Fine-Tuned ResNet-18", "sub": "Adapt the pretrained network to EuroSAT", "color": TRAINABLE_COLOR},
    ]
    return _draw_stack_diagram(steps, "The Learning Journey")
