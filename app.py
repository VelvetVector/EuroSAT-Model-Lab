"""
EuroSAT Model Lab — From CNNs to ResNet and Transfer Learning
================================================================
An inference + architecture + analysis + visualization laboratory for
four already-trained EuroSAT classifiers. This app does NOT train or
retrain anything — it loads finished .pth checkpoints and demonstrates
the progression from a from-scratch CNN to full transfer-learning
fine-tuning.

Run locally:    streamlit run app.py
Deploy:         Streamlit Community Cloud (see README.md)
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from utils.model_loader import (
    APP_DIR,
    CLASS_NAMES,
    MODEL_FILES,
    TEST_CSV,
    TEST_DIR,
    CheckpointLoadError,
    backbone_frozen_status,
    count_layer_types,
    count_parameters,
    count_residual_blocks,
    get_device,
    load_all_models,
    model_size_mb,
)
from utils.preprocessing import preprocess_uploaded_image
from utils.evaluation import (
    compute_confusion_matrix,
    compute_per_class_metrics,
    evaluate_model_on_test_set,
    get_wrong_predictions,
    overall_summary,
    time_single_image_inference,
)
from utils import visualization as viz

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="EuroSAT Model Lab",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = [
    "1. Overview",
    "2. Architecture Explorer",
    "3. Model Parameters",
    "4. Training Progress",
    "5. Model Comparison",
    "6. Confusion Matrix",
    "7. Per-Class Analysis",
    "8. Error Analysis",
    "9. Image Prediction Lab",
    "10. Model Efficiency",
    "11. Learning Journey",
]

MODEL_ORDER = ["Simple CNN", "Custom ResNet-18", "Frozen ResNet-18", "Fine-Tuned ResNet-18"]


# ---------------------------------------------------------------------------
# Model loading (cached) — surfaced clearly if checkpoints are missing/broken
# ---------------------------------------------------------------------------

def _try_load_models():
    try:
        models = load_all_models()
        return models, None
    except CheckpointLoadError as exc:
        return None, str(exc)
    except Exception as exc:  # pragma: no cover
        return None, f"Unexpected error while loading models: {exc}"


def _test_assets_present() -> bool:
    if not (TEST_CSV.exists() and TEST_DIR.exists()):
        return False
    return any(TEST_DIR.iterdir())


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("🛰️ EuroSAT Model Lab")
st.sidebar.caption("From CNNs to ResNet and Transfer Learning")
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")

st.sidebar.divider()
st.sidebar.caption(f"Device: `{get_device()}`")
st.sidebar.caption(f"App directory:\n`{APP_DIR}`")

models_dict, load_error = _try_load_models()

if load_error:
    st.sidebar.error("⚠️ One or more model checkpoints failed to load.")


# ---------------------------------------------------------------------------
# Shared helper widgets
# ---------------------------------------------------------------------------

def missing_models_banner():
    st.error(
        "**Model checkpoints could not be loaded.**\n\n"
        "This dashboard expects the four already-trained checkpoint files "
        "directly inside `models/`:\n\n"
        "- `models/model1.pth` → Simple CNN\n"
        "- `models/model2.pth` → Custom ResNet-18\n"
        "- `models/model3.pth` → Frozen ResNet-18\n"
        "- `models/model4.pth` → Fine-Tuned ResNet-18\n\n"
        f"Details:\n```\n{load_error}\n```"
    )


def missing_test_data_banner():
    st.warning(
        "**Test data not found.** This section evaluates all four models on "
        "the shared test set, which requires:\n\n"
        f"- `{TEST_CSV.relative_to(APP_DIR)}` (columns: `img_id,label`)\n"
        f"- `{TEST_DIR.relative_to(APP_DIR)}/` containing the referenced images\n\n"
        "Add these files to the project root and refresh."
    )


def model_select_widget(key: str, default_index: int = 0) -> str:
    return st.selectbox("Choose a model", MODEL_ORDER, index=default_index, key=key)


# ===========================================================================
# 1. OVERVIEW
# ===========================================================================

def page_overview():
    st.title("🛰️ EuroSAT Model Lab")
    st.subheader("From CNNs to ResNet and Transfer Learning")
    st.markdown(
        "An interactive laboratory demonstrating how an image-classification "
        "system evolves — from a convolutional network trained from scratch, "
        "through a hand-built residual network, to a pretrained backbone that "
        "is first frozen and then fully fine-tuned. All four models below are "
        "**already trained** — this app performs inference, architecture "
        "inspection, and analysis, not training."
    )

    if load_error:
        missing_models_banner()
        return

    cols = st.columns(4)
    for col, model_name in zip(cols, MODEL_ORDER):
        loaded = models_dict[model_name]
        counts = count_parameters(loaded.model)
        frozen_status = backbone_frozen_status(model_name, loaded.model)

        with col:
            st.markdown(f"#### {loaded.label}")
            st.caption(loaded.description)
            st.metric("Total parameters", f"{counts['total']:,}")
            st.metric("Trainable parameters", f"{counts['trainable']:,}")
            st.caption(f"Input size: {loaded.input_size}×{loaded.input_size}")
            if frozen_status is None:
                st.caption("Pretrained: No · Trained from scratch")
            else:
                st.caption(f"Pretrained: Yes · Backbone frozen: {frozen_status}")

    st.divider()
    st.plotly_chart(viz.learning_journey_diagram(), use_container_width=True)


# ===========================================================================
# 2. ARCHITECTURE EXPLORER  (incl. layer inspector)
# ===========================================================================

def page_architecture_explorer():
    st.title("Architecture Explorer")
    if load_error:
        missing_models_banner()
        return

    model_name = model_select_widget("arch_explorer_model")
    loaded = models_dict[model_name]
    model = loaded.model

    st.markdown(f"### {loaded.label}")
    st.caption(loaded.description)

    diagram_col, info_col = st.columns([2, 1])
    with diagram_col:
        if model_name == "Simple CNN":
            st.plotly_chart(viz.simple_cnn_diagram(), use_container_width=True)
        elif model_name == "Custom ResNet-18":
            st.plotly_chart(viz.custom_resnet_diagram(), use_container_width=True)
            st.plotly_chart(viz.basic_block_diagram(), use_container_width=True)
        else:
            frozen = backbone_frozen_status(model_name, model)
            st.plotly_chart(viz.transfer_resnet_diagram(frozen=bool(frozen)), use_container_width=True)

    with info_col:
        counts = count_parameters(model)
        st.metric("Total parameters", f"{counts['total']:,}")
        st.metric("Trainable", f"{counts['trainable']:,}")
        st.metric("Frozen", f"{counts['frozen']:,}")
        st.metric("Model size", f"{model_size_mb(model):.2f} MB")

        frozen_status = backbone_frozen_status(model_name, model)
        if frozen_status is not None:
            tag = "🔒 FROZEN" if frozen_status else "🔓 TRAINABLE"
            st.info(f"Backbone: **{tag}**")

        residual_blocks = count_residual_blocks(model)
        if residual_blocks:
            st.caption(f"Residual (BasicBlock) count: {residual_blocks}")

    st.divider()
    st.markdown("#### Layer Inspector")
    st.caption("Derived live from `named_modules()` / `named_parameters()` — nothing hard-coded.")

    rows = []
    for name, module in model.named_modules():
        cls_name = module.__class__.__name__
        if cls_name in ("Conv2d", "BatchNorm2d", "Linear", "MaxPool2d", "AdaptiveAvgPool2d", "Dropout"):
            params = sum(p.numel() for p in module.parameters())
            trainable = any(p.requires_grad for p in module.parameters()) if params > 0 else None
            row = {
                "Layer": name or cls_name,
                "Layer Type": cls_name,
                "Parameters": params,
                "Trainable?": trainable,
            }
            if cls_name == "Conv2d":
                row["Kernel Size"] = str(module.kernel_size)
                row["Stride"] = str(module.stride)
                row["Padding"] = str(module.padding)
                row["In→Out Channels"] = f"{module.in_channels}→{module.out_channels}"
            elif cls_name == "Linear":
                row["In→Out Features"] = f"{module.in_features}→{module.out_features}"
            rows.append(row)

    layer_df = pd.DataFrame(rows)
    st.dataframe(layer_df, use_container_width=True, height=420)


# ===========================================================================
# 3. MODEL PARAMETERS
# ===========================================================================

def page_model_parameters():
    st.title("Model Parameters")
    if load_error:
        missing_models_banner()
        return

    model_name = model_select_widget("params_model")
    loaded = models_dict[model_name]
    counts = count_parameters(loaded.model)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", f"{counts['total']:,}")
    c2.metric("Trainable", f"{counts['trainable']:,}")
    c3.metric("Frozen", f"{counts['frozen']:,}")
    trainable_pct = (counts["trainable"] / counts["total"] * 100) if counts["total"] else 0.0
    c4.metric("Trainable %", f"{trainable_pct:.1f}%")

    st.plotly_chart(viz.parameter_breakdown_chart(counts), use_container_width=True)

    st.markdown("#### Parameters by top-level component")
    component_rows = []
    for name, module in loaded.model.named_children():
        n_params = sum(p.numel() for p in module.parameters())
        n_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        if n_params > 0:
            component_rows.append(
                {"Component": name, "Total": n_params, "Trainable": n_trainable, "Frozen": n_params - n_trainable}
            )
    if component_rows:
        st.dataframe(pd.DataFrame(component_rows), use_container_width=True)

    st.markdown("#### Layer type counts")
    st.json(count_layer_types(loaded.model))


# ===========================================================================
# 4. TRAINING PROGRESS
# ===========================================================================

def page_training_progress():
    st.title("Training Progress")
    st.info(
        "Training history unavailable from supplied inputs.\n\n"
        "This project loads four already-trained checkpoints; no training "
        "logs, loss curves, or learning-rate schedules were supplied "
        "alongside them, so none are fabricated here."
    )
    st.markdown(
        "If you'd like this page to show real training/validation loss and "
        "accuracy curves, supply a training-history file (e.g. a CSV with "
        "per-epoch metrics) for each model and this section can be extended "
        "to plot it directly — nothing will be estimated or invented."
    )


# ===========================================================================
# 5. MODEL COMPARISON
# ===========================================================================

def page_model_comparison():
    st.title("Model Comparison")
    if load_error:
        missing_models_banner()
        return

    rows = []
    for model_name in MODEL_ORDER:
        loaded = models_dict[model_name]
        model = loaded.model
        counts = count_parameters(model)
        layer_counts = count_layer_types(model)
        frozen_status = backbone_frozen_status(model_name, model)
        rows.append(
            {
                "Model": loaded.label,
                "Architecture": model.__class__.__name__,
                "Input Size": f"{loaded.input_size}×{loaded.input_size}",
                "Total Parameters": counts["total"],
                "Trainable Parameters": counts["trainable"],
                "Frozen Parameters": counts["frozen"],
                "Conv Layers": layer_counts.get("Conv2d", 0),
                "BatchNorm Layers": layer_counts.get("BatchNorm2d", 0),
                "Residual Blocks": count_residual_blocks(model),
                "Classifier": "Linear",
                "Pretrained?": "No" if frozen_status is None else "Yes",
                "Backbone Frozen?": "N/A" if frozen_status is None else str(frozen_status),
            }
        )

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, height=200)

    st.divider()
    param_rows = [
        {"model": r["Model"], "trainable": r["Trainable Parameters"], "frozen": r["Frozen Parameters"]}
        for r in rows
    ]
    st.plotly_chart(viz.parameter_comparison_chart(param_rows), use_container_width=True)


# ===========================================================================
# 6. CONFUSION MATRIX
# ===========================================================================

def page_confusion_matrix():
    st.title("Confusion Matrix")
    if load_error:
        missing_models_banner()
        return
    if not _test_assets_present():
        missing_test_data_banner()
        return

    model_name = model_select_widget("cm_model")
    result = evaluate_model_on_test_set(model_name)
    cm = compute_confusion_matrix(result)

    summary = overall_summary(result)
    c1, c2, c3 = st.columns(3)
    c1.metric("Test images evaluated", summary["num_images"])
    c2.metric("Accuracy", f"{summary['accuracy']:.3f}")
    c3.metric("Avg inference", f"{summary['avg_inference_ms_per_image']:.2f} ms/img")

    st.plotly_chart(
        viz.confusion_matrix_heatmap(cm, f"Confusion Matrix — {model_name}"),
        use_container_width=True,
    )

    with st.expander("Compare against another model"):
        other_model = st.selectbox(
            "Second model", [m for m in MODEL_ORDER if m != model_name], key="cm_other_model"
        )
        other_result = evaluate_model_on_test_set(other_model)
        other_cm = compute_confusion_matrix(other_result)
        st.plotly_chart(
            viz.confusion_matrix_heatmap(other_cm, f"Confusion Matrix — {other_model}"),
            use_container_width=True,
        )


# ===========================================================================
# 7. PER-CLASS ANALYSIS
# ===========================================================================

def page_per_class_analysis():
    st.title("Per-Class Analysis")
    if load_error:
        missing_models_banner()
        return
    if not _test_assets_present():
        missing_test_data_banner()
        return

    metric = st.radio("Metric", ["Precision", "Recall", "F1"], horizontal=True)

    combined_rows = []
    single_model_tabs = st.tabs(MODEL_ORDER)
    for tab, model_name in zip(single_model_tabs, MODEL_ORDER):
        result = evaluate_model_on_test_set(model_name)
        per_class = compute_per_class_metrics(result)
        with tab:
            st.dataframe(per_class, use_container_width=True)
            st.plotly_chart(viz.per_class_metric_chart(per_class, metric), use_container_width=True)
        per_class = per_class.copy()
        per_class["Model"] = model_name
        combined_rows.append(per_class)

    st.divider()
    st.markdown("#### Cross-model comparison")
    combined_df = pd.concat(combined_rows, ignore_index=True)
    st.plotly_chart(viz.multi_model_metric_chart(combined_df, metric), use_container_width=True)


# ===========================================================================
# 8. ERROR ANALYSIS
# ===========================================================================

def page_error_analysis():
    st.title("Error Analysis — Wrong Predictions")
    if load_error:
        missing_models_banner()
        return
    if not _test_assets_present():
        missing_test_data_banner()
        return

    model_name = model_select_widget("err_model")
    result = evaluate_model_on_test_set(model_name)
    wrong_df = get_wrong_predictions(result)

    if wrong_df.empty:
        st.success("No misclassified test images for this model.")
        return

    c1, c2 = st.columns(2)
    with c1:
        sort_by = st.selectbox(
            "Sort by",
            ["Highest-confidence wrong prediction", "Lowest-confidence prediction", "Class"],
        )
    with c2:
        max_shown = st.slider("Number of examples to display", 4, 40, 12, step=4)

    if sort_by == "Highest-confidence wrong prediction":
        wrong_df = wrong_df.sort_values("confidence", ascending=False)
    elif sort_by == "Lowest-confidence prediction":
        wrong_df = wrong_df.sort_values("confidence", ascending=True)
    else:
        wrong_df = wrong_df.sort_values("true_class")

    st.caption(f"{len(wrong_df)} misclassified out of {result.num_images} test images "
               f"({len(wrong_df) / result.num_images:.1%} error rate)")

    display_df = wrong_df.head(max_shown)
    grid_cols = st.columns(4)
    for i, (_, row) in enumerate(display_df.iterrows()):
        col = grid_cols[i % 4]
        image_path = TEST_DIR / row["img_id"]
        with col:
            if image_path.exists():
                st.image(str(image_path), use_container_width=True)
            st.caption(
                f"**{row['img_id']}**\n\n"
                f"True: `{row['true_class']}`\n\n"
                f"Pred: `{row['pred_class']}` ({row['confidence']:.1%})"
            )

    st.divider()
    st.markdown("#### Compare this image across all four models")
    if len(display_df) > 0:
        chosen_id = st.selectbox("Choose an image ID", display_df["img_id"].tolist())
        cross_rows = []
        for other_name in MODEL_ORDER:
            other_result = evaluate_model_on_test_set(other_name)
            if chosen_id in other_result.img_ids:
                idx = other_result.img_ids.index(chosen_id)
                cross_rows.append(
                    {
                        "Model": other_name,
                        "True class": CLASS_NAMES[int(other_result.y_true[idx])],
                        "Predicted class": CLASS_NAMES[int(other_result.y_pred[idx])],
                        "Confidence": f"{other_result.y_conf[idx]:.1%}",
                        "Correct?": bool(other_result.y_true[idx] == other_result.y_pred[idx]),
                    }
                )
        st.dataframe(pd.DataFrame(cross_rows), use_container_width=True)

    st.markdown("#### Full misclassification table")
    st.dataframe(wrong_df, use_container_width=True, height=300)


# ===========================================================================
# 9. IMAGE PREDICTION LAB
# ===========================================================================

def page_image_prediction_lab():
    st.title("Image Prediction Lab")
    st.caption("Upload one image — it will be run through all four models, each with its own preprocessing.")
    if load_error:
        missing_models_banner()
        return

    uploaded = st.file_uploader("Upload a satellite image tile", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Upload an image to run it through all four models.")
        return

    image = Image.open(uploaded)
    st.image(image, caption="Uploaded image", width=250)

    device = get_device()
    rows = []
    top3_by_model = {}

    for model_name in MODEL_ORDER:
        loaded = models_dict[model_name]
        tensor = preprocess_uploaded_image(image, model_name)
        elapsed_ms, probs = time_single_image_inference(loaded.model, tensor, device)

        top_idx = int(np.argmax(probs))
        top3_idx = np.argsort(probs)[::-1][:3]
        top3_by_model[model_name] = [(CLASS_NAMES[i], float(probs[i])) for i in top3_idx]

        rows.append(
            {
                "Model": loaded.label,
                "Prediction": CLASS_NAMES[top_idx],
                "Confidence": f"{probs[top_idx]:.1%}",
                "Inference Time (ms)": f"{elapsed_ms:.2f}",
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.markdown("#### Top-3 predictions per model")
    cols = st.columns(4)
    for col, model_name in zip(cols, MODEL_ORDER):
        with col:
            st.markdown(f"**{model_name}**")
            for cls_name, prob in top3_by_model[model_name]:
                st.progress(min(max(prob, 0.0), 1.0), text=f"{cls_name} — {prob:.1%}")


# ===========================================================================
# 10. MODEL EFFICIENCY
# ===========================================================================

def page_model_efficiency():
    st.title("Model Efficiency")
    if load_error:
        missing_models_banner()
        return

    rows = []
    device = get_device()
    for model_name in MODEL_ORDER:
        loaded = models_dict[model_name]
        counts = count_parameters(loaded.model)
        size_mb = model_size_mb(loaded.model)

        # Measure single-image inference time with a dummy tensor of the
        # correct input size (dynamic — never hard-coded).
        dummy = torch.zeros(1, 3, loaded.input_size, loaded.input_size)
        # Warm-up run (first call can include lazy-init overhead).
        time_single_image_inference(loaded.model, dummy, device)
        elapsed_ms, _ = time_single_image_inference(loaded.model, dummy, device)

        rows.append(
            {
                "Model": model_name,
                "Total Parameters": counts["total"],
                "Trainable Parameters": counts["trainable"],
                "Frozen Parameters": counts["frozen"],
                "Model Size (MB)": round(size_mb, 2),
                "Input Resolution": f"{loaded.input_size}×{loaded.input_size}",
                "Measured Inference (ms)": round(elapsed_ms, 3),
            }
        )

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    st.caption(
        "Inference times are measured live on this session's device "
        f"(`{device}`) with a single dummy image per model and will vary "
        "by hardware. Test-set-wide averages are available on the "
        "Confusion Matrix page once evaluated."
    )

    if _test_assets_present() and not load_error:
        st.divider()
        st.markdown("#### Accuracy vs. inference time (from full test-set evaluation)")
        eff_rows = []
        for model_name in MODEL_ORDER:
            try:
                result = evaluate_model_on_test_set(model_name)
                counts = count_parameters(models_dict[model_name].model)
                eff_rows.append(
                    {
                        "model": model_name,
                        "accuracy": result.accuracy,
                        "inference_ms": result.avg_inference_ms_per_image,
                        "params": counts["total"],
                    }
                )
            except Exception:
                continue
        if eff_rows:
            st.plotly_chart(viz.efficiency_scatter(eff_rows), use_container_width=True)
    else:
        missing_test_data_banner()


# ===========================================================================
# 11. LEARNING JOURNEY
# ===========================================================================

def page_learning_journey():
    st.title("The Learning Journey")
    st.markdown(
        "This is the central narrative of the EuroSAT Model Lab: watching one "
        "classification problem get solved four different ways, each building "
        "on ideas the previous model didn't have."
    )
    st.plotly_chart(viz.learning_journey_diagram(), use_container_width=True)

    st.markdown(
        """
| Step | Model | Core Idea |
|---|---|---|
| 1 | **Simple CNN** | Learn convolutional feature extraction from scratch |
| 2 | **Custom ResNet-18** | Learn residual connections — deeper networks without vanishing gradients |
| 3 | **Frozen ResNet-18** | Reuse pretrained (ImageNet) visual representations directly |
| 4 | **Fine-Tuned ResNet-18** | Adapt the entire pretrained network specifically to EuroSAT |
"""
    )

    if not load_error:
        st.divider()
        st.markdown("#### Where each step lands, in this session's measurements")
        if _test_assets_present():
            acc_rows = []
            for model_name in MODEL_ORDER:
                try:
                    result = evaluate_model_on_test_set(model_name)
                    acc_rows.append({"model": model_name, "accuracy": result.accuracy})
                except Exception:
                    continue
            if acc_rows:
                st.plotly_chart(viz.accuracy_bar_chart(acc_rows), use_container_width=True)
        else:
            missing_test_data_banner()


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

PAGE_FUNCS = {
    "1. Overview": page_overview,
    "2. Architecture Explorer": page_architecture_explorer,
    "3. Model Parameters": page_model_parameters,
    "4. Training Progress": page_training_progress,
    "5. Model Comparison": page_model_comparison,
    "6. Confusion Matrix": page_confusion_matrix,
    "7. Per-Class Analysis": page_per_class_analysis,
    "8. Error Analysis": page_error_analysis,
    "9. Image Prediction Lab": page_image_prediction_lab,
    "10. Model Efficiency": page_model_efficiency,
    "11. Learning Journey": page_learning_journey,
}

PAGE_FUNCS[page]()
