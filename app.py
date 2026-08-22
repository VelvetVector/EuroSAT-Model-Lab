from pathlib import Path
import time

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image

from models.cnn import SimpleCNN
from models.custom_resnet import CustomResNet18
from models.transfer_resnet import build_transfer_resnet18

from utils.model_loader import load_all_models
from utils.preprocessing import TRANSFORMS
from utils.evaluation import (
    parameter_stats,
    count_modules,
    evaluate_model,
    layer_rows,
)
from utils.visualization import (
    plot_confusion_matrix,
    show_image_grid,
)


# ============================================================
# PATHS / CONFIGURATION
# ============================================================

APP_DIR = Path(__file__).resolve().parent

# The four .pth files are stored directly in models/
MODEL_DIR = APP_DIR / "models"

# Flat test images
TEST_DIR = APP_DIR / "test"

# Ground-truth CSV
TEST_CSV = APP_DIR / "test.csv"


MODEL_FILES = {
    "Simple CNN": "model1.pth",
    "Custom ResNet-18": "model2.pth",
    "Frozen ResNet-18": "model3.pth",
    "Fine-Tuned ResNet-18": "model4.pth",
}


# EuroSAT label mapping used by test.csv
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


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EuroSAT Model Lab",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# TEST DATASET
# ============================================================

class EuroSATTestDataset(torch.utils.data.Dataset):
    """
    EuroSAT test dataset where:
        test/
            img_00000.jpg
            img_00001.jpg
            ...

    and labels are stored in:

        test.csv

    with columns:

        img_id,label
    """

    def __init__(self, image_dir, csv_path, transform=None):
        self.image_dir = Path(image_dir)
        self.csv_path = Path(csv_path)
        self.transform = transform

        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Test image directory not found: {self.image_dir}"
            )

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Test CSV not found: {self.csv_path}"
            )

        self.df = pd.read_csv(self.csv_path)

        required_columns = {"img_id", "label"}

        if not required_columns.issubset(self.df.columns):
            raise ValueError(
                f"test.csv must contain columns: "
                f"{sorted(required_columns)}. "
                f"Found: {list(self.df.columns)}"
            )

        self.df["label"] = self.df["label"].astype(int)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image_name = str(row["img_id"])
        label = int(row["label"])

        image_path = self.image_dir / image_name

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image listed in test.csv was not found: {image_path}"
            )

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# ============================================================
# MODEL-SPECIFIC TEST LOADER
# ============================================================

@st.cache_resource(show_spinner=False)
def make_test_dataset(model_name):
    """
    Creates a test dataset using the preprocessing appropriate
    for the selected model.

    The underlying images and labels remain exactly the same.
    """

    return EuroSATTestDataset(
        image_dir=TEST_DIR,
        csv_path=TEST_CSV,
        transform=TRANSFORMS[model_name],
    )


def make_test_loader(dataset, batch_size=64):

    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


# ============================================================
# MODEL LOADING
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


@st.cache_resource(show_spinner="Loading trained models...")
def load_models():

    loaded = {}
    errors = {}

    # load_all_models expects the directory containing model1.pth ... model4.pth
    loaded, errors = load_all_models(
        MODEL_DIR,
        DEVICE,
        num_classes=len(CLASS_NAMES),
    )

    return loaded, errors


loaded, load_errors = load_models()


# ============================================================
# HELPERS
# ============================================================

def fmt_int(value):
    return f"{int(value):,}"


def model_summary_df():

    rows = []

    for name, info in loaded.items():

        model = info["model"]
        stats = parameter_stats(model)

        if name == "Simple CNN":

            architecture = "Custom CNN"
            input_size = "64×64"
            residual_blocks = 0
            pretrained = "No"
            frozen = "No"

        elif name == "Custom ResNet-18":

            architecture = "Custom ResNet-18"
            input_size = "64×64"
            residual_blocks = 8
            pretrained = "No"
            frozen = "No"

        elif name == "Frozen ResNet-18":

            architecture = "torchvision ResNet-18"
            input_size = "224×224"
            residual_blocks = 8
            pretrained = "ImageNet"
            frozen = "Yes"

        else:

            architecture = "torchvision ResNet-18"
            input_size = "224×224"
            residual_blocks = 8
            pretrained = "ImageNet"
            frozen = "No"

        rows.append(
            {
                "Model": name,
                "Architecture": architecture,
                "Input": input_size,
                "Total Parameters": fmt_int(stats["total"]),
                "Trainable Parameters": fmt_int(stats["trainable"]),
                "Frozen Parameters": fmt_int(stats["frozen"]),
                "Trainable %": f"{stats['trainable_pct']:.2f}%",
                "Conv Layers": count_modules(model, nn.Conv2d),
                "BatchNorm Layers": count_modules(model, nn.BatchNorm2d),
                "Residual Blocks": residual_blocks,
                "Pretrained": pretrained,
                "Backbone Frozen": frozen,
            }
        )

    return pd.DataFrame(rows)


def architecture_description(name):

    descriptions = {

        "Simple CNN": """
### Simple CNN — trained from scratch

```text
Input: 3 × 64 × 64

Conv2D 3 → 32
BatchNorm
ReLU
MaxPool

Conv2D 32 → 64
BatchNorm
ReLU
MaxPool

Conv2D 64 → 128
BatchNorm
ReLU
MaxPool

Conv2D 128 → 256
BatchNorm
ReLU
MaxPool

AdaptiveAvgPool
Flatten
Linear 256 → 128
ReLU
Dropout 0.5
Linear 128 → 10
```

This model contains no residual/skip connections.
""",

        "Custom ResNet-18": """
### Custom ResNet-18 — trained from scratch

```text
Input: 3 × 64 × 64

3×3 Conv
3 → 64
Stride 1
BatchNorm
ReLU

Layer 1
BasicBlock × 2

Layer 2
BasicBlock × 2
64 → 128
Stride 2

Layer 3
BasicBlock × 2
128 → 256
Stride 2

Layer 4
BasicBlock × 2
256 → 512
Stride 2

AdaptiveAvgPool
Flatten
Linear 512 → 10
```

Each BasicBlock contains two 3×3 convolutions,
BatchNorm, ReLU and an identity/projection skip path.
""",

        "Frozen ResNet-18": """
### Frozen ResNet-18 — transfer learning

Standard torchvision ResNet-18 initialized with ImageNet
pretrained weights.

```text
Image
  ↓
ImageNet-pretrained ResNet-18 backbone
  ↓
FROZEN
  ↓
512 features
  ↓
Linear 512 → 10
  ↓
EuroSAT classes
```

Only the replacement EuroSAT classification head is trainable.
""",

        "Fine-Tuned ResNet-18": """
### Fine-Tuned ResNet-18 — transfer learning

Standard torchvision ResNet-18 initialized with ImageNet
pretrained weights.

```text
Image
  ↓
ImageNet-pretrained ResNet-18 backbone
  ↓
TRAINABLE
  ↓
512 features
  ↓
Linear 512 → 10
  ↓
EuroSAT classes
```

The backbone and the new EuroSAT classification head are trainable.
""",
    }

    return descriptions[name]


def get_top_predictions(model, image, transform, k=3):

    tensor = transform(image).unsqueeze(0).to(DEVICE)

    start = time.perf_counter()

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    elapsed = time.perf_counter() - start

    values, indices = torch.topk(
        probabilities,
        min(k, len(CLASS_NAMES)),
    )

    predictions = []

    for value, index in zip(values.cpu(), indices.cpu()):

        predictions.append(
            (
                CLASS_NAMES[int(index)],
                float(value),
            )
        )

    return predictions, elapsed


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🛰️ EuroSAT Model Lab")
st.sidebar.caption("From CNNs to ResNet and Transfer Learning")

PAGES = [
    "Overview",
    "Architecture Explorer",
    "Model Parameters",
    "Training Progress",
    "Model Comparison",
    "Confusion Matrix",
    "Per-Class Analysis",
    "Error Analysis",
    "Image Prediction Lab",
    "Model Efficiency",
    "Learning Journey",
]

page = st.sidebar.radio(
    "Navigate",
    PAGES,
)

st.sidebar.divider()

st.sidebar.write(f"**Device:** `{DEVICE}`")
st.sidebar.write(
    f"**Models loaded:** {len(loaded)}/{len(MODEL_FILES)}"
)

st.sidebar.write(
    f"**Test images:** "
    f"{len(pd.read_csv(TEST_CSV)) if TEST_CSV.exists() else 'N/A'}"
)

if load_errors:

    with st.sidebar.expander("Model loading issues"):

        for name, error in load_errors.items():

            st.error(
                f"**{name}**\n\n{error}"
            )


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.title("EuroSAT Model Lab")

    st.subheader(
        "From CNNs to ResNet and Transfer Learning"
    )

    st.markdown(
        """
        An interactive computer-vision laboratory showing the progression
        from a CNN trained from scratch to residual learning, pretrained
        feature extraction, and full fine-tuning.
        """
    )

    st.divider()

    columns = st.columns(4)

    for column, name in zip(columns, MODEL_FILES.keys()):

        with column:

            st.markdown(f"### {name}")

            if name == "Simple CNN":

                st.write("Custom CNN")
                st.write("Trained from scratch")
                st.write("64×64 input")
                st.write("No skip connections")

            elif name == "Custom ResNet-18":

                st.write("Custom residual network")
                st.write("Trained from scratch")
                st.write("64×64 input")
                st.write("8 BasicBlocks")

            elif name == "Frozen ResNet-18":

                st.write("torchvision ResNet-18")
                st.write("ImageNet pretrained")
                st.write("Backbone frozen")
                st.write("512 → 10 head")

            else:

                st.write("torchvision ResNet-18")
                st.write("ImageNet pretrained")
                st.write("Full fine-tuning")
                st.write("512 → 10 head")

    st.divider()

    st.markdown("## Learning progression")

    progression = [
        (
            "01",
            "Simple CNN",
            "Learn convolutional feature extraction."
        ),
        (
            "02",
            "Custom ResNet-18",
            "Introduce residual blocks and skip connections."
        ),
        (
            "03",
            "Frozen ResNet-18",
            "Reuse ImageNet features and train only a new classifier."
        ),
        (
            "04",
            "Fine-Tuned ResNet-18",
            "Adapt the pretrained network to EuroSAT."
        ),
    ]

    for number, title, description in progression:

        left, right = st.columns([1, 6])

        with left:
            st.metric(number, "")

        with right:

            st.markdown(
                f"**{title}**"
            )

            st.write(description)

    st.divider()

    st.info(
        "The application does not retrain any model. "
        "It loads the supplied checkpoints and analyzes their "
        "architecture, parameters, predictions and errors."
    )


# ============================================================
# ARCHITECTURE EXPLORER
# ============================================================

elif page == "Architecture Explorer":

    st.title("Architecture Explorer")

    if not loaded:

        st.error(
            "No models were loaded. Check the four .pth files "
            "inside the models/ folder."
        )

    else:

        selected = st.selectbox(
            "Select model",
            list(loaded.keys()),
        )

        st.markdown(
            architecture_description(selected)
        )

        st.divider()

        st.markdown("### Actual PyTorch architecture")

        st.code(
            str(loaded[selected]["model"]),
            language="text",
        )

        st.divider()

        st.markdown("### What makes this model different?")

        if selected == "Simple CNN":

            st.write(
                "A conventional convolutional feature extractor. "
                "The model progressively increases channel capacity "
                "from 32 to 256 while pooling reduces spatial resolution."
            )

        elif selected == "Custom ResNet-18":

            st.write(
                "A deeper custom network using residual BasicBlocks. "
                "The skip connection provides an identity/projection "
                "path around the two convolutional layers in each block."
            )

        elif selected == "Frozen ResNet-18":

            st.write(
                "The architecture is the standard torchvision ResNet-18. "
                "Its ImageNet-pretrained convolutional representation "
                "is reused as a fixed feature extractor."
            )

        else:

            st.write(
                "The architecture is the same torchvision ResNet-18 "
                "used for frozen transfer learning. The difference is "
                "that the pretrained backbone is allowed to update."
            )


# ============================================================
# MODEL PARAMETERS
# ============================================================

elif page == "Model Parameters":

    st.title("Model Parameters")

    if not loaded:

        st.error("No models were loaded.")

    else:

        st.dataframe(
            model_summary_df(),
            use_container_width=True,
            hide_index=True,
        )

        st.divider()

        selected = st.selectbox(
            "Inspect model",
            list(loaded.keys()),
        )

        model = loaded[selected]["model"]

        stats = parameter_stats(model)

        a, b, c, d = st.columns(4)

        a.metric(
            "Total parameters",
            fmt_int(stats["total"]),
        )

        b.metric(
            "Trainable",
            fmt_int(stats["trainable"]),
        )

        c.metric(
            "Frozen",
            fmt_int(stats["frozen"]),
        )

        d.metric(
            "Trainable %",
            f"{stats['trainable_pct']:.2f}%",
        )

        st.divider()

        st.markdown(
            "### Layer-by-layer inspector"
        )

        input_size = (
            64
            if selected in (
                "Simple CNN",
                "Custom ResNet-18",
            )
            else 224
        )

        with st.spinner("Inspecting layers..."):

            rows = layer_rows(
                model,
                DEVICE,
                input_size,
            )

        if rows:

            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.info(
                "Layer inspection could not be generated."
            )


# ============================================================
# TRAINING PROGRESS
# ============================================================

elif page == "Training Progress":

    st.title("Training Progress")

    st.info(
        "No training-history files are required by this project. "
        "The four models are already trained, and training-history "
        "arrays were not supplied as an input."
    )

    st.write(
        "Therefore this page deliberately does not fabricate "
        "training or validation curves."
    )

    st.markdown(
        """
        The rest of the dashboard can still analyze the trained models
        directly through their checkpoints and the EuroSAT test set.
        """
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

elif page == "Model Comparison":

    st.title("Model Comparison")

    if not loaded:

        st.error("No models were loaded.")

    else:

        st.dataframe(
            model_summary_df(),
            use_container_width=True,
            hide_index=True,
        )

        st.divider()

        st.markdown("### What changes between the four models?")

        st.markdown(
            """
            **Simple CNN → Custom ResNet-18**

            The network becomes deeper and introduces residual/skip
            connections.

            **Custom ResNet-18 → Frozen ResNet-18**

            The model moves from a custom scratch implementation to
            the standard torchvision ResNet-18 initialized with
            ImageNet-pretrained weights.

            **Frozen ResNet-18 → Fine-Tuned ResNet-18**

            The architecture is essentially the same. The important
            difference is the parameter-update strategy.

            - Frozen: only the new EuroSAT classifier is trained.
            - Fine-tuned: the pretrained backbone and classifier are trained.
            """
        )


# ============================================================
# CONFUSION MATRIX
# ============================================================

elif page == "Confusion Matrix":

    st.title("Confusion Matrix")

    if not TEST_DIR.exists():

        st.error(
            f"Test image folder not found:\n\n`{TEST_DIR}`"
        )

    elif not TEST_CSV.exists():

        st.error(
            f"Test CSV not found:\n\n`{TEST_CSV}`"
        )

    elif not loaded:

        st.error("No models were loaded.")

    else:

        selected = st.selectbox(
            "Select model",
            list(loaded.keys()),
        )

        dataset = make_test_dataset(selected)

        loader = make_test_loader(dataset)

        with st.spinner(
            f"Evaluating {selected} on the test set..."
        ):

            result = evaluate_model(
                loaded[selected]["model"],
                loader,
                DEVICE,
                CLASS_NAMES,
                max_images=0,
            )

        st.metric(
            "Test images evaluated",
            len(result["y_true"]),
        )

        fig = plot_confusion_matrix(
            result["confusion_matrix"],
            CLASS_NAMES,
            f"{selected} — Confusion Matrix",
        )

        st.pyplot(
            fig,
            clear_figure=True,
        )


# ============================================================
# PER-CLASS ANALYSIS
# ============================================================

elif page == "Per-Class Analysis":

    st.title("Per-Class Analysis")

    if not TEST_DIR.exists():

        st.error(
            f"Test image folder not found:\n\n`{TEST_DIR}`"
        )

    elif not TEST_CSV.exists():

        st.error(
            f"Test CSV not found:\n\n`{TEST_CSV}`"
        )

    elif not loaded:

        st.error("No models were loaded.")

    else:

        selected = st.selectbox(
            "Select model",
            list(loaded.keys()),
        )

        dataset = make_test_dataset(selected)

        loader = make_test_loader(dataset)

        with st.spinner(
            f"Evaluating {selected}..."
        ):

            result = evaluate_model(
                loaded[selected]["model"],
                loader,
                DEVICE,
                CLASS_NAMES,
                max_images=0,
            )

        report = result["report"]

        rows = []

        for class_name in CLASS_NAMES:

            metrics = report[class_name]

            rows.append(
                {
                    "Class": class_name,
                    "Precision": metrics["precision"],
                    "Recall": metrics["recall"],
                    "F1": metrics["f1-score"],
                    "Support": int(metrics["support"]),
                }
            )

        df = pd.DataFrame(rows)

        st.dataframe(
            df.style.format(
                {
                    "Precision": "{:.3f}",
                    "Recall": "{:.3f}",
                    "F1": "{:.3f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# ERROR ANALYSIS
# ============================================================

elif page == "Error Analysis":

    st.title("Wrong-Prediction Analysis")

    if not TEST_DIR.exists():

        st.error(
            f"Test image folder not found:\n\n`{TEST_DIR}`"
        )

    elif not TEST_CSV.exists():

        st.error(
            f"Test CSV not found:\n\n`{TEST_CSV}`"
        )

    elif not loaded:

        st.error("No models were loaded.")

    else:

        selected = st.selectbox(
            "Select model",
            list(loaded.keys()),
        )

        sort_mode = st.selectbox(
            "Sort errors by",
            [
                "Highest-confidence wrong predictions",
                "Lowest-confidence predictions",
            ],
        )

        dataset = make_test_dataset(selected)

        loader = make_test_loader(dataset)

        with st.spinner(
            f"Finding wrong predictions for {selected}..."
        ):

            result = evaluate_model(
                loaded[selected]["model"],
                loader,
                DEVICE,
                CLASS_NAMES,
                max_images=None,
            )

        errors = [
            sample
            for sample in result["samples"]
            if sample["true_idx"] != sample["pred_idx"]
        ]

        if sort_mode.startswith("Highest"):

            errors.sort(
                key=lambda x: x["confidence"],
                reverse=True,
            )

        else:

            errors.sort(
                key=lambda x: x["confidence"]
            )

        if not errors:

            st.success(
                "No incorrect predictions were found."
            )

        else:

            max_images = min(
                32,
                len(errors),
            )

            number = st.slider(
                "Images to display",
                min_value=4,
                max_value=max_images,
                value=min(12, max_images),
                step=4,
            )

            show_image_grid(
                errors[:number],
                columns=4,
            )


# ============================================================
# IMAGE PREDICTION LAB
# ============================================================

elif page == "Image Prediction Lab":

    st.title("Image Prediction Lab")

    st.write(
        "Upload an image and run it independently through all four "
        "models. Each model receives its own input resolution and "
        "model-specific preprocessing."
    )

    uploaded = st.file_uploader(
        "Upload an image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
    )

    if uploaded:

        image = Image.open(
            uploaded
        ).convert("RGB")

        st.image(
            image,
            caption="Uploaded image",
            width=400,
        )

        if loaded:

            rows = []

            prediction_cache = {}

            for name, info in loaded.items():

                predictions, elapsed = get_top_predictions(
                    info["model"],
                    image,
                    TRANSFORMS[name],
                    k=3,
                )

                prediction_cache[name] = predictions

                rows.append(
                    {
                        "Model": name,
                        "Prediction": predictions[0][0],
                        "Confidence": predictions[0][1],
                        "Inference ms": elapsed * 1000,
                    }
                )

            df = pd.DataFrame(rows)

            st.markdown(
                "### Four-model prediction"
            )

            st.dataframe(
                df.style.format(
                    {
                        "Confidence": "{:.2%}",
                        "Inference ms": "{:.2f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

            st.markdown(
                "### Top-3 predictions"
            )

            columns = st.columns(
                len(loaded)
            )

            for column, name in zip(
                columns,
                loaded.keys(),
            ):

                with column:

                    st.markdown(
                        f"**{name}**"
                    )

                    for label, probability in prediction_cache[name]:

                        st.progress(
                            probability,
                            text=(
                                f"{label}: "
                                f"{probability:.1%}"
                            ),
                        )

        else:

            st.error(
                "No models were loaded."
            )


# ============================================================
# MODEL EFFICIENCY
# ============================================================

elif page == "Model Efficiency":

    st.title("Model Efficiency")

    if not loaded:

        st.error("No models were loaded.")

    else:

        rows = []

        for name, info in loaded.items():

            model = info["model"]

            stats = parameter_stats(model)

            input_size = (
                64
                if name in (
                    "Simple CNN",
                    "Custom ResNet-18",
                )
                else 224
            )

            dummy = torch.randn(
                1,
                3,
                input_size,
                input_size,
                device=DEVICE,
            )

            model.eval()

            # Warmup
            with torch.no_grad():

                for _ in range(2):

                    model(dummy)

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            with torch.no_grad():

                for _ in range(5):

                    model(dummy)

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            elapsed = (
                time.perf_counter() - start
            ) / 5

            rows.append(
                {
                    "Model": name,
                    "Total Parameters": stats["total"],
                    "Trainable Parameters": stats["trainable"],
                    "Frozen Parameters": stats["frozen"],
                    "Input": f"{input_size}×{input_size}",
                    "Inference ms": elapsed * 1000,
                }
            )

        df = pd.DataFrame(rows)

        st.dataframe(
            df.style.format(
                {
                    "Total Parameters": "{:,}",
                    "Trainable Parameters": "{:,}",
                    "Frozen Parameters": "{:,}",
                    "Inference ms": "{:.2f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Inference time is measured on the current machine and "
            "therefore depends on the hardware and execution backend."
        )


# ============================================================
# LEARNING JOURNEY
# ============================================================

elif page == "Learning Journey":

    st.title("Learning Journey")

    steps = [
        (
            "01 — Simple CNN",
            "Convolutional feature extraction",
            "Learn convolution, BatchNorm, ReLU, pooling and classification."
        ),
        (
            "02 — Custom ResNet-18",
            "Residual learning",
            "Increase depth and introduce BasicBlocks with skip connections."
        ),
        (
            "03 — Frozen ResNet-18",
            "Feature extraction with pretrained representations",
            "Reuse ImageNet features while training only a new EuroSAT classifier."
        ),
        (
            "04 — Fine-Tuned ResNet-18",
            "Domain adaptation",
            "Allow the pretrained backbone and classifier to adapt to EuroSAT."
        ),
    ]

    for title, subtitle, description in steps:

        st.markdown(
            f"## {title}"
        )

        st.markdown(
            f"**{subtitle}**"
        )

        st.write(
            description
        )

        st.divider()

    st.success(
        "The key progression is learning convolutional feature extraction, "
        "residual connections, pretrained representations, frozen feature "
        "extraction, and finally full fine-tuning."
    )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "EuroSAT Model Lab • Inference & Architecture Laboratory"
)
