# EuroSAT Model Lab

### From CNNs to ResNet and Transfer Learning

## Objective

A progressive, visual study of image-classification architectures on the
EuroSAT satellite-imagery dataset — from a convolutional network trained
from scratch, through a hand-built residual network, to a pretrained
backbone that is first frozen and then fully fine-tuned.

This is an **inference + architecture + analysis + visualization
laboratory**, not a training pipeline. All four models are already
trained; the app never retrains or downloads data at runtime.

## Models

1. **Simple CNN** — four-stage convolutional network, trained from scratch.
2. **Custom ResNet-18** — hand-implemented residual network (8 BasicBlocks), trained from scratch.
3. **Frozen ResNet-18** — `torchvision.models.resnet18`, ImageNet-pretrained, backbone frozen, classifier trained.
4. **Fine-Tuned ResNet-18** — the same `torchvision.models.resnet18` architecture as Model 3, but the whole backbone is fine-tuned.

Models 3 and 4 use **the same architecture**. The only difference between
them is which parameters were updated during training (frozen backbone vs.
trainable backbone) — the app makes this distinction explicit throughout.

## Concepts covered

convolution · feature extraction · pooling · BatchNorm · residual /
skip connections · pretrained representations · transfer learning ·
frozen backbones · fine-tuning · parameter analysis · inference ·
error analysis

## Project structure

```
eurosat-model-lab/
├── app.py                    # Streamlit entrypoint
├── README.md
├── requirements.txt
├── test.csv                  # ground truth: img_id,label  (you provide)
│
├── train/                    # not used at inference time (you may leave empty)
│
├── test/                     # flat directory of test images (you provide)
│   ├── img_00000.jpg
│   ├── img_00001.jpg
│   └── ...
│
├── models/
│   ├── model1.pth            # Simple CNN weights            (you provide)
│   ├── model2.pth            # Custom ResNet-18 weights      (you provide)
│   ├── model3.pth            # Frozen ResNet-18 weights      (you provide)
│   ├── model4.pth            # Fine-Tuned ResNet-18 weights  (you provide)
│   ├── cnn.py                # Simple CNN architecture
│   ├── custom_resnet.py      # Custom ResNet-18 architecture
│   └── transfer_resnet.py    # torchvision ResNet-18 wrapper (frozen/fine-tuned)
│
└── utils/
    ├── model_loader.py       # paths, checkpoint loading, parameter introspection
    ├── preprocessing.py      # per-model transforms + flat-directory test dataset
    ├── evaluation.py         # test-set inference, confusion matrices, metrics
    └── visualization.py      # architecture diagrams and charts
```

### Where the four `.pth` files belong

Place your four trained checkpoint files **directly inside `models/`** —
no subfolder, no `checkpoints/` directory:

```
models/model1.pth   → Simple CNN
models/model2.pth   → Custom ResNet-18
models/model3.pth   → Frozen ResNet-18 (torchvision)
models/model4.pth   → Fine-Tuned ResNet-18 (torchvision)
```

Each checkpoint may be a raw `state_dict`, or a dict containing one under
a `state_dict` / `model_state_dict` key — the loader handles all three
without you needing to convert anything.

### Test data

The test images are **not** organized into per-class folders, so the app
uses a custom `EuroSATTestDataset` (not `ImageFolder`). Provide:

- `test.csv` at the project root, with columns `img_id,label`
- `test/` containing every image referenced in `test.csv`

Class ordering (fixed, used consistently across all four models):

```
0 AnnualCrop   1 Forest   2 HerbaceousVegetation   3 Highway   4 Industrial
5 Pasture      6 PermanentCrop   7 Residential   8 River   9 SeaLake
```

If `test.csv` / `test/` aren't present yet, the app still runs — sections
that depend on the test set (Confusion Matrix, Per-Class Analysis, Error
Analysis, part of Model Efficiency) will show a clear notice instead of
fabricating numbers. Everything architecture- and parameter-related works
as soon as the four `.pth` files are in place.

## Local setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploying to Streamlit Community Cloud

1. Push this repository to GitHub.
2. Ensure `app.py` is in the repository root.
3. Ensure `requirements.txt` is in the repository root.
4. Ensure the four `.pth` files are in `models/`.
5. Ensure `test.csv` is in the repository root.
6. Ensure test images are in `test/`.
7. Connect the GitHub repository to [Streamlit Community Cloud](https://streamlit.io/cloud).
8. Select **Branch: `main`**, **Main file: `app.py`**, then **Deploy**.

No secrets or environment variables are required.

> **Large file note:** the four `.pth` checkpoints and up to several
> thousand test images can make this repository large. If you hit
> GitHub's size limits, consider Git LFS for the checkpoints and images —
> this is left as an optional step and is not configured automatically.

## Notes

- All paths resolve relative to `Path(__file__).resolve().parent`, so the
  app behaves identically locally and on Streamlit Cloud.
- Device selection prefers CUDA, then MPS, then falls back to CPU — CPU is
  fully supported and required for Streamlit Cloud.
- Parameter counts, layer counts, inference times, and metrics are all
  computed dynamically from the loaded models and evaluated test data —
  nothing is hard-coded or fabricated. Where information isn't available
  (e.g. training history), the app says so explicitly.
