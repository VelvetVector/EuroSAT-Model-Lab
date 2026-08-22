# EuroSAT Model Lab

### From CNNs to ResNet and Transfer Learning

EuroSAT Model Lab is an interactive **computer-vision model laboratory** built around four already-trained image-classification models on the EuroSAT dataset.

The project is designed to demonstrate a progression in image classification:

```text
Simple CNN
    ↓
Custom ResNet-18
    ↓
Frozen Pretrained ResNet-18
    ↓
Fully Fine-Tuned ResNet-18
```

The focus is not just on comparing accuracy. The dashboard lets you inspect the **architecture, layers, parameters, trainable/frozen weights, predictions, errors, class-wise behavior, and inference efficiency** of the four models.

---

# 1. Models

The project contains four already-trained models.

| Model | Architecture | Training strategy | Input |
|---|---|---|---|
| `model1.pth` | Custom Simple CNN | Trained from scratch | 64×64 |
| `model2.pth` | Custom ResNet-18 | Trained from scratch | 64×64 |
| `model3.pth` | torchvision ResNet-18 | ImageNet pretrained + frozen backbone | 224×224 |
| `model4.pth` | torchvision ResNet-18 | ImageNet pretrained + full fine-tuning | 224×224 |

## Model 1 — Simple CNN

A custom convolutional neural network trained from scratch.

```text
Input: 3 × 64 × 64

Conv 3 → 32
BatchNorm
ReLU
MaxPool

Conv 32 → 64
BatchNorm
ReLU
MaxPool

Conv 64 → 128
BatchNorm
ReLU
MaxPool

Conv 128 → 256
BatchNorm
ReLU
MaxPool

AdaptiveAvgPool
Flatten
Linear 256 → 128
ReLU
Dropout
Linear 128 → 10
```

This model does not use residual/skip connections.

---

## Model 2 — Custom ResNet-18

A custom implementation of ResNet-18 trained from scratch.

```text
Input: 3 × 64 × 64

3×3 Conv
3 → 64
Stride 1
BatchNorm
ReLU

BasicBlock × 2
BasicBlock × 2
BasicBlock × 2
BasicBlock × 2

AdaptiveAvgPool
Linear 512 → 10
```

The four residual stages use:

```text
64 → 128 → 256 → 512 channels
```

Each BasicBlock contains two 3×3 convolutional layers and a residual/skip path.

**Important:** this is a custom ResNet-18 implementation. It must not be replaced by torchvision ResNet-18.

---

## Model 3 — Frozen ResNet-18

This model uses the standard torchvision ResNet-18 initialized with ImageNet pretrained weights.

Conceptually:

```text
Image
  ↓
ImageNet-pretrained ResNet-18
  ↓
Frozen backbone
  ↓
512 features
  ↓
Linear 512 → 10
  ↓
EuroSAT classes
```

Only the new EuroSAT classification head is trainable.

---

## Model 4 — Fine-Tuned ResNet-18

This model uses the same standard torchvision ResNet-18 architecture and ImageNet pretrained initialization as Model 3.

The difference is the training strategy:

```text
Image
  ↓
ImageNet-pretrained ResNet-18
  ↓
Trainable backbone
  ↓
512 features
  ↓
Linear 512 → 10
  ↓
EuroSAT classes
```

The backbone and the new EuroSAT classification head are trainable.

### Frozen vs Fine-Tuned

The frozen and fine-tuned models are **not two fundamentally different architectures**.

They use the same torchvision ResNet-18 architecture.

The key difference is:

```text
                 ResNet-18
                     │
          ┌──────────┴──────────┐
          │                     │
       Frozen              Fine-Tuned
          │                     │
 Backbone locked        Backbone trainable
          │                     │
 Classifier trains      Classifier trains
```

This distinction is one of the main concepts demonstrated by the project.

---

# 2. Project Structure

The project uses the following structure:

```text
eurosat-model-lab/
│
├── app.py
│
├── train/
│   └── ... training data ...
│
├── test/
│   ├── img_00000.jpg
│   ├── img_00001.jpg
│   ├── img_00002.jpg
│   ├── ...
│   └── img_03999.jpg
│
├── test.csv
│
├── models/
│   ├── model1.pth
│   ├── model2.pth
│   ├── model3.pth
│   ├── model4.pth
│   ├── cnn.py
│   ├── custom_resnet.py
│   └── transfer_resnet.py
│
└── utils/
    ├── model_loader.py
    ├── preprocessing.py
    ├── evaluation.py
    └── visualization.py
```

There is **no need for a `checkpoints/` directory**.

The four trained `.pth` files are stored directly inside `models/`.

---

# 3. Model Checkpoints

The four checkpoint filenames are fixed as follows:

```text
models/model1.pth
models/model2.pth
models/model3.pth
models/model4.pth
```

Their mapping is:

```text
model1.pth → Simple CNN
model2.pth → Custom ResNet-18
model3.pth → Frozen ResNet-18
model4.pth → Fine-Tuned ResNet-18
```

The application reconstructs the appropriate architecture and loads the corresponding checkpoint.

The application does **not** retrain these models.

---

# 4. EuroSAT Test Data

The test images are stored directly inside the `test/` directory.

They are **not organized into class folders**.

For example:

```text
test/
├── img_00000.jpg
├── img_00001.jpg
├── img_00002.jpg
├── ...
└── img_03999.jpg
```

The ground-truth labels are stored in:

```text
test.csv
```

The CSV is expected to contain:

```text
img_id,label
```

For example:

```text
img_00000.jpg,9
img_00001.jpg,3
img_00002.jpg,3
```

The application uses:

- `img_id` to locate the image in `test/`
- `label` as the ground-truth class

The label mapping is:

```text
0 → AnnualCrop
1 → Forest
2 → HerbaceousVegetation
3 → Highway
4 → Industrial
5 → Pasture
6 → PermanentCrop
7 → Residential
8 → River
9 → SeaLake
```

The application therefore does **not** use `torchvision.datasets.ImageFolder` for the test set.

---

# 5. Model-Specific Preprocessing

The four models do not necessarily use the same input resolution.

The application therefore preprocesses each image according to the selected model.

```text
Simple CNN
    ↓
64 × 64 RGB

Custom ResNet-18
    ↓
64 × 64 RGB

Frozen ResNet-18
    ↓
224 × 224 RGB

Fine-Tuned ResNet-18
    ↓
224 × 224 RGB
```

For uploaded images, each model receives its own appropriately resized and normalized version.

The application must not force every model to use the same input resolution.

---

# 6. Dashboard

Run the application using Streamlit.

The dashboard contains:

```text
1. Overview
2. Architecture Explorer
3. Model Parameters
4. Training Progress
5. Model Comparison
6. Confusion Matrix
7. Per-Class Analysis
8. Error Analysis
9. Image Prediction Lab
10. Model Efficiency
11. Learning Journey
```

---

# 7. Overview

The Overview page introduces the four-model progression:

```text
CNN
 ↓
ResNet-18
 ↓
Pretrained Features
 ↓
Frozen Backbone
 ↓
Full Fine-Tuning
```

It provides a short description of each model and explains the overall learning progression.

---

# 8. Architecture Explorer

The Architecture Explorer lets the user select any of the four models and inspect its architecture.

It displays:

- model architecture
- major layers
- residual blocks
- classifier
- training strategy
- pretrained/frozen status

The actual PyTorch architecture is also displayed so that the user can inspect the loaded model directly.

---

# 9. Model Parameters

The Model Parameters page calculates information directly from the loaded PyTorch model.

It displays:

```text
Total parameters
Trainable parameters
Frozen parameters
Percentage trainable
Percentage frozen
```

It also provides a layer-by-layer inspector containing information such as:

```text
Layer
Type
Input shape
Output shape
Parameters
Trainable?
```

Parameter counts are calculated from the actual loaded models rather than being manually entered.

---

# 10. Training Progress

The models are already trained.

No training-history files are currently required by the project.

Therefore the application does **not fabricate training curves**.

If training-history files are added later, this section can be extended to display:

- training loss
- validation loss
- training accuracy
- validation accuracy
- learning rate

---

# 11. Model Comparison

The Model Comparison page compares the four models using architectural information.

It includes:

```text
Architecture
Input size
Total parameters
Trainable parameters
Frozen parameters
Convolutional layers
BatchNorm layers
Residual blocks
Pretrained status
Frozen backbone status
```

The purpose is to understand how the models differ structurally and in training strategy.

---

# 12. Confusion Matrix

The application evaluates the selected model on the same EuroSAT test images.

For every test image:

```text
image → model → prediction
```

The true class comes from `test.csv`.

A confusion matrix is then generated using the ten EuroSAT classes.

This allows the user to inspect which classes are commonly confused.

---

# 13. Per-Class Analysis

The application calculates per-class:

```text
Precision
Recall
F1-score
Support
```

for the selected model.

This makes it possible to compare how different models behave on individual EuroSAT classes rather than relying only on an overall metric.

---

# 14. Wrong-Prediction Analysis

The Error Analysis page identifies incorrectly classified test images.

For every selected model, the dashboard can display:

```text
Image
True class
Predicted class
Prediction confidence
```

Wrong predictions can be sorted by confidence.

This makes it possible to inspect cases where a model is:

- confidently wrong
- uncertain
- confused between visually similar classes

---

# 15. Image Prediction Lab

The user can upload a single image.

The image is passed independently through all four models.

The dashboard displays:

```text
Model
Prediction
Confidence
Inference time
```

It also displays the top-3 predictions for each model.

Each model receives its own model-specific preprocessing.

This allows direct visual comparison of how the four models interpret the same image.

---

# 16. Model Efficiency

The Model Efficiency page compares:

```text
Total parameters
Trainable parameters
Frozen parameters
Input resolution
Inference time
```

Inference time is measured dynamically on the current machine.

Therefore inference-time comparisons are hardware-dependent and should not be interpreted as universal benchmarks.

---

# 17. Learning Journey

The Learning Journey page explains the conceptual progression:

### Step 1 — Simple CNN

Learn:

- convolution
- feature extraction
- BatchNorm
- ReLU
- pooling
- classification heads

### Step 2 — Custom ResNet-18

Learn:

- deeper networks
- residual blocks
- skip connections
- identity/projection paths

### Step 3 — Frozen ResNet-18

Learn:

- pretrained representations
- transfer learning
- feature extraction
- frozen backbones
- training a new classifier

### Step 4 — Fine-Tuned ResNet-18

Learn:

- adapting pretrained features
- training the backbone
- domain-specific fine-tuning

The purpose is to demonstrate that the project is a progression in understanding image classification rather than merely four unrelated models.

---

# 18. Important Technical Rules

## Do not retrain

The application is an:

> **Inference + Architecture + Analysis + Visualization laboratory**

It is not a training pipeline.

---

## Do not replace the architectures

The application must preserve the distinction between:

```text
Custom CNN
Custom ResNet-18
torchvision ResNet-18
```

The custom ResNet-18 must not be silently replaced with torchvision ResNet-18.

---

## Frozen vs fine-tuned

Do not describe the frozen and fine-tuned models as different network architectures.

They use the same torchvision ResNet-18 architecture.

Their main difference is whether the pretrained backbone is trainable.

---

## Do not fabricate results

The application must never invent:

- accuracy
- precision
- recall
- F1
- parameter counts
- confidence values
- inference times
- training curves
- predictions

These must be calculated from the actual models and data.

---

# 19. Installation

Create a Python environment if desired, then install the requirements:

```bash
pip install -r requirements.txt
```

The main dependencies are:

```text
torch
torchvision
streamlit
numpy
pandas
matplotlib
plotly
scikit-learn
Pillow
```

---

# 20. Running the Application

From inside the project directory:

```bash
cd ~/Desktop/eurosat-model-lab
```

Then:

```bash
streamlit run app.py
```

Streamlit will start the dashboard locally.

---

# 21. Required Inputs

The application requires only:

```text
models/model1.pth
models/model2.pth
models/model3.pth
models/model4.pth

test/
    img_00000.jpg
    img_00001.jpg
    ...

test.csv
```

The `train/` directory can remain in the project because it contains the EuroSAT training data, but the current dashboard does not retrain the models and does not require the training images for test-set evaluation.

---

# 22. Conceptual Goal

The final project should communicate the following progression:

```text
I started with a CNN.
        ↓
I learned how convolutional feature extraction works.
        ↓
I built a deeper residual network.
        ↓
I learned how skip connections enable ResNet-style architectures.
        ↓
I reused ImageNet-pretrained features.
        ↓
I froze the pretrained backbone and trained a new classifier.
        ↓
I then fine-tuned the complete pretrained network.
        ↓
I can now inspect how architecture and training strategy
affect the resulting image-classification system.
```

The project is therefore intended to showcase a progression in **computer-vision and image-classification understanding**, not simply a collection of four trained models.
