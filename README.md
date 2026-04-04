# 😷 Face Mask Detector — IWMI Data Science Intern Assessment

> **Custom CNN built from scratch with TensorFlow/Keras, deployed via Streamlit.**

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Architecture](#architecture)
- [Setup & Installation](#setup--installation)
- [Training the Model](#training-the-model)
- [Running the Streamlit App](#running-the-streamlit-app)
- [Results](#results)
- [Performance Metrics & Justification](#performance-metrics--justification)
- [Failure Mode Analysis](#failure-mode-analysis)

---

## Project Overview

Binary image classifier that detects whether a person in a photo is wearing a
face mask. The pipeline covers:

| Stage | Details |
|---|---|
| **Data** | Face-mask dataset — two classes: `with_mask`, `without_mask` |
| **Preprocessing** | Resize to 128×128, normalise, augment (flip, rotate, zoom, brightness) |
| **Model** | Custom 4-block CNN (~2.8 M params) — no pretrained weights |
| **Inference** | OpenCV Haarcascade face detection → CNN classification per face |
| **Deployment** | Streamlit web app (upload image → annotated result + confidence chart) |

---

## Repository Structure

```
senadhi-thimanya-iwmi-assessment/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── .gitignore
│
├── src/
│   └── model.py                # All 3 classes + main()
│         ├── BasicPreprocessing    — Task 1
│         ├── ModelDevelopment      — Task 2
│         └── BasicInference        — Task 3
│
├── app/
│   └── streamlit_app.py        # Task 4 — Streamlit UI
│
├── models/
│   └── best_model.keras        # Saved weights (after training)
│
└── results/
    ├── training_curves.png
    ├── confusion_matrix.png
    ├── sample_images.png
    ├── metrics.json
    └── class_distribution.png


```

---

## Architecture

**MaskDetectorCNN** — designed and trained entirely from scratch.

```
Input (128 × 128 × 3)
│
├─ ConvBlock 1:  Conv2D(32)  → BN → ReLU → Conv2D(32)  → BN → ReLU → MaxPool → Dropout(0.20)
├─ ConvBlock 2:  Conv2D(64)  → BN → ReLU → Conv2D(64)  → BN → ReLU → MaxPool → Dropout(0.25)
├─ ConvBlock 3:  Conv2D(128) → BN → ReLU → Conv2D(128) → BN → ReLU → MaxPool → Dropout(0.30)
├─ ConvBlock 4:  Conv2D(256) → BN → ReLU → Conv2D(256) → BN → ReLU → MaxPool → Dropout(0.35)
│
├─ GlobalAveragePooling2D
│
├─ Dense(512) → BN → ReLU → Dropout(0.50)
├─ Dense(256) → BN → ReLU → Dropout(0.30)
│
└─ Dense(2, softmax)     ← with_mask | without_mask
```

**Design choices:**

- **Dual Conv per block** — captures richer spatial features before downsampling.
- **Batch Normalisation** — stabilises training, reduces sensitivity to learning rate.
- **Graduated Dropout** — deeper blocks drop more aggressively to prevent overfitting.
- **GlobalAveragePooling** — replaces Flatten+Dense, significantly reduces parameter count and overfitting.
- **L2 regularisation** (1e-4) on Conv and Dense kernels — additional weight decay.
- **Adam + ReduceLROnPlateau** — adaptive optimisation with automatic LR halving on plateau.

---

## Setup & Installation

### Prerequisites

- Python 3.9 – 3.11
- pip ≥ 23
- (Optional) NVIDIA GPU with CUDA 11.8+ for faster training

### 1. Clone the repository

```bash
git clone https://github.com/senadhi-Thimanya/senadhi-thimanya-iwmi-assessment.git
cd senadhi-thimanya-iwmi-assessment
```

### 2. Create a virtual environment

```bash
python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download the dataset

Download the dataset from the provided link: [https://drive.google.com/file/d/1Dw0DGHwdmiblqzk8u1LeMzMCqo87sJhN/view?usp=sharing](https://drive.google.com/file/d/1Dw0DGHwdmiblqzk8u1LeMzMCqo87sJhN/view?usp=sharing)

```
dataset/
    with_mask/
        image1.jpg
        image2.jpg
        ...
    without_mask/
        image1.jpg
        image2.jpg
        ...
```

---

## Training the Model

```bash
python src/model.py
```

This will:
1. Load and split the dataset (70 / 15 / 15 train/val/test).
2. Apply augmentations and build generators.
3. Build, compile, and train the CNN for up to 50 epochs.
4. Save the best checkpoint to `models/best_model.keras`.
5. Save training curves to `results/training_curves.png`.
6. Evaluate on the test set and save `results/confusion_matrix.png` and `results/metrics.json`.

**Typical training time:** ~15 min on CPU · ~3 min on a single GPU.

---

## Running the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

**Features:**
- Upload `.jpg`, `.jpeg`, or `.png` images.
- Annotated image with bounding boxes around detected faces.
- Predicted class label + confidence percentage per face.
- Horizontal bar chart of class confidence scores.
- Sidebar showing model architecture summary, test accuracy, and training curves.

**Live demo:** [https://senadhi-thimanya-iwmi-assessment.streamlit.app/](https://senadhi-thimanya-iwmi-assessment.streamlit.app/)

---

## Results

| Metric | Score |
|---|---|
| Test Accuracy | ~97% |
| Precision (weighted) | ~97% |
| Recall (weighted) | ~97% |
| F1-Score (weighted) | ~97% |

> Actual numbers are written to `results/metrics.json` after training.

---

## Performance Metrics & Justification

| Metric | Why it matters here |
|---|---|
| **Accuracy** | Good baseline when classes are balanced. |
| **Precision** | Limits false alarms (predicting "masked" when not). |
| **Recall** | Safety-critical — catches all actual no-mask cases. |
| **F1-Score** | Balances precision & recall; single go-to metric for binary tasks. |
| **Confusion Matrix** | Shows the direction of errors (FP vs FN split). |

---

## Failure Mode Analysis

**Where the model succeeds:**
- Well-lit, front-facing, single-person images.
- Standard surgical / N95 / cloth masks that match the training distribution.

**Where the model struggles:**
- Non-standard face coverings (scarves, bandanas, neck gaiters).
- Profile or heavily tilted faces (Haarcascade misses the face → full-image fallback).
- Very low resolution or heavily compressed images.
- Demographically unbalanced training data → lower recall on underrepresented groups.

**Mitigation strategies:**
- Collect harder edge cases and re-train with oversampling.
- Replace Haarcascade with a modern detector (MediaPipe Face Mesh, MTCNN).
- Apply test-time augmentation (TTA) for better confidence calibration.

---

*IWMI Data Science Intern Assessment · 2025*
