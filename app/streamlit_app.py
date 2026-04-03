"""
streamlit_app.py
----------------
IWMI Data Science Intern Assessment — Task 4
Streamlit web application for real-time face mask detection.

Run:
    streamlit run app/streamlit_app.py
"""

import os
import sys
import json
import tempfile

import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ── Make src/ importable when running from the project root ───────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import BasicInference, ModelDevelopment   # noqa: E402

import tensorflow as tf
from tensorflow import keras

# ═════════════════════════════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════════════════════════════
MODEL_PATH   = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.keras")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "metrics.json")
CURVES_PATH  = os.path.join(os.path.dirname(__file__), "..", "results", "training_curves.png")
IMG_SIZE     = (128, 128)
CLASS_NAMES  = ["with_mask", "without_mask"]
CLASS_EMOJIS = {"with_mask": "😷", "without_mask": "😐"}
CLASS_COLORS = {"with_mask": "#2ecc71", "without_mask": "#e74c3c"}

# ═════════════════════════════════════════════════════════════════════
# Page setup
# ═════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Face Mask Detector — IWMI",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.4rem;
            font-weight: 800;
            color: #1a1a2e;
            margin-bottom: 0.2rem;
        }
        .sub-title {
            font-size: 1.05rem;
            color: #555;
            margin-bottom: 1.5rem;
        }
        .result-card {
            border-radius: 12px;
            padding: 1.4rem 1.8rem;
            margin-top: 1rem;
            font-size: 1.2rem;
            font-weight: 600;
            text-align: center;
        }
        .with-mask    { background-color: #d4efdf; color: #1e8449; }
        .without-mask { background-color: #fadbd8; color: #c0392b; }
        .metric-box {
            background: #f4f6f9;
            border-radius: 10px;
            padding: 0.9rem 1.2rem;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ═════════════════════════════════════════════════════════════════════
# Helpers (cached)
# ═════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading model…")
def load_model_cached():
    """Load the trained Keras model — cached so it only runs once."""
    if not os.path.exists(MODEL_PATH):
        return None
    return keras.models.load_model(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_metrics():
    """Load persisted evaluation metrics from JSON."""
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as fh:
            return json.load(fh)
    return None


def run_inference(model, uploaded_file) -> dict:
    """
    Save the uploaded file to a temp path, run detect_images, return results.

    Parameters
    ----------
    model         : keras.Model
    uploaded_file : Streamlit UploadedFile

    Returns
    -------
    dict  from BasicInference.detect_images
    """
    suffix = "." + uploaded_file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    detector = BasicInference(model=model, img_size=IMG_SIZE)
    result = detector.detect_images(tmp_path)
    os.unlink(tmp_path)
    return result


def confidence_bar_chart(probabilities: list) -> plt.Figure:
    """
    Horizontal bar chart of class probabilities.

    Parameters
    ----------
    probabilities : list  [p_with_mask, p_without_mask]

    Returns
    -------
    matplotlib Figure
    """
    # Pad to 3 items so the visualisation always shows ≥3 bars
    # (For a 2-class model we add an "Uncertain" placeholder)
    labels = CLASS_NAMES + ["uncertain"]
    values = list(probabilities) + [0.0]
    colors = [CLASS_COLORS.get(l, "#95a5a6") for l in labels]

    fig, ax = plt.subplots(figsize=(6, 2.8))
    bars = ax.barh(labels, values, color=colors, height=0.45, edgecolor="white")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Confidence", fontsize=11)
    ax.set_title("Class Confidence Distribution", fontsize=12, fontweight="bold")

    for bar, val in zip(bars, values):
        ax.text(
            min(val + 0.02, 0.95),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1%}",
            va="center", ha="left", fontsize=10, fontweight="bold",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════════════
# Sidebar
# ═════════════════════════════════════════════════════════════════════

def render_sidebar(model):
    with st.sidebar:
        st.image(
            "https://www.iwmi.cgiar.org/wp-content/uploads/2022/10/IWMI-logo.png",
            width=180,
        )
        st.markdown("## ℹ️ Model Info")

        # ── Architecture summary ──────────────────────────────────────
        with st.expander("🏗 Architecture Summary", expanded=False):
            if model is not None:
                dev = ModelDevelopment()
                dev.model = model
                st.code(dev.get_architecture_summary(), language="text")
            else:
                st.warning("Model not loaded.")

        # ── Evaluation metrics ────────────────────────────────────────
        st.markdown("### 📊 Test-Set Performance")
        metrics = load_metrics()
        if metrics:
            col1, col2 = st.columns(2)
            col1.metric("Accuracy",  f"{metrics['accuracy']:.2%}")
            col1.metric("Precision", f"{metrics['precision']:.2%}")
            col2.metric("Recall",    f"{metrics['recall']:.2%}")
            col2.metric("F1-Score",  f"{metrics['f1_score']:.2%}")
        else:
            st.info("No metrics.json found — run training first.")

        # ── Training curves ───────────────────────────────────────────
        st.markdown("### 📈 Training Curves")
        if os.path.exists(CURVES_PATH):
            st.image(CURVES_PATH, use_container_width=True)
        else:
            st.info("training_curves.png not found.")

        st.markdown("---")
        st.caption("IWMI Data Science Intern Assessment · 2025")


# ═════════════════════════════════════════════════════════════════════
# Main page
# ═════════════════════════════════════════════════════════════════════

def main():
    model = load_model_cached()

    # ── Render sidebar ────────────────────────────────────────────────
    render_sidebar(model)

    # ── Header ───────────────────────────────────────────────────────
    st.markdown(
        '<p class="main-title">😷 Face Mask Detector</p>'
        '<p class="sub-title">Upload a photo to detect whether faces are wearing masks '
        '— powered by a custom CNN trained from scratch.</p>',
        unsafe_allow_html=True,
    )

    if model is None:
        st.error(
            f"⚠️ No trained model found at `{MODEL_PATH}`.  \n"
            "Please train the model first by running `python src/model.py` "
            "from the project root."
        )
        st.stop()

    # ── File uploader ─────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        st.info("👆 Upload a **.jpg / .png / .jpeg** image to get started.")
        return

    # ── Run inference ─────────────────────────────────────────────────
    with st.spinner("Analysing image…"):
        result = run_inference(model, uploaded_file)

    faces       = result["faces"]
    annotated   = result["annotated_image"]
    face_count  = result["face_count"]

    # ── Layout: image | results ────────────────────────────────────────
    col_img, col_res = st.columns([1.1, 1], gap="large")

    with col_img:
        st.markdown("#### 🖼 Annotated Image")
        st.image(annotated, use_container_width=True, caption=uploaded_file.name)
        st.caption(f"Detected {face_count} face(s).")

    with col_res:
        st.markdown(f"#### 🔍 Results  —  {face_count} face(s) found")

        if not faces:
            st.warning("No faces detected. Showing full-image classification.")

        for idx, face in enumerate(faces):
            label      = face["prediction"]
            confidence = face["confidence"]
            probs      = face["probabilities"]
            emoji      = CLASS_EMOJIS.get(label, "")
            css_class  = label.replace("_", "-")

            st.markdown(
                f'<div class="result-card {css_class}">'
                f'{emoji}  Face {idx + 1}: <b>{label.replace("_", " ").title()}</b>'
                f'<br><span style="font-size:0.95rem">Confidence: {confidence:.1%}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )

            # ── Bar chart (top 3 class predictions) ──────────────────
            st.markdown("##### Class Confidence Distribution")
            fig = confidence_bar_chart(probs)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            if idx < len(faces) - 1:
                st.divider()

    # ── Footer ────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "Model: MaskDetectorCNN (custom CNN, TensorFlow/Keras) · "
        "Face detection: OpenCV Haarcascade · "
        "IWMI Data Science Intern Assessment"
    )


if __name__ == "__main__":
    main()
