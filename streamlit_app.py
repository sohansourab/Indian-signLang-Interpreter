"""Streamlit front-end for the ISL fingerspelling letter recognizer.

This file only adds a hosting/UI layer. It does not change the model, the
training code or the evaluation pipeline: it loads the SAME saved
``ISL_Interpreter.keras`` that the webcam script and FastAPI app use and
reuses the same 128x128 grayscale preprocessing.
"""

import base64
import json
import os
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

IMG_SIZE = 128
ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT / "ISL_Interpreter.keras"
DEFAULT_LABELS_PATH = ROOT / "labels.json"
METRICS_PATHS = [ROOT / "results" / "metrics_clean.json", ROOT / "results" / "metrics.json"]


st.set_page_config(
    page_title="ISL Fingerspelling Interpreter",
    page_icon="✋",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
    .stApp {
        background: radial-gradient(1200px 600px at 10% 0%, #1e293b 0%, #0f172a 45%, #020617 100%);
        color: #e2e8f0;
    }
    .hero {
        padding: 1.2rem 1.6rem;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(56,189,248,0.18), rgba(168,85,247,0.18));
        border: 1px solid rgba(148,163,184,0.25);
        margin-bottom: 1.2rem;
    }
    .hero h1 { margin: 0; font-size: 2.1rem; }
    .hero p { margin: 0.3rem 0 0; color: #cbd5e1; }
    .card {
        background: rgba(15,23,42,0.7);
        border: 1px solid rgba(148,163,184,0.2);
        border-radius: 16px;
        padding: 1.1rem 1.2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    }
    .pred-letter {
        font-size: 7rem;
        font-weight: 800;
        line-height: 1;
        text-align: center;
        background: linear-gradient(135deg, #38bdf8, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .pred-conf { text-align: center; color: #cbd5e1; font-size: 1.1rem; }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        background: rgba(56,189,248,0.15);
        color: #7dd3fc;
        border: 1px solid rgba(56,189,248,0.35);
    }
    .footer-note { color: #94a3b8; font-size: 0.85rem; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def resolve_model_url() -> str | None:
    # Allow deploying without committing the 75 MB model: point MODEL_URL at a
    # hosted copy (Streamlit secrets, env var, or sidebar).
    for source in (st.secrets.get("MODEL_URL") if hasattr(st, "secrets") else None,
                   os.environ.get("MODEL_URL")):
        if source:
            return str(source)
    return None


def ensure_model(model_path: str, model_url: str | None) -> Path:
    mp = Path(model_path)
    if mp.exists():
        return mp
    if not model_url:
        raise FileNotFoundError(
            f"Model not found at {mp}. Set MODEL_URL (env var or Streamlit secret) "
            "to a hosted .keras file, or place the model next to this app."
        )
    mp.parent.mkdir(parents=True, exist_ok=True)
    progress = st.progress(0.0, text="Downloading model (first run only)...")
    def hook(block, block_size, total):
        if total and total > 0:
            progress.progress(min(block * block_size / total, 1.0))
    urlretrieve(model_url, str(mp), reporthook=hook)
    progress.progress(1.0, text="Model downloaded.")
    return mp


@st.cache_resource(show_spinner="Loading model (first run only)...")
def load_model_and_labels(model_path: str, labels_path: str):
    mp = Path(model_path)
    lp = Path(labels_path)
    if not mp.exists():
        raise FileNotFoundError(f"Model not found: {mp}")
    if not lp.exists():
        raise FileNotFoundError(f"Labels not found: {lp}")
    model = tf.keras.models.load_model(str(mp))
    labels = json.loads(lp.read_text(encoding="utf-8"))
    if not isinstance(labels, list) or not labels:
        raise ValueError(f"Labels file must contain a non-empty list: {lp}")
    return model, labels


def load_metrics():
    for path in METRICS_PATHS:
        if path.exists():
            try:
                return path, json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
    return None, None


def decode_uploaded(file) -> np.ndarray | None:
    if file is None:
        return None
    data = file.getvalue()
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame


def preprocess(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)


def predict(model, labels, frame, threshold: float):
    processed = preprocess(frame)
    scores = model.predict(processed, verbose=0)[0]
    idx = int(np.argmax(scores))
    conf = float(scores[idx])
    label = labels[idx] if conf >= threshold else "uncertain"
    order = np.argsort(scores)[::-1][:5]
    top5 = [(labels[i], float(scores[i])) for i in order]
    return label, conf, top5, scores


def render_prediction(label, conf, top5, frame):
    col_img, col_pred = st.columns([1, 1])
    with col_img:
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Input image", use_container_width=True)
    with col_pred:
        st.markdown(f'<div class="pred-letter">{label}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="pred-conf">Confidence: {conf:.2%}</div>', unsafe_allow_html=True)
        st.progress(min(max(conf, 0.0), 1.0))
        st.markdown("#### Top-5 scores")
        for name, score in top5:
            st.write(f"**{name}**")
            st.progress(float(score), text=f"{score:.4f}")


def sidebar():
    st.sidebar.markdown("## ⚙️ Settings")
    model_path = st.sidebar.text_input("Model path", value=str(DEFAULT_MODEL_PATH))
    labels_path = st.sidebar.text_input("Labels path", value=str(DEFAULT_LABELS_PATH))
    model_url = st.sidebar.text_input("Model URL (optional)", value=resolve_model_url() or "")
    threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown(
        "Static **single-letter** ISL fingerspelling recognizer. "
        "Not sentence-level translation. Model: 6.57M params, input 128×128 grayscale."
    )
    metrics_path, metrics = load_metrics()
    if metrics is not None:
        splits = metrics.get("splits", {})
        if splits:
            st.sidebar.markdown("### Recorded metrics")
            for name, m in splits.items():
                st.sidebar.metric(name, f"{m.get('accuracy', 0):.4f}", f"macro-F1 {m.get('macro_f1', 0):.4f}")
    return model_path, labels_path, model_url, threshold


def main():
    st.markdown(
        '<div class="hero"><h1>✋ ISL Fingerspelling Interpreter</h1>'
        '<p>Real-time single-letter recognition for Indian Sign Language. '
        'Upload a photo or capture from your webcam.</p></div>',
        unsafe_allow_html=True,
    )

    model_path, labels_path, model_url, threshold = sidebar()

    try:
        ensure_model(model_path, model_url or None)
        model, labels = load_model_and_labels(model_path, labels_path)
    except Exception as exc:
        st.error(f"Could not load model/labels: {exc}")
        st.stop()

    st.markdown('<span class="badge">Model loaded · 26 classes (A–Z)</span>', unsafe_allow_html=True)

    tab_cam, tab_upload, tab_info = st.tabs(["📷 Webcam", "🖼️ Upload image", "📊 Info"])

    with tab_cam:
        st.markdown("### Capture a hand sign")
        cam_file = st.camera_input("Take a picture", key="cam")
        if cam_file is not None:
            frame = decode_uploaded(cam_file)
            if frame is None:
                st.error("Could not decode the captured image.")
            else:
                with st.spinner("Predicting..."):
                    label, conf, top5, _ = predict(model, labels, frame, threshold)
                render_prediction(label, conf, top5, frame)

    with tab_upload:
        st.markdown("### Upload an image")
        up = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
        if up is not None:
            frame = decode_uploaded(up)
            if frame is None:
                st.error("Could not decode the uploaded image.")
            else:
                with st.spinner("Predicting..."):
                    label, conf, top5, _ = predict(model, labels, frame, threshold)
                render_prediction(label, conf, top5, frame)

    with tab_info:
        st.markdown("### Project status")
        st.markdown(
            "- Trains a small CNN (Conv2D 64/64/128 → Dense 256/128 → 26-way softmax).\n"
            "- Evaluation numbers come from `results/` (seed 42, saved model).\n"
            "- Testing (full): 5,198 images, acc 0.8671. Testing (duplicate-free): 4,587 images, acc 0.8494.\n"
            "- Validation: 2,579 images, acc 0.7980."
        )
        st.markdown("### Recorded metrics")
        _, metrics = load_metrics()
        if metrics is not None:
            st.json(metrics.get("splits", metrics))
        else:
            st.info("No metrics files found under results/.")
        st.markdown(
            '<p class="footer-note">Live-webcam accuracy is [NOT MEASURED]. '
            'This demo classifies static single letters only.</p>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
