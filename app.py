import streamlit as st
import torch
from PIL import Image
import pandas as pd
import numpy as np
import pathlib
import platform
import os

# ── 1. SYSTEM CONFIGURATION ───────────────────────────────────────────────────
# Fixes "PosixPath" error when a Windows-trained model is loaded on Linux
if platform.system() == "Linux":
    pathlib.WindowsPath = pathlib.PosixPath

st.set_page_config(page_title="Pneumonia CDSS (V1)", layout="wide")

# ── 2. HEADER & STYLING ───────────────────────────────────────────────────────
st.title("🫁 Pneumonia- Clinical Desicion Support System(v1)")
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer     {visibility: hidden;}
</style>
""", unsafe_allow_html=True)
st.markdown("---")

# ── 3. MODEL LOADING ENGINE ───────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """
    Loads best.pt using the YOLOv5 torch.hub API.

    WHY torch.hub and NOT ultralytics.YOLO():
      best.pt is a YOLOv5 model trained with ultralytics/yolov5 (the original
      YOLOv5 repo). The newer ultralytics package (YOLOv8+) explicitly rejects
      YOLOv5 weights with a forward-compatibility error.

    WHY trust_repo=True:
      Permanently silences the "untrusted repository" UserWarning introduced in
      PyTorch 1.12+. Without it, future torch versions will make this a hard
      error. trust_repo=True is the correct, documented permanent fix.

    WHY NOT force_reload=True:
      force_reload re-downloads the entire YOLOv5 GitHub repo on every cold
      start, wasting time and hitting GitHub rate limits. Without it, torch.hub
      caches the repo in ~/.cache/torch/hub after the first download — fast
      on all subsequent starts.
    """
    model_path = "best.pt"

    if not os.path.exists(model_path):
        st.error(
            f"❌ Model file '{model_path}' not found. "
            "Ensure best.pt is committed to the repository root."
        )
        return None

    try:
        model = torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            path=model_path,
            trust_repo=True,        # Permanent fix for UserWarning / future hard error
            verbose=False,
        )
        return model

    except Exception as e:
        st.error(f"⚠️ Model Loading Failed: {e}")
        return None


with st.spinner("Initializing AI Engine..."):
    model = load_model()

# ── 4. SIDEBAR SETTINGS ───────────────────────────────────────────────────────
st.sidebar.header("⚙️ Diagnostics Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
st.sidebar.info(f"Current Sensitivity: **{int(conf_threshold * 100)}%**")

# ── 5. MAIN INTERFACE ─────────────────────────────────────────────────────────
if model:
    uploaded_file = st.file_uploader(
        "Upload Chest X-Ray Image", type=["png", "jpg", "jpeg"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📄 Patient Scan")
            st.image(image, caption="Original Input", use_container_width=True)

        with col2:
            st.subheader("🔍 AI Analysis")

            # YOLOv5 inference — set conf on model then run
            model.conf = conf_threshold
            results = model(image)

            # YOLOv5 API: render() draws boxes directly onto results.ims
            results.render()
            out_img = Image.fromarray(results.ims[0])
            st.image(out_img, caption="Detected Opacities", use_container_width=True)

        # ── 6. CLINICAL REPORTING ─────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📋 Clinical Report Data")

        # YOLOv5 API: .pandas().xyxy[0] returns a clean DataFrame
        df = results.pandas().xyxy[0]

        if not df.empty:
            report_df = df[["name", "confidence", "xmin", "ymin", "xmax", "ymax"]].copy()
            report_df.columns = ["Finding", "Confidence", "X-min", "Y-min", "X-max", "Y-max"]
            report_df["Confidence"] = report_df["Confidence"].apply(lambda x: f"{x:.2%}")
            report_df[["X-min","Y-min","X-max","Y-max"]] = (
                report_df[["X-min","Y-min","X-max","Y-max"]].astype(int)
            )

            st.warning(
                f"⚠️ **Findings:** {len(report_df)} potential opacity region(s) detected."
            )
            st.dataframe(report_df, use_container_width=True)

            csv = report_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Report (CSV)",
                csv,
                "pneumonia_screening_report.csv",
                "text/csv",
            )
        else:
            st.success(
                "✅ **Negative:** No pulmonary opacities detected above threshold."
            )

else:
    st.warning("⚠️ Application is waiting for the model to load.")