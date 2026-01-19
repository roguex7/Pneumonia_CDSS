import streamlit as st
import torch
from PIL import Image
import os
import pathlib
from pathlib import Path

# --- 1. Page Config ---
st.set_page_config(page_title="Pneumonia Detection CDSS", layout="wide")
st.title("🫁 Pneumonia Detection - Clinical Decision Support System")
st.markdown("---")

# --- 2. Load Model (Cloud Optimized) ---
@st.cache_resource
def load_model():
    # Use 'ultralytics/yolov5' to load from the web, not 'source=local'
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True) 
    return model

try:
    model = load_model()
    st.sidebar.success("✅ Model Loaded Successfully")
except Exception as e:
    st.sidebar.error("Model failed to load.")
    st.error(f"Error: {e}")

# --- 3. Sidebar Controls ---
st.sidebar.header("🔬 Clinical Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.10, 1.0, 0.25)
st.sidebar.info("Adjust sensitivity to catch subtle cases.")

# --- 4. Main Interface ---
uploaded_file = st.file_uploader("Upload Chest X-ray (PNG/JPG)", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖼️ Patient X-ray")
        st.image(img, width="stretch", caption="Original Upload") # Updated for 2026 standards

    with col2:
        st.subheader("🎯 AI Analysis")
        # Run inference
        model.conf = conf_threshold
        results = model(img)
        
        # Render and Show
        results.render()
        res_img = Image.fromarray(results.ims[0])
        st.image(res_img, width="stretch", caption=f"Detections at {conf_threshold*100}% Confidence")

    # --- 5. Clinical Report & Export ---
    st.markdown("---")
    st.subheader("📋 Clinical Report")
    df = results.pandas().xyxy[0]
    
    if not df.empty:
        st.warning(f"⚠️ Findings: {len(df)} opacity regions detected.")
        st.dataframe(df[['name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']], use_container_width=True)
        
        # CSV Download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Findings (CSV)", 
            csv, 
            "pneumonia_report.csv", 
            "text/csv"
        )
    else:
        st.success("✅ No pneumonia detected at this threshold.")

