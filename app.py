import streamlit as st
import torch
import pandas as pd
from PIL import Image
import os

# --- 1. Page Configuration ---
# 'wide' layout uses the full screen, perfect for side-by-side X-ray comparison
st.set_page_config(page_title="Pneumonia Detection CDSS", layout="wide")

st.title("🫁 Pneumonia Detection - Clinical Decision Support System")
st.write("Upload a Chest X-ray from your `dataset/images/val/` folder for AI-assisted analysis.")

# --- 2. Load Model Function ---
@st.cache_resource
def load_model():
    # UPDATED: Points exactly to your successful 50-epoch run
    weights_path = 'src/yolov5/runs/train/exp14/weights/best.pt'
    
    # Path safety check
    if not os.path.exists(weights_path):
        st.error(f"❌ Weights not found at: {weights_path}")
        st.info("Check if your training folder name is correct in the script.")
        st.stop()
        
    # Loading model from your local YOLOv5 folder
    model = torch.hub.load('src/yolov5', 'custom', path=weights_path, source='local')
    return model

# Initialize the model
model = load_model()

# --- 3. Sidebar: Clinical Controls ---
st.sidebar.header("🔬 Clinical Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.10, 1.0, 0.25, help="Lower values increase sensitivity; higher values increase specificity.")
st.sidebar.markdown("---")
st.sidebar.write("Developed by Annant R Gautam")

# --- 4. Main Interface ---
uploaded_file = st.file_uploader("Upload Chest X-ray (PNG/JPG)", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖼️ Patient X-ray")
        # UPDATED: Replaced deprecated use_column_width with width="stretch"
        st.image(img, width="stretch", caption="Uploaded Image")

    with col2:
        st.subheader("🎯 AI Analysis")
        
        # Run detection with the selected threshold
        model.conf = conf_threshold
        results = model(img)
        
        # Render the boxes onto the image
        results.render()
        res_img = Image.fromarray(results.ims[0])
        
        # UPDATED: Replaced deprecated use_column_width with width="stretch"
        st.image(res_img, width="stretch", caption=f"Detections at {conf_threshold*100}% Confidence")

    # --- 5. Analysis Report ---
    st.markdown("---")
    st.subheader("📋 Detailed Clinical Report")
    
    # Extract results as a Pandas DataFrame
    df_results = results.pandas().xyxy[0]
    
    if not df_results.empty:
        st.warning(f"⚠️ Findings: {len(df_results)} area(s) of suspected Pneumonia detected.")
        st.dataframe(df_results[['name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']], use_container_width=True)
        
        # --- NEW: Download Findings as CSV ---
        # Cache the conversion to avoid re-running on every click
        csv = df_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Clinical Findings (CSV)",
            data=csv,
            file_name='pneumonia_analysis_report.csv',
            mime='text/csv',
            help="Click to save these coordinates and confidence scores for patient records."
        )
    else:
        st.success("✅ No pneumonia detected. The lungs appear clear at this confidence level.")