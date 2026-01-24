import streamlit as st
import torch
from PIL import Image
import pandas as pd
import pathlib
import platform
import os

# --- 1. SYSTEM CONFIGURATION (Crucial for Cloud) ---
# This fixes the "PosixPath" error when loading a Windows-trained model on Linux
if platform.system() == 'Linux':
    pathlib.WindowsPath = pathlib.PosixPath

st.set_page_config(page_title="Pneumonia CDSS (V1 Legacy)", layout="wide")

# --- 2. HEADER & STYLING ---
st.title("🫁 Pneumonia- Clinical Desicion Support System(v1)")
st.markdown("""
<style>
    .reportview-container { margin-top: -2em; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)
st.markdown("---")

# --- 3. MODEL LOADING ENGINE ---
@st.cache_resource
def load_model():
    """
    Smart Loader: Tries to load locally first (fastest/stable),
    falls back to GitHub download if local files aren't found.
    """
    model_path = 'best.pt'
    
    # Check if the model file actually exists
    if not os.path.exists(model_path):
        st.error(f"❌ Error: '{model_path}' not found. Please upload it to your GitHub repository.")
        return None

    try:
        # OPTION A: Try loading from local 'yolov5' folder (Best for avoiding API limits)
        if os.path.exists('yolov5'):
            return torch.hub.load('yolov5', 'custom', path=model_path, source='local')
        
        # OPTION B: Fallback to downloading from GitHub (Might hit rate limits)
        return torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
        
    except Exception as e:
        # If loading fails, print the error clearly
        st.error(f"⚠️ Model Loading Failed: {e}")
        return None

# Load the model
with st.spinner("Initializing Legacy AI Engine..."):
    model = load_model()

# --- 4. SIDEBAR SETTINGS ---
st.sidebar.header("⚙️ Diagnostics Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
st.sidebar.info(f"Current Sensitivity: **{int(conf_threshold * 100)}%**")

# --- 5. MAIN INTERFACE ---
if model:
    uploaded_file = st.file_uploader("Upload X-Ray Image", type=['png', 'jpg', 'jpeg'])

    if uploaded_file:
        # Layout: Left (Input), Right (Output)
        col1, col2 = st.columns(2)
        
        image = Image.open(uploaded_file)
        
        with col1:
            st.subheader("📄 Patient Scan")
            st.image(image, caption="Original Input", use_container_width=True)

        with col2:
            st.subheader("🔍 AI Analysis")
            
            # --- INFERENCE ---
            model.conf = conf_threshold
            results = model(image)
            
            # Render detections
            results.render()  # Updates the image with boxes
            out_img = Image.fromarray(results.ims[0])
            st.image(out_img, caption="Detected Opacities", use_container_width=True)

        # --- 6. CLINICAL REPORTING ---
        st.markdown("---")
        st.subheader("📋 Clinical Report Data")
        
        # Extract data to Pandas
        df = results.pandas().xyxy[0]
        
        if not df.empty:
            # Format the dataframe for better readability
            report_df = df[['name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']]
            report_df['confidence'] = report_df['confidence'].apply(lambda x: f"{x:.2%}")
            
            st.warning(f"⚠️ **Findings:** {len(df)} potential opacity regions detected.")
            st.dataframe(report_df, use_container_width=True)
            
            # CSV Download
            csv = report_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Report (CSV)",
                csv,
                "pneumonia_screening_report.csv",
                "text/csv"
            )
        else:
            st.success("✅ **Negative:** No pulmonary opacities detected above threshold.")

else:
    st.warning("⚠️ Application is waiting for the model to load.")

