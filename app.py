import streamlit as st
from PIL import Image
from ultralytics import YOLO  # <--- Using the installed library instead of torch.hub
import pandas as pd

# --- 1. Page Configuration ---
st.set_page_config(page_title="Pneumonia Detection CDSS", layout="wide")

st.title("🫁 Pneumonia Detection - Clinical Decision Support System")
st.markdown("---")

# --- 2. Load Model (Offline / Safe Mode) ---
@st.cache_resource
def load_model():
    # This loads directly from the file 'best.pt' without downloading anything from GitHub
    model = YOLO('best.pt')
    return model

try:
    with st.spinner("Loading AI Model..."):
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
        st.image(img, width=None, caption="Original Upload", use_container_width=True)

    with col2:
        st.subheader("🎯 AI Analysis")
        
        # Run inference using the modern Ultralytics API
        results = model.predict(img, conf=conf_threshold)
        
        # Plot results (returns a numpy array)
        res_plotted = results[0].plot()
        st.image(res_plotted, caption=f"Detections at {conf_threshold*100:.0f}% Confidence", use_container_width=True)

    # --- 5. Clinical Report & Export ---
    st.markdown("---")
    st.subheader("📋 Clinical Report")
    
    # Extract data from the new Ultralytics result format
    boxes = results[0].boxes
    if boxes:
        # Create a clean DataFrame
        data = []
        for box in boxes:
            data.append({
                "Class": model.names[int(box.cls)],
                "Confidence": float(box.conf),
                "xmin": float(box.xyxy[0][0]),
                "ymin": float(box.xyxy[0][1]),
                "xmax": float(box.xyxy[0][2]),
                "ymax": float(box.xyxy[0][3])
            })
        df = pd.DataFrame(data)
        
        st.warning(f"⚠️ Findings: {len(df)} opacity regions detected.")
        st.dataframe(df, use_container_width=True)
        
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
