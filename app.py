import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# 1. System Configuration
st.set_page_config(page_title="MRI Classification System", layout="wide")

# 2. Optimized Model Loading
@st.cache_resource
def load_classification_model():
    # Loading the trained H5 model for inference
    return tf.keras.models.load_model('brain_tumor_model.h5')

model = load_classification_model()

# 3. Technical Sidebar
with st.sidebar:
    st.markdown("### SYSTEM LOGS")
    st.code("BACKEND: TensorFlow\nREADY: True\nSHAPE: 256x256")
    st.divider()
    st.markdown("**Performance Benchmarks:**")
    st.write("- Accuracy: 92.8%")
    st.write("- Task: Binary Classification")
    st.divider()
    st.caption("Developed by Deepneet")

# 4. Main Interface
st.title("MRI Screening Prototype")
st.markdown("##### A software interface for assisting in the analysis of medical imaging data.")
st.divider()

# 5. Image Processing and Inference
uploaded_file = st.file_uploader("Upload MRI Scan", type=["jpg", "png", "tif"])

if uploaded_file:
    # Creating a split-view workspace
    left_col, right_col = st.columns(2, gap="large")
    
    # --- Step 1: Input Handling ---
    img = Image.open(uploaded_file).convert('RGB')
    resized_img = img.resize((256, 256))
    
    with left_col:
        st.subheader("I. Input Data")
        st.image(img, use_container_width=True, caption="Source Scan")

    # --- Step 2: Data Normalization & Prediction ---
    # Normalizing pixel values to a 0-1 scale
    input_tensor = np.array(resized_img) / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0) # Add batch dimension
    
    # Benchmarking inference time
    t0 = time.time()
    raw_prediction = model.predict(input_tensor)
    t1 = time.time()
    latency_ms = round((t1 - t0) * 1000, 2)
    
    # --- Step 3: Result Logic ---
    raw_val = float(raw_prediction[0][0])
    is_anomaly = raw_val > 0.5
    conf_score = raw_val if is_anomaly else (1 - raw_val)

    with right_col:
        st.subheader("II. System Output")
        
        if is_anomaly:
            st.error("#### [CLASSIFICATION: ANOMALY DETECTED]")
        else:
            st.success("#### [CLASSIFICATION: NO ANOMALY]")
            
        # Displaying metrics
        c1, c2 = st.columns(2)
        c1.metric("CONFIDENCE", f"{conf_score:.2%}")
        c2.metric("LATENCY", f"{latency_ms} ms")
        
        st.progress(conf_score)
        
        # Professional Engineering Observability
        with st.expander("III. Technical Metadata"):
            st.json({
                "model_output": raw_val,
                "inference_time": f"{latency_ms}ms",
                "normalization": "1/255.0",
                "status": "Success"
            })
else:
    st.info("System Ready. Please upload imaging data for processing.")

st.divider()
st.caption("Technical Prototype | SDE Portfolio Project | Deepneet")