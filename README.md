# Brain Tumor Classification System
**A Production-Grade Medical Imaging Pipeline**

This repository contains an end-to-end AI system designed to classify MRI scans into 'Anomaly' (Tumor) or 'Normal' states. The project focuses on high-inference speed and scalable deployment.

## Technical Overview
* **Architectures:** ResNet50 and VGG19 (Fine-tuned via Transfer Learning).
* **Frameworks:** TensorFlow 2.x, Keras.
* **Input Layer:** 256x256 RGB normalized tensors.
* **Optimization:** Adam Optimizer with Categorical Cross-Entropy loss.
* **Deployment:** Streamlit-based interface hosted on Hugging Face Spaces.

## Key Engineering Features
* **Inference Latency:** Benchmarked at sub-100ms to ensure real-time clinical utility.
* **Memory Management:** Implemented `@st.cache_resource` to optimize weights loading and prevent redundant memory allocation.
* **Data Pipelines:** Integrated data augmentation and normalization layers to handle variance in clinical imaging data.
* **Observability:** Technical metadata expander included in the UI for monitoring raw model outputs and latency metrics.

## Performance Metrics
* **Validation Accuracy:** 92.8%
* **Testing Precision:** Highly robust across heterogeneous MRI datasets.
* **Status:** Deployed and operational.

## Repository Structure
* `app.py`: Main application logic and Streamlit UI.
* `brain_tumor_model.h5`: Trained model weights (managed via Git LFS).
* `requirements.txt`: Environment dependencies.

---
**Developed by Deepneet Nanda | IIT Kharagpur**
