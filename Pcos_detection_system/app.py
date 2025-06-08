import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tempfile
from ultralytics import YOLO  

# Page config
st.set_page_config(page_title="PCOS Detection", layout="centered")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "PCOS Prediction"])

# Load model only once
@st.cache_resource
def load_model():
    model_path = r"C:\pcos_detection\assets\best (1).onnx"
    return YOLO(model_path)

model = load_model()
class_names = ["Normal", "PCOS"]

# Overview Page
if page == "Overview":
    st.title("PCOS Detection from Ultrasound Images")
    st.image(r"C:\pcos_detection\assets\pcos app image.jpg", use_container_width=True)
    
    st.markdown("""
    ## What is PCOS?
    **Polycystic Ovary Syndrome (PCOS)** is a hormonal disorder common among women of reproductive age. It can cause:
    - Irregular menstrual periods
    - Excess hair growth
    - Acne
    - Ovarian cysts

    ## Why Detection Matters
    Early detection can help in better management of symptoms, reduce long-term complications like infertility and diabetes, and improve quality of life.

    """)

# Prediction Page
elif page == "PCOS Prediction":
    st.title("PCOS Detection from Ultrasound Image")
    st.write("Upload an ultrasound image to detect if it indicates PCOS or is Normal.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption='Uploaded Image', width=400)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            img.save(tmp_file.name)
            tmp_path = tmp_file.name

        results = model.predict(source=tmp_path, save=False, show=False, conf=0.25)
        result_image = results[0].plot()
        result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        st.image(result_image_rgb, caption="Prediction Result", width=400)

        st.subheader("Detections:")
        if results[0].boxes:
            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                st.write(f"**{class_names[cls_id]}** detected with {conf:.2%} confidence")
            if class_names[cls_id] == "PCOS":
                st.write("**PCOS detected!** It's recommended to seek medical consultation for a comprehensive diagnosis and personalized treatment plan.")
            else:
                st.write("**No PCOS detected.** Routine health monitoring is always a good practice.")
        else:
            st.write("No detections found.")

