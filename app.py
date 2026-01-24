import streamlit as st
from PIL import Image
import os
from predict_utils import run_detection

# ---------- Page Config ----------
st.set_page_config(
    page_title="Weapon Detection System",
    page_icon="🔍",
    layout="centered"
)

with st.sidebar:
    st.title("📌 Project Overview")
    st.write(
        """
        **Weapon Detection System**

        The main purpose of this project is to demonstrate how  
        **computer vision and deep learning** can be used to detect
        potentially dangerous objects in images.

        The system detects:
        - 🔫 Guns
        - 🔪 Knives

        It is designed as an **end-to-end ML application**, covering
        data preparation, model training, inference, and deployment.
        """
    )

    st.markdown("---")

    st.title("🧠 Core Concepts Used")
    st.write(
        """
        - Object Detection using **YOLOv8**
        - Deep Learning with **Convolutional Neural Networks (CNNs)**
        - Multi-model inference pipeline
        - Bounding box prediction & confidence scoring
        - Model evaluation and inference optimization
        """
    )

    st.markdown("---")

    st.title("🛠️ Tools & Technologies")
    st.write(
        """
        - **Python**
        - **YOLOv8 (Ultralytics)**
        - **Streamlit** (Web Interface)
        - **OpenCV** (Image processing)
        - **PyTorch**
        """
    )


# ---------- Main UI ----------
st.markdown(
    """
    <h1 style='text-align: center;'>🔍 Weapon Detection</h1>
    <p style='text-align: center; color: grey;'>
    Upload an image to detect <b>guns</b> and <b>knives</b>
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ---------- File Upload ----------
uploaded_file = st.file_uploader(
    "📤 Upload an image",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear image containing a gun or knife"
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Uploaded Image", width="stretch")

    # Save temp image
    temp_path = "temp_input.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.markdown("")

    # ---------- Detect Button ----------
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        detect_clicked = st.button("🚀 Detect Weapon",use_container_width=True)

    if detect_clicked:
        with st.spinner("🔎 Analyzing image..."):
            output_path = run_detection(temp_path)

        st.success("✅ Detection completed!")

        result_img = Image.open(output_path)
        st.image(result_img, caption="📌 Detection Result", width="stretch")

        # Cleanup
        os.remove(temp_path)

        st.markdown("---")
        st.info(
            "⚠️ **Disclaimer:** This system is for educational/demo purposes only."
        )

# ---------- Footer ----------
st.markdown(
    """
    <hr>
    <p style='text-align: center; color: grey; font-size: 0.85em;'>
    © 2026 <b>Sanchit</b>. All rights reserved.<br>
    This project is intended for <b>educational and demonstration purposes only</b>.<br>
    Misuse of this system for harmful or unethical activities is strictly discouraged.
    </p>
    """,
    unsafe_allow_html=True
)

