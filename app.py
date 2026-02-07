import streamlit as st
from PIL import Image
import os
from predict_utils import run_detection

# ---------- Page Config ----------
st.set_page_config(
    page_title="Weapon Detection System",
    page_icon="🔍",
    layout="wide"
)

# ---------- Global CSS ----------
st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(180deg, #0b0f19, #020617);
    color: #e5e7eb;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #020617;
    border-right: 1px solid #1f2937;
}

/* Headings */
h1, h2, h3 {
    letter-spacing: 0.6px;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #2563eb, #3b82f6);
    color: white;
    border-radius: 10px;
    padding: 8px 14px;
    font-size: 14px;
    font-weight: 600;
    border: none;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #1d4ed8, #2563eb);
    transform: scale(1.02);
}

/* Upload box */
[data-testid="stFileUploader"] {
    border: 2px dashed #3b82f6;
    border-radius: 15px;
    padding: 20px;
}

/* Images */
img {
    border-radius: 15px;
}

/* Alerts */
.stAlert {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("""
    ### 📌 Project Overview
    - 🔍 Detects **guns & knives**
    - 🧠 Deep Learning based
    - ⚡ YOLOv8 object detection
    - 🚀 End-to-end ML pipeline
    """)

    st.markdown("---")

    st.markdown("""
    ### 🧠 Core Concepts
    - Object Detection using **YOLOv8**
    - CNN-based deep learning
    - Bounding box & confidence scoring
    - Optimized inference pipeline
    """)

    st.markdown("---")

    st.markdown("""
    ### 🛠️ Tools & Tech
    - Python
    - YOLOv8 (Ultralytics)
    - Streamlit
    - OpenCV
    - PyTorch
    """)

# ---------- Hero Section ----------
st.markdown("""
<div style="text-align:center; padding:10px 0;">
    <h1 style="font-size:34px;">🔍 Weapon Detection System</h1>
    <p style="font-size:15px; opacity:0.75;">
        AI-powered detection of <b>guns</b> and <b>knives</b> using YOLOv8
    </p>
    <div style="margin-top:6px;">
        <span style="background:#1f2937;padding:4px 10px;border-radius:16px;font-size:13px;">YOLOv8</span>
        <span style="background:#1f2937;padding:4px 10px;border-radius:16px;font-size:13px;margin-left:6px;">Computer Vision</span>
        <span style="background:#1f2937;padding:4px 10px;border-radius:16px;font-size:13px;margin-left:6px;">Deep Learning</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------- Upload Card ----------
st.markdown("""
<div style="background:#020617; padding:14px; border-radius:16px; border:1px solid #1f2937;">
<h3 style="margin-bottom:4px;">📤 Upload Image</h3>
<p style="opacity:0.7; font-size:14px;">
Upload a clear image containing a gun or knife
</p>
</div>
""", unsafe_allow_html=True)


uploaded_file = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

    # Save temporary image
    temp_path = "temp_input.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.markdown("")

    # ---------- Detect Button ----------
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        detect_clicked = st.button("🚀 Detect Weapon", use_container_width=True)

    if detect_clicked:
        with st.spinner("🔎 Analyzing image..."):
            output_path = run_detection(temp_path)

        st.success("🎯 Detection completed successfully!")

        st.markdown("### 🔎 Detection Result")
        result_img = Image.open(output_path)
        st.image(result_img, caption="📌 Detection Output", use_container_width=True)

        # Cleanup
        os.remove(temp_path)

        st.markdown("---")
        st.info("⚠️ This system is intended for educational and demo purposes only.")

# ---------- Footer ----------
st.markdown("""
<hr>
<p style="text-align:center; font-size:13px; opacity:0.65;">
© 2026 <b>Sanchit</b>. All rights reserved.<br>
This project is intended for educational and demonstration purposes only.
</p>
""", unsafe_allow_html=True)
