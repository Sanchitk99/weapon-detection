import os
from datetime import datetime, timezone

import streamlit as st
from PIL import Image

from predict_utils import run_detection

# ---------- Page Config ----------
st.set_page_config(
    page_title="Weapon Detection System",
    layout="wide"
)

# ---------- Global CSS ----------
st.markdown("""
<style>
:root {
    --bg: #0a0f1a;
    --bg-2: #0c1221;
    --panel: #111827;
    --panel-2: #0f172a;
    --border: #1f2a44;
    --text: #e9eef7;
    --muted: #b1bdce;
    --accent: #38bdf8;
    --accent-2: #22d3ee;
    --shadow: 0 18px 40px rgba(2, 8, 23, 0.45);
    --radius: 16px;
}

@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700&display=swap');

/* Main background */
.stApp {
    background:
        radial-gradient(900px 600px at 85% -10%, rgba(34, 211, 238, 0.14), transparent 60%),
        radial-gradient(800px 500px at 10% 10%, rgba(56, 189, 248, 0.10), transparent 60%),
        linear-gradient(180deg, var(--bg), #05070f 70%);
    color: var(--text);
    font-family: 'Sora', sans-serif;
}

/* Content width */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1100px;
}

/* Top deploy bar */
header[data-testid="stHeader"] {
    background: transparent;
    border-bottom: none;
}
header[data-testid="stHeader"] [data-testid="stToolbar"] {
    color: var(--text);
}
header[data-testid="stHeader"] [data-testid="stDeployButton"] button {
    background: transparent;
    color: var(--text);
    border: none;
    box-shadow:none;
    border-radius: 10px;
}
header[data-testid="stHeader"] [data-testid="stDeployButton"] button:hover {
    background: #111a33;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--bg-2);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] h3 {
    color: var(--text);
}

/* Headings */
h1, h2, h3 {
    letter-spacing: 0.2px;
    color: var(--text);
}
h1 {
    font-size: 2.2rem;
    font-weight: 700;
}
h2 {
    font-size: 1.4rem;
    font-weight: 600;
}
h3 {
    font-size: 1.05rem;
    font-weight: 600;
}

/* Text */
p, li, span {
    color: var(--muted);
}

/* Cards */
.card {
    background: linear-gradient(180deg, rgba(17, 24, 39, 0.95), rgba(11, 18, 32, 0.95));
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px;
    box-shadow: var(--shadow);
    transition: transform 140ms ease, box-shadow 140ms ease;
}
.card-subtle {
    background: var(--panel-2);
}
.section-title {
    font-size: 0.95rem;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: var(--muted);
    margin-bottom: 6px;
}
.badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    border: 1px solid #22324f;
    color: #c7d4e7;
    font-size: 12px;
    margin-right: 6px;
    background: rgba(15, 23, 42, 0.6);
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #3b82f6, #22d3ee);
    color: #f8fafc;
    border-radius: 12px;
    padding: 10px 16px;
    font-size: 14px;
    font-weight: 700;
    border: 1px solid rgba(56, 189, 248, 0.35);
    box-shadow: 0 10px 22px rgba(56, 189, 248, 0.22);
    transition: transform 140ms ease, box-shadow 140ms ease;
    text-shadow: 0 1px 2px rgba(3, 7, 18, 0.5);
}
.stButton>button:hover {
    background: linear-gradient(90deg, #5aa4ff, #5eead4);
    transform: translateY(-1px);
    box-shadow: 0 14px 26px rgba(56, 189, 248, 0.28);
}

/* Upload box */
[data-testid="stFileUploader"] {
    background: linear-gradient(180deg, rgba(17, 24, 39, 0.92), rgba(12, 20, 34, 0.92));
    border: 1px dashed #2a3a56;
    border-radius: var(--radius);
    padding: 18px;
    box-shadow: var(--shadow);
}
[data-testid="stFileUploader"] label {
    font-size: 13px;
    color: var(--muted);
}

/* Images */
img {
    border-radius: var(--radius);
}

/* Alerts */
.stAlert {
    border-radius: var(--radius);
    border: 1px solid var(--border);
}
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("""
    ### Project Overview
    - Detects guns and knives
    - Deep learning based
    - YOLOv8 object detection
    - End-to-end ML pipeline
    """)

    st.markdown("---")

    st.markdown("""
    ### Core Concepts
    - Object detection with YOLOv8
    - CNN-based model
    - Bounding box and confidence scoring
    - Optimized inference pipeline
    """)

    st.markdown("---")

    st.markdown("""
    ### Tools and Tech
    - Python
    - YOLOv8 (Ultralytics)
    - Streamlit
    - OpenCV
    - PyTorch
    """)

# ---------- Hero Section ----------
st.markdown("""
<div class="card" style="text-align:left;">
    <div class="section-title">Modern Tech</div>
    <h1>Weapon Detection System</h1>
    <p>
        Real-time visual inference for firearms and bladed weapons using YOLOv8.
        Optimized for fast demos and clean operational workflows.
    </p>
    <div style="margin-top:10px;">
        <span class="badge">YOLOv8</span>
        <span class="badge">Realtime Inference</span>
        <span class="badge">Edge Ready</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------- Input Section ----------
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

col_title, col_reset = st.columns([4, 1])
with col_title:
    st.markdown("""
    <div class="section-title">Input</div>
    <h3 style="margin-top:0;">Upload Image</h3>
    <p style="margin-top:-6px;">
    Provide a clear image. For best results, use well-lit scenes with minimal motion blur.
    </p>
    """, unsafe_allow_html=True)

reset_slot = col_reset.empty()

uploaded_file = st.file_uploader(
    "Image file",
    type=["jpg", "jpeg", "png"],
    key=f"uploader_{st.session_state['uploader_key']}"
)

if uploaded_file:
    with reset_slot.container():
        if st.button("Reset", use_container_width=True, key="reset_btn"):
            st.session_state["uploader_key"] += 1
            st.rerun()
else:
    reset_slot.empty()

conf_threshold = st.slider(
    "Confidence threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.3,
    step=0.05
)
st.caption("Higher threshold reduces false positives but may miss smaller objects.")

st.caption("Privacy: Images are processed locally in this session. Temporary files are deleted after inference.")

input_image = None
input_path = None
temp_path = None
input_label = None

if uploaded_file:
    input_label = "Uploaded Image"
    input_image = Image.open(uploaded_file)

if input_image:
    st.image(input_image, caption=input_label, use_container_width=True)

    if uploaded_file and input_path is None:
    # Save temporary image
        temp_path = "temp_input.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        input_path = temp_path

    # ---------- Detect Button ----------
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        detect_clicked = st.button("Run Detection", use_container_width=True)

    if detect_clicked:
        with st.spinner("Analyzing image..."):
            output_path, counts = run_detection(input_path, conf=conf_threshold)

        st.success(
            f"Detection completed. Guns: {counts['gun']} | Knives: {counts['knife']}"
        )

        st.markdown("### Detection Result")
        metric_a, metric_b, metric_c = st.columns(3)
        metric_a.metric("Guns", counts["gun"])
        metric_b.metric("Knives", counts["knife"])
        metric_c.metric("Threshold", f"{conf_threshold:.2f}")

        result_img = Image.open(output_path)
        st.image(result_img, caption="Detection Output", use_container_width=True)

        # Cleanup
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

        st.markdown("---")
        st.info("This system is intended for educational and demo purposes only.")

# ---------- Footer ----------
build_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
st.markdown(f"""
<hr>
<p style="text-align:center; font-size:13px; opacity:0.65;">
Model: YOLOv8n (yolov8n.pt) | Build: {build_time}<br>
Copyright 2026 <b>Sanchit</b>. All rights reserved.<br>
This project is intended for educational and demonstration purposes only.
</p>
""", unsafe_allow_html=True)
