# 🔍 Weapon Detection System (YOLOv8)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Live-brightgreen)](https://weapon-detection-sanchitk99.streamlit.app/)

An end-to-end **computer vision application** that detects **guns and knives** in images using **YOLOv8** and a **Streamlit web interface**.

---

## 🌐 Live Demo

👉 **Try the app here:**  
https://weapon-detection-sanchitk99.streamlit.app/

---

## 🎯 Project Purpose
This project demonstrates how **deep learning and object detection** can be applied to identify potentially dangerous objects in images.  
It covers the full ML lifecycle: data preparation, model training, inference, and deployment.

---

## 🚀 Demo

![Weapon Detection Demo](assets/demo/weapon_detection_demo.gif)

---

## 📸 Screenshots

### 🖥️ Application Interface
![UI](assets/screenshots/ui.png)

### 📤 Image Upload
![Upload](assets/screenshots/upload.png)

### 📌 Detection Result
![Result](assets/screenshots/result.png)

---

## 🧠 Core Concepts Used
- Object Detection with **YOLOv8**
- Convolutional Neural Networks (CNNs)
- Transfer Learning
- Multi-model inference pipeline
- Bounding box prediction & confidence scoring

---

## 🛠️ Tech Stack
- Python
- YOLOv8 (Ultralytics)
- PyTorch
- OpenCV
- Streamlit

---

## 📂 Project Structure

```text
weapon-detection/
│── app.py                 # Streamlit application
│── predict_utils.py       # Inference logic (gun + knife models)
│── train.py               # Gun model training
│── predict.py             # CLI inference script
│
│── datasets/              # Weapon datasets
│── knife_dataset/         # Knife-specific dataset
│── models/                # Trained model weights
│
│── assets/
│   ├── screenshots/       # UI screenshots
│   └── demo/              # Demo GIF
│
│── requirements.txt
│── README.md
│── .gitignore
```
---

## 📥 Installation Guide (Local Setup)

Follow these steps to download and run the project on your machine.

### 1) Clone the repository
```bash
git clone https://github.com/Sanchitk99/weapon-detection.git
cd weapon-detection
```

### 2) Create a virtual environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

### 4) Run the Streamlit app
```bash
streamlit run app.py
```

The app should open in your browser. If it doesn't, visit:
```
http://localhost:8501
```