# 🚨 Vision-Based Automatic Highway Accident Detection & Emergency Command System

> **IEEE Computer Vision Hackathon 2026** | **Runtime Collision Command Center**

A real-time computer vision & deep learning system that monitors highway CCTV footage, detects traffic collisions, eliminates false alarms using dual-engine spatio-temporal fusion, and automatically generates emergency dispatch reports and 2D trajectory reconstructions.

---

## 🚀 Quick Start Guide: How to Run the Project

### 1. Prerequisite Setup
Ensure Python 3.10+ and FFmpeg are installed, then install Python dependencies:

```bash
pip install -r requirements.txt
```
*(Dependencies: `flask`, `flask-cors`, `ultralytics`, `tensorflow`, `opencv-python`, `matplotlib`, `seaborn`, `scikit-learn`, `numpy`)*

---

### 2. 🌐 Option A: Run the Live Command Center Web Dashboard

Launch the Flask backend server:

```bash
python web_server.py
```

Then open your browser to **`http://127.0.0.1:5000`**.

* **Drag & Drop Video Ingest:** Upload any MP4/MOV/AVI CCTV footage or select pre-packaged sample videos (`sample4.mp4`, `s1.mp4`).
* **Run AI Detection:** Click **`RUN V7 DUAL-ENGINE ACCIDENT DETECTION`** to process video and populate the 10 real-time emergency dispatch deliverables.
* **Interactive Map & Simulation:** Click **`START SIMULATION`** or **`TRIGGER DEMO CRASH`** to view live highway markers on Leaflet map.
* **2D Trajectory Vector Canvas:** View birds-eye vehicle physics reconstruction diagrams and speed vectors (km/h).

---

### 3. 🎥 Option B: Run Standalone CLI Video Inference

Process any video feed and generate an annotated `.mp4` video with vehicle bounding boxes (RED for colliding, GREEN for normal), top status HUD banner, and collision count:

```bash
# Infer any input video:
python video_inference.py --input sample4.mp4 --output output_sample4.mp4

# Infer sample collision video:
python video_inference.py --input s1.mp4
```

---

### 4. 📊 Option C: Run Full Model Evaluation & Benchmark Plots

Run the complete evaluation suite across the test video dataset (`s1..s8` and `n1..n6`):

```bash
python evaluate_all_metrics.py
```

Outputs generated in **`Evaluation_Outputs/`**:
- 📊 **`confusion_matrix.png`**: High-resolution Seaborn Confusion Matrix Heatmap
- 📈 **`precision_recall_curve.png`**: Precision vs. Recall Curve Plot
- ⚡ **`f1_confidence_curve.png`**: F1-Score vs. Confidence Threshold Curve
- 🖼️ **`val_batch_predictions.png`**: Validation Batch Prediction Thumbnail Grid
- 📄 **`evaluation_report.md`**: Performance Summary Document (Accuracy, Precision, Recall, Specificity)

---

### 5. 🎬 Option D: Run Batch Processing on All Test Videos

Process all 14 test videos (`s1..s8` collisions + `n1..n6` normal) in batch mode:

```bash
python batch_infer_all.py
```

All 14 annotated `.mp4` output files are saved into **`Inferred_Outputs/`**.

---

## 🏆 Model Benchmark Performance

| Evaluation Metric | Score | Detail / Key System Advantage |
| :--- | :---: | :--- |
| **Accuracy** | **100.00%** | Perfect classification across all 14 test video feeds |
| **Precision** | **100.00%** | **Zero False Alarms** on normal traffic (`n1..n6`) |
| **Recall (Sensitivity)** | **100.00%** | **Zero Missed Collisions** across crash feeds (`s1..s8`) |
| **Specificity** | **100.00%** | All 6 normal traffic video feeds correctly cleared |
| **F1-Score** | **100.00%** | Ideal balanced detection metric |

---

## 📁 Repository Structure

```text
.
├── web_server.py                 # Flask REST API & Dashboard Web Server
├── video_inference.py             # CLI Video Processing Script
├── keras_accident_pipeline.py    # Keras CNN + YOLO Vehicle Collision Engine
├── evaluate_all_metrics.py       # Evaluation Suite & Matplotlib Plot Generator
├── batch_infer_all.py            # Batch Video Inference Runner
├── organize_dataset.py           # Dataset Organizer & FFmpeg Converter
├── model.json                    # Keras Model Architecture Definition
├── templates/
│   └── index.html                # Stitch Cyber Command Center UI
├── static/                       # Static Assets & Styling
├── Evaluation_Outputs/           # Generated Evaluation Plots & Metrics
│   ├── confusion_matrix.png
│   ├── precision_recall_curve.png
│   ├── f1_confidence_curve.png
│   ├── val_batch_predictions.png
│   └── evaluation_report.md
├── Test_Dataset/                 # Structured Test Dataset
│   ├── s_collisions/             # Collision Feeds (s1.mp4 ... s8.mp4)
│   └── n_normal/                 # Normal Traffic Feeds (n1.mp4 ... n6.mp4)
└── README.md
```
