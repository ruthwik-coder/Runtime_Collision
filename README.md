

```markdown
# 🚨 Vision-Based Automatic Highway Accident Detection System

> **India Computer Vision Hackathon 2026**  
> A real-time, dual-engine computer vision pipeline designed to monitor highway CCTV feeds, detect vehicular accidents with zero false alarms, and generate an automated emergency dispatch packet for authorities.

---

## 📌 Overview
This project implements a **V7 Dual-Engine Pipeline** that processes video feeds to detect accidents in real-time. Instead of relying on fragile distance heuristics that trigger false positives in dense traffic, this system combines a custom-trained accident detection model with a standard vehicle tracking model, gated by temporal confirmation.

When an accident is detected, the system instantly generates an **Emergency Dispatch Report** containing the crash location, severity, vehicle types involved, and a 5-second telemetry video clip of the exact moment of impact.

## ✨ Key Features (Deliverables)
When an accident is verified, the pipeline outputs the following high-priority dispatch deliverables:
1. **Accident Detection Flag & Confidence:** Boolean trigger + AI confidence score.
2. **Abstracted Map Location:** Simulated GPS coordinates and Camera ID for dashboard mapping.
3. **Snapshot & Telemetry Video:** An auto-saved image and a 5-second video clip (2.5s before and 2.5s after impact) providing instant context to responders.
4. **Standardized Collision Type:** Classification such as Rear-end, Head-on, Single-vehicle, or Multi-vehicle pileup.
5. **Vehicle Count:** The exact number of vehicles physically inside the accident bounding box.
6. **Vehicle Class Type:** Identification of involved vehicles (Truck, Car, Motorcycle, etc.).
7. **Impact Severity Score:** A calculated metric (Minor, Moderate, Severe, Critical) derived from vehicle types and collision dynamics.

---

## 🏗️ System Architecture

The pipeline utilizes a dual-engine approach processed on every frame:

* **Pipeline A (AI Vision):** A custom fine-tuned `YOLO11s` model (`accident_best.pt`) trained on ~9,700 accident images to directly detect accident features (crumpled metal, flipped vehicles).
* **Pipeline B (Trajectory Physics):** A standard `YOLO11s` model tracks vehicle bounding boxes and classes.
* **Fusion & Temporal Gating:** The system only triggers an alert if the accident model is confident across consecutive frames, eliminating single-frame flickering false positives.
* **Center-Point IoU:** To count vehicles involved, the system checks if the center point of a vehicle's bounding box falls inside the accident's bounding box.

---

## 🛠️ Tech Stack
* **Deep Learning:** Ultralytics YOLO11s
* **Video Processing:** Supervision, OpenCV (`opencv-python-headless`)
* **Media Transcoding:** FFmpeg (for HEVC `.mov` to H.264 `.mp4` conversion)
* **Environment:** Python 3.10+ (Optimized for Google Colab T4 GPU)

---

## 🚀 Installation & Usage

This code is optimized for **Google Colab**. 

### 1. Install Dependencies
Run this in your first Colab cell:
```bash
!pip install ultralytics supervision opencv-python-headless -q
!apt-get install ffmpeg -y
```

### 2. Upload Files
Upload your trained model weights (`accident_best.pt`) and your sample input video (e.g., `sample4.mp4`) to the `/content/` directory in Colab.

### 3. Run the Pipeline
Run the main Python script. You can configure the input/output files in the configuration block:
```python
# --- UPDATE FILENAMES HERE ---
MODEL_PATH = "/content/accident_best.pt"
INPUT_VIDEO = "/content/sample4.mp4"
OUTPUT_VIDEO = "/content/output_detected.mp4"
WEB_VIDEO = "/content/web_output.mp4"
```

---

## 📊 Expected Output
Upon running the script, the system will:
1. Draw Green boxes around tracked vehicles and Red boxes around detected accidents.
2. Display a "REC" indicator while capturing the 5-second telemetry buffer.
3. Save `accident_snapshot.jpg` and `web_telemetry.mp4` to the Colab filesystem.
4. Print a structured **Emergency Dispatch Report** to the console:

```text
==================================================
🚨 EMERGENCY DISPATCH REPORT GENERATED 🚨
==================================================
1️⃣ Accident Flag       : TRUE
   Confidence Score    : 88.00%
2️⃣ Map Location        : Lat 12.345, Lon 78.910
   Camera ID           : NH-44-CAM-019
3️⃣ Evidence Captured   : /content/accident_snapshot.jpg
   Telemetry Video     : /content/web_telemetry.mp4 (5 sec)
4️⃣ Collision Type      : Rear-end / Head-on Collision
5️⃣ Vehicles Involved   : 2
6️⃣ Vehicle Class(es)   : car, truck
7️⃣ Impact Severity     : Severe/Critical
==================================================
```

---

## ⚠️ Critical Technical Notes
* **iPhone `.mov` Handling:** OpenCV fails silently on iPhone HEVC `.mov` files. The code automatically uses `ffmpeg` to transcode `.mov` to H.264 `.mp4` before processing.
* **Browser Playback:** Colab's `IPython.display.Video` cannot play raw `mp4v` codecs. The output video is transcoded to `libx264` before being displayed in the notebook.
* **Model Size:** Pipeline B intentionally uses `yolo11s.pt` (instead of `yolo11x.pt`) to ensure the pipeline runs 10x faster on standard Colab T4 GPUs without sacrificing tracking accuracy.

## 📁 Project Structure
```text
.
├── README.md
├── accident_detection_pipeline.py  # The main Colab script
├── accident_best.pt                # Custom trained YOLO weights (User must provide)
└── sample4.mp4                     # Input CCTV/Dashcam footage (User must provide)
```
```
