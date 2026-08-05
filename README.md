# Vision-Based Automatic Highway Accident Detection System

> **India Computer Vision Hackathon 2026**

A real-time computer vision system that monitors highway CCTV footage, detects road accidents, and automatically generates an emergency dispatch report for first responders.

---

# Overview

This project implements a dual-engine accident detection pipeline for highway surveillance. Instead of relying solely on distance- or motion-based heuristics that often generate false alarms in dense traffic, the system combines a custom-trained accident detection model with a vehicle detection and tracking model. A temporal verification stage confirms detections across multiple consecutive frames before raising an alert, significantly improving reliability.

Once an accident is verified, the system automatically generates an emergency dispatch report containing the estimated location, collision details, vehicle information, severity assessment, and visual evidence.

---

# Features

When an accident is confirmed, the system generates the following information:

1. **Accident Detection**

   * Boolean accident flag
   * Model confidence score

2. **Estimated Location**

   * Simulated GPS coordinates
   * Camera ID for dashboard mapping

3. **Visual Evidence**

   * Snapshot of the accident
   * Five-second telemetry video (2.5 seconds before and after detection)

4. **Collision Classification**

   * Rear-end
   * Head-on
   * Single-vehicle
   * Multi-vehicle collision

5. **Vehicle Count**

   * Number of vehicles involved in the detected accident region

6. **Vehicle Classification**

   * Vehicle categories such as car, truck, motorcycle, bus, etc.

7. **Impact Severity**

   * Severity level classified as Minor, Moderate, Severe, or Critical

---

# System Architecture

The system processes every frame using two parallel pipelines.

### Pipeline A – Accident Detection

A custom fine-tuned **YOLO11s** model (`accident_best.pt`) trained on approximately 9,700 accident images detects accident-specific visual features such as damaged vehicles, overturned vehicles, and collision scenes.

### Pipeline B – Vehicle Detection and Tracking

A standard **YOLO11s** model detects and tracks vehicles throughout the video while identifying their classes.

### Temporal Verification

Accident detections are validated across multiple consecutive frames before an alert is generated. This reduces false positives caused by temporary detection errors.

### Vehicle Association

Vehicles involved in the accident are identified by checking whether the center point of each tracked vehicle lies within the detected accident bounding box.

---

# Technology Stack

* **Deep Learning:** Ultralytics YOLO11s
* **Video Processing:** OpenCV, Supervision
* **Video Transcoding:** FFmpeg
* **Programming Language:** Python 3.10+
* **Execution Environment:** Google Colab (optimized for NVIDIA T4 GPU)

---

# Installation

This project is designed to run on **Google Colab**.

## 1. Install Dependencies

```bash
!pip install ultralytics supervision opencv-python-headless -q
!apt-get install ffmpeg -y
```

## 2. Upload Required Files

Upload the following files to the `/content/` directory:

* `accident_best.pt`
* Input video (for example, `sample4.mp4`)

## 3. Configure File Paths

```python
MODEL_PATH = "accident_best.pt"
INPUT_VIDEO = "s1.mp4"
OUTPUT_VIDEO = "output_detected.mp4"
WEB_VIDEO = "web_output.mp4"
```

Run the main pipeline after updating the paths if necessary.

---

# Expected Output

During execution, the system:

* Draws green bounding boxes around detected vehicles
* Draws red bounding boxes around detected accident regions
* Displays a recording indicator while capturing the telemetry buffer
* Saves an accident snapshot and telemetry video
* Prints a structured emergency dispatch report

Example:

```text
==================================================
EMERGENCY DISPATCH REPORT
==================================================
Accident Detected : TRUE
Confidence Score  : 88.00%

Location
  Latitude        : 12.345
  Longitude       : 78.910
  Camera ID       : NH-44-CAM-019

Evidence
  Snapshot        : /content/accident_snapshot.jpg
  Telemetry Video : /content/web_telemetry.mp4

Collision Type    : Rear-end
Vehicles Involved : 2
Vehicle Classes   : Car, Truck
Impact Severity   : Severe
==================================================
```

---

# Technical Notes

* iPhone-recorded HEVC (`.mov`) videos are automatically transcoded to H.264 (`.mp4`) using FFmpeg before processing, ensuring compatibility with OpenCV.

* Output videos are encoded using **H.264** (`libx264`) to enable playback within Google Colab notebooks.

* The tracking pipeline uses **YOLO11s** instead of larger variants to achieve faster inference on standard Colab T4 GPUs while maintaining reliable detection performance.

---

# Project Structure

```text
.
├── README.md
├── accident_detection_pipeline.py
├── accident_best.pt          # Custom-trained model (user provided)
└── sample4.mp4               # Sample input video (user provided)
```
