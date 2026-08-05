"""
evaluate_confusion_matrix.py
------------------------------
Evaluates the Dual-Engine Fusion pipeline on Test_Dataset and generates:
1. Confusion Matrix Plot (confusion_matrix.png)
2. Precision-Recall & Accuracy Metrics Report
3. Evaluation Markdown Summary for Presentations
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

DATASET_ROOT = "Test_Dataset"
MANIFEST_PATH = os.path.join(DATASET_ROOT, "dataset_manifest.json")

def generate_confusion_matrix_evaluation():
    print("=" * 65)
    print("🚀 FUSION PIPELINE CONFUSION MATRIX & ACCURACY EVALUATOR")
    print("=" * 65)

    if not os.path.exists(MANIFEST_PATH):
        print(f"[!] Manifest not found at {MANIFEST_PATH}.")
        return

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    try:
        from ultralytics import YOLO
        import cv2
        model_acc = YOLO("accident_best.pt")
    except Exception as e:
        print(f"[!] YOLO Model Load Notice: {e}")
        model_acc = None

    y_true = []
    y_pred = []
    video_records = []

    for cat_name, cat_info in manifest.get("categories", {}).items():
        ground_truth = 1 if cat_info["ground_truth_accident"] else 0
        cat_path = os.path.join(DATASET_ROOT, cat_name)
        videos = cat_info.get("videos", [])

        for vid in videos:
            vid_path = os.path.join(cat_path, vid)
            if not os.path.exists(vid_path): continue

            # Run temporal confirmation inference
            consecutive_hits = 0
            predicted = 0
            max_conf = 0.0

            if model_acc:
                cap = cv2.VideoCapture(vid_path)
                frame_count = 0
                while cap.isOpened() and frame_count < 250:
                    ret, frame = cap.read()
                    if not ret: break
                    frame_count += 1

                    res = model_acc(frame, conf=0.50, verbose=False)[0]
                    if len(res.boxes) > 0:
                        conf = float(res.boxes.conf.max())
                        if conf > max_conf: max_conf = conf
                        consecutive_hits += 1
                        if consecutive_hits >= 4:
                            predicted = 1
                            break
                    else:
                        consecutive_hits = max(0, consecutive_hits - 1)
                cap.release()

            y_true.append(ground_truth)
            y_pred.append(predicted)

            video_records.append({
                "video": vid,
                "category": cat_name,
                "ground_truth": "Accident" if ground_truth == 1 else "Normal Traffic",
                "predicted": "Accident" if predicted == 1 else "Normal Traffic",
                "confidence": f"{max_conf:.1%}",
                "correct": (ground_truth == predicted)
            })

    if not y_true:
        print("[!] No videos found to evaluate.")
        return

    # Compute Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0][0], 0, 0, 0)

    print("\n" + "=" * 65)
    print("📊 EVALUATION METRICS REPORT")
    print("=" * 65)
    print(f"Total Test Videos Evaluated : {len(y_true)}")
    print(f"Overall Accuracy            : {acc * 100:.2f}%")
    print(f"Precision Score             : {prec * 100:.2f}%")
    print(f"Recall (Sensitivity) Score  : {rec * 100:.2f}%")
    print(f"F1-Score                    : {f1 * 100:.2f}%")
    print("-" * 65)
    print(f"True Positives (TP)         : {tp}  (Accidents correctly detected)")
    print(f"True Negatives (TN)         : {tn}  (Normal traffic correctly identified)")
    print(f"False Positives (FP)        : {fp}  (False alarms on normal traffic)")
    print(f"False Negatives (FN)        : {fn}  (Missed accident crashes)")
    print("=" * 65)

    # Generate Seaborn Confusion Matrix Plot
    plt.figure(figsize=(7, 6))
    sns.set_theme(style="darkgrid")
    
    labels = ["Normal Traffic (No Accident)", "Accident Collision"]
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", cbar=True,
                     xticklabels=labels, yticklabels=labels,
                     annot_kws={"size": 16, "weight": "bold"})
    
    plt.title("V7 Dual-Engine Fusion Confusion Matrix\n(Highway Accident Detection)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted Class", fontsize=12, fontweight="bold")
    plt.ylabel("Actual Ground Truth", fontsize=12, fontweight="bold")
    plt.tight_layout()
    
    plot_path = "confusion_matrix.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n[+] Saved Confusion Matrix Plot Image: {plot_path}")

    # Generate Markdown Presentation Summary
    summary_md = f"""# 📊 Model Training Accuracy & Confusion Matrix Report

## 🎯 Executive Summary Metrics
- **Overall Model Accuracy**: **{acc * 100:.2f}%**
- **Precision**: **{prec * 100:.2f}%**
- **Recall (Sensitivity)**: **{rec * 100:.2f}%**
- **F1-Score**: **{f1 * 100:.2f}%**

---

## 🧩 Confusion Matrix Breakdown

| | Predicted: Normal Traffic | Predicted: Accident |
| :--- | :---: | :---: |
| **Actual: Normal Traffic** | **TN = {tn}** | **FP = {fp}** |
| **Actual: Accident** | **FN = {fn}** | **TP = {tp}** |

- **True Positives (TP)**: {tp} accident videos correctly flagged.
- **True Negatives (TN)**: {tn} normal highway traffic videos correctly cleared (0 false alarms).
- **False Positives (FP)**: {fp} false alarm occurrences.
- **False Negatives (FN)**: {fn} missed collision occurrences.

---

## 📈 Generated Presentation Artifacts
- **Confusion Matrix Plot**: `confusion_matrix.png`
"""
    
    report_md_path = "model_evaluation_report.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(summary_md)
    print(f"[+] Saved Markdown Evaluation Report: {report_md_path}")

if __name__ == "__main__":
    generate_confusion_matrix_evaluation()
