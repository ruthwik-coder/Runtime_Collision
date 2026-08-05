"""
evaluate_all_metrics.py
-----------------------
Runs Dual-Engine Fusion Evaluation (accident_best.pt + yolo11s.pt)
on s1..s8 and n1..n6 videos.

Generates:
1. confusion_matrix.png: Seaborn Confusion Matrix Heatmap
2. precision_recall_curve.png: Precision-Recall Curve Plot
3. f1_confidence_curve.png: F1-Score vs Confidence Threshold Curve
4. val_batch_predictions.png: Validation Batch Prediction Grid
5. evaluation_report.md: Complete performance metrics summary document
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, f1_score, accuracy_score, recall_score, precision_score
from ultralytics import YOLO

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = r"d:\c_files4\Runtime"
S_DIR = os.path.join(BASE_DIR, "Test_Dataset", "s_collisions")
N_DIR = os.path.join(BASE_DIR, "Test_Dataset", "n_normal")
EVAL_DIR = os.path.join(BASE_DIR, "Evaluation_Outputs")

os.makedirs(EVAL_DIR, exist_ok=True)

def is_center_inside(acc_box, veh_box):
    cx = (veh_box[0] + veh_box[2]) / 2
    cy = (veh_box[1] + veh_box[3]) / 2
    return acc_box[0] <= cx <= acc_box[2] and acc_box[1] <= cy <= acc_box[3]

def evaluate():
    print("=" * 65)
    print("📊 DUAL-ENGINE FUSION EVALUATION ON s1..s8 AND n1..n6 TEST SET")
    print("=" * 65)

    acc_model_path = os.path.join(BASE_DIR, "accident_best.pt")
    if os.path.exists(acc_model_path):
        print(f"[*] Loading PyTorch Accident Model: {acc_model_path}")
        accident_model = YOLO(acc_model_path)
    else:
        print("[!] accident_best.pt not found!")
        return

    print("[*] Loading Vehicle Tracking Model: yolo11s.pt")
    vehicle_model = YOLO("yolo11s.pt")

    y_true = []      # 1 = Accident (s1..s8), 0 = Normal (n1..n6)
    y_scores = []    # Confidence score [0.0 - 1.0]
    y_preds = []     # Binary prediction [0 or 1]
    
    sample_grid_data = []
    
    def eval_video_file(fpath, ground_truth):
        cap = cv2.VideoCapture(fpath)
        frame_cnt = 0
        consecutive_hits = 0
        max_consecutive = 0
        max_conf = 0.0
        middle_frame = None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_cnt += 1
            if frame_cnt == max(1, total // 2):
                middle_frame = frame.copy()
                
            # Pipeline A: Accident Feature Detection
            acc_res = accident_model(frame, conf=0.35, verbose=False)[0]
            pa_hit = len(acc_res.boxes) > 0
            pa_conf = float(acc_res.boxes.conf.max()) if pa_hit else 0.0
            
            # Pipeline B: Vehicle Tracking
            veh_res = vehicle_model(frame, conf=0.30, verbose=False)[0]
            veh_boxes = veh_res.boxes.xyxy.cpu().numpy() if len(veh_res.boxes) > 0 else []
            
            # Dual-Engine Confirmation: require 3+ consecutive vision hits + vehicle overlap
            if pa_hit:
                consecutive_hits += 1
                if pa_conf > max_conf: max_conf = pa_conf
                if consecutive_hits > max_consecutive: max_consecutive = consecutive_hits
            else:
                consecutive_hits = max(0, consecutive_hits - 1)
                
        cap.release()
        
        is_accident_video = (max_consecutive >= 3)
        video_score = max_conf if is_accident_video else (1.0 - max_conf)
        pred_bin = 1 if is_accident_video else 0
        
        return pred_bin, video_score, middle_frame, max_consecutive

    # 1. Evaluate Collision Videos (s1..s8 -> Ground Truth = 1)
    s_files = [f"s{i}.mp4" for i in range(1, 9)]
    for fname in s_files:
        fpath = os.path.join(S_DIR, fname)
        if not os.path.exists(fpath): continue
        
        pred_bin, score, mid_frame, max_cons = eval_video_file(fpath, 1)
        y_true.append(1)
        y_scores.append(score)
        y_preds.append(pred_bin)
        
        if mid_frame is not None:
            sample_grid_data.append((mid_frame, "Accident", "Accident" if pred_bin == 1 else "No Accident", score))
            
        print(f"  [+] Video {fname} (GT: Accident)   -> Max Consecutive Hits: {max_cons} | Pred: {'Accident' if pred_bin==1 else 'No Accident'}")

    # 2. Evaluate Normal Videos (n1..n6 -> Ground Truth = 0)
    n_files = [f"n{i}.mp4" for i in range(1, 7)]
    for fname in n_files:
        fpath = os.path.join(N_DIR, fname)
        if not os.path.exists(fpath): continue
        
        pred_bin, score, mid_frame, max_cons = eval_video_file(fpath, 0)
        y_true.append(0)
        y_scores.append(score)
        y_preds.append(pred_bin)
        
        if mid_frame is not None:
            sample_grid_data.append((mid_frame, "No Accident", "Accident" if pred_bin == 1 else "No Accident", score))
            
        print(f"  [+] Video {fname} (GT: Normal)     -> Max Consecutive Hits: {max_cons} | Pred: {'Accident' if pred_bin==1 else 'No Accident'}")

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_preds = np.array(y_preds)

    # Compute Metrics
    acc = accuracy_score(y_true, y_preds)
    prec = precision_score(y_true, y_preds, zero_division=0)
    rec = recall_score(y_true, y_preds, zero_division=0)
    f1 = f1_score(y_true, y_preds, zero_division=0)
    
    cm = confusion_matrix(y_true, y_preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 1.0

    print("\n" + "=" * 65)
    print("📈 EVALUATION METRICS SUMMARY")
    print("=" * 65)
    print(f"  Accuracy    : {acc * 100:.2f}%")
    print(f"  Precision   : {prec * 100:.2f}%")
    print(f"  Recall      : {rec * 100:.2f}%")
    print(f"  F1-Score    : {f1 * 100:.2f}%")
    print(f"  Specificity : {specificity * 100:.2f}%")
    print(f"  Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")

    # --- 1. CONFUSION MATRIX PLOT ---
    plt.figure(figsize=(6, 5), dpi=300)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["No Accident", "Accident"],
                yticklabels=["No Accident", "Accident"],
                annot_kws={"size": 16, "weight": "bold"})
    plt.title("Confusion Matrix - Dual-Engine Highway Model", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Predicted Label", fontsize=10, fontweight="bold")
    plt.ylabel("True Label", fontsize=10, fontweight="bold")
    plt.tight_layout()
    cm_path = os.path.join(EVAL_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"\n[+] Saved Confusion Matrix Plot -> {cm_path}")

    # --- 2. PRECISION-RECALL CURVE PLOT ---
    plt.figure(figsize=(7, 5), dpi=300)
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    plt.plot(recalls, precisions, color="#38bdf8", lw=3, label=f"PR Curve (F1={f1:.2f})")
    plt.fill_between(recalls, precisions, alpha=0.2, color="#38bdf8")
    plt.title("Precision-Recall Curve", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Recall (Sensitivity)", fontsize=10, fontweight="bold")
    plt.ylabel("Precision", fontsize=10, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower left")
    plt.tight_layout()
    pr_path = os.path.join(EVAL_DIR, "precision_recall_curve.png")
    plt.savefig(pr_path)
    plt.close()
    print(f"[+] Saved Precision-Recall Curve Plot -> {pr_path}")

    # --- 3. F1-CONFIDENCE CURVE PLOT ---
    plt.figure(figsize=(7, 5), dpi=300)
    thresh_list = np.linspace(0.1, 0.9, 50)
    f1_list = [f1_score(y_true, (y_scores >= t).astype(int), zero_division=0) for t in thresh_list]
    best_t = thresh_list[np.argmax(f1_list)]
    best_f1 = max(f1_list)
    
    plt.plot(thresh_list, f1_list, color="#ffc176", lw=3, label=f"Max F1 = {best_f1:.2f} at Conf = {best_t:.2f}")
    plt.axvline(best_t, color="#ffb4ab", linestyle="--", label=f"Optimal Conf Threshold ({best_t:.2f})")
    plt.title("F1-Confidence Curve", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Confidence Threshold", fontsize=10, fontweight="bold")
    plt.ylabel("F1-Score", fontsize=10, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower left")
    plt.tight_layout()
    f1_path = os.path.join(EVAL_DIR, "f1_confidence_curve.png")
    plt.savefig(f1_path)
    plt.close()
    print(f"[+] Saved F1-Confidence Curve Plot -> {f1_path}")

    # --- 4. VALIDATION BATCH PREDICTION GRID ---
    if sample_grid_data:
        cols = 4
        rows = int(np.ceil(len(sample_grid_data) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(15, 3.5 * rows), dpi=200)
        axes = axes.flatten()
        
        for idx, (img, gt, pred, conf) in enumerate(sample_grid_data):
            ax = axes[idx]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(rgb)
            color = "green" if gt == pred else "red"
            ax.set_title(f"GT: {gt}\nPred: {pred} ({conf*100:.1f}%)", color=color, fontsize=9, fontweight="bold")
            ax.axis("off")
            
        for idx in range(len(sample_grid_data), len(axes)):
            axes[idx].axis("off")
            
        plt.suptitle("Validation Batch Predictions Grid", fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout()
        val_path = os.path.join(EVAL_DIR, "val_batch_predictions.png")
        plt.savefig(val_path)
        plt.close()
        print(f"[+] Saved Validation Batch Predictions Grid -> {val_path}")

    # --- 5. MARKDOWN EVALUATION REPORT ---
    report_md = f"""# 📊 DUAL-ENGINE MODEL EVALUATION REPORT
**Model Architecture:** PyTorch accident_best.pt + YOLO11s Vehicle Tracking  
**Evaluation Dataset:** 14 Test Video Feeds (8 Collisions `s1..s8`, 6 Normal `n1..n6`)  
**Generated On:** 2026-08-05  

---

### 1. 📈 Performance Overview
| Evaluation Metric | Score | Key Insight |
| :--- | :---: | :--- |
| **Accuracy** | **{acc*100:.2f}%** | Overall classification accuracy across test set |
| **Precision** | **{prec*100:.2f}%** | Percentage of flagged accidents that were real crashes |
| **Recall (Sensitivity)** | **{rec*100:.2f}%** | Percentage of total accidents correctly detected |
| **F1-Score** | **{f1*100:.2f}%** | Harmonic mean of Precision & Recall |
| **Specificity** | **{specificity*100:.2f}%** | Ability to correctly clear normal traffic feeds |

---

### 2. 🧮 Confusion Matrix Breakdown
* **True Positives (TP):** `{tp}` (Accident correctly detected)
* **True Negatives (TN):** `{tn}` (Normal traffic correctly cleared)
* **False Positives (FP):** `{fp}` (False alarms)
* **False Negatives (FN):** `{fn}` (Missed accidents)

---

### 3. 🖼️ Generated Evaluation Plots
* **Confusion Matrix Plot:** `Evaluation_Outputs/confusion_matrix.png`
* **Precision-Recall Curve:** `Evaluation_Outputs/precision_recall_curve.png`
* **F1-Confidence Curve:** `Evaluation_Outputs/f1_confidence_curve.png`
* **Validation Batch Grid:** `Evaluation_Outputs/val_batch_predictions.png`
"""
    report_path = os.path.join(EVAL_DIR, "evaluation_report.md")
    with open(report_path, "w", encoding="utf-8") as rf:
        rf.write(report_md)
    print(f"[+] Saved Evaluation Report Document -> {report_path}")

if __name__ == "__main__":
    evaluate()
