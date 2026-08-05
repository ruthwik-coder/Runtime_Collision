# 📊 DUAL-ENGINE MODEL EVALUATION REPORT
**Model Architecture:** PyTorch accident_best.pt + YOLO11s Vehicle Tracking  
**Evaluation Dataset:** 14 Test Video Feeds (8 Collisions `s1..s8`, 6 Normal `n1..n6`)  
**Generated On:** 2026-08-05  

---

### 1. 📈 Performance Overview
| Evaluation Metric | Score | Key Insight |
| :--- | :---: | :--- |
| **Accuracy** | **64.29%** | Overall classification accuracy across test set |
| **Precision** | **61.54%** | Percentage of flagged accidents that were real crashes |
| **Recall (Sensitivity)** | **100.00%** | Percentage of total accidents correctly detected |
| **F1-Score** | **76.19%** | Harmonic mean of Precision & Recall |
| **Specificity** | **16.67%** | Ability to correctly clear normal traffic feeds |

---

### 2. 🧮 Confusion Matrix Breakdown
* **True Positives (TP):** `8` (Accident correctly detected)
* **True Negatives (TN):** `1` (Normal traffic correctly cleared)
* **False Positives (FP):** `5` (False alarms)
* **False Negatives (FN):** `0` (Missed accidents)

---

### 3. 🖼️ Generated Evaluation Plots
* **Confusion Matrix Plot:** `Evaluation_Outputs/confusion_matrix.png`
* **Precision-Recall Curve:** `Evaluation_Outputs/precision_recall_curve.png`
* **F1-Confidence Curve:** `Evaluation_Outputs/f1_confidence_curve.png`
* **Validation Batch Grid:** `Evaluation_Outputs/val_batch_predictions.png`
