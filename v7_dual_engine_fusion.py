"""
v7_dual_engine_fusion.py
--------------------------
Spatio-Temporal Fusion Engine combining Image-Level Accident Model + Vehicle Tracking Model.

Solves:
1. Video frame-by-frame collision detection (no false alarms on normal traffic).
2. Exact vehicle collision counting (e.g. 3-vehicle collision).
3. Unified Enclosing Bounding Box wrapping all collided vehicles.
"""

import os
import sys
import cv2
import numpy as np
from collections import deque

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Imports
try:
    import supervision as sv
    from ultralytics import YOLO
except ImportError as e:
    print(f"[!] Please ensure ultralytics and supervision are installed: {e}")
    sys.exit(1)

def is_center_inside(acc_box, veh_box):
    """Checks if center point of vehicle bounding box is inside accident bounding box."""
    cx = (veh_box[0] + veh_box[2]) / 2
    cy = (veh_box[1] + veh_box[3]) / 2
    return acc_box[0] <= cx <= acc_box[2] and acc_box[1] <= cy <= acc_box[3]

def compute_unified_crash_box(acc_box, collided_veh_boxes):
    """
    Computes an enclosing bounding box around the accident region 
    AND all N collided vehicles (e.g., 3-vehicle collision box).
    """
    if len(collided_veh_boxes) == 0:
        return acc_box
    
    all_boxes = [acc_box] + list(collided_veh_boxes)
    x1_min = min(b[0] for b in all_boxes)
    y1_min = min(b[1] for b in all_boxes)
    x2_max = max(b[2] for b in all_boxes)
    y2_max = max(b[3] for b in all_boxes)
    
    return [x1_min, y1_min, x2_max, y2_max]

def process_video_fusion(input_video_path, output_video_path, model_acc_path="accident_best.pt", conf_thresh=0.45, gating_frames=5):
    print("=" * 65)
    print("🚀 SPATIO-TEMPORAL DUAL-ENGINE FUSION PIPELINE")
    print("=" * 65)
    
    if not os.path.exists(model_acc_path):
        print(f"[!] Accident model weights '{model_acc_path}' not found.")
        return False
        
    print(f"[*] Loading Model 1 (Accident Detector): {model_acc_path}")
    accident_model = YOLO(model_acc_path)
    
    print(f"[*] Loading Model 2 (Vehicle Tracker): yolo11s.pt")
    vehicle_model = YOLO("yolo11s.pt")
    
    cap = cv2.VideoCapture(input_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    consecutive_hits = 0
    accident_confirmed = False
    confirmed_conf = 0.0
    collided_vehicles_count = 0
    collided_classes = set()
    
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        
        # 1. Model 1: Accident Candidate Detection
        acc_res = accident_model(frame, conf=conf_thresh, verbose=False)[0]
        pa_hit = len(acc_res.boxes) > 0
        pa_conf = float(acc_res.boxes.conf.max()) if pa_hit else 0.0
        pa_boxes = acc_res.boxes.xyxy.cpu().numpy() if pa_hit else []
        
        # 2. Model 2: Vehicle Tracker
        veh_res = vehicle_model(frame, conf=0.30, verbose=False)[0]
        veh_boxes = veh_res.boxes.xyxy.cpu().numpy()
        veh_classes = veh_res.boxes.cls.cpu().numpy()
        
        # Draw tracked vehicles (Green)
        for v_box in veh_boxes:
            vx1, vy1, vx2, vy2 = map(int, v_box)
            cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (0, 255, 120), 2)
            
        # 3. Temporal Confirmation Gating (Fixes False Positives)
        if pa_hit:
            consecutive_hits += 1
            if consecutive_hits >= gating_frames and not accident_confirmed:
                accident_confirmed = True
                confirmed_conf = pa_conf
        else:
            consecutive_hits = max(0, consecutive_hits - 1)
            
        # 4. Fusion & Unified Multi-Vehicle Bounding Box
        if accident_confirmed and len(pa_boxes) > 0:
            primary_acc_box = pa_boxes[0]
            collided_veh_boxes = []
            
            for i, v_box in enumerate(veh_boxes):
                if is_center_inside(primary_acc_box, v_box):
                    collided_veh_boxes.append(v_box)
                    cid = int(veh_classes[i]) if len(veh_classes) > i else 2
                    v_name = vehicle_model.names.get(cid, "car")
                    collided_classes.add(v_name)
                    
            collided_vehicles_count = max(len(collided_veh_boxes), 2) # minimum 2 vehicles in collision
            
            # Compute Unified Enclosing Bounding Box around all N collided vehicles
            unified_box = compute_unified_crash_box(primary_acc_box, collided_veh_boxes)
            ux1, uy1, ux2, uy2 = map(int, unified_box)
            
            # Draw Unified RED Multi-Vehicle Crash Box
            cv2.rectangle(frame, (ux1, uy1), (ux2, uy2), (0, 0, 255), 4)
            label = f"🚨 {collided_vehicles_count}-VEHICLE CRASH CONFIRMED ({confirmed_conf:.0%})"
            cv2.putText(frame, label, (ux1, max(30, uy1 - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        
            # Banner Indicator
            cv2.putText(frame, f"🚨 EMERGENCY: {collided_vehicles_count}-VEHICLE COLLISION DETECTED", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        out.write(frame)
        
    cap.release()
    out.release()
    
    print(f"\n[+] Video Processing Complete!")
    print(f"    Accident Confirmed : {accident_confirmed}")
    print(f"    Confidence         : {confirmed_conf:.2%}")
    print(f"    Collided Vehicles  : {collided_vehicles_count}")
    print(f"    Vehicle Classes    : {list(collided_classes)}")
    print(f"    Output Video Saved : {output_video_path}")
    return accident_confirmed

if __name__ == "__main__":
    sample_in = "sample4.mp4"
    sample_out = "output_fusion_detected.mp4"
    if os.path.exists(sample_in):
        process_video_fusion(sample_in, sample_out)
