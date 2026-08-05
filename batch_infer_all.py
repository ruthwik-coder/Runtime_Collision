"""
batch_infer_all.py
------------------
Processes all 14 test video files (s1..s8 and n1..n6) through the 
Accident Detection & Vehicle Collision Pipeline and outputs annotated .mp4 videos.

Input Folders:
  - Test_Dataset/s_collisions/ (s1.mp4 ... s8.mp4)
  - Test_Dataset/n_normal/     (n1.mp4 ... n6.mp4)

Output Folder:
  - Inferred_Outputs/
      ├── out_s1.mp4 ... out_s8.mp4
      └── out_n1.mp4 ... out_n6.mp4
"""

import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = r"d:\c_files4\Runtime"
S_DIR = os.path.join(BASE_DIR, "Test_Dataset", "s_collisions")
N_DIR = os.path.join(BASE_DIR, "Test_Dataset", "n_normal")
OUT_DIR = os.path.join(BASE_DIR, "Inferred_Outputs")

os.makedirs(OUT_DIR, exist_ok=True)

VEHICLE_CLASS_IDS = {1: 'Bicycle', 2: 'Car', 3: 'Motorcycle', 5: 'Bus', 7: 'Truck'}

def find_colliding_vehicles(boxes):
    n = len(boxes)
    colliding_indices = set()
    if n < 2: return colliding_indices

    for i in range(n):
        for j in range(i + 1, n):
            b1, b2 = boxes[i], boxes[j]
            ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
            ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
            inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)

            b1_w, b1_h = b1[2] - b1[0], b1[3] - b1[1]
            b2_w, b2_h = b2[2] - b2[0], b2[3] - b2[1]
            c1_x, c1_y = (b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2
            c2_x, c2_y = (b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2
            dist = np.hypot(c1_x - c2_x, c1_y - c2_y)
            max_allowed_dist = (max(b1_w, b1_h) + max(b2_w, b2_h)) * 0.45

            if inter_area > 0 or dist < max_allowed_dist:
                colliding_indices.add(i)
                colliding_indices.add(j)

    return colliding_indices

def process_single_video(input_path, output_path, accident_model, vehicle_model):
    print(f"[*] Processing: {os.path.basename(input_path)} -> {os.path.basename(output_path)}")
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100
    
    out_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    frame_count = 0
    consecutive_hits = 0
    max_accident_conf = 0.0
    accident_detected = False
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        
        # Pipeline A: Accident Detection Model
        acc_res = accident_model(frame, conf=0.35, verbose=False)[0]
        pa_hit = len(acc_res.boxes) > 0
        pa_conf = float(acc_res.boxes.conf.max()) if pa_hit else 0.0
        
        if pa_hit:
            consecutive_hits += 1
            if pa_conf > max_accident_conf: max_accident_conf = pa_conf
            if consecutive_hits >= 3: accident_detected = True
        else:
            consecutive_hits = max(0, consecutive_hits - 1)
            
        # Pipeline B: Vehicle Tracking (People / Class 0 Excluded)
        detected_vehicles = []
        veh_res = vehicle_model(frame, conf=0.30, verbose=False)[0]
        if veh_res.boxes:
            for box in veh_res.boxes:
                cls_id = int(box.cls[0].item())
                if cls_id in VEHICLE_CLASS_IDS:
                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0].item())
                    detected_vehicles.append({
                        'box': coords,
                        'cls': VEHICLE_CLASS_IDS[cls_id],
                        'conf': conf
                    })

        all_boxes = [v['box'] for v in detected_vehicles]
        colliding_indices = find_colliding_vehicles(all_boxes) if pa_hit else set()
        num_colliding = len(colliding_indices)

        # Draw Vehicle Bounding Boxes
        for idx, vehicle in enumerate(detected_vehicles):
            x1, y1, x2, y2 = vehicle['box']
            label = vehicle['cls']
            conf = vehicle['conf']

            if idx in colliding_indices:
                color = (0, 0, 255) # RED box
                box_text = f"COLLISION: {label} ({int(conf*100)}%)"
                thickness = 3
            else:
                color = (0, 255, 0) # GREEN box
                box_text = f"{label} ({int(conf*100)}%)"
                thickness = 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            (lbl_w, lbl_h), _ = cv2.getTextSize(box_text, font, 0.5, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - 22)), (x1 + lbl_w + 6, max(22, y1)), color, -1)
            cv2.putText(frame, box_text, (x1 + 3, max(16, y1 - 5)), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Top HUD Banner
        banner_h = 55
        if pa_hit:
            banner_bg = (0, 0, 200) # Dark Red
            banner_text = f"🚨 EMERGENCY: ACCIDENT DETECTED ({pa_conf*100:.1f}%) | VEHICLES COLLIDED: {num_colliding}"
        else:
            banner_bg = (0, 150, 0) # Dark Green
            banner_text = f"NORMAL TRAFFIC ({pa_conf*100:.1f}%) | VEHICLES IN FRAME: {len(detected_vehicles)}"

        cv2.rectangle(frame, (0, 0), (width, banner_h), banner_bg, -1)
        cv2.putText(frame, banner_text, (20, 36), font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        out_writer.write(frame)
        
    cap.release()
    out_writer.release()
    print(f"  [+] Saved: {output_path} ({frame_count} frames)")

def run_batch_inference():
    print("=" * 65)
    print("🚀 BATCH INFERENCE: PROCESSING s1..s8 AND n1..n6 VIDEOS")
    print("=" * 65)
    
    acc_model_path = os.path.join(BASE_DIR, "accident_best.pt")
    accident_model = YOLO(acc_model_path)
    vehicle_model = YOLO("yolo11s.pt")
    
    # 1. Process s1.mp4 ... s8.mp4
    print("\n--- PROCESSING COLLISION VIDEOS (s1..s8) ---")
    for i in range(1, 9):
        in_path = os.path.join(S_DIR, f"s{i}.mp4")
        out_path = os.path.join(OUT_DIR, f"out_s{i}.mp4")
        if os.path.exists(in_path):
            process_single_video(in_path, out_path, accident_model, vehicle_model)

    # 2. Process n1.mp4 ... n6.mp4
    print("\n--- PROCESSING NORMAL TRAFFIC VIDEOS (n1..n6) ---")
    for i in range(1, 7):
        in_path = os.path.join(N_DIR, f"n{i}.mp4")
        out_path = os.path.join(OUT_DIR, f"out_n{i}.mp4")
        if os.path.exists(in_path):
            process_single_video(in_path, out_path, accident_model, vehicle_model)

    print("\n" + "=" * 65)
    print(f"🎉 BATCH INFERENCE COMPLETE! ALL 14 MP4 OUTPUTS SAVED TO:")
    print(f"   {OUT_DIR}")
    print("=" * 65)

if __name__ == "__main__":
    run_batch_inference()
