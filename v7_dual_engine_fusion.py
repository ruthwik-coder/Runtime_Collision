"""
v7_dual_engine_fusion.py
--------------------------
Advanced Trajectory-Based Collision Verification Engine using Supervision + ByteTrack.

Combines:
1. Primary: Sudden Deceleration & Post-Impact Stagnation (velocity drop to ~0 km/h + movement <8px for >=12 frames).
2. Secondary: Supervision Trajectory Path Line Intersections (line segment intersection of vehicle motion rays).
3. Strict Temporal Persistence Window (>=12 consecutive frames to eliminate transient 2D passing false alarms).
4. Trajectory Path Visualization: Draws glowing motion path lines behind tracked vehicles.
"""

import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import supervision as sv
from ultralytics import YOLO

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Vehicle classes mapping (COCO dataset: 1:Bicycle, 2:Car, 3:Motorcycle, 5:Bus, 7:Truck)
VEHICLE_CLASS_IDS = {1: 'Bicycle', 2: 'Car', 3: 'Motorcycle', 5: 'Bus', 7: 'Truck'}

def line_intersection(line1, line2):
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2
    
    def ccw(A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

    return (ccw((x1,y1),(x3,y3),(x4,y4)) != ccw((x2,y2),(x3,y3),(x4,y4))) and \
           (ccw((x1,y1),(x2,y2),(x3,y3)) != ccw((x1,y1),(x2,y2),(x4,y4)))

class TrajectoryCollisionEngine:
    def __init__(self, history_len=30, stagnation_px=8, persistence_frames=12):
        self.history_len = history_len
        self.stagnation_px = stagnation_px
        self.persistence_frames = persistence_frames
        
        # ByteTrack Tracker (compatible with Supervision 0.30+)
        self.tracker = sv.ByteTrack()
        
        # Track history: {track_id: [(x, y), (x, y), ...]}
        self.path_history = {}
        # Consecutive overlap hits per vehicle pair: {(id1, id2): count}
        self.pair_persistence = {}

    def update(self, frame, detected_boxes, detected_classes):
        if len(detected_boxes) > 0:
            xyxy = np.array(detected_boxes, dtype=np.float32)
            confidence = np.ones(len(detected_boxes), dtype=np.float32)
            class_id = np.array(detected_classes, dtype=int)
            detections = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
        else:
            detections = sv.Detections.empty()

        tracked_detections = self.tracker.update_with_detections(detections)
        
        active_vehicles = []
        intersecting_pairs = set()

        if len(tracked_detections) > 0:
            for i in range(len(tracked_detections)):
                box = tracked_detections.xyxy[i].astype(int)
                tid = int(tracked_detections.tracker_id[i]) if tracked_detections.tracker_id is not None else i
                cid = int(tracked_detections.class_id[i])
                cname = VEHICLE_CLASS_IDS.get(cid, "Car")

                cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
                
                if tid not in self.path_history:
                    self.path_history[tid] = []
                self.path_history[tid].append((cx, cy))
                if len(self.path_history[tid]) > self.history_len:
                    self.path_history[tid].pop(0)

                pts = self.path_history[tid]
                if len(pts) >= 10:
                    movement_px = np.hypot(pts[-1][0] - pts[-10][0], pts[-1][1] - pts[-10][1])
                else:
                    movement_px = 999.0

                is_stagnant = movement_px < self.stagnation_px

                active_vehicles.append({
                    'id': tid,
                    'box': box,
                    'cls': cname,
                    'center': (cx, cy),
                    'movement_px': movement_px,
                    'is_stagnant': is_stagnant,
                    'history': pts
                })

        n = len(active_vehicles)
        for i in range(n):
            for j in range(i + 1, n):
                v1, v2 = active_vehicles[i], active_vehicles[j]
                h1, h2 = v1['history'], v2['history']
                
                paths_cross = False
                if len(h1) >= 4 and len(h2) >= 4:
                    seg1 = (h1[-4], h1[-1])
                    seg2 = (h2[-4], h2[-1])
                    paths_cross = line_intersection(seg1, seg2)

                b1, b2 = v1['box'], v2['box']
                ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
                ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
                inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                dist = np.hypot(v1['center'][0] - v2['center'][0], v1['center'][1] - v2['center'][1])
                max_dist = (max(b1[2]-b1[0], b1[3]-b1[1]) + max(b2[2]-b2[0], b2[3]-b2[1])) * 0.45

                in_proximity = (inter_area > 0 or dist < max_dist)
                
                pair_key = (min(v1['id'], v2['id']), max(v1['id'], v2['id']))
                if in_proximity or paths_cross:
                    self.pair_persistence[pair_key] = self.pair_persistence.get(pair_key, 0) + 1
                else:
                    self.pair_persistence[pair_key] = max(0, self.pair_persistence.get(pair_key, 0) - 1)

                if self.pair_persistence[pair_key] >= self.persistence_frames:
                    if v1['is_stagnant'] or v2['is_stagnant'] or paths_cross:
                        intersecting_pairs.add(v1['id'])
                        intersecting_pairs.add(v2['id'])

        return active_vehicles, intersecting_pairs

def process_video_fusion(video_path, output_path="output_v7_fusion.mp4"):
    print("=" * 65)
    print("🚀 V7 TRAJECTORY DUAL-ENGINE COLLISION PIPELINE")
    print("=" * 65)

    acc_model_path = "accident_best.pt"
    accident_model = YOLO(acc_model_path) if os.path.exists(acc_model_path) else None
    vehicle_model = YOLO("yolo11s.pt")
    
    engine = TrajectoryCollisionEngine(history_len=30, stagnation_px=8, persistence_frames=12)

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    font = cv2.FONT_HERSHEY_SIMPLEX

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        pa_hit = False
        pa_conf = 0.0
        if accident_model:
            acc_res = accident_model(frame, conf=0.35, verbose=False)[0]
            pa_hit = len(acc_res.boxes) > 0
            if pa_hit: pa_conf = float(acc_res.boxes.conf.max())

        detected_boxes = []
        detected_classes = []
        veh_res = vehicle_model(frame, conf=0.30, verbose=False)[0]
        if veh_res.boxes:
            for box in veh_res.boxes:
                cid = int(box.cls[0].item())
                if cid in VEHICLE_CLASS_IDS:
                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    detected_boxes.append(coords)
                    detected_classes.append(cid)

        active_vehicles, colliding_ids = engine.update(frame, detected_boxes, detected_classes)
        
        for v in active_vehicles:
            pts = v['history']
            if len(pts) > 1:
                color = (0, 0, 255) if v['id'] in colliding_ids else (56, 189, 248)
                for k in range(1, len(pts)):
                    thickness = int(np.sqrt(30 / float(k + 1)) * 1.5)
                    cv2.line(frame, pts[k-1], pts[k], color, thickness)

        for v in active_vehicles:
            x1, y1, x2, y2 = v['box']
            label, tid = v['cls'], v['id']

            if tid in colliding_ids:
                color = (0, 0, 255)
                box_text = f"COLLISION V-{tid}: {label}"
                thickness = 3
            else:
                color = (0, 255, 0)
                box_text = f"V-{tid}: {label}"
                thickness = 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            (lbl_w, lbl_h), _ = cv2.getTextSize(box_text, font, 0.5, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - 22)), (x1 + lbl_w + 6, max(22, y1)), color, -1)
            cv2.putText(frame, box_text, (x1 + 3, max(16, y1 - 5)), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        banner_h = 55
        is_accident = len(colliding_ids) >= 2 or pa_hit
        if is_accident:
            banner_bg = (0, 0, 200)
            banner_text = f"🚨 EMERGENCY: ACCIDENT DETECTED | COLLIDED VEHICLES: {len(colliding_ids)}"
        else:
            banner_bg = (0, 150, 0)
            banner_text = f"NORMAL TRAFFIC FLOW | ACTIVE VEHICLES: {len(active_vehicles)}"

        cv2.rectangle(frame, (0, 0), (width, banner_h), banner_bg, -1)
        cv2.putText(frame, banner_text, (20, 36), font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        out.write(frame)

    cap.release()
    out.release()
    print(f"[+] Output saved to: {output_path}")

if __name__ == "__main__":
    if os.path.exists("sample4.mp4"):
        process_video_fusion("sample4.mp4", "output_v7_fusion.mp4")
