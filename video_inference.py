"""
video_inference.py
------------------
CLI Video Inference Script for Highway Accident & Vehicle Collision Detection.

Usage:
  python video_inference.py --input sample4.mp4 --output output_sample4.mp4
  python video_inference.py --input s1.mp4

Features:
- Loads Keras Accident Detection Model (model.json + model_weights.h5)
- Loads YOLO Vehicle Tracking Model (yolo11s.pt)
- Applies find_colliding_vehicles() proximity & overlap calculation
- Draws Red Bounding Boxes on colliding vehicles, Green Bounding Boxes on normal vehicles
- Renders Top Status HUD Banner (Accident % & Vehicle Count)
"""

import os
import sys
import argparse
import cv2
import numpy as np

# Import Keras Model & Collision Engine
from keras_accident_pipeline import AccidentDetectionModel, find_colliding_vehicles, preprocess_frame, VEHICLE_CLASS_IDS

def run_video_inference(input_path, output_path=None, json_file="model.json", weights_file="model_weights.h5"):
    print("=" * 65)
    print("🚀 CLI VIDEO INFERENCE: HIGHWAY ACCIDENT DETECTION")
    print("=" * 65)
    
    if not os.path.exists(input_path):
        print(f"[!] Error: Input video file '{input_path}' not found.")
        return

    if output_path is None:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = f"output_{base_name}.mp4"

    print(f"[*] Input Video  : {input_path}")
    print(f"[*] Output Video : {output_path}")

    # 1. Load Keras Model
    accident_model = AccidentDetectionModel(json_file, weights_file)
    
    # 2. Load YOLO Model
    try:
        from ultralytics import YOLO
        yolo_model = YOLO("yolo11s.pt")
        print("[+] Loaded YOLO vehicle tracking model: yolo11s.pt")
    except Exception as e:
        print(f"[!] YOLO Model Load Notice: {e}")
        yolo_model = None

    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    out_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    frame_count = 0
    accident_frames = 0
    max_colliding_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        
        # Step 1: Keras Inference (BGR -> RGB -> 250x250 -> batch)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(rgb_frame, (250, 250))
        pred_label, prob = accident_model.predict_accident(roi[np.newaxis, :, :])
        
        is_accident = (pred_label == "Accident")
        if is_accident:
            accident_frames += 1
            confidence_pct = round(float(prob[0][0]) * 100, 1)
        else:
            confidence_pct = round(float(prob[0][1]) * 100, 1)

        # Step 2: YOLO Vehicle Detection (People / Class 0 Excluded!)
        detected_vehicles = []
        if yolo_model:
            yolo_results = yolo_model(frame, verbose=False, conf=0.35)[0]
            if yolo_results.boxes:
                for box in yolo_results.boxes:
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
        colliding_indices = find_colliding_vehicles(all_boxes) if is_accident else set()
        num_colliding = len(colliding_indices)
        if num_colliding > max_colliding_count:
            max_colliding_count = num_colliding

        # Step 3: Draw Bounding Boxes
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

        # Step 4: Top HUD Status Banner
        banner_h = 55
        if is_accident:
            banner_bg = (0, 0, 200)
            banner_text = f"🚨 EMERGENCY: ACCIDENT DETECTED ({confidence_pct}%) | VEHICLES COLLIDED: {num_colliding}"
        else:
            banner_bg = (0, 150, 0)
            banner_text = f"NORMAL TRAFFIC ({confidence_pct}%) | VEHICLES IN FRAME: {len(detected_vehicles)}"

        cv2.rectangle(frame, (0, 0), (width, banner_h), banner_bg, -1)
        cv2.putText(frame, banner_text, (20, 36), font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        out_writer.write(frame)
        
        if frame_count % 30 == 0 or frame_count == total_frames:
            print(f"  -> Progress: Frame {frame_count}/{total_frames} ({frame_count/max(1,total_frames):.0%})")

    cap.release()
    out_writer.release()

    print("\n" + "=" * 65)
    print("📊 INFERENCE COMPLETE!")
    print("=" * 65)
    print(f"Total Frames Processed : {frame_count}")
    print(f"Accident Frames        : {accident_frames} ({accident_frames/max(1,frame_count):.1%})")
    print(f"Max Vehicles Collided  : {max_colliding_vehicles}")
    print(f"Annotated Output Saved : {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI Video Inference Script for Highway Accident Detection")
    parser.add_argument("--input", "-i", type=str, default="sample4.mp4", help="Path to input video file")
    parser.add_argument("--output", "-o", type=str, default=None, help="Path to save annotated output video file")
    args = parser.parse_args()

    run_video_inference(args.input, args.output)
