"""
keras_accident_pipeline.py
----------------------------
Integrated Keras (.h5 / .json) + YOLO Vehicle Collision Detection Pipeline.

Features from D:\\c_files4\\Accident-Detection-System:
1. AccidentDetectionModel: Loads model.json & model_weights.h5 (with automatic Sequential fallback if json fails).
2. Frame Preprocessing: BGR -> RGB, resize to (250, 250), batch dim -> [1, 250, 250, 3].
3. find_colliding_vehicles(): Bounding box intersection & center proximity distance calculation.
4. Bounding Boxes: Draws RED boxes around collided vehicles and GREEN boxes around normal vehicles.
5. Calculates exact number of collided vehicles and accident/normal confidence percentages.
"""

import os
import sys
import cv2
import numpy as np
import tensorflow as tf

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Vehicle classes mapping for YOLO
VEHICLE_CLASS_IDS = {1: 'Bicycle', 2: 'Car', 3: 'Motorcycle', 5: 'Bus', 7: 'Truck'}

# --- 1. KERAS ACCIDENT DETECTION MODEL CLASS ---
class AccidentDetectionModel(object):
    class_nums = ['Accident', "No Accident"]

    def __init__(self, model_json_file="model.json", model_weights_file="model_weights.h5"):
        loaded = False
        if os.path.exists(model_json_file):
            try:
                from keras.models import model_from_json
                with open(model_json_file, "r") as json_file:
                    loaded_model_json = json_file.read()
                    self.loaded_model = model_from_json(loaded_model_json)
                    loaded = True
            except Exception:
                loaded = False

        if not loaded:
            print("[*] Reconstructing CNN Sequential architecture for model.json...")
            self.loaded_model = tf.keras.models.Sequential([
                tf.keras.layers.BatchNormalization(input_shape=(250, 250, 3)),
                tf.keras.layers.Conv2D(32, 3, activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(64, 3, activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(128, 3, activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(256, 3, activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dense(2, activation='softmax')
            ])

        if os.path.exists(model_weights_file):
            self.loaded_model.load_weights(model_weights_file)
            print(f"[+] Successfully loaded weights from {model_weights_file}")
        else:
            print(f"[!] Warning: {model_weights_file} not found!")

    def predict_accident(self, img):
        """
        Fast prediction using loaded_model(img, training=False).
        Input img shape: [1, 250, 250, 3] RGB
        """
        self.preds = self.loaded_model(img, training=False).numpy()
        class_idx = int(np.argmax(self.preds[0]))
        return AccidentDetectionModel.class_nums[class_idx], self.preds

# --- 2. COLLIDING VEHICLES PROXIMITY & OVERLAP CALCULATOR ---
def find_colliding_vehicles(boxes):
    """
    Given a list of bounding boxes [(x1, y1, x2, y2), ...],
    returns a set of indices of vehicles involved in a collision based on 
    bounding box intersection area OR center proximity distance.
    """
    n = len(boxes)
    colliding_indices = set()
    if n < 2:
        return colliding_indices

    for i in range(n):
        for j in range(i + 1, n):
            b1 = boxes[i]
            b2 = boxes[j]

            # Bounding box intersection area
            ix1 = max(b1[0], b2[0])
            iy1 = max(b1[1], b2[1])
            ix2 = min(b1[2], b2[2])
            iy2 = min(b1[3], b2[3])

            inter_w = max(0, ix2 - ix1)
            inter_h = max(0, iy2 - iy1)
            inter_area = inter_w * inter_h

            # Proximity check using center distances
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

# --- 3. VIDEO PROCESSING PIPELINE ---
def process_video_pipeline(video_path, output_path="output_keras_detection.mp4", json_file="model.json", weights_file="model_weights.h5"):
    print("=" * 65)
    print("🚀 ACCIDENT DETECTION & VEHICLE COLLISION PIPELINE (KERAS + YOLO)")
    print("=" * 65)
    
    if not os.path.exists(video_path):
        print(f"[!] Error: Video file '{video_path}' does not exist.")
        return

    # Load Keras Accident Model
    accident_model = AccidentDetectionModel(json_file, weights_file)
    
    # Load YOLO Vehicle Tracking Model
    try:
        from ultralytics import YOLO
        yolo_model = YOLO("yolo11s.pt")
        print("[+] Loaded YOLO vehicle tracking model: yolo11s.pt")
    except Exception as e:
        print(f"[!] Warning: YOLO model loading issue: {e}")
        yolo_model = None

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    out_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    frame_count = 0
    accident_frames = 0
    max_colliding_vehicles = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1

        # Step 1: Preprocess Frame for Keras (BGR -> RGB -> 250x250 -> batch dim)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(rgb_frame, (250, 250))
        roi_batch = roi[np.newaxis, :, :] # shape: [1, 250, 250, 3]

        # Keras Inference
        pred_label, prob = accident_model.predict_accident(roi_batch)
        is_accident = (pred_label == "Accident")

        if is_accident:
            accident_frames += 1
            confidence_pct = round(float(prob[0][0]) * 100, 1)
        else:
            confidence_pct = round(float(prob[0][1]) * 100, 1)

        # Step 2: Vehicle Object Detection using YOLO
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
        
        # Step 3: Find colliding vehicle indices if accident detected
        colliding_indices = find_colliding_vehicles(all_boxes) if is_accident else set()
        num_colliding = len(colliding_indices)
        if num_colliding > max_colliding_vehicles:
            max_colliding_vehicles = num_colliding

        # Step 4: Draw Vehicle Bounding Boxes (RED = Collision, GREEN = Normal)
        for idx, vehicle in enumerate(detected_vehicles):
            x1, y1, x2, y2 = vehicle['box']
            label = vehicle['cls']
            conf = vehicle['conf']

            if idx in colliding_indices:
                color = (0, 0, 255) # RED box for colliding vehicle
                box_text = f"COLLISION: {label} ({int(conf*100)}%)"
                thickness = 3
            else:
                color = (0, 255, 0) # GREEN box for normal vehicle
                box_text = f"{label} ({int(conf*100)}%)"
                thickness = 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Label background banner
            (lbl_w, lbl_h), _ = cv2.getTextSize(box_text, font, 0.5, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - 22)), (x1 + lbl_w + 6, max(22, y1)), color, -1)
            cv2.putText(frame, box_text, (x1 + 3, max(16, y1 - 5)), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Step 5: HUD Banner at Top of Screen
        if is_accident:
            banner_bg = (0, 0, 200) # Dark Red Banner
            banner_text = f"🚨 EMERGENCY: ACCIDENT DETECTED ({confidence_pct}%) | VEHICLES COLLIDED: {num_colliding}"
        else:
            banner_bg = (0, 150, 0) # Dark Green Banner
            banner_text = f"NORMAL TRAFFIC ({confidence_pct}%) | VEHICLES IN FRAME: {len(detected_vehicles)}"

        cv2.rectangle(frame, (0, 0), (width, 50), banner_bg, -1)
        cv2.putText(frame, banner_text, (20, 33), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        out_writer.write(frame)

    cap.release()
    out_writer.release()

    print("\n" + "=" * 65)
    print("📊 VIDEO INFERENCE SUMMARY")
    print("=" * 65)
    print(f"Total Frames Processed : {frame_count}")
    print(f"Accident Frames        : {accident_frames} ({accident_frames/max(1,frame_count):.1%})")
    print(f"Max Vehicles Collided  : {max_colliding_vehicles}")
    print(f"Output Saved To        : {output_path}")

if __name__ == "__main__":
    test_video = "sample4.mp4"
    if os.path.exists(test_video):
        process_video_pipeline(test_video, "output_keras_detection.mp4")
