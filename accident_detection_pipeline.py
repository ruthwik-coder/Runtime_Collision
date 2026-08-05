import os, subprocess, cv2, random
import supervision as sv
from ultralytics import YOLO
from collections import deque
from IPython.display import Video, Image, display

# --- UPDATE FILENAMES HERE ---
MODEL_PATH = "/content/accident_best.pt"
INPUT_VIDEO = "/content/sample4.mp4"  # Input video filename
OUTPUT_VIDEO = "/content/output_detected.mp4"
WEB_VIDEO = "/content/web_output.mp4"

# Load models
print("⏳ Loading models...")
accident_model = YOLO(MODEL_PATH)
# Using 's' model for vehicles to keep Colab T4 running fast
vehicle_model = YOLO("yolo11s.pt")

# Auto-convert iPhone .mov to H.264 .mp4 if needed
if not os.path.exists(INPUT_VIDEO) and os.path.exists(INPUT_VIDEO.replace(".mp4", ".mov")):
    mov_file = INPUT_VIDEO.replace(".mp4", ".mov")
    print(f"⚙️ Converting {mov_file} to MP4...")
    subprocess.run(["ffmpeg", "-y", "-i", mov_file, "-vcodec", "libx264", "-acodec", "aac", INPUT_VIDEO],
                   stdout=subprocess.DEVNULL, stderr=None)

info = sv.VideoInfo.from_video_path(INPUT_VIDEO)
fps = int(info.fps) if info.fps > 0 else 30

# --- STATE VARIABLES ---
accident_detected = False
accident_conf = 0.0
confirmed_frame_idx = -1
involved_vehicles = set()
accident_box_coords = []
confirmed_veh_boxes = [] 
total_vehicles_involved = 0 # To store the count for the text report

# Buffers for 5-second telemetry video (2.5s before, 2.5s after)
frame_buffer = deque(maxlen=int(fps * 2.5))
telemetry_frames = []
tel_capture_remaining = 0

# --- HELPER FUNCTIONS ---
def is_center_inside(acc_box, veh_box):
    cx = (veh_box[0] + veh_box[2]) / 2
    cy = (veh_box[1] + veh_box[3]) / 2
    return acc_box[0] <= cx <= acc_box[2] and acc_box[1] <= cy <= acc_box[3]

def assess_severity(vehicles):
    if any(v in ["truck", "bus"] for v in vehicles): return "Severe/Critical"
    if "motorcycle" in vehicles: return "Critical (Vulnerable User)"
    if len(vehicles) >= 3: return "Severe"
    if "car" in vehicles: return "Moderate"
    return "Minor"

def get_collision_type(acc_box, veh_boxes):
    if len(acc_box) == 0: return "Unknown / No Primary Accident Box"
    if len(veh_boxes) == 0: return "Unknown / No Vehicles Detected"
    if len(veh_boxes) == 1: return "Single-vehicle (Barrier/Object hit)"

    overlaps = 0
    for i in range(len(veh_boxes)):
        for j in range(i+1, len(veh_boxes)):
            xA = max(veh_boxes[i][0], veh_boxes[j][0])
            yA = max(veh_boxes[i][1], veh_boxes[j][1])
            xB = min(veh_boxes[i][2], veh_boxes[j][2])
            yB = min(veh_boxes[i][3], veh_boxes[j][3])
            if xB - xA > 0 and yB - yA > 0: overlaps += 1

    if overlaps > 0: return "Rear-end / Head-on Collision"
    return "Multi-vehicle Pileup / Side-swipe"

# --- FRAME PROCESSING CALLBACK ---
def process_frame(frame, idx):
    global accident_detected, accident_conf, confirmed_frame_idx, tel_capture_remaining, confirmed_veh_boxes, accident_box_coords, total_vehicles_involved

    # 1. Accident Detection (Pipeline A)
    acc_res = accident_model(frame, conf=0.35, verbose=False)[0]
    pa_hit = len(acc_res.boxes) > 0
    pa_conf = float(acc_res.boxes.conf.max()) if pa_hit else 0.0
    pa_boxes = acc_res.boxes.xyxy.cpu().numpy() if pa_hit else []

    # 2. Vehicle Tracking (Pipeline B)
    veh_res = vehicle_model(frame, conf=0.30, verbose=False)[0]
    veh_boxes = veh_res.boxes.xyxy.cpu().numpy()
    veh_classes = veh_res.boxes.cls.cpu().numpy()

    # Draw vehicles (Green)
    for v_box in veh_boxes:
        x1, y1, x2, y2 = map(int, v_box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # If accident is detected and we haven't triggered the alert yet
    if pa_hit and not accident_detected:
        accident_detected = True
        accident_conf = pa_conf
        confirmed_frame_idx = idx
        tel_capture_remaining = int(fps * 2.5) # Capture next 2.5 secs

        # --- Save Snapshot (only accident box) ---
        snapshot_frame = frame.copy() 
        for box in pa_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(snapshot_frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
            cv2.putText(snapshot_frame, f"ACCIDENT {pa_conf:.0%}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        snapshot_path = "/content/accident_snapshot.jpg"
        cv2.imwrite(snapshot_path, snapshot_frame)

        # 3. Extract Deliverables (4, 5, 6, 7)
        if len(pa_boxes) > 0 and acc_res.boxes.conf is not None and len(acc_res.boxes.conf) > 0:
            best_acc_idx = acc_res.boxes.conf.argmax()
            accident_box_coords = pa_boxes[best_acc_idx]
        else:
            accident_box_coords = [] 

        confirmed_veh_boxes = veh_boxes 

        # Count vehicles involved for the text report
        total_vehicles_involved = 0
        for i, veh_box in enumerate(veh_boxes):
            if len(accident_box_coords) > 0 and is_center_inside(accident_box_coords, veh_box):
                total_vehicles_involved += 1 # Increment count
                cid = int(veh_classes[i])
                v_type = vehicle_model.names[cid]
                involved_vehicles.add(v_type)

    # Draw Accident (Red) on the main frame 
    for box in pa_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
        cv2.putText(frame, f"ACCIDENT {pa_conf:.0%}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Handle telemetry video buffer
    if accident_detected and tel_capture_remaining > 0:
        telemetry_frames.append(frame)
        tel_capture_remaining -= 1
        cv2.circle(frame, (info.width - 30, 30), 10, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (info.width - 70, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if accident_detected:
        cv2.putText(frame, "🚨 ACCIDENT DETECTED", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    frame_buffer.append(frame)
    return frame

# --- EXECUTE PIPELINE ---
print("🚀 Processing video...")
sv.process_video(source_path=INPUT_VIDEO, target_path=OUTPUT_VIDEO, callback=process_frame)

# Save Telemetry Video
if accident_detected and len(telemetry_frames) > 0:
    telemetry_path = "/content/telemetry_clip.mp4"
    h, w = telemetry_frames[0].shape[:2]
    out_vid = cv2.VideoWriter(telemetry_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for t_frame in frame_buffer:
        out_vid.write(t_frame)
    for t_frame in telemetry_frames:
        out_vid.write(t_frame)
    out_vid.release()
    subprocess.run(["ffmpeg", "-y", "-i", telemetry_path, "-vcodec", "libx264", "-acodec", "aac", "/content/web_telemetry.mp4"],
                   stdout=subprocess.DEVNULL, stderr=None)

# Transcode main output for browser preview
subprocess.run(["ffmpeg", "-y", "-i", OUTPUT_VIDEO, "-vcodec", "libx264", "-acodec", "aac", WEB_VIDEO],
               stdout=subprocess.DEVNULL, stderr=None)

# --- FINAL DISPATCH REPORT (TEXT OUTPUT) ---
print("\n" + "="*50)
if accident_detected:
    print("🚨 EMERGENCY DISPATCH REPORT GENERATED 🚨")
    print("="*50)
    print(f"1️⃣ Accident Flag       : TRUE")
    print(f"   Confidence Score    : {accident_conf:.2%}")
    
    sim_lat = round(random.uniform(8.0, 37.0), 4)
    sim_lon = round(random.uniform(68.0, 97.0), 4)
    print(f"2️⃣ Map Location        : Lat {sim_lat}, Lon {sim_lon}")
    print(f"   Camera ID           : NH-44-CAM-019")
    
    print(f"3️⃣ Evidence Captured   : /content/accident_snapshot.jpg")
    if len(telemetry_frames) > 0: print(f"   Telemetry Video     : /content/web_telemetry.mp4 (5 sec)")
    
    collision_type = get_collision_type(accident_box_coords, confirmed_veh_boxes)
    print(f"4️⃣ Collision Type      : {collision_type}")
    
    # Output the exact count of vehicles involved
    print(f"5️⃣ Vehicles Involved   : {total_vehicles_involved}")
    
    print(f"6️⃣ Vehicle Class(es)   : {', '.join(involved_vehicles) if involved_vehicles else 'Unknown'}")
    
    severity = assess_severity(involved_vehicles)
    print(f"7️⃣ Impact Severity     : {severity}")
    print("="*50)
else:
    print("✅ No accident detected in footage.")
    print("="*50)

# Display Main Video
display(Video(WEB_VIDEO, embed=True, width=720))

# Display Snapshot and Telemetry
if accident_detected:
    print("\n📸 Accident Snapshot Captured:")
    display(Image(filename="/content/accident_snapshot.jpg", width=640))
    if os.path.exists("/content/web_telemetry.mp4"):
        print("\n🎬 5-Second Telemetry Video Clip:")
        display(Video("/content/web_telemetry.mp4", embed=True, width=640))
