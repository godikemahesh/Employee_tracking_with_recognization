import cv2
import torch
import numpy as np
import json
import os
import time
from datetime import datetime
from ultralytics import YOLO
import torchreid
from torchvision import transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
import urllib.parse
import sys
import threading
import math

# -------- SETTINGS --------
ZONE_FILE = "zones.json"
EMBEDDING_FILE = "embeddings.npy"
REPORT_FILE = "report.json"

INTERVAL_SECONDS = 10
FRAMES_PER_CYCLE = 10

# Weighted scoring thresholds
MATCH_THRESHOLD_COMBINED = 0.55  # When face + body available
MATCH_THRESHOLD_BODY_ONLY = 0.68  # When only body available
MIN_BREAK_SECONDS = 30  # Only count as break if absent longer than this
FACE_WEIGHT = 0.6
BODY_WEIGHT = 0.4

# Auto-update: max daily body embeddings to capture automatically
MAX_DAILY_BODY_UPDATE = 7
FACE_CONFIDENCE_FOR_AUTO_UPDATE = 0.5  # Minimum face score to trigger body auto-update

USERNAME = "admin1"
PASSWORD = "admin@123"
NVR_IP = "192.168.88.2"
RTSP_PORT = 554
STREAM_TYPE = 0
MAX_CAMERAS = 8
# --------------------------

# -------- LOAD DATA --------
with open(ZONE_FILE, "r") as f:
    zones = json.load(f)

employee_embeddings = np.load(EMBEDDING_FILE, allow_pickle=True).item()

# Build list of all employee names (unique across all cameras)
all_employee_names = list(zones.keys())

employee_states = {}
for name in all_employee_names:
    employee_states[name] = {
        "state": "ABSENT",
        "last_change": time.time(),
        "presence_counter": 0,
        "absence_counter": 0
    }

today_date = datetime.now().strftime("%d-%m-%Y")

# Load existing report file (preserves all previous dates)
if os.path.exists(REPORT_FILE):
    with open(REPORT_FILE, "r") as f:
        try:
            loaded_data = json.load(f)
        except:
            loaded_data = {}

    # Migrate old format {"date": "...", "employees": {...}} to new format
    if "date" in loaded_data and "employees" in loaded_data:
        old_date = loaded_data["date"]
        all_report_data = {old_date: loaded_data["employees"]}
        print(f"Migrated old report format for date {old_date}.")
    else:
        all_report_data = loaded_data
else:
    all_report_data = {}

# Initialize today's data (resume if already exists, else fresh start)
if today_date not in all_report_data:
    all_report_data[today_date] = {}
# -------- EMPLOYEE COLORS --------
np.random.seed(42)  # same colors every run

employee_colors = {}
for name in all_employee_names:
    color = tuple(np.random.randint(50, 255, size=3).tolist())
    employee_colors[name] = color

for name in all_employee_names:
    if name not in all_report_data[today_date]:
        all_report_data[today_date][name] = {
            "in_seat_seconds": 0,
            "out_seat_seconds": 0,
            "total_breaks": 0,
            "current_state": "ABSENT"
        }
    elif "in_seat_seconds" not in all_report_data[today_date][name]:
        all_report_data[today_date][name] = {
            "in_seat_seconds": 0,
            "out_seat_seconds": 0,
            "total_breaks": all_report_data[today_date][name].get("total_breaks", 0),
            "current_state": all_report_data[today_date][name].get("current_state", "ABSENT")
        }

# Shortcut reference to today's data for easy access
report_data = all_report_data[today_date]

# -------- LOAD MODELS --------
detector = YOLO("person_detection.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Body ReID model (OSNet)
reid_model = torchreid.models.build_model(
    name='osnet_x1_0',
    num_classes=1000,
    pretrained=True
)
reid_model.eval()
reid_model.to(device)
 
body_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Face Detection + Recognition models
face_detector = MTCNN(keep_all=False, device=device, min_face_size=40)
face_recognizer = InceptionResnetV1(pretrained='vggface2').eval().to(device)

print("All models loaded (YOLO + OSNet + MTCNN + FaceNet).")

# Daily body cache: auto-updated body embeddings for today
# Format: {name: {cam_key: [body_embeddings]}}
daily_body_cache = {}

# -------- HELPER FUNCTIONS --------
def get_zones_for_camera(cam_number):
    cam_key = str(cam_number)
    cam_zones = {}
    for name, data in zones.items():
        if "cameras" in data and cam_key in data["cameras"]:
            cam_zones[name] = data["cameras"][cam_key]
    return cam_zones

def get_cameras_for_employee(name):
    if name not in zones or "cameras" not in zones[name]:
        return []
    return list(zones[name]["cameras"].keys())

def get_stored_embeddings(name, cam_key):
    """Get face and body embeddings for an employee on a camera.
    Handles both old format (flat list) and new format (dict with face/body)."""
    if name not in employee_embeddings or cam_key not in employee_embeddings[name]:
        return [], []

    data = employee_embeddings[name][cam_key]

    # Old format: flat list of body embeddings
    if isinstance(data, list):
        return [], data

    # New format: dict with face and body
    face_embs = data.get("face", [])
    body_embs = data.get("body", [])
    return face_embs, body_embs

def format_time(seconds):
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    return f"{hours}h {minutes}m"

def save_report():
    save_data = {}
    for date_key, employees in all_report_data.items():
        save_data[date_key] = {}
        for name, data in employees.items():
            if "in_seat_seconds" in data:
                save_data[date_key][name] = {
                    "in_seat_seconds": data["in_seat_seconds"],
                    "out_seat_seconds": data["out_seat_seconds"],
                    "in_seat_time": format_time(data["in_seat_seconds"]),
                    "out_seat_time": format_time(data["out_seat_seconds"]),
                    "total_breaks": data["total_breaks"],
                    "current_state": data["current_state"]
                }
            else:
                save_data[date_key][name] = data

    with open(REPORT_FILE, "w") as f:
        json.dump(save_data, f, indent=4)

# -------- EMBEDDING EXTRACTION --------
def extract_body_embedding(img):
    img = body_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = reid_model(img)
    embedding = embedding.cpu().numpy().flatten()
    embedding = embedding / np.linalg.norm(embedding)
    return embedding

def extract_face_embedding(person_crop):
    """Detect face in person crop and extract face embedding.
    Returns face_embedding or None if no face found."""
    try:
        rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        boxes, probs = face_detector.detect(rgb_crop)

        if boxes is None or len(boxes) == 0:
            return None

        best_idx = probs.argmax()
        box = boxes[best_idx].astype(int)
        x1, y1, x2, y2 = box

        h, w = person_crop.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 - x1 < 10 or y2 - y1 < 10:
            return None

        face_crop = rgb_crop[y1:y2, x1:x2]
        face_crop = cv2.resize(face_crop, (160, 160))

        face_tensor = torch.from_numpy(face_crop).permute(2, 0, 1).float() / 255.0
        face_tensor = face_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = face_recognizer(face_tensor)

        embedding = embedding.cpu().numpy().flatten()
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    except Exception:
        return None

def cosine_similarity(a, b):
    return np.dot(a, b)
def is_new_embedding(new_emb, existing_embs, threshold=0.95):
    """
    Check if the new embedding is sufficiently different from existing ones.
    If similarity > threshold, it means it is almost the same embedding.
    """
    for emb in existing_embs:
        sim = cosine_similarity(new_emb, emb)
        if sim > threshold:
            return False
    return True
def compute_match_score(body_emb, face_emb, stored_face_embs, stored_body_embs, daily_body_embs=None):
    """Compute weighted match score using face + body embeddings.
    Returns (final_score, face_score, body_score, used_face)."""

    # Combine registered body embeddings with daily auto-updated ones
    all_body_embs = list(stored_body_embs)
    if daily_body_embs:
        all_body_embs.extend(daily_body_embs)

    # Body score
    if all_body_embs:
        body_sims = [cosine_similarity(body_emb, e) for e in all_body_embs]
        body_sims.sort(reverse=True)
        gap = 0
        if len(body_sims) > 1:
            gap = body_sims[0] - body_sims[1]
        body_score=max(body_sims)        #np.mean(body_sims[:2])  # average of top 2 matches
    else:
        body_score = 0.0

    # Face score
    face_score = 0.0
    used_face = False

    if face_emb is not None and stored_face_embs:
        face_sims = [cosine_similarity(face_emb, e) for e in stored_face_embs]
        face_sims.sort(reverse=True)
        face_score = max(face_sims)        #np.mean(face_sims[:2])  # average of best 2
        used_face = True

    # Weighted combination
    if used_face:
        final_score = FACE_WEIGHT * face_score + BODY_WEIGHT * body_score
        threshold = MATCH_THRESHOLD_COMBINED
    else:
        final_score = body_score
        threshold = MATCH_THRESHOLD_BODY_ONLY

    return final_score, face_score, body_score, used_face, threshold, gap

def inside_zone(box, zone):
    x1,y1,x2,y2 = box
    zx1,zy1,zx2,zy2 = zone
    cx = (x1+x2)//2
    cy = (y1+y2)//2
    return zx1 < cx < zx2 and zy1 < cy < zy2

def update_state(name, detected):
    state_data = employee_states[name]
    current_time = time.time()

    if detected:
        state_data["presence_counter"] += 1
        state_data["absence_counter"] = 0

        if state_data["presence_counter"] >= 2 and state_data["state"] == "ABSENT":
            duration = current_time - state_data["last_change"]

            report_data[name]["out_seat_seconds"] += duration
            report_data[name]["current_state"] = "PRESENT"

            # Only count as a break if absent > 25 seconds
            if duration > MIN_BREAK_SECONDS:
                report_data[name]["total_breaks"] += 1
                print(f"{name} CONFIRMED PRESENT (break counted: {int(duration)}s)")
            else:
                print(f"{name} CONFIRMED PRESENT (brief absence ignored: {int(duration)}s)")

            state_data["state"] = "PRESENT"
            state_data["last_change"] = current_time

    else:
        state_data["absence_counter"] += 1
        state_data["presence_counter"] = 0

        if state_data["absence_counter"] >= 2 and state_data["state"] == "PRESENT":
            duration = current_time - state_data["last_change"]

            report_data[name]["in_seat_seconds"] += duration
            report_data[name]["current_state"] = "ABSENT"

            state_data["state"] = "ABSENT"
            state_data["last_change"] = current_time

            print(f"{name} CONFIRMED ABSENT")

def save_report_with_live_time():
    
    """Save report with live running time included (not just state-change snapshots)."""
    with open("time.json", "w") as f:
        f.write(datetime.now().strftime("%H:%M:%S"))
    current_time = time.time()

    for name, state_data in employee_states.items():
        ongoing_duration = current_time - state_data["last_change"]

        if state_data["state"] == "PRESENT":
            report_data[name]["current_state"] = "PRESENT"
            # Temporarily add ongoing duration for saving
            report_data[name]["_live_in_seat"] = report_data[name]["in_seat_seconds"] + ongoing_duration
            report_data[name]["_live_out_seat"] = report_data[name]["out_seat_seconds"]
        else:
            report_data[name]["current_state"] = "ABSENT"
            report_data[name]["_live_in_seat"] = report_data[name]["in_seat_seconds"]
            report_data[name]["_live_out_seat"] = report_data[name]["out_seat_seconds"] + ongoing_duration

    # Save with live times
    save_data = {}
    for date_key, employees in all_report_data.items():
        save_data[date_key] = {}
        for name, data in employees.items():
            if "in_seat_seconds" in data:
                # Use live times for today, raw times for other dates
                in_secs = data.get("_live_in_seat", data["in_seat_seconds"])
                out_secs = data.get("_live_out_seat", data["out_seat_seconds"])
                save_data[date_key][name] = {
                    "in_seat_seconds": in_secs,
                    "out_seat_seconds": out_secs,
                    "in_seat_time": format_time(in_secs),
                    "out_seat_time": format_time(out_secs),
                    "total_breaks": data["total_breaks"],
                    "current_state": data["current_state"]
                }
            else:
                save_data[date_key][name] = data

    with open(REPORT_FILE, "w") as f:
        json.dump(save_data, f, indent=4)

    # Clean up temp keys
    for name in all_employee_names:
        report_data[name].pop("_live_in_seat", None)
        report_data[name].pop("_live_out_seat", None)

# -------- MULTI-CAMERA MAIN --------
def main(active_cameras):
    num_cameras = len(active_cameras)

    base_frames = FRAMES_PER_CYCLE // num_cameras
    extra_frames = FRAMES_PER_CYCLE % num_cameras

    frames_per_cam = {}
    for i, cam in enumerate(active_cameras):
        frames_per_cam[cam] = base_frames + (1 if i < extra_frames else 0)

    print(f"\n--- Multi-Camera Monitoring (Face + Body) ---")
    print(f"Active cameras: {active_cameras}")
    print(f"Total frames per cycle: {FRAMES_PER_CYCLE}")
    print(f"Scoring: {FACE_WEIGHT:.0%} Face + {BODY_WEIGHT:.0%} Body")
    for cam, nf in frames_per_cam.items():
        print(f"  Camera {cam}: {nf} frames")

    # Build per-camera zone maps
    cam_zones = {}
    for cam in active_cameras:
        cz = get_zones_for_camera(cam)
        cam_zones[cam] = cz
        if cz:
            print(f"  Camera {cam} employees: {', '.join(cz.keys())}")
        else:
            print(f"  Camera {cam}: No zones assigned!")

    # Open all camera streams
    caps = {}
    encoded_pass = urllib.parse.quote(PASSWORD)
    for cam in active_cameras:
        rtsp_url = (
            f"rtsp://{USERNAME}:{encoded_pass}@{NVR_IP}:{RTSP_PORT}"
            f"/cam/realmonitor?channel={cam}&subtype={STREAM_TYPE}"
        )
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            print(f"Failed to open Camera {cam}. Skipping.")
            continue
        caps[cam] = cap
        print(f"Camera {cam} stream opened.")

    if not caps:
        print("No cameras could be opened.")
        return
    #--------check this if they reset--------
    # Reset timers to NOW (not program start time)
    start_now = time.time()
    for name in all_employee_names:
        employee_states[name]["last_change"] = start_now
        employee_states[name]["presence_counter"] = 0
        employee_states[name]["absence_counter"] = 0

    lock = threading.Lock()
    shared_frames = {cam: {"frame": None} for cam in caps.keys()}
    last_detections = {cam: [] for cam in caps.keys()}
    running = True

    # -------- FRAME GRABBERS --------
    def frame_grabber(cam, cap):
        while running:
            try:
                ret, frame = cap.read()
                if not running:
                    break
                if ret:
                    with lock:
                        shared_frames[cam]["frame"] = frame.copy()
            except cv2.error:
                break  # Suppress "Unknown C++ exception" when cap.release() is called
            except Exception:
                break

    grabber_threads = []
    for cam, cap in caps.items():
        t = threading.Thread(target=frame_grabber, args=(cam, cap), daemon=True)
        t.start()
        grabber_threads.append(t)

    # -------- PROCESSOR --------
    def processor():
        nonlocal last_detections

        while running:
            start_time = time.time()

            cycle_results = {name: [] for name in all_employee_names}
            display_detections = {cam: {} for cam in caps.keys()}

            for cam in caps.keys():
                cam_frame_count = frames_per_cam.get(cam, 0)
                this_cam_zones = cam_zones.get(cam, {})

                if not this_cam_zones or cam_frame_count == 0:
                    continue

                cam_key = str(cam)

                for frame_index in range(cam_frame_count):
                    with lock:
                        frame = shared_frames[cam]["frame"]

                    if frame is None:
                        continue

                    results = detector(frame, verbose=False)
                    current_frame_detections = {}

                    for result in results:
                        for box in result.boxes:

                            if int(box.cls[0]) != 0:
                                continue

                            x1,y1,x2,y2 = map(int, box.xyxy[0])
                            crop = frame[y1:y2, x1:x2]
                            if crop.size == 0:
                                continue

                            # Extract both embeddings
                            body_emb = extract_body_embedding(crop)
                            face_emb = extract_face_embedding(crop)

                            for name, data in this_cam_zones.items():
                                if inside_zone((x1,y1,x2,y2), data["zone"]):
                                    if name not in employee_embeddings:
                                        continue
                                    if cam_key not in employee_embeddings[name]:
                                        continue

                                    stored_face, stored_body = get_stored_embeddings(name, cam_key)

                                    if not stored_body and not stored_face:
                                        continue

                                    # Get daily auto-updated body embeddings
                                    daily_body = daily_body_cache.get(name, {}).get(cam_key, [])

                                    # Compute weighted match score
                                    final_score, face_score, body_score, used_face, threshold, gap = \
                                        compute_match_score(body_emb, face_emb,
                                                           stored_face, stored_body,
                                                           daily_body)

                                    if final_score > threshold:
                                        cycle_results[name].append(True)

                                        # Display info
                                        score_type = "F+B" if used_face else "B"
                                        current_frame_detections[name] = (
                                            x1, y1, x2, y2, name,
                                            final_score, score_type
                                        )

                                        # Auto-update body embeddings
                                        # If face confidently identifies the person,
                                        # capture today's body appearance
                                        if used_face and face_score >= FACE_CONFIDENCE_FOR_AUTO_UPDATE:

                                            if name not in daily_body_cache:
                                                daily_body_cache[name] = {}

                                            if cam_key not in daily_body_cache[name]:
                                                daily_body_cache[name][cam_key] = []

                                            existing_embs = daily_body_cache[name][cam_key]

                                            # Only add if it is a new appearance
                                            if len(existing_embs) < MAX_DAILY_BODY_UPDATE and is_new_embedding(body_emb, existing_embs):
                                                existing_embs.append(body_emb)
                                                print(f"[AUTO UPDATE] Added new body embedding for {name} on Camera {cam_key}")

                    # Keep last frame detections for display
                    if frame_index == cam_frame_count - 1:
                        display_detections[cam] = current_frame_detections.copy()

            # Update state using voting
            for name in all_employee_names:
                employee_cams = get_cameras_for_employee(name)
                total_frames_for_employee = sum(
                    frames_per_cam.get(int(c), 0) for c in employee_cams if int(c) in caps
                )
                threshold = max(1, total_frames_for_employee // 2)
                detected = sum(cycle_results[name]) >= threshold
                update_state(name, detected)
            
            save_report_with_live_time()

            # Update display detections
            with lock:
                for cam in caps.keys():
                    last_detections[cam] = list(display_detections.get(cam, {}).values())

            elapsed = time.time() - start_time
            if elapsed < INTERVAL_SECONDS:
                time.sleep(INTERVAL_SECONDS - elapsed)

    threading.Thread(target=processor, daemon=True).start()

    print("\nLive Monitoring Started... Press Q to stop")
    print(f"Auto-updating body embeddings when face is confident (max {MAX_DAILY_BODY_UPDATE}/day per person)")

    # -------- DISPLAY LOOP --------
    while True:
        for cam in list(caps.keys()):
            with lock:
                frame = shared_frames[cam]["frame"]
                detections = last_detections[cam].copy() if cam in last_detections else []

            if frame is not None:
                display = frame.copy()

                # Draw detections (now includes score type)
                for detection in detections:
                    x1, y1, x2, y2, name, score, score_type = detection
                    color = employee_colors.get(name, (0, 255, 0))
                    cv2.rectangle(
                        display,
                        (x1, y1),
                        (x2, y2),
                        color,
                        1,
                        lineType=cv2.LINE_AA
                    )

                    cv2.putText(
                        display,
                        f"{name} {score:.2f}",
                        (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0,255,0),
                        1,
                        cv2.LINE_AA
                    )

                # Show camera label
                cv2.putText(display, f"Camera {cam}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

                cv2.imshow(f"Camera {cam}", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # -------- CLEANUP --------
    # Flush final durations into raw seconds
    current_time = time.time()

    for name, state_data in employee_states.items():
        duration = current_time - state_data["last_change"]

        if state_data["state"] == "PRESENT":
            report_data[name]["in_seat_seconds"] += duration
        else:
            report_data[name]["out_seat_seconds"] += duration

        # Update last_change so no double-counting if save_report_with_live_time is called
        state_data["last_change"] = current_time

    save_report()
    print("Final report saved.")

    # Show daily body cache stats
    if daily_body_cache:
        print("\nAuto-updated body embeddings today:")
        for name, cams in daily_body_cache.items():
            for cam_key, embs in cams.items():
                print(f"  {name} Camera {cam_key}: {len(embs)} new body embeddings")

    running = False

    # Wait for grabber threads to exit gracefully if possible
    for t in grabber_threads:
        t.join(timeout=0.2)

    for cap in caps.values():
        try:
            cap.release()
        except:
            pass
    cv2.destroyAllWindows()

# -------- CAMERA SELECT --------
if __name__ == "__main__":
    print(f"\n{'='*55}")
    print("  EMPLOYEE TRACKING - FACE + BODY MONITOR")
    print(f"{'='*55}")

    # Show which cameras have zones
    print("\nZone assignments:")
    for name, data in zones.items():
        if "cameras" in data:
            cams = list(data["cameras"].keys())
            print(f"  {name}: Camera(s) {', '.join(cams)}")

    print(f"\nScoring: {FACE_WEIGHT:.0%} Face + {BODY_WEIGHT:.0%} Body")
    print(f"Thresholds: Combined={MATCH_THRESHOLD_COMBINED}, Body-only={MATCH_THRESHOLD_BODY_ONLY}")
    print(f"\nEnter camera numbers to monitor (1-{MAX_CAMERAS})")
    print("Examples: '1' for single camera, '1,5' for cameras 1 and 5")

    while True:
        try:
            cam_input = input("\nSelect cameras (comma-separated) or 0 to exit: ").strip()

            if cam_input == "0":
                break

            active_cameras = [int(c.strip()) for c in cam_input.split(",")]
            active_cameras = [c for c in active_cameras if 1 <= c <= MAX_CAMERAS]

            if not active_cameras:
                print("No valid cameras selected. Try again.")
                continue

            seen = set()
            unique_cameras = []
            for c in active_cameras:
                if c not in seen:
                    seen.add(c)
                    unique_cameras.append(c)
            active_cameras = unique_cameras

            main(active_cameras)

        except Exception as e:
            print(f"Error: {e}")
            continue