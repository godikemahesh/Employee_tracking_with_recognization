import cv2
import torch
import numpy as np
import json
import os
import threading
from ultralytics import YOLO
import torchreid
from torchvision import transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
import urllib.parse

# -------- SETTINGS --------
ZONE_FILE = "zones.json"
EMBEDDING_FILE = "embeddings.npy"
MAX_FACE_EMBEDDINGS = 10
MAX_BODY_EMBEDDINGS = 10
SIMILARITY_THRESHOLD = 0.7
# --------------------------
# RTSP CONFIG
USERNAME = "admin1"
PASSWORD = "admin@123"
NVR_IP = "192.168.88.2"
RTSP_PORT = 554
STREAM_TYPE = 0
MAX_CAMERAS = 8

# Load zones
with open(ZONE_FILE, "r") as f:
    zones = json.load(f)

# Load existing embeddings if exist
# New format: {name: {cam_key: {"face": [...], "body": [...]}}}
if os.path.exists(EMBEDDING_FILE):
    employee_embeddings = np.load(EMBEDDING_FILE, allow_pickle=True).item()
else:
    employee_embeddings = {}

# Load YOLO
detector = YOLO("person_det_v10.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Body ReID model (OSNet)
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

# Load Face Detection + Recognition models
face_detector = MTCNN(keep_all=False, device=device, min_face_size=40)
face_recognizer = InceptionResnetV1(pretrained='vggface2').eval().to(device)

print("All models loaded.")

# -------- EMBEDDING UTILS --------
def extract_body_embedding(img):
    img = body_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = reid_model(img)
    embedding = embedding.cpu().numpy().flatten()
    embedding = embedding / np.linalg.norm(embedding)
    return embedding

def extract_face_embedding(person_crop):
    """Detect face in person crop and extract face embedding.
    Returns (face_embedding, face_box) or (None, None) if no face found."""
    try:
        # Convert BGR to RGB for facenet
        rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)

        # Detect face
        boxes, probs = face_detector.detect(rgb_crop)

        if boxes is None or len(boxes) == 0:
            return None, None

        # Get the best face
        best_idx = probs.argmax()
        box = boxes[best_idx].astype(int)
        x1, y1, x2, y2 = box

        # Clamp to image bounds
        h, w = person_crop.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 - x1 < 10 or y2 - y1 < 10:
            return None, None

        face_crop = rgb_crop[y1:y2, x1:x2]
        face_crop = cv2.resize(face_crop, (160, 160))

        # To tensor
        face_tensor = torch.from_numpy(face_crop).permute(2, 0, 1).float() / 255.0
        face_tensor = face_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = face_recognizer(face_tensor)

        embedding = embedding.cpu().numpy().flatten()
        embedding = embedding / np.linalg.norm(embedding)
        return embedding, (x1, y1, x2, y2)

    except Exception:
        return None, None

def inside_zone(box, zone):
    x1, y1, x2, y2 = box
    zx1, zy1, zx2, zy2 = zone
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    return zx1 < cx < zx2 and zy1 < cy < zy2

def get_cameras_for_employee(name):
    """Get all camera numbers and zones for an employee."""
    if name not in zones or "cameras" not in zones[name]:
        return {}
    return zones[name]["cameras"]

def get_counts(name, cam_key):
    """Get face and body embedding counts for a name+camera."""
    if name not in employee_embeddings or cam_key not in employee_embeddings[name]:
        return 0, 0

    data = employee_embeddings[name][cam_key]

    # Handle old format (flat list = body only)
    if isinstance(data, list):
        return 0, len(data)

    face_count = len(data.get("face", []))
    body_count = len(data.get("body", []))
    return face_count, body_count

def is_cam_done(name, cam_key):
    """Check if both face and body are fully registered for a camera."""
    face_count, body_count = get_counts(name, cam_key)
    return face_count >= MAX_FACE_EMBEDDINGS and body_count >= MAX_BODY_EMBEDDINGS

def get_rtsp_url(cam_number):
    encoded_pass = urllib.parse.quote(PASSWORD)
    return (
        f"rtsp://{USERNAME}:{encoded_pass}@{NVR_IP}:{RTSP_PORT}"
        f"/cam/realmonitor?channel={cam_number}&subtype={STREAM_TYPE}"
    )

def init_embedding_structure(name, cam_key):
    """Ensure the embedding structure exists for a name+camera."""
    if name not in employee_embeddings:
        employee_embeddings[name] = {}

    if cam_key not in employee_embeddings[name]:
        employee_embeddings[name][cam_key] = {"face": [], "body": []}
    elif isinstance(employee_embeddings[name][cam_key], list):
        # Migrate old format (flat list) to new format
        old_body = employee_embeddings[name][cam_key]
        employee_embeddings[name][cam_key] = {"face": [], "body": old_body}
        print(f"  Migrated old embeddings for {name} Camera {cam_key}")

def main(target_name):
    """Register one employee across all their cameras simultaneously."""

    cam_data = get_cameras_for_employee(target_name)

    if not cam_data:
        print(f"'{target_name}' not found in zones.json.")
        return

    cam_numbers = list(cam_data.keys())
    print(f"\n  '{target_name}' has zones on Camera(s): {', '.join(cam_numbers)}")

    # Initialize embedding structures
    for cam_key in cam_numbers:
        init_embedding_structure(target_name, cam_key)

    # Check if all already done
    all_already_done = all(is_cam_done(target_name, ck) for ck in cam_numbers)

    if all_already_done:
        print(f"  '{target_name}' already fully registered on all cameras.")
        redo = input("  Reset and re-register all cameras? (y/n): ").strip().lower()
        if redo == 'y':
            for cam_key in cam_numbers:
                employee_embeddings[target_name][cam_key] = {"face": [], "body": []}
            print("  Embeddings cleared. Starting fresh.")
        else:
            return

    # Open all camera streams
    caps = {}
    for cam_key in cam_numbers:
        cam_num = int(cam_key)
        rtsp_url = get_rtsp_url(cam_num)
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            print(f"  Failed to open Camera {cam_key}. Skipping.")
            continue
        caps[cam_key] = cap
        print(f"  Camera {cam_key} opened.")

    if not caps:
        print("  No cameras could be opened.")
        return

    # Frame grabber threads
    lock = threading.Lock()
    shared_frames = {cam_key: None for cam_key in caps.keys()}
    stop_threads = False
    def frame_grabber(cam_key, cap):
        while not stop_threads:
            try:
                if not cap.isOpened():
                    print(f"Camera {cam_key}: Stream not opened. Reconnecting...")
                    cap.open(get_rtsp_url(int(cam_key)))
                    continue

                ret, frame = cap.read()

                if not ret or frame is None:
                    print(f"Camera {cam_key}: Frame read failed. Reconnecting...")
                    cap.release()
                    cap.open(get_rtsp_url(int(cam_key)))
                    continue

                with lock:
                    shared_frames[cam_key] = frame.copy()

            except Exception as e:
                print(f"Camera {cam_key}: Exception in frame grabber:", e)
                break

    for cam_key, cap in caps.items():
        t = threading.Thread(target=frame_grabber, args=(cam_key, cap), daemon=True)
        t.start()

    print(f"\n{'='*55}")
    print(f"  REGISTERING: {target_name.upper()}")
    for cam_key in caps.keys():
        fc, bc = get_counts(target_name, cam_key)
        print(f"  Camera {cam_key}: Face {fc}/{MAX_FACE_EMBEDDINGS} | Body {bc}/{MAX_BODY_EMBEDDINGS}")
    print(f"{'='*55}")
    print("  SPACE  = Capture from all cameras at once")
    print("  Q      = Quit and save")
    print("")
    print("  Tip: Face the camera directly for face capture")
    print("       Change angle between each SPACE press")
    print(f"{'='*55}\n")

    capture_flash = {cam_key: 0 for cam_key in caps.keys()}

    while True:
        # Check if all cameras are done
        all_done = all(is_cam_done(target_name, ck) for ck in caps.keys())
        if all_done:
            np.save(EMBEDDING_FILE, employee_embeddings)
            print(f"\n  {target_name} FULLY REGISTERED on all cameras! Auto-saved.")
            break

        # Display all camera feeds
        for cam_key in list(caps.keys()):
            with lock:
                frame = shared_frames[cam_key]

            if frame is None:
                continue

            display = frame.copy()
            face_count, body_count = get_counts(target_name, cam_key)
            zone = cam_data[cam_key]["zone"]
            zx1, zy1, zx2, zy2 = zone

            # Zone color
            done = is_cam_done(target_name, cam_key)
            zone_color = (0, 200, 255) if done else (0, 255, 0)
            cv2.rectangle(display, (zx1, zy1), (zx2, zy2), zone_color, 1,lineType=cv2.LINE_AA)
            cv2.putText(display, target_name, (zx1, zy1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, zone_color, 1,lineType=cv2.LINE_AA)

            # Camera label
            cv2.putText(display, f"Camera {cam_key} | {target_name}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Face progress bar (blue)
            bar_x, bar_y, bar_w, bar_h = 10, 60, 250, 20
            face_progress = min(face_count / MAX_FACE_EMBEDDINGS, 1.0)
            cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
            cv2.rectangle(display, (bar_x, bar_y), (bar_x + int(bar_w * face_progress), bar_y + bar_h), (255, 150, 0), -1)
            cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1)
            cv2.putText(display, f"Face: {face_count}/{MAX_FACE_EMBEDDINGS}",
                        (bar_x + bar_w + 10, bar_y + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 150, 0), 2)

            # Body progress bar (green)
            bar_y2 = bar_y + bar_h + 8
            body_progress = min(body_count / MAX_BODY_EMBEDDINGS, 1.0)
            cv2.rectangle(display, (bar_x, bar_y2), (bar_x + bar_w, bar_y2 + bar_h), (50, 50, 50), -1)
            cv2.rectangle(display, (bar_x, bar_y2), (bar_x + int(bar_w * body_progress), bar_y2 + bar_h), (0, 255, 0), -1)
            cv2.rectangle(display, (bar_x, bar_y2), (bar_x + bar_w, bar_y2 + bar_h), (255, 255, 255), 1)
            cv2.putText(display, f"Body: {body_count}/{MAX_BODY_EMBEDDINGS}",
                        (bar_x + bar_w + 10, bar_y2 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            # Status text
            status_y = bar_y2 + bar_h + 25
            if done:
                cv2.putText(display, "DONE!", (10, status_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            else:
                cv2.putText(display, "Press SPACE to capture", (10, status_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            # Flash effect
            if capture_flash[cam_key] > 0:
                overlay = display.copy()
                cv2.rectangle(overlay, (0, 0), (display.shape[1], display.shape[0]), (0, 255, 0), -1)
                alpha = capture_flash[cam_key] / 10.0
                display = cv2.addWeighted(overlay, alpha * 0.3, display, 1 - alpha * 0.3, 0)
                capture_flash[cam_key] -= 1

            cv2.imshow(f"Camera {cam_key} - {target_name}", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):  # SPACE — capture from ALL cameras at once
            captured_any = False

            for cam_key in list(caps.keys()):
                if is_cam_done(target_name, cam_key):
                    continue

                with lock:
                    frame = shared_frames[cam_key]

                if frame is None:
                    continue

                zone = cam_data[cam_key]["zone"]
                results = detector(frame, verbose=False)
                found = False

                for result in results:
                    for box in result.boxes:
                        if int(box.cls[0]) != 0:
                            continue

                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        if inside_zone((x1, y1, x2, y2), zone):
                            person_crop = frame[y1:y2, x1:x2]

                            if person_crop.size == 0:
                                continue

                            # Extract body embedding (always works)
                            body_emb = extract_body_embedding(person_crop)

                            fc, bc = get_counts(target_name, cam_key)
                            body_captured = False
                            face_captured = False

                            # Store body embedding if needed
                            if bc < MAX_BODY_EMBEDDINGS:
                                employee_embeddings[target_name][cam_key]["body"].append(body_emb)
                                body_captured = True

                            # Try to extract face embedding
                            face_emb, face_box = extract_face_embedding(person_crop)

                            if face_emb is not None and fc < MAX_FACE_EMBEDDINGS:
                                employee_embeddings[target_name][cam_key]["face"].append(face_emb)
                                face_captured = True

                            # Print status
                            fc_new, bc_new = get_counts(target_name, cam_key)
                            face_msg = f"Face {fc_new}/{MAX_FACE_EMBEDDINGS}"
                            body_msg = f"Body {bc_new}/{MAX_BODY_EMBEDDINGS}"
                            face_indicator = "✓" if face_captured else "✗ no face"
                            print(f"  Camera {cam_key}: {body_msg} | {face_msg} [{face_indicator}]")

                            capture_flash[cam_key] = 10
                            found = True
                            captured_any = True

                            if is_cam_done(target_name, cam_key):
                                print(f"  Camera {cam_key}: COMPLETE!")
                            np.save(EMBEDDING_FILE, employee_embeddings)
                            print("Embeddings auto-saved.")
                            break
                    if found:
                        break

                if not found:
                    print(f"  Camera {cam_key}: No person detected in zone.")

            if not captured_any:
                print("  No person detected in any zone. Make sure person is in the zone.")

        elif key == ord('q'):
            print("Saving embeddings before exit...")
            np.save(EMBEDDING_FILE, employee_embeddings)
            stop_threads = True
            break

    # Cleanup
    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()

    # Save on exit
    np.save(EMBEDDING_FILE, employee_embeddings)
    print(f"\nEmbeddings saved for {target_name}.")
    for cam_key in cam_numbers:
        fc, bc = get_counts(target_name, cam_key)
        print(f"  Camera {cam_key}: Face {fc}/{MAX_FACE_EMBEDDINGS} | Body {bc}/{MAX_BODY_EMBEDDINGS}")

# -------- MAIN LOOP --------
if __name__ == "__main__":
    print(f"\n{'='*55}")
    print("  EMPLOYEE EMBEDDING REGISTRATION (FACE + BODY)")
    print(f"{'='*55}")

    while True:
        # Show overall registration status
        print("\nRegistration status:")
        all_names = list(zones.keys())
        for name in all_names:
            if "cameras" in zones[name]:
                cams = list(zones[name]["cameras"].keys())
                statuses = []
                for cam_key in cams:
                    fc, bc = get_counts(name, cam_key)
                    face_s = "✓" if fc >= MAX_FACE_EMBEDDINGS else f"{fc}/{MAX_FACE_EMBEDDINGS}"
                    body_s = "✓" if bc >= MAX_BODY_EMBEDDINGS else f"{bc}/{MAX_BODY_EMBEDDINGS}"
                    statuses.append(f"Cam {cam_key}: F:{face_s} B:{body_s}")
                print(f"  {name} — {', '.join(statuses)}")

        target_name = input("\nEnter employee name (or 'exit' to quit): ").strip().lower()

        if target_name == 'exit':
            print("exited!")
            break
        
        if target_name not in zones:
            print(f"'{target_name}' not found in zones.json.")
            print(f"Available names: {', '.join(zones.keys())}")
            continue
        
        main(target_name)