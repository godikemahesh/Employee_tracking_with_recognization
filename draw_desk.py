import cv2
import json
import os
import urllib.parse

ZONE_FILE = "zones.json"

# RTSP CONFIG
USERNAME = "admin1"
PASSWORD = "admin@123"
NVR_IP = "192.168.88.2"
RTSP_PORT = 554
STREAM_TYPE = 0
MAX_CAMERAS = 8

zones = {}
drawing = False
editing = False
moving = False
selected_corner = None

start_point = None
end_point = None
rect = None

def load_zones():
    global zones
    if os.path.exists(ZONE_FILE):
        with open(ZONE_FILE, "r") as f:
            try:
                zones = json.load(f)
                print("Loaded existing zones.")
            except:
                print("Zones file corrupted. Starting fresh.")
                zones = {}
    else:
        zones = {}

def save_zones():
    with open(ZONE_FILE, "w") as f:
        json.dump(zones, f, indent=4)
    print("Zones saved successfully.")

def get_all_employee_names():
    """Get all unique employee names across all cameras."""
    return list(zones.keys())

def draw_existing_zones(frame, cam_number):
    """Draw only zones belonging to this camera."""
    cam_key = str(cam_number)
    for name, data in zones.items():
        if "cameras" in data and cam_key in data["cameras"]:
            x1, y1, x2, y2 = data["cameras"][cam_key]["zone"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1,lineType=cv2.LINE_AA)
            cv2.putText(frame, name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,lineType=cv2.LINE_AA)

def draw_edit_rect(frame):
    global rect
    if rect:
        x1, y1, x2, y2 = rect
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1,lineType=cv2.LINE_AA)

        # Draw corner points
        for (x, y) in [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]:
            cv2.circle(frame, (x, y), 6, (0,0,255), -1)

def get_corner(x, y):
    global rect
    if not rect:
        return None

    x1, y1, x2, y2 = rect
    corners = {
        "tl": (x1,y1),
        "tr": (x2,y1),
        "bl": (x1,y2),
        "br": (x2,y2)
    }

    for name, (cx, cy) in corners.items():
        if abs(x - cx) < 10 and abs(y - cy) < 10:
            return name

    return None

def inside_rect(x, y):
    global rect
    if not rect:
        return False
    x1,y1,x2,y2 = rect
    return x1 < x < x2 and y1 < y < y2

def mouse_callback(event, x, y, flags, param):
    global drawing, start_point, rect
    global editing, selected_corner, moving

    if event == cv2.EVENT_LBUTTONDOWN:

        if rect:
            corner = get_corner(x, y)
            if corner:
                editing = True
                selected_corner = corner
                return
            elif inside_rect(x, y):
                moving = True
                start_point = (x, y)
                return

        drawing = True
        start_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:

        if drawing:
            rect = [start_point[0], start_point[1], x, y]

        elif editing:
            x1,y1,x2,y2 = rect

            if selected_corner == "tl":
                rect[0], rect[1] = x, y
            elif selected_corner == "tr":
                rect[2], rect[1] = x, y
            elif selected_corner == "bl":
                rect[0], rect[3] = x, y
            elif selected_corner == "br":
                rect[2], rect[3] = x, y

        elif moving:
            dx = x - start_point[0]
            dy = y - start_point[1]
            rect = [rect[0]+dx, rect[1]+dy,
                    rect[2]+dx, rect[3]+dy]
            start_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        editing = False
        moving = False
        selected_corner = None

def main(rtsp_url, cam_number):
    global rect, zones

    load_zones()

    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print("Cannot open camera.")
        return

    cv2.namedWindow("Desk Zone Setup")
    cv2.setMouseCallback("Desk Zone Setup", mouse_callback)

    cam_key = str(cam_number)

    print(f"\n--- Camera {cam_number} Zone Setup ---")
    print("Instructions:")
    print("- Drag to draw rectangle")
    print("- Drag corners to resize")
    print("- Drag inside to move")
    print("- Press ENTER to confirm and assign employee name")
    print("- Press R to reset current box")
    print("- Press D to delete an existing zone for this camera")
    print("- Press Q to quit")

    # Show existing zones for this camera
    existing = []
    for name, data in zones.items():
        if "cameras" in data and cam_key in data["cameras"]:
            existing.append(name)
    if existing:
        print(f"\nExisting zones on Camera {cam_number}: {', '.join(existing)}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()

        draw_existing_zones(display, cam_number)
        draw_edit_rect(display)

        # Show camera number on frame
        cv2.putText(display, f"Camera {cam_number}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        cv2.imshow("Desk Zone Setup", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            rect = None

        elif key == 13:  # ENTER key
            if rect:
                x1,y1,x2,y2 = rect
                x1,x2 = min(x1,x2), max(x1,x2)
                y1,y2 = min(y1,y2), max(y1,y2)

                name = input("Enter employee name: ").strip().lower()
                if not name:
                    print("Name cannot be empty. Try again.")
                    continue

                # Check if this name already has a zone on THIS camera
                if name in zones and "cameras" in zones[name] and cam_key in zones[name]["cameras"]:
                    overwrite = input(f"'{name}' already has a zone on Camera {cam_number}. Overwrite? (y/n): ").strip().lower()
                    if overwrite != 'y':
                        print("Skipped.")
                        rect = None
                        continue

                # Create or update the employee entry
                if name not in zones:
                    zones[name] = {"cameras": {}}
                elif "cameras" not in zones[name]:
                    # Migrate old format if needed
                    zones[name] = {"cameras": {}}

                zones[name]["cameras"][cam_key] = {"zone": [x1, y1, x2, y2]}

                save_zones()
                print(f"Zone for '{name}' saved on Camera {cam_number}.")
                rect = None

        elif key == ord('d'):
            # Delete a zone for this camera
            name = input("Enter employee name to delete zone for this camera: ").strip().lower()
            if name in zones and "cameras" in zones[name] and cam_key in zones[name]["cameras"]:
                del zones[name]["cameras"][cam_key]
                # If no cameras left, remove the employee entirely
                if not zones[name]["cameras"]:
                    del zones[name]
                save_zones()
                print(f"Zone for '{name}' deleted from Camera {cam_number}.")
            else:
                print(f"No zone found for '{name}' on Camera {cam_number}.")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    while True:
        try:
            cam = int(input(f"\nSelect camera (1-{MAX_CAMERAS}) or 0 to exit: "))
        except:
            continue

        if cam == 0:
            break

        encoded_pass = urllib.parse.quote(PASSWORD)
        rtsp_url = (
            f"rtsp://{USERNAME}:{encoded_pass}@{NVR_IP}:{RTSP_PORT}"
            f"/cam/realmonitor?channel={cam}&subtype={STREAM_TYPE}"
        )

        print(f"Opening Camera {cam}...")
        print(rtsp_url)
        main(rtsp_url, cam)