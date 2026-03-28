# from ultralytics import YOLO
# import cv2
# USERNAME = "admin1"
# PASSWORD = "admin@123"
# NVR_IP = "192.168.88.2"
# RTSP_PORT = 554
# STREAM_TYPE = 0
# MAX_CAMERAS = 8
# model = YOLO("live_monitoring\croed_ofc_head_det.pt")
# import numpy as np

# def letterbox(im, new_shape=(640,480), color=(114,114,114)):
    
#     shape = im.shape[:2]  # current shape [height, width]

#     # scale ratio
#     r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

#     # new size without padding
#     new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))

#     # compute padding
#     dw = new_shape[1] - new_unpad[0]
#     dh = new_shape[0] - new_unpad[1]

#     dw /= 2
#     dh /= 2

#     # resize
#     im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

#     # add padding
#     top, bottom = int(round(dh-0.1)), int(round(dh+0.1))
#     left, right = int(round(dw-0.1)), int(round(dw+0.1))

#     im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

#     return im
# rtsp_url = (
#             f"rtsp://{USERNAME}:{PASSWORD}@{NVR_IP}:{RTSP_PORT}"
#             f"/cam/realmonitor?channel={5}&subtype={STREAM_TYPE}"
#         )
# cap = cv2.VideoCapture(rtsp_url)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = letterbox(frame,(640,640))

#     h,w = frame.shape[:2]
#     print(w,h)

#     results = model(frame, classes=[0], verbose=False)

#     for r in results:
#         boxes = r.boxes.xyxy.cpu().numpy()
#         total_detections = len(boxes)
#         print(total_detections)

#     annotated_frame = results[0].plot()

#     cv2.imshow("YOLO Detection", annotated_frame)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()


# ===========================================================
# code 2

import cv2
from ultralytics import YOLO

# Load model
model = YOLO("live_monitoring/croed_ofc_head_det.pt")   # change if you use another model

# Open video or webcam
cap = cv2.VideoCapture(r"C:\Users\valkontek 010\Downloads\gettyimages-1129218254-640_adpp.mp4")   # 0 = webcam OR put video path

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create VideoWriter to save output
out = cv2.VideoWriter(
    "out_detection2.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (frame_width, frame_height)
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detection
    results = model(frame, classes=[0]) 
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()  # get box coordinates
        total_detections = len(boxes)
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)

            # draw only bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 1)  # detect only person class
    cv2.putText(frame,
                f"Total people: {total_detections}",
                (20,40),                     # position
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2)
    # Draw bounding boxes
    # annotated_frame = results[0].plot()

    # Save frame to video
    out.write(frame)

    # Show video
    cv2.imshow("YOLO Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()





# import cv2
# from ultralytics import YOLO

# # Load model
# model = YOLO("crowd_head_detection.pt")

# # Open video
# cap = cv2.VideoCapture(r"C:\Users\valkontek 010\Downloads\medaram.mp4")

# # Video properties
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))
# fps = int(cap.get(cv2.CAP_PROP_FPS))

# # Output video
# out = cv2.VideoWriter(
#     "out_detection_tracking.mp4",
#     cv2.VideoWriter_fourcc(*"mp4v"),
#     fps,
#     (frame_width, frame_height)
# )

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # YOLO detection + ByteTrack
#     results = model.track(frame, persist=True, classes=[0], tracker="bytetrack.yaml")

#     total_people = 0

#     for r in results:

#         boxes = r.boxes

#         if boxes.id is not None:

#             ids = boxes.id.cpu().numpy()
#             coords = boxes.xyxy.cpu().numpy()

#             total_people = len(ids)

#             for box, track_id in zip(coords, ids):

#                 x1, y1, x2, y2 = map(int, box)

#                 # draw bounding box
#                 cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),1)

#                 # show tracking ID
#                 cv2.putText(frame,
#                             f"ID {int(track_id)}",
#                             (x1,y1-10),
#                             cv2.FONT_HERSHEY_SIMPLEX,
#                             0.5,
#                             (0,255,0),
#                             2)

#     # show total people
#     cv2.putText(frame,
#                 f"Total People: {total_people}",
#                 (20,40),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 1,
#                 (0,255,0),
#                 2)

#     # save frame
#     out.write(frame)

#     cv2.imshow("Crowd Tracking", frame)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# out.release()
# cv2.destroyAllWindows()