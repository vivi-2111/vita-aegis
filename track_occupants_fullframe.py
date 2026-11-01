
import cv2
import csv
from ultralytics import YOLO

# ---------------- CONFIGURATION ----------------

VIDEO_PATH = "accident_clip.mp4"
MODEL_PATH = "yolov8n-pose.pt"

OUTPUT_VIDEO = "occupants_fullframe_tracking.mp4"
OUTPUT_CSV = "occupants_fullframe_tracking.csv"

START_TIME = 3.0
END_TIME = 7.5

CONFIDENCE = 0.15
IMAGE_SIZE = 640

# Previously selected accident ROI
ROI_X1 = 609
ROI_Y1 = 275
ROI_X2 = 872
ROI_Y2 = 426

# Expand ROI to avoid excluding partially visible occupants
PADDING = 60

# ------------------------------------------------

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Could not open input video.")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

start_frame = int(START_TIME * fps)
end_frame = int(END_TIME * fps)

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

records = []
frame_number = start_frame

print("Starting full-frame occupant tracking...")

while frame_number <= end_frame:

    success, frame = cap.read()

    if not success:
        break

    timestamp = frame_number / fps

    # Full-frame detection and tracking
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],
        conf=CONFIDENCE,
        imgsz=IMAGE_SIZE,
        verbose=False
    )

    # Expanded ROI
    rx1 = max(0, ROI_X1 - PADDING)
    ry1 = max(0, ROI_Y1 - PADDING)
    rx2 = min(width, ROI_X2 + PADDING)
    ry2 = min(height, ROI_Y2 + PADDING)

    # Draw ROI
    cv2.rectangle(
        frame,
        (rx1, ry1),
        (rx2, ry2),
        (255, 0, 0),
        2
    )

    result = results[0]

    if result.boxes is not None and len(result.boxes) > 0:

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        if result.boxes.id is not None:
            track_ids = result.boxes.id.int().cpu().tolist()
        else:
            track_ids = [None] * len(boxes)

        keypoints = None

        if result.keypoints is not None:
            keypoints = result.keypoints.data.cpu().numpy()

        for i, box in enumerate(boxes):

            x1, y1, x2, y2 = map(float, box)

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # Keep detections whose centers fall inside expanded ROI
            if not (
                rx1 <= center_x <= rx2
                and ry1 <= center_y <= ry2
            ):
                continue

            track_id = track_ids[i]
            confidence = float(confidences[i])

            label = (
                f"ID {track_id}"
                if track_id is not None
                else "Untracked"
            )

            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"{label} {confidence:.2f}",
                (int(x1), max(20, int(y1) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

            person_keypoints = (
                keypoints[i]
                if keypoints is not None
                else None
            )

            records.append({
                "time_sec": round(timestamp, 3),
                "frame": frame_number,
                "track_id": track_id,
                "confidence": round(confidence, 3),
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2)
            })

    cv2.putText(
        frame,
        f"Time: {timestamp:.2f}s",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    writer.write(frame)

    frame_number += 1

cap.release()
writer.release()

fieldnames = [
    "time_sec", "frame", "track_id", "confidence",
    "x1", "y1", "x2", "y2"
]

with open(OUTPUT_CSV, "w", newline="") as file:
    csv_writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )
    csv_writer.writeheader()
    csv_writer.writerows(records)

print("\nTracking completed.")
print("Output video:", OUTPUT_VIDEO)
print("Tracking CSV:", OUTPUT_CSV)
print("Total detection records:", len(records))

unique_ids = sorted({
    row["track_id"]
    for row in records
    if row["track_id"] is not None
})

print("Track IDs:", unique_ids)
