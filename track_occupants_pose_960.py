
import cv2
import csv
from ultralytics import YOLO

# ==========================================
# VITA AEGIS - HIGH RESOLUTION POSE TRACKING
# ==========================================

VIDEO_PATH = "accident_clip.mp4"
MODEL_PATH = "yolov8n-pose.pt"

OUTPUT_VIDEO = "occupants_pose_960.mp4"
OUTPUT_CSV = "occupants_pose_960.csv"

START_TIME = 3.0
END_TIME = 7.5

CONFIDENCE = 0.15
IMAGE_SIZE = 960

# Original accident ROI
ROI_X1 = 609
ROI_Y1 = 275
ROI_X2 = 872
ROI_Y2 = 426

PADDING = 60

# COCO pose skeleton connections
SKELETON = [
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (0, 1), (0, 2),
    (1, 3), (2, 4)
]

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle"
]

# ==========================================
# INITIALIZATION
# ==========================================

print("Loading YOLOv8 Pose model...")

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Could not open accident video.")

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

if not writer.isOpened():
    raise RuntimeError("Could not create output video.")

# Expanded accident region
rx1 = max(0, ROI_X1 - PADDING)
ry1 = max(0, ROI_Y1 - PADDING)
rx2 = min(width, ROI_X2 + PADDING)
ry2 = min(height, ROI_Y2 + PADDING)

records = []
frame_number = start_frame

print("Starting high-resolution tracking...")

# ==========================================
# PROCESS VIDEO
# ==========================================

while frame_number <= end_frame:

    success, frame = cap.read()

    if not success:
        break

    timestamp = frame_number / fps

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],
        conf=CONFIDENCE,
        imgsz=IMAGE_SIZE,
        verbose=False
    )

    result = results[0]

    annotated = frame.copy()

    # Draw expanded accident ROI
    cv2.rectangle(
        annotated,
        (rx1, ry1),
        (rx2, ry2),
        (255, 0, 0),
        2
    )

    selected_count = 0

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

            # Filter detections by expanded accident ROI
            if not (
                rx1 <= center_x <= rx2
                and ry1 <= center_y <= ry2
            ):
                continue

            selected_count += 1

            confidence = float(confidences[i])
            track_id = track_ids[i]

            label = (
                f"ID {track_id}"
                if track_id is not None
                else "Untracked"
            )

            # Draw bounding box
            cv2.rectangle(
                annotated,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )

            cv2.putText(
                annotated,
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

            # Save pose keypoints
            keypoint_data = {}

            if person_keypoints is not None:

                for j, point in enumerate(person_keypoints):

                    px, py, pc = map(float, point)

                    name = KEYPOINT_NAMES[j]

                    keypoint_data[f"{name}_x"] = round(px, 2)
                    keypoint_data[f"{name}_y"] = round(py, 2)
                    keypoint_data[f"{name}_conf"] = round(pc, 3)

                    # Draw visible keypoints
                    if pc >= 0.25:

                        cv2.circle(
                            annotated,
                            (int(px), int(py)),
                            3,
                            (0, 0, 255),
                            -1
                        )

                # Draw skeleton connections
                for start, end in SKELETON:

                    if (
                        person_keypoints[start][2] >= 0.25
                        and person_keypoints[end][2] >= 0.25
                    ):

                        p1 = tuple(
                            person_keypoints[start][:2].astype(int)
                        )

                        p2 = tuple(
                            person_keypoints[end][:2].astype(int)
                        )

                        cv2.line(
                            annotated,
                            p1,
                            p2,
                            (0, 255, 255),
                            2
                        )

            else:
                keypoint_data = {
                    f"{name}_{axis}": ""
                    for name in KEYPOINT_NAMES
                    for axis in ["x", "y", "conf"]
                }

            record = {
                "time_sec": round(timestamp, 3),
                "frame": frame_number,
                "track_id": track_id if track_id is not None else "",
                "confidence": round(confidence, 3),
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2)
            }

            record.update(keypoint_data)

            records.append(record)

    # Display information
    cv2.putText(
        annotated,
        f"Time: {timestamp:.2f}s",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        annotated,
        f"ROI detections: {selected_count}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    writer.write(annotated)

    frame_number += 1

# ==========================================
# SAVE RESULTS
# ==========================================

cap.release()
writer.release()

fieldnames = [
    "time_sec", "frame", "track_id", "confidence",
    "x1", "y1", "x2", "y2"
]

for name in KEYPOINT_NAMES:
    fieldnames.extend([
        f"{name}_x",
        f"{name}_y",
        f"{name}_conf"
    ])

with open(OUTPUT_CSV, "w", newline="") as file:

    csv_writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    csv_writer.writeheader()
    csv_writer.writerows(records)

unique_ids = sorted({
    row["track_id"]
    for row in records
    if row["track_id"] != ""
})

print("\n================================")
print("HIGH-RESOLUTION TRACKING DONE")
print("================================")

print("Output video:", OUTPUT_VIDEO)
print("Output CSV:", OUTPUT_CSV)
print("Total detection records:", len(records))
print("Track IDs:", unique_ids)

print("\nTracking records per ID:")

for track_id in unique_ids:

    count = sum(
        1 for row in records
        if row["track_id"] == track_id
    )

    print(f"ID {track_id}: {count} records")

print("\nPose keypoint data saved successfully.")
