
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO

VIDEO = "accident_clip.mp4"
MODEL = "yolov8n-pose.pt"

START_TIME = 3.5
END_TIME = 7.0

CONFIDENCE = 0.15
IMAGE_SIZE = 640

OUTPUT_VIDEO = "multi_person_tracking.mp4"
OUTPUT_CSV = "multi_person_tracking.csv"

print("Loading YOLOv8 Pose model...")

model = YOLO(MODEL)

cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise RuntimeError("Cannot open accident video.")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

start_frame = int(START_TIME * fps)
end_frame = int(END_TIME * fps)

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

records = []
frame_number = 0

print("\nStarting multi-person tracking...\n")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    if frame_number < start_frame:
        frame_number += 1
        continue

    if frame_number > end_frame:
        break

    timestamp = frame_number / fps

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=CONFIDENCE,
        imgsz=IMAGE_SIZE,
        verbose=False
    )

    result = results[0]

    annotated = result.plot()

    boxes = result.boxes
    keypoints = result.keypoints

    if boxes is not None and keypoints is not None:

        if boxes.id is not None:

            ids = boxes.id.cpu().numpy().astype(int)
            xyxy = boxes.xyxy.cpu().numpy()
            points = keypoints.data.cpu().numpy()

            for i, track_id in enumerate(ids):

                x1, y1, x2, y2 = xyxy[i]

                person_points = points[i]

                valid = (
                    len(person_points) >= 17
                    and np.all(
                        np.isfinite(person_points[:17, :2])
                    )
                )

                def get_point(index):
                    if valid:
                        return person_points[index]
                    return [np.nan, np.nan, np.nan]

                left_shoulder = get_point(5)
                right_shoulder = get_point(6)
                left_hip = get_point(11)
                right_hip = get_point(12)

                records.append({
                    "time_sec": timestamp,
                    "frame": frame_number,
                    "track_id": track_id,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "box_height": y2 - y1,
                    "left_shoulder_x": left_shoulder[0],
                    "left_shoulder_y": left_shoulder[1],
                    "right_shoulder_x": right_shoulder[0],
                    "right_shoulder_y": right_shoulder[1],
                    "left_hip_x": left_hip[0],
                    "left_hip_y": left_hip[1],
                    "right_hip_x": right_hip[0],
                    "right_hip_y": right_hip[1],
                    "keypoints_valid": int(valid)
                })

                cv2.putText(
                    annotated,
                    f"Person ID: {track_id}",
                    (int(x1), max(20, int(y1) - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

    cv2.putText(
        annotated,
        f"Time: {timestamp:.2f}s",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    writer.write(annotated)

    if frame_number % 15 == 0:
        print(f"Processed: {timestamp:.2f}s")

    frame_number += 1

cap.release()
writer.release()

df = pd.DataFrame(records)
df.to_csv(OUTPUT_CSV, index=False)

print("\n===================================")
print("MULTI-PERSON TRACKING COMPLETED")
print("===================================")
print("Video:", OUTPUT_VIDEO)
print("CSV:", OUTPUT_CSV)
print("Total tracking records:", len(df))

if not df.empty:
    print("\nUnique tracking IDs:")
    print(sorted(df["track_id"].unique()))

    print("\nRecords per tracking ID:")
    print(df["track_id"].value_counts().sort_index())

print("===================================")
