
import cv2
import pandas as pd
from ultralytics import YOLO
from pathlib import Path

VIDEO_PATH = "accident_clip.mp4"
MODEL_PATH = "yolov8n-pose.pt"

OUTPUT_DIR = Path("filtered_detection_inspection")
OUTPUT_DIR.mkdir(exist_ok=True)

TIMES = [4.80, 4.90, 5.00, 5.20, 5.40, 5.60, 5.633, 5.667]

# Selected accident region
ROI_X1 = 609
ROI_Y1 = 275
ROI_X2 = 872
ROI_Y2 = 426

# Padding around the accident region
PADDING = 40

CONFIDENCE = 0.15
IMAGE_SIZE = 960

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Cannot open accident_clip.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)

records = []

# Expanded accident region
rx1 = max(0, ROI_X1 - PADDING)
ry1 = max(0, ROI_Y1 - PADDING)
rx2 = ROI_X2 + PADDING
ry2 = ROI_Y2 + PADDING


def overlaps_roi(box):

    x1, y1, x2, y2 = box

    intersection_x1 = max(x1, rx1)
    intersection_y1 = max(y1, ry1)
    intersection_x2 = min(x2, rx2)
    intersection_y2 = min(y2, ry2)

    return (
        intersection_x2 > intersection_x1
        and intersection_y2 > intersection_y1
    )


for time_sec in TIMES:

    frame_number = round(time_sec * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    success, frame = cap.read()

    if not success:
        continue

    result = model.predict(
        frame,
        classes=[0],
        conf=CONFIDENCE,
        imgsz=IMAGE_SIZE,
        max_det=30,
        verbose=False
    )[0]

    annotated = frame.copy()

    # Draw selected accident region
    cv2.rectangle(
        annotated,
        (rx1, ry1),
        (rx2, ry2),
        (255, 200, 0),
        3
    )

    raw_count = 0
    filtered_count = 0

    if result.boxes is not None:

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        raw_count = len(boxes)

        for i, box in enumerate(boxes):

            x1, y1, x2, y2 = box.tolist()

            if not overlaps_roi(box):
                continue

            filtered_count += 1

            confidence = float(confidences[i])

            cv2.rectangle(
                annotated,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                3
            )

            cv2.putText(
                annotated,
                f"ROI Person {filtered_count}: {confidence:.2f}",
                (int(x1), max(20, int(y1) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

            records.append({
                "time_sec": time_sec,
                "frame": frame_number,
                "detection_number": filtered_count,
                "confidence": confidence,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            })

    cv2.putText(
        annotated,
        f"Raw: {raw_count} | ROI: {filtered_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    output_path = OUTPUT_DIR / f"frame_{frame_number}.jpg"

    cv2.imwrite(str(output_path), annotated)

    print(
        f"{time_sec:.3f}s | "
        f"Raw detections: {raw_count} | "
        f"ROI detections: {filtered_count}"
    )

cap.release()

pd.DataFrame(records).to_csv(
    OUTPUT_DIR / "filtered_results.csv",
    index=False
)

print("\nFiltered inspection completed.")
print("Images:", OUTPUT_DIR)
