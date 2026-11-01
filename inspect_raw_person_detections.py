
import cv2
import pandas as pd
from ultralytics import YOLO
from pathlib import Path

VIDEO_PATH = "accident_clip.mp4"
MODEL_PATH = "yolov8n-pose.pt"

OUTPUT_DIR = Path("raw_detection_inspection")
OUTPUT_DIR.mkdir(exist_ok=True)

# Inspect the period before both occupants receive tracking IDs
TIMES = [4.80, 4.90, 5.00, 5.20, 5.40, 5.60, 5.633, 5.667]

CONFIDENCE = 0.10
IMAGE_SIZE = 1280

print("Loading YOLOv8 Pose model...")
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Cannot open accident_clip.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)

records = []

for time_sec in TIMES:

    frame_number = round(time_sec * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    success, frame = cap.read()

    if not success:
        print("Could not read frame:", frame_number)
        continue

    results = model.predict(
        frame,
        classes=[0],
        conf=CONFIDENCE,
        imgsz=IMAGE_SIZE,
        max_det=30,
        verbose=False
    )

    result = results[0]

    annotated = frame.copy()

    count = 0

    if result.boxes is not None:

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        count = len(boxes)

        for i, box in enumerate(boxes):

            x1, y1, x2, y2 = map(int, box)

            confidence = float(confidences[i])

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            label = f"Person {i + 1}: {confidence:.2f}"

            cv2.putText(
                annotated,
                label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            records.append({
                "time_sec": time_sec,
                "frame": frame_number,
                "detection_number": i + 1,
                "confidence": confidence,
                "x1": float(box[0]),
                "y1": float(box[1]),
                "x2": float(box[2]),
                "y2": float(box[3])
            })

    cv2.putText(
        annotated,
        f"Time: {time_sec:.3f}s | Detections: {count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2
    )

    output_path = OUTPUT_DIR / f"frame_{frame_number}.jpg"

    cv2.imwrite(str(output_path), annotated)

    print(
        f"{time_sec:.3f}s | "
        f"Frame {frame_number} | "
        f"Detections: {count} | "
        f"Saved: {output_path}"
    )

cap.release()

df = pd.DataFrame(records)

df.to_csv(
    OUTPUT_DIR / "raw_detection_results.csv",
    index=False
)

print("\nInspection completed.")
print("Images saved in:", OUTPUT_DIR)
print("CSV saved in:", OUTPUT_DIR / "raw_detection_results.csv")
