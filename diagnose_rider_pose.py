import cv2
from ultralytics import YOLO

VIDEO = "accident_clip.mp4"
OUTPUT = "rider_fall_pose_diagnostic.mp4"

# Pretrained human-pose model; Ultralytics downloads it if needed.
model = YOLO("yolov8n-pose.pt")

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    raise SystemExit("Cannot open accident_clip.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    raise SystemExit("Could not read video FPS.")

# Approximate region around the rider previously observed.
# Adjust these coordinates if the rider is outside this crop.
x1, y1, x2, y2 = 450, 180, 850, 520

frame_no = 0
saved_frames = 0
writer = None

while True:
    ok, frame = cap.read()
    if not ok:
        break

    time_sec = frame_no / fps

    # Inspect only the accident window.
    if 3.5 <= time_sec <= 7.0:
        h, w = frame.shape[:2]
        xa, ya = max(0, x1), max(0, y1)
        xb, yb = min(w, x2), min(h, y2)
        crop = frame[ya:yb, xa:xb].copy()

        if crop.size:
            result = model.predict(
                crop,
                conf=0.15,
                imgsz=640,
                verbose=False
            )[0]

            # Draw pose keypoints and skeletons in the crop only.
            annotated = result.plot()

            if writer is None:
                ch, cw = annotated.shape[:2]
                writer = cv2.VideoWriter(
                    OUTPUT,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (cw, ch)
                )

            cv2.putText(
                annotated,
                f"Clip time: {time_sec:.2f}s",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            writer.write(annotated)
            saved_frames += 1

    frame_no += 1

cap.release()
if writer is not None:
    writer.release()

print("Pose diagnostic saved:", OUTPUT)
print("Frames analyzed:", saved_frames)
print("Time window: 3.5 to 7.0 seconds")
