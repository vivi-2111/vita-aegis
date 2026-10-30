
import cv2
import csv
from ultralytics import YOLO

VIDEO = "accident_clip.mp4"
OUTPUT = "motorcycle_rider_tracking_botsort.mp4"
CSV_OUTPUT = "crash_tracking_botsort.csv"

# Detect people and motorcycles
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise SystemExit(f"Cannot open video: {VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    cap.release()
    raise SystemExit("Could not read video FPS.")

writer = cv2.VideoWriter(
    OUTPUT,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

frame_no = 0
rows = []

while True:
    ok, frame = cap.read()
    if not ok:
        break

    result = model.track(
        frame,
        persist=True,
        tracker="botsort.yaml",
        classes=[0, 3],  # person, motorcycle
        conf=0.25,
        verbose=False
    )[0]

    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()

        ids = result.boxes.id
        ids = ids.int().cpu().tolist() if ids is not None else (
            [None] * len(boxes)
        )

        for box, cls, score, track_id in zip(
            boxes, classes, scores, ids
        ):
            x1, y1, x2, y2 = map(int, box)
            class_id = int(cls)
            label = model.names[class_id]

            if track_id is None:
                display = f"{label} {score:.2f} (untracked)"
            else:
                display = f"{label} ID:{track_id} {score:.2f}"

            color = (0, 255, 0) if class_id == 0 else (255, 150, 0)

            cv2.rectangle(
                frame, (x1, y1), (x2, y2), color, 2
            )
            cv2.putText(
                frame,
                display,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2
            )

            rows.append([
                frame_no / fps,
                frame_no,
                label,
                track_id if track_id is not None else "",
                round(float(score), 3),
                x1, y1, x2, y2,
                round((x1 + x2) / 2, 2),
                round((y1 + y2) / 2, 2),
                x2 - x1,
                y2 - y1
            ])

    cv2.putText(
        frame,
        f"Clip time: {frame_no / fps:.2f}s",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    writer.write(frame)
    frame_no += 1

cap.release()
writer.release()

with open(CSV_OUTPUT, "w", newline="") as file:
    output = csv.writer(file)
    output.writerow([
        "time_sec", "frame", "object", "track_id", "confidence",
        "x1", "y1", "x2", "y2", "center_x", "center_y",
        "box_width", "box_height"
    ])
    output.writerows(rows)

print("Tracking complete.")
print("Annotated video:", OUTPUT)
print("Tracking data:", CSV_OUTPUT)
print("Frames processed:", frame_no)
print("Detections recorded:", len(rows))

