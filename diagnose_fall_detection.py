import cv2
from ultralytics import YOLO

VIDEO = "accident_clip.mp4"
OUTPUT = "fall_detection_diagnostic.mp4"

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise SystemExit("Cannot open accident_clip.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    OUTPUT, cv2.VideoWriter_fourcc(*"mp4v"),
    fps, (width, height)
)

frame_no = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # Detect people and motorcycles without assigning tracking IDs.
    result = model.predict(
        frame,
        classes=[0, 3],
        conf=0.10,
        verbose=False
    )[0]

    person_count = 0
    motorcycle_count = 0

    if result.boxes is not None:
        for box in result.boxes:
            cls = int(box.cls.item())
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            if cls == 0:
                label = f"Person {confidence:.2f}"
                color = (0, 255, 0)
                person_count += 1
            elif cls == 3:
                label = f"Motorcycle {confidence:.2f}"
                color = (255, 150, 0)
                motorcycle_count += 1
            else:
                continue

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, label, (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
            )

    clip_time = frame_no / fps
    cv2.putText(
        frame,
        f"Time: {clip_time:.2f}s | People: {person_count} | Motorcycles: {motorcycle_count}",
        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
        (255, 255, 255), 2
    )

    writer.write(frame)
    frame_no += 1

cap.release()
writer.release()

print("Diagnostic video saved:", OUTPUT)
print("Frames processed:", frame_no)
