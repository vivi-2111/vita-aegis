
import cv2
from ultralytics import YOLO

VIDEO_PATH = "accident_clip.mp4"

# Original selected ROI
with open("accident_roi.txt", "r") as file:
    x, y, w, h = map(int, file.read().split(","))

# Add a 60-pixel margin around the selected region
margin = 60

rx1 = max(0, x - margin)
ry1 = max(0, y - margin)
rx2 = x + w + margin
ry2 = y + h + margin

model = YOLO("yolov8n-pose.pt")

cap = cv2.VideoCapture(VIDEO_PATH)

timestamps = [
    3.5, 4.0, 4.5, 5.0,
    5.5, 6.0, 6.5, 7.0
]

for timestamp in timestamps:

    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)

    ret, frame = cap.read()

    if not ret:
        continue

    # Detect on the ORIGINAL full-resolution frame
    result = model.predict(
        frame,
        conf=0.15,
        imgsz=640,
        verbose=False
    )[0]

    selected = []

    if result.boxes is not None:

        for box in result.boxes:

            confidence = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # Keep detections whose center is in expanded ROI
            if (
                rx1 <= center_x <= rx2
                and ry1 <= center_y <= ry2
            ):
                selected.append(
                    (x1, y1, x2, y2, confidence)
                )

    print(f"\nTime: {timestamp:.2f}s")
    print("Filtered people:", len(selected))

    for i, (x1, y1, x2, y2, conf) in enumerate(selected):

        print(
            f"Person {i+1}: "
            f"Confidence={conf:.3f}, "
            f"Box=({x1:.1f}, {y1:.1f}, "
            f"{x2:.1f}, {y2:.1f})"
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
            f"{conf:.2f}",
            (int(x1), max(20, int(y1) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # Draw expanded ROI
    cv2.rectangle(
        frame,
        (rx1, ry1),
        (rx2, ry2),
        (255, 0, 0),
        2
    )

    cv2.putText(
        frame,
        f"Time: {timestamp:.2f}s",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    cv2.imshow("Full Frame ROI Detection", frame)

    if cv2.waitKey(1000) == 27:
        break

cap.release()
cv2.destroyAllWindows()

print("\nFull-frame ROI detection completed.")
