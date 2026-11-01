
import cv2
from ultralytics import YOLO

with open("accident_roi.txt", "r") as file:
    x, y, w, h = map(int, file.read().split(","))

model = YOLO("yolov8n-pose.pt")

cap = cv2.VideoCapture("accident_clip.mp4")

timestamps = [
    3.5, 4.0, 4.5, 5.0,
    5.5, 6.0, 6.5, 7.0
]

for timestamp in timestamps:

    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)

    ret, frame = cap.read()

    if not ret:
        continue

    roi = frame[y:y+h, x:x+w]

    result = model.predict(
        roi,
        conf=0.05,
        imgsz=640,
        verbose=False
    )[0]

    print(f"\n========== {timestamp:.2f}s ==========")

    if result.boxes is None or len(result.boxes) == 0:
        print("No detections")
        continue

    for i, box in enumerate(result.boxes):

        confidence = float(box.conf[0])
        class_id = int(box.cls[0])

        class_name = model.names[class_id]

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        print(
            f"Detection {i+1}: "
            f"Class={class_name}, "
            f"Confidence={confidence:.3f}, "
            f"Box=({x1:.1f}, {y1:.1f}, "
            f"{x2:.1f}, {y2:.1f})"
        )

    cv2.imshow(
        "ROI Confidence Inspection",
        result.plot()
    )

    if cv2.waitKey(1000) == 27:
        break

cap.release()
cv2.destroyAllWindows()

print("\nConfidence inspection completed.")
