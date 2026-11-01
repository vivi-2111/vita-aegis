
import cv2
from ultralytics import YOLO

VIDEO_PATH = "accident_clip.mp4"
MODEL_PATH = "yolov8n-pose.pt"

# Read selected ROI
with open("accident_roi.txt", "r") as file:
    x, y, w, h = map(int, file.read().split(","))

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)

timestamps = [3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]

for timestamp in timestamps:

    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    ret, frame = cap.read()

    if not ret:
        continue

    # Crop selected accident region
    roi = frame[y:y+h, x:x+w]

    results = model.predict(
        roi,
        conf=0.15,
        imgsz=640,
        verbose=False
    )

    result = results[0]
    count = len(result.boxes)

    annotated = result.plot()

    print(
        f"Time: {timestamp:.2f}s | "
        f"People detected inside ROI: {count}"
    )

    cv2.imshow("ROI Person Detection", annotated)

    key = cv2.waitKey(1500)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
