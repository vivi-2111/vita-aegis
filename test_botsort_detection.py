import cv2
from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")

cap = cv2.VideoCapture("accident_clip.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)

# Inspect frames around the accident
for timestamp in [3.5, 4.0, 4.5, 5.0, 6.0, 7.0]:

    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)

    ret, frame = cap.read()

    if not ret:
        print("Could not read:", timestamp)
        continue

    results = model.predict(
        frame,
        conf=0.10,
        imgsz=640,
        verbose=False
    )

    result = results[0]

    print(
        f"Time: {timestamp}s | "
        f"People detected: {len(result.boxes)}"
    )

    annotated = result.plot()

    cv2.imshow("YOLO Person Detection", annotated)

    cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()