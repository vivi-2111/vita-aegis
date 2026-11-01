
import cv2
from ultralytics import YOLO

VIDEO_PATH = "accident_clip.mp4"
MODEL_PATH = "yolov8n-pose.pt"

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

timestamps = [4.833, 4.900, 5.000, 5.333, 5.633, 5.667]

fps = cap.get(cv2.CAP_PROP_FPS)

for timestamp in timestamps:

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(timestamp * fps))
    success, frame = cap.read()

    if not success:
        print("Could not read:", timestamp)
        continue

    print(f"\nTime: {timestamp:.3f}s")

    for size in [640, 960, 1280]:

        results = model.predict(
            frame,
            conf=0.15,
            imgsz=size,
            verbose=False
        )

        result = results[0]

        count = (
            len(result.boxes)
            if result.boxes is not None
            else 0
        )

        print(f"Image size {size}: {count} detections")

        annotated = result.plot()

        output_name = (
            f"resolution_{int(timestamp * 1000)}_{size}.jpg"
        )

        cv2.imwrite(output_name, annotated)

cap.release()

print("\nResolution comparison completed.")
