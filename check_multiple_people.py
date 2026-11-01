
import cv2
from ultralytics import YOLO

VIDEO = "accident_clip.mp4"
MODEL = "yolov8n-pose.pt"

START_TIME = 3.5
END_TIME = 7.0

model = YOLO(MODEL)

cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise RuntimeError("Cannot open accident video.")

fps = cap.get(cv2.CAP_PROP_FPS)

start_frame = int(START_TIME * fps)
end_frame = int(END_TIME * fps)

frame_number = 0

print("\nMULTI-PERSON DETECTION REPORT\n")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    if start_frame <= frame_number <= end_frame:

        results = model.predict(
            frame,
            conf=0.15,
            imgsz=640,
            verbose=False
        )

        result = results[0]

        if result.keypoints is not None:
            person_count = len(result.keypoints)

            print(
                f"Time: {frame_number/fps:.2f}s | "
                f"People detected: {person_count}"
            )

            annotated = result.plot()

            cv2.imshow(
                "Multiple Person Pose Detection",
                annotated
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        else:
            print(
                f"Time: {frame_number/fps:.2f}s | "
                "People detected: 0"
            )

    frame_number += 1

cap.release()
cv2.destroyAllWindows()

print("\nDetection inspection completed.")
