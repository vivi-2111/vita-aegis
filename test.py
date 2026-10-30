
import cv2
from ultralytics import YOLO

VIDEO_PATH = "Road (3).mp4"
MODEL_PATH = "yolov8n.pt"

print("Loading YOLO model...")
model = YOLO(MODEL_PATH)

print("Opening CCTV video...")
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open the video.")
    raise SystemExit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video FPS: {fps:.2f}")
print(f"Total frames: {total_frames}")
print(f"Duration: {total_frames / fps:.2f} seconds" if fps > 0 else "Duration unavailable")

# Test one frame first
success, frame = cap.read()

if not success:
    print("ERROR: Could not read the first frame.")
    cap.release()
    raise SystemExit(1)

print("Running vehicle detection...")
results = model.predict(frame, imgsz=640, conf=0.25, device="cpu", verbose=False)

annotated_frame = results[0].plot()
cv2.imwrite("test_detection.jpg", annotated_frame)

print("SUCCESS: Detection test completed.")
print("Saved annotated frame as test_detection.jpg")

cap.release()
