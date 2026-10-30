
import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolov8n.pt")

# Open the road video
video = cv2.VideoCapture("Road (2).mp4")

if not video.isOpened():
    print("Error: Could not open road.mp4")
    exit()

# Vehicle class IDs
vehicle_classes = [2, 3, 5, 7]

print("Vehicle tracking started!")
print("Press Q to stop.")

while True:
    success, frame = video.read()

    if not success:
        break

    # Detect and track vehicles
    results = model.track(
        frame,
        persist=True,
        classes=vehicle_classes,
        conf=0.4,
        tracker="bytetrack.yaml",
        verbose=False
    )

    # Draw bounding boxes and tracking IDs
    output = results[0].plot()

    # Display tracking result
    cv2.imshow("Vita Aegis - Vehicle Tracking", output)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()

print("Tracking completed!")