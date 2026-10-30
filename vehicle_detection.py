
import cv2
from ultralytics import YOLO

# Load a lightweight YOLO model
model = YOLO("yolov8n.pt")

# Open the road video
video = cv2.VideoCapture("Road (2).mp4")

if not video.isOpened():
    print("Error: Could not open road.mp4")
    exit()

# COCO class IDs for vehicles:
# 2 = car, 3 = motorcycle, 5 = bus, 7 = truck
vehicle_classes = [2, 3, 5, 7]

print("Vehicle detection started!")
print("Press Q to stop.")

while True:
    success, frame = video.read()

    if not success:
        break

    # Detect vehicles in the current frame
    results = model(
        frame,
        classes=vehicle_classes,
        conf=0.4,
        verbose=False
    )

    # Draw bounding boxes and labels
    output = results[0].plot()

    # Display the result
    cv2.imshow("Vita Aegis - Vehicle Detection", output)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()

print("Detection completed!")