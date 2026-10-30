
import cv2
import os

video = "motorcycle_rider_tracking.mp4"
output_folder = "tracking_inspection"

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video)

if not cap.isOpened():
    print("Could not open tracking video.")
    raise SystemExit

fps = cap.get(cv2.CAP_PROP_FPS)

for seconds in [3, 4, 5, 6, 7]:
    cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    success, frame = cap.read()

    if success:
        path = os.path.join(
            output_folder, f"tracking_{seconds}s.jpg"
        )
        cv2.imwrite(path, frame)
        print("Saved:", path)

cap.release()
print("Finished.")
