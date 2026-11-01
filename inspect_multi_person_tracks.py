import cv2
import os

VIDEO_FILE = "multi_person_tracking.mp4"
OUTPUT_FOLDER = "multi_track_inspection"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_FILE)

if not cap.isOpened():
    print("Error: Cannot open tracking video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)

# Original accident_clip.mp4 timestamps
original_times = [4.93, 5.50, 5.67, 6.00, 6.50, 7.00]

# Tracking video starts at original timestamp 3.5 seconds
offset = 3.5

for original_time in original_times:

    video_time = original_time - offset
    frame_number = int(video_time * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    success, frame = cap.read()

    if success:
        filename = os.path.join(
            OUTPUT_FOLDER,
            f"original_{original_time:.2f}s.jpg"
        )

        cv2.imwrite(filename, frame)

        print("Saved:", filename)

cap.release()

print("\nInspection snapshots created successfully.")