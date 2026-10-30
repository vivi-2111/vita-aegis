
import cv2
import os

video_path = "accident_clip.mp4"
output_folder = "accident_frames"

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open accident_clip.mp4")
    raise SystemExit

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps if fps > 0 else 0

print(f"FPS: {fps}")
print(f"Duration: {duration:.2f} seconds")

timestamps = [0, 2, 4, 5, 6, 8, 10]

for seconds in timestamps:
    if seconds >= duration:
        continue

    frame_number = int(seconds * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    success, frame = cap.read()

    if success:
        filename = os.path.join(
            output_folder, f"frame_{seconds:02d}s.jpg"
        )
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")

cap.release()
print("Finished extracting screenshots.")
