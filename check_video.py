
import cv2

video_path = "Road (4).mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Could not open the video.")
else:
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frames / fps if fps > 0 else 0

    print(f"Video: {video_path}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"FPS: {fps:.2f}")
    print(f"Total frames: {int(frames)}")

cap.release()
