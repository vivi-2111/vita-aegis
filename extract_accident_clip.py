
import cv2

video_path = "Road (3).mp4"
output_path = "accident_clip.mp4"

start_seconds = 15
end_seconds = 25

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open", video_path)
    raise SystemExit

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    print("ERROR: Could not read video FPS.")
    cap.release()
    raise SystemExit

cap.set(cv2.CAP_PROP_POS_MSEC, start_seconds * 1000)

writer = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

while cap.get(cv2.CAP_PROP_POS_MSEC) < end_seconds * 1000:
    success, frame = cap.read()

    if not success:
        break

    writer.write(frame)

cap.release()
writer.release()

print("Clip created:", output_path)
print("Clip covers approximately 00:00:15 to 00:00:25")
