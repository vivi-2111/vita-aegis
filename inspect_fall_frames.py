
import cv2
import os

INPUT_VIDEO = "fall_sequence_analysis.mp4"
OUTPUT_VIDEO = "fall_inspection_clip.mp4"

# These are ORIGINAL video timestamps
ORIGINAL_START = 3.8
ORIGINAL_END = 5.0

# Annotated video starts at original timestamp 3.5
VIDEO_START_OFFSET = 3.5

START_TIME = ORIGINAL_START - VIDEO_START_OFFSET
END_TIME = ORIGINAL_END - VIDEO_START_OFFSET

cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():
    raise RuntimeError("Could not open annotated video.")

fps = cap.get(cv2.CAP_PROP_FPS)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

start_frame = int(START_TIME * fps)
end_frame = int(END_TIME * fps)

os.makedirs("inspection_frames", exist_ok=True)

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

frame_number = 0
saved_frames = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if start_frame <= frame_number <= end_frame:

        original_timestamp = (
            frame_number / fps + VIDEO_START_OFFSET
        )

        cv2.putText(
            frame,
            f"Original time: {original_timestamp:.2f}s",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        writer.write(frame)

        if frame_number % 3 == 0:
            filename = (
                f"inspection_frames/"
                f"frame_{original_timestamp:.2f}.jpg"
            )
            cv2.imwrite(filename, frame)
            saved_frames += 1

    frame_number += 1

cap.release()
writer.release()

print("Inspection completed.")
print("Output:", OUTPUT_VIDEO)
print("Frames saved:", saved_frames)
print("Original time range:", ORIGINAL_START, "-", ORIGINAL_END)
