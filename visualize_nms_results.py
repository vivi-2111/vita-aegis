
import cv2
import pandas as pd
from pathlib import Path

VIDEO_FILE = "accident_clip.mp4"
CSV_FILE = "nms_detection_results.csv"
OUTPUT_DIR = Path("nms_visual_inspection")

OUTPUT_DIR.mkdir(exist_ok=True)

# Load video and NMS results
df = pd.read_csv(CSV_FILE)

cap = cv2.VideoCapture(VIDEO_FILE)

if not cap.isOpened():
    raise RuntimeError("Could not open accident_clip.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)

print("Video FPS:", fps)

# Process every frame that has NMS results
for frame_number, frame_data in df.groupby("frame"):

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))

    success, frame = cap.read()

    if not success:
        print("Could not read frame:", frame_number)
        continue

    # Draw retained bounding boxes
    for index, box in enumerate(
        frame_data.sort_values("confidence", ascending=False).to_dict("records"),
        start=1
    ):

        x1 = int(box["x1"])
        y1 = int(box["y1"])
        x2 = int(box["x2"])
        y2 = int(box["y2"])

        confidence = float(box["confidence"])

        # Different colours for retained detections
        color = (0, 255, 0) if index == 1 else (255, 165, 0)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            3
        )

        label = f"Box {index} | Conf: {confidence:.2f}"

        cv2.putText(
            frame,
            label,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2
        )

    timestamp = frame_number / fps

    cv2.putText(
        frame,
        f"Time: {timestamp:.3f}s | NMS boxes: {len(frame_data)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    output_file = OUTPUT_DIR / f"frame_{frame_number}.jpg"

    cv2.imwrite(str(output_file), frame)

    print(
        f"Saved {output_file} | "
        f"Time: {timestamp:.3f}s | "
        f"Boxes: {len(frame_data)}"
    )

cap.release()

print("\nVisualization completed.")
print("Images saved in:", OUTPUT_DIR)

