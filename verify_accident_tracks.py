
import cv2
import pandas as pd

VIDEO = "accident_clip.mp4"
CSV = "crash_tracking.csv"
OUTPUT = "accident_track_verification.mp4"

df = pd.read_csv(CSV)

cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise SystemExit("Cannot open accident_clip.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    OUTPUT,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

# Highlight the suspected rider and motorcycle candidates.
targets = {
    ("person", 55),
    ("motorcycle", 9),
    ("motorcycle", 277),
}

by_frame = {
    frame: group
    for frame, group in df.groupby("frame")
}

frame_number = 0

while True:
    ok, image = cap.read()

    if not ok:
        break

    rows = by_frame.get(frame_number)

    if rows is not None:
        for _, row in rows.iterrows():
            key = (row["object"], int(row["track_id"]))

            if key not in targets:
                continue

            x1 = int(row["x1"])
            y1 = int(row["y1"])
            x2 = int(row["x2"])
            y2 = int(row["y2"])

            label = f"{key[0]} ID {key[1]}"

            cv2.rectangle(
                image, (x1, y1), (x2, y2),
                (0, 255, 255), 2
            )

            cv2.putText(
                image, label, (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 255), 2
            )

    cv2.putText(
        image,
        f"Time: {frame_number / fps:.2f}s",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (255, 255, 255), 2
    )

    writer.write(image)
    frame_number += 1

cap.release()
writer.release()

print("Verification video saved:", OUTPUT)
print("Open the video and inspect the fall around 4–6 seconds.")
