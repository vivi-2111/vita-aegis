import cv2
import pandas as pd

VIDEO = "accident_clip.mp4"
CSV = "crash_tracking_botsort.csv"
OUTPUT = "falling_person_id53.mp4"
TARGET_ID = 53

df = pd.read_csv(CSV)
df = df[
    (df["object"] == "person") &
    (df["track_id"] == TARGET_ID) &
    (df["time_sec"].between(3.0, 7.5))
]

by_frame = {int(f): g for f, g in df.groupby("frame")}

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    raise SystemExit("Cannot open accident_clip.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    OUTPUT, cv2.VideoWriter_fourcc(*"mp4v"),
    fps, (width, height)
)

frame_no = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    for _, row in by_frame.get(frame_no, pd.DataFrame()).iterrows():
        x1, y1, x2, y2 = [
            int(row[c]) for c in ("x1", "y1", "x2", "y2")
        ]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(
            frame, "PERSON ID 53",
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
        )

    cv2.putText(
        frame, f"Time: {frame_no / fps:.2f}s",
        (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (255, 255, 255), 2
    )

    writer.write(frame)
    frame_no += 1

cap.release()
writer.release()

print("Saved:", OUTPUT)
