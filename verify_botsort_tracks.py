import cv2
import pandas as pd

VIDEO = "accident_clip.mp4"
CSV = "crash_tracking_botsort.csv"
OUTPUT = "accident_track_verification_botsort.mp4"

df = pd.read_csv(CSV)

# Find the person track IDs present around the fall.
person_rows = df[
    (df["object"] == "person") &
    (df["time_sec"].between(4.0, 6.5))
]

print("Person track IDs around the accident:")
print(
    person_rows.groupby("track_id")
    .agg(
        frames=("frame", "count"),
        first_time=("time_sec", "min"),
        last_time=("time_sec", "max")
    )
    .sort_values("frames", ascending=False)
    .head(15)
)

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

by_frame = {int(f): g for f, g in df.groupby("frame")}
frame_no = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    rows = by_frame.get(frame_no)

    if rows is not None:
        for _, row in rows.iterrows():
            if row["object"] not in ("person", "motorcycle"):
                continue

            x1, y1, x2, y2 = [
                int(row[c]) for c in ("x1", "y1", "x2", "y2")
            ]
            tid = row["track_id"]
            label = f'{row["object"]} ID:{tid}'

            color = (0, 255, 255) if row["object"] == "person" else (255, 150, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, label, (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
            )

    cv2.putText(
        frame, f"Clip time: {frame_no / fps:.2f}s",
        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
    )
    writer.write(frame)
    frame_no += 1

cap.release()
writer.release()

print("Verification video saved:", OUTPUT)
